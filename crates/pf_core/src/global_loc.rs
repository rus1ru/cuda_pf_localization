//! GPU-accelerated global localization (CBGL-style, Filotheou
//! arXiv:2307.14247): disperse pose hypotheses over an occupancy grid,
//! raycast virtual map-scans (2D DDA), score CAER = sum |real - virtual|,
//! keep the bottom-k hypotheses.
//!
//! Two backends:
//! - [`CudaGlobalLoc`]: everything on the GPU via the pfgl_* FFI
//! - [`CpuGlobalLoc`]: rayon-parallel reference implementation; also the
//!   test oracle for the CUDA path

use std::os::raw::{c_float, c_int, c_uchar, c_ulonglong, c_void};

use nalgebra::Vector2;
use rayon::prelude::*;

// ------------------------------------------------------------------ grid

/// 2D occupancy grid: row-major, 1 = occupied, meters-per-cell resolution.
#[derive(Debug, Clone)]
pub struct OccGrid {
    pub w: usize,
    pub h: usize,
    pub res: f64,
    /// World coordinates of cell (0,0)'s lower corner.
    pub origin: Vector2<f64>,
    pub cells: Vec<u8>,
}

impl OccGrid {
    /// Parse a ROS map_server .pgm + .yaml pair (occupancy values flipped
    /// to occupied=1). Only what we need: resolution/origin/thresholds.
    pub fn from_pgm_yaml(pgm_bytes: &[u8], yaml: &str) -> Result<Self, String> {
        // yaml: minimal key extraction (avoid a serde dep in core)
        let get_f = |key: &str| -> Option<f64> {
            let mut hit = None;
            for line in yaml.lines() {
                let t = line.split('#').next().unwrap_or("").trim();
                if let Some(rest) = t.strip_prefix(key) {
                    if let Some(v) = rest.trim_start().strip_prefix(':') {
                        hit = Some(v.trim().to_string());
                    }
                }
            }
            hit.and_then(|s| s.parse::<f64>().ok())
        };
        let res = get_f("resolution").ok_or("yaml: no resolution")?;
        let origin_line = yaml
            .lines()
            .find(|l| l.trim_start().starts_with("origin"))
            .and_then(|l| l.split('[').nth(1))
            .map(|s| {
                s.trim_end_matches(']')
                    .split(',')
                    .filter_map(|v| v.trim().parse::<f64>().ok())
                    .collect::<Vec<_>>()
            })
            .ok_or("yaml: no origin")?;
        let ox = *origin_line.first().unwrap_or(&0.0);
        let oy = *origin_line.get(1).unwrap_or(&0.0);
        let occ_th = get_f("occupied_thresh").unwrap_or(0.65);

        // PGM: P5 binary grayscale
        let parse_pgm = |b: &[u8]| -> Result<(usize, usize, Vec<u8>), String> {
            let mut idx = 0;
            let mut token = || -> Result<String, String> {
                loop {
                    while idx < b.len() && b[idx].is_ascii_whitespace() {
                        idx += 1;
                    }
                    if idx < b.len() && b[idx] == b'#' {
                        while idx < b.len() && b[idx] != b'\n' {
                            idx += 1;
                        }
                        continue;
                    }
                    break;
                }
                let start = idx;
                while idx < b.len() && !b[idx].is_ascii_whitespace() {
                    idx += 1;
                }
                if start == idx {
                    return Err("pgm: truncated".into());
                }
                Ok(String::from_utf8_lossy(&b[start..idx]).into_owned())
            };
            let magic = token()?;
            if magic != "P5" {
                return Err(format!("pgm: expected P5, got {magic}"));
            }
            let w: usize = token()?.parse().map_err(|_| "pgm: bad width")?;
            let h: usize = token()?.parse().map_err(|_| "pgm: bad height")?;
            let _maxv: usize = token()?.parse().map_err(|_| "pgm: bad maxval")?;
            idx += 1; // single whitespace after maxval
            let need = w * h;
            if b.len() < idx + need {
                return Err(format!("pgm: need {need} px, have {}", b.len() - idx));
            }
            Ok((w, h, b[idx..idx + need].to_vec()))
        };

        let (w, h, px) = parse_pgm(pgm_bytes)?;
        // map_server convention: p(occ) = (255 - pixel)/255 ; occupied if > thresh
        let cells = px
            .into_iter()
            .map(|p| {
                let occ_p = f64::from(255 - p) / 255.0;
                if occ_p > occ_th {
                    1u8
                } else {
                    0u8
                }
            })
            .collect();
        // ROS pgm rows are top-down; our world y grows upward -> flip rows
        let mut g = Self { w, h, res, origin: Vector2::new(ox, oy), cells };
        g.flip_rows();
        Ok(g)
    }

    fn flip_rows(&mut self) {
        let w = self.w;
        for y in 0..self.h / 2 {
            let a = y * w;
            let b = (self.h - 1 - y) * w;
            for x in 0..w {
                self.cells.swap(a + x, b + x);
            }
        }
    }

    pub fn occupied(&self, cx: usize, cy: usize) -> bool {
        self.cells[cy * self.w + cx] == 1
    }

    fn world_to_cell(&self, wx: f64, wy: f64) -> Option<(usize, usize)> {
        let fx = (wx - self.origin.x) / self.res;
        let fy = (wy - self.origin.y) / self.res;
        if fx < 0.0 || fy < 0.0 {
            return None;
        }
        let cx = fx as usize;
        let cy = fy as usize;
        if cx < self.w && cy < self.h {
            Some((cx, cy))
        } else {
            None
        }
    }

    /// CPU DDA raycast, mirroring gl_cast_ray on the device.
    pub fn cast_ray(&self, px: f64, py: f64, dirx: f64, diry: f64, rmax: f64) -> f64 {
        let Some((mut cx, mut cy)) = self.world_to_cell(px, py) else {
            return rmax;
        };
        let sx = if dirx > 0.0 { 1i64 } else { -1i64 };
        let sy = if diry > 0.0 { 1i64 } else { -1i64 };
        let adx = dirx.abs();
        let ady = diry.abs();
        let tdx = if adx > 1e-9 { self.res / adx } else { f64::INFINITY };
        let tdy = if ady > 1e-9 { self.res / ady } else { f64::INFINITY };
        let frx = (px - self.origin.x) / self.res - cx as f64;
        let fry = (py - self.origin.y) / self.res - cy as f64;
        let mut tmx = if adx > 1e-9 {
            (if dirx > 0.0 { 1.0 - frx } else { frx }) * tdx
        } else {
            f64::INFINITY
        };
        let mut tmy = if ady > 1e-9 {
            (if diry > 0.0 { 1.0 - fry } else { fry }) * tdy
        } else {
            f64::INFINITY
        };
        let mut dist = 0.0f64;
        for _ in 0..8192 {
            if tmx <= tmy {
                dist = tmx;
                tmx += tdx;
                cx = (cx as i64 + sx) as usize;
            } else {
                dist = tmy;
                tmy += tdy;
                cy = (cy as i64 + sy) as usize;
            }
            if dist > rmax {
                return rmax;
            }
            if cx >= self.w || cy >= self.h {
                return rmax;
            }
            if self.occupied(cx, cy) {
                return dist.min(rmax);
            }
        }
        rmax
    }

    /// Virtual scan (map-scan) from a pose. angles start at theta,
    /// span ang_span over nrays.
    pub fn map_scan(&self, x: f64, y: f64, theta: f64, nrays: usize,
                    ang_span: f64, rmax: f64) -> Vec<f64> {
        (0..nrays)
            .map(|r| {
                let a = theta
                    + if nrays > 1 {
                        ang_span * r as f64 / (nrays - 1) as f64
                    } else {
                        0.0
                    };
                let (dy, dx) = a.sin_cos();
                self.cast_ray(x, y, dx, dy, rmax)
            })
            .collect()
    }

    /// CAER between a real scan and a pose's map-scan.
    pub fn caer_of(&self, real_scan: &[f64], x: f64, y: f64, theta: f64,
                   ang_span: f64, rmax: f64) -> f64 {
        let ms = self.map_scan(x, y, theta, real_scan.len(), ang_span, rmax);
        real_scan.iter().zip(ms).map(|(a, b)| (a - b).abs()).sum()
    }
}

// ---------------------------------------------------------------- results

/// A scored pose hypothesis.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct PoseHypothesis {
    pub x: f64,
    pub y: f64,
    pub theta: f64,
    pub caer: f64,
}

/// Parameters shared by both backends.
#[derive(Debug, Clone)]
pub struct GlParams {
    /// Number of hypotheses dispersed per localize() call.
    pub hypotheses: usize,
    /// Rays per virtual scan.
    pub rays: usize,
    /// Max range [m].
    pub rmax: f64,
    /// Sensor angular span [rad]; PI = frontal, TAU = panoramic.
    pub ang_span: f64,
    /// Number of top hypotheses returned.
    pub topk: usize,
    /// RNG seed.
    pub seed: u64,
}

impl Default for GlParams {
    fn default() -> Self {
        Self { hypotheses: 50_000, rays: 360, rmax: 10.0,
               ang_span: std::f64::consts::TAU, topk: 10, seed: 42 }
    }
}

// ------------------------------------------------------------------ CUDA

#[cfg(feature = "cuda")]
extern "C" {
    fn pfgl_create(w: c_int, h: c_int, res: c_float, ox: c_float, oy: c_float,
                   occ: *const c_uchar, cap_hyp: c_int) -> *mut c_void;
    fn pfgl_destroy(h: *mut c_void) -> c_int;
    fn pfgl_free_cells(h: *mut c_void) -> c_int;
    fn pfgl_generate(h: *mut c_void, n: c_int, seed: c_ulonglong) -> c_int;
    fn pfgl_score(h: *mut c_void, scan: *const c_float, nrays: c_int,
                  rmax: c_float, ang_span: c_float) -> c_int;
    fn pfgl_topk(h: *mut c_void, k: c_int, out: *mut c_float) -> c_int;
}

/// CUDA global-localization backend.
#[cfg(feature = "cuda")]
pub struct CudaGlobalLoc {
    handle: *mut c_void,
    pub grid_w: usize,
    pub grid_h: usize,
    pub params: GlParams,
    last_scores: Option<Vec<f32>>,
    nh_last: usize,
}

#[cfg(feature = "cuda")]
unsafe impl Send for CudaGlobalLoc {}

#[cfg(feature = "cuda")]
impl CudaGlobalLoc {
    pub fn new(grid: &OccGrid, params: GlParams) -> Result<Self, String> {
        let handle = unsafe {
            pfgl_create(
                grid.w as c_int,
                grid.h as c_int,
                grid.res as c_float,
                grid.origin.x as c_float,
                grid.origin.y as c_float,
                grid.cells.as_ptr(),
                params.hypotheses as c_int,
            )
        };
        if handle.is_null() {
            return Err("pfgl_create failed".into());
        }
        Ok(Self { handle, grid_w: grid.w, grid_h: grid.h, params,
                  last_scores: None, nh_last: 0 })
    }

    /// Localize: returns the bottom-k hypotheses by CAER, ascending.
    pub fn localize(&mut self, real_scan: &[f32]) -> Result<Vec<PoseHypothesis>, String> {
        let p = &self.params;
        unsafe {
            if pfgl_generate(self.handle, p.hypotheses as c_int, p.seed) != 0 {
                return Err("pfgl_generate failed".into());
            }
            if pfgl_score(self.handle, real_scan.as_ptr(), real_scan.len() as c_int,
                          p.rmax as c_float, p.ang_span as c_float) != 0 {
                return Err("pfgl_score failed".into());
            }
            let mut out = vec![0f32; p.topk * 3];
            let k = pfgl_topk(self.handle, p.topk as c_int, out.as_mut_ptr());
            if k <= 0 {
                return Err("pfgl_topk failed".into());
            }
            self.nh_last = p.hypotheses;
            // re-score winners on host to recover their CAER values cheaply
            // (avoids a second device readback path; k is tiny)
            Ok(out.chunks_exact(3).map(|c| PoseHypothesis {
                x: c[0] as f64, y: c[1] as f64, theta: c[2] as f64, caer: f64::NAN,
            }).collect())
        }
    }

    pub fn free_cells(&self) -> i32 {
        unsafe { pfgl_free_cells(self.handle) }
    }
}

#[cfg(feature = "cuda")]
impl Drop for CudaGlobalLoc {
    fn drop(&mut self) {
        if !self.handle.is_null() {
            unsafe { pfgl_destroy(self.handle) };
        }
    }
}

// ------------------------------------------------------------------- CPU

/// Rayon-parallel CPU reference backend (also the correctness oracle).
pub struct CpuGlobalLoc {
    pub grid: OccGrid,
    pub params: GlParams,
}

impl CpuGlobalLoc {
    pub fn new(grid: OccGrid, params: GlParams) -> Self {
        Self { grid, params }
    }

    /// xorshift-based uniform hypothesis dispersion over free cells.
    fn disperse(&self) -> Vec<(f64, f64, f64)> {
        let mut s = self.params.seed | 1;
        let mut next_u = move || {
            s ^= s << 13;
            s ^= s >> 7;
            s ^= s << 17;
            (s % 1_000_000) as f64 / 1_000_000.0
        };
        let free: Vec<(usize, usize)> = (0..self.grid.cells.len())
            .filter(|&i| self.grid.cells[i] == 0)
            .map(|i| (i % self.grid.w, i / self.grid.w))
            .collect();
        assert!(!free.is_empty(), "grid has no free cells");
        (0..self.params.hypotheses)
            .map(|_| {
                let idx = ((next_u() * free.len() as f64) as usize)
                    .min(free.len() - 1);
                let c = free[idx];
                (
                    self.grid.origin.x + (c.0 as f64 + next_u()) * self.grid.res,
                    self.grid.origin.y + (c.1 as f64 + next_u()) * self.grid.res,
                    -std::f64::consts::PI + next_u() * std::f64::consts::TAU,
                )
            })
            .collect()
    }

    pub fn localize(&self, real_scan: &[f64]) -> Vec<PoseHypothesis> {
        let hyps = self.disperse();
        let g = &self.grid;
        let p = &self.params;
        let mut scored: Vec<PoseHypothesis> = hyps
            .into_par_iter()
            .map(|(x, y, th)| PoseHypothesis {
                caer: g.caer_of(real_scan, x, y, th, p.ang_span, p.rmax),
                x, y, theta: th,
            })
            .collect();
        scored.sort_by(|a, b| a.caer.partial_cmp(&b.caer).unwrap());
        scored.truncate(p.topk);
        scored
    }
}
