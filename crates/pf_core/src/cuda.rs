//! CUDA backend: FFI to the pf_kernels library (built by cmake from
//! cuda/pf_kernels.cu). Enabled with the `cuda` cargo feature.

#![allow(clippy::too_many_arguments)]

use std::os::raw::{c_int, c_uint};
use std::sync::Mutex;

use nalgebra::{Matrix4, Vector3};

use crate::config::PfConfig;
use crate::landmark_map::LandmarkMap;
use crate::particle_filter::Backend;
use crate::types::{
    quat_mean, Estimate, Observation, OdomDelta, Particle, Pose,
};

#[repr(C)]
#[derive(Debug, Clone, Copy)]
struct CObservation {
    id: c_int,
    mode: c_int, // 0 cartesian, 1 range-bearing
    dx: f32,
    dy: f32,
    dz: f32,
    range: f32,
}

/// Output of the covariance pass: packed upper triangles
/// [Pxx,Pxy,Pxz,Pyy,Pyz,Pzz | Axx,Axy,Axz,Ayy,Ayz,Azz].
#[repr(C)]
#[derive(Debug, Clone, Copy, Default)]
struct CCovs {
    v: [f32; 12],
}

extern "C" {
    fn pfc_create(cap_obs: c_int) -> *mut core::ffi::c_void;
    fn pfc_destroy(h: *mut core::ffi::c_void);
    fn pfc_init(h: *mut core::ffi::c_void, n: c_int, seed: c_uint) -> c_int;
    fn pfc_upload_landmarks(
        h: *mut core::ffi::c_void,
        data: *const f32, // 3 floats per entry, dense by id
        count: c_int,
    ) -> c_int;
    fn pfc_reinit(
        h: *mut core::ffi::c_void,
        x: f32,
        y: f32,
        z: f32,
        qw: f32,
        qx: f32,
        qy: f32,
        qz: f32,
        pos_std: *const f32,
        rot_std: *const f32,
    ) -> c_int;
    fn pfc_predict(
        h: *mut core::ffi::c_void,
        tx: f32, ty: f32, tz: f32,
        qw: f32, qx: f32, qy: f32, qz: f32,
        tstd: *const f32,
        rstd: *const f32,
    ) -> c_int;
    fn pfc_weight(
        h: *mut core::ffi::c_void,
        obs: *const CObservation,
        nobs: c_int,
        sigma_r: f32,
        sigma_b: f32,
        gate: f32,
    ) -> c_int;
    fn pfc_resample(h: *mut core::ffi::c_void, random_inject_ratio: f32) -> c_int;
    fn pfc_est_sums(h: *mut core::ffi::c_void, out: *mut f32) -> c_int;
    fn pfc_covs(
        h: *mut core::ffi::c_void,
        mx: f32,
        my: f32,
        mz: f32,
        qw: f32,
        qx: f32,
        qy: f32,
        qz: f32,
        out: *mut CCovs,
    ) -> c_int;
    fn pfc_snapshot(
        h: *mut core::ffi::c_void,
        pos: *mut f32, // 3n
        quat: *mut f32, // 4n
    ) -> c_int;
    fn pfc_device_available() -> c_int;
}

pub fn device_available() -> bool {
    unsafe { pfc_device_available() == 1 }
}

/// Observation capacity per weight() call (must match pfc_create).
const OBS_CAP: usize = 256;

pub struct CudaBackend {
    handle: *mut core::ffi::c_void,
    n: usize,
    cfg: PfConfig,
    /// Last landmark table uploaded to the device; skips redundant PCIe
    /// transfers when the map is unchanged between cycles.
    lm_cache: Vec<f32>,
    /// Serialize GPU access (a filter is single-tenant, but the node may clone).
    _lock: Mutex<()>,
}

// The CUDA library owns its device memory; the handle is only used from Rust.
unsafe impl Send for CudaBackend {}

impl CudaBackend {
    pub fn new() -> Result<Self, String> {
        if !device_available() {
            return Err("no CUDA device".to_string());
        }
        let handle = unsafe { pfc_create(OBS_CAP as c_int) };
        if handle.is_null() {
            return Err("pfc_create failed".to_string());
        }
        Ok(Self {
            handle,
            n: 0,
            cfg: PfConfig::default(),
            lm_cache: Vec::new(),
            _lock: Mutex::new(()),
        })
    }
}

impl Drop for CudaBackend {
    fn drop(&mut self) {
        if !self.handle.is_null() {
            unsafe { pfc_destroy(self.handle) };
        }
    }
}

impl Backend for CudaBackend {
    fn name(&self) -> &'static str {
        "cuda"
    }

    fn init(&mut self, n: usize, cfg: PfConfig) {
        self.n = n;
        self.cfg = cfg.clone();
        let rc = unsafe { pfc_init(self.handle, n as c_int, cfg.seed as c_uint) };
        assert_eq!(rc, 0, "pfc_init failed");
    }

    fn reinit(&mut self, pose: &Pose, pos_std: &Vector3<f64>, rot_std: &Vector3<f64>) {
        let ps: Vec<f32> = pos_std.iter().map(|v| *v as f32).collect();
        let rs: Vec<f32> = rot_std.iter().map(|v| *v as f32).collect();
        let q = pose.attitude.quaternion();
        unsafe {
            pfc_reinit(
                self.handle,
                pose.position.x as f32,
                pose.position.y as f32,
                pose.position.z as f32,
                q.w as f32,
                q.i as f32,
                q.j as f32,
                q.k as f32,
                ps.as_ptr(),
                rs.as_ptr(),
            );
        }
    }

    fn predict(&mut self, odom: &OdomDelta) {
        let t: Vec<f32> = odom.trans_noise.iter().map(|v| *v as f32).collect();
        let r: Vec<f32> = odom.rot_noise.iter().map(|v| *v as f32).collect();
        let q = odom.rotation.quaternion();
        unsafe {
            pfc_predict(
                self.handle,
                odom.translation.x as f32,
                odom.translation.y as f32,
                odom.translation.z as f32,
                q.w as f32,
                q.i as f32,
                q.j as f32,
                q.k as f32,
                t.as_ptr(),
                r.as_ptr(),
            );
        }
    }

    fn weight(&mut self, obs: &[Observation], map: &LandmarkMap) {
        let _g = self._lock.lock().unwrap();
        // Mirror the CPU backend: drop observations without a map entry.
        let valid: Vec<&Observation> =
            obs.iter().filter(|o| map.has(o.landmark_id)).collect();
        if valid.is_empty() {
            // nobs == 0 resets the GPU weights to uniform, matching cpu.rs.
            unsafe {
                pfc_weight(
                    self.handle,
                    std::ptr::null(),
                    0,
                    self.cfg.range_noise as f32,
                    self.cfg.bearing_noise as f32,
                    self.cfg.gate_sigma as f32,
                );
            }
            return;
        }
        // Upload the landmark table (dense to max id) only when it changed.
        let cap = map.capacity().max(1);
        let mut flat = vec![0.0f32; cap * 3];
        for (id, p) in map.iter() {
            flat[(id as usize) * 3] = p.x as f32;
            flat[(id as usize) * 3 + 1] = p.y as f32;
            flat[(id as usize) * 3 + 2] = p.z as f32;
        }
        if flat != self.lm_cache {
            unsafe {
                pfc_upload_landmarks(self.handle, flat.as_ptr(), cap as c_int);
            }
            self.lm_cache = flat;
        }
        let cobs: Vec<CObservation> = valid
            .iter()
            .take(OBS_CAP)
            .map(|o| CObservation {
                id: o.landmark_id,
                mode: match o.mode {
                    crate::types::ObsMode::Cartesian => 0,
                    crate::types::ObsMode::RangeBearing => 1,
                },
                dx: o.vector.x as f32,
                dy: o.vector.y as f32,
                dz: o.vector.z as f32,
                range: o.range as f32,
            })
            .collect();
        unsafe {
            pfc_weight(
                self.handle,
                cobs.as_ptr(),
                cobs.len() as c_int,
                self.cfg.range_noise as f32,
                self.cfg.bearing_noise as f32,
                self.cfg.gate_sigma as f32,
            );
        }
    }

    fn resample(&mut self, _ess_ratio: f64, random_inject_ratio: f64) {
        unsafe {
            pfc_resample(self.handle, random_inject_ratio as f32);
        }
    }

    fn estimate(&self) -> Estimate {
        if self.n == 0 {
            return Estimate::default();
        }
        // Pass 1: weighted sums from the device.
        let mut sums = [0.0f32; 14];
        unsafe {
            let rc = pfc_est_sums(self.handle, sums.as_mut_ptr());
            assert_eq!(rc, 0, "pfc_est_sums failed");
        }

        // Markley quaternion mean on the host — the exact same code path as
        // the CPU backend, so both agree by construction.
        let mut m = Matrix4::zeros();
        for r in 0..4usize {
            for c in r..4usize {
                let off = 4 * r - r * (r - 1) / 2 + (c - r);
                let v = sums[3 + off] as f64;
                m[(r, c)] = v;
                m[(c, r)] = v;
            }
        }
        let mean_att = quat_mean(&m);
        let q = mean_att.quaternion();

        // Pass 2: covariances around the means.
        let mut cv = CCovs::default();
        unsafe {
            let rc = pfc_covs(
                self.handle,
                sums[0],
                sums[1],
                sums[2],
                q.w as f32,
                q.i as f32,
                q.j as f32,
                q.k as f32,
                &mut cv,
            );
            assert_eq!(rc, 0, "pfc_covs failed");
        }
        // unpack a symmetric 3x3 from a packed upper triangle at offset o
        let sym3 = |v: &[f32; 12], o: usize| -> [[f64; 3]; 3] {
            let mut m = [[0.0f64; 3]; 3];
            let at = |r: usize, c: usize| o + 3 * r - r * (r - 1) / 2 + (c - r);
            for r in 0..3usize {
                for c in r..3usize {
                    let x = v[at(r, c)] as f64;
                    m[r][c] = x;
                    m[c][r] = x;
                }
            }
            m
        };

        Estimate {
            mean: Pose {
                position: Vector3::new(sums[0] as f64, sums[1] as f64, sums[2] as f64),
                attitude: mean_att,
            },
            pos_cov: sym3(&cv.v, 0),
            att_cov: sym3(&cv.v, 6),
            ess: if sums[13] > 0.0 { 1.0 / sums[13] as f64 } else { 0.0 },
            valid: true,
        }
    }

    fn snapshot(&self) -> Vec<Particle> {
        let mut pos = vec![0.0f32; self.n * 3];
        let mut quat = vec![0.0f32; self.n * 4];
        unsafe {
            pfc_snapshot(self.handle, pos.as_mut_ptr(), quat.as_mut_ptr());
        }
        (0..self.n)
            .map(|i| Particle {
                pose: Pose {
                    position: Vector3::new(
                        pos[i * 3] as f64,
                        pos[i * 3 + 1] as f64,
                        pos[i * 3 + 2] as f64,
                    ),
                    attitude: nalgebra::UnitQuaternion::from_quaternion(
                        nalgebra::Quaternion::new(
                            quat[i * 4] as f64,
                            quat[i * 4 + 1] as f64,
                            quat[i * 4 + 2] as f64,
                            quat[i * 4 + 3] as f64,
                        ),
                    ),
                },
                weight: 1.0 / self.n as f64,
            })
            .collect()
    }
}

