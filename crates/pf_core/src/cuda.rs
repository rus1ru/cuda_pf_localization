//! CUDA backend: FFI to the pf_kernels library (built by cmake from
//! cuda/pf_kernels.cu). Enabled with the `cuda` cargo feature.

#![allow(clippy::too_many_arguments)]

use std::os::raw::{c_int, c_uint};
use std::sync::Mutex;

use nalgebra::{Matrix4, Vector3};

use crate::config::PfConfig;
use crate::landmark_map::LandmarkMap;
use crate::particle_filter::Backend;
use crate::types::{quat_mean, quat_moment, Estimate, Observation, OdomDelta, Particle, Pose};

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

#[repr(C)]
#[derive(Debug, Clone, Copy)]
struct CEstimate {
    x: f32,
    y: f32,
    z: f32,
    qw: f32,
    qx: f32,
    qy: f32,
    qz: f32,
    ess: f32,
    valid: c_int,
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
    fn pfc_resample(h: *mut core::ffi::c_void) -> c_int;
    fn pfc_estimate(h: *mut core::ffi::c_void, out: *mut CEstimate) -> c_int;
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

pub struct CudaBackend {
    handle: *mut core::ffi::c_void,
    n: usize,
    cfg: PfConfig,
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
        let handle = unsafe { pfc_create(64) };
        if handle.is_null() {
            return Err("pfc_create failed".to_string());
        }
        Ok(Self {
            handle,
            n: 0,
            cfg: PfConfig::default(),
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
        if obs.is_empty() {
            // GPU weights stay uniform from init/resample
            return;
        }
        // upload landmark table dense to max id
        let cap = map.capacity().max(1);
        let mut flat = vec![0.0f32; cap * 3];
        for (id, p) in map.iter() {
            flat[(id as usize) * 3] = p.x as f32;
            flat[(id as usize) * 3 + 1] = p.y as f32;
            flat[(id as usize) * 3 + 2] = p.z as f32;
        }
        unsafe {
            pfc_upload_landmarks(self.handle, flat.as_ptr(), cap as c_int);
        }
        let cobs: Vec<CObservation> = obs
            .iter()
            .take(64)
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

    fn resample(&mut self, _ess_ratio: f64, _random_inject_ratio: f64) {
        unsafe {
            pfc_resample(self.handle);
        }
    }

    fn estimate(&self) -> Estimate {
        let mut out = CEstimate {
            x: 0.0,
            y: 0.0,
            z: 0.0,
            qw: 1.0,
            qx: 0.0,
            qy: 0.0,
            qz: 0.0,
            ess: 0.0,
            valid: 0,
        };
        unsafe {
            pfc_estimate(self.handle, &mut out);
        }
        let mut e = Estimate {
            mean: Pose {
                position: Vector3::new(out.x as f64, out.y as f64, out.z as f64),
                attitude: nalgebra::UnitQuaternion::from_quaternion(
                    nalgebra::Quaternion::new(out.qw as f64, out.qx as f64, out.qy as f64, out.qz as f64),
                ),
            },
            pos_cov: [[0.25; 3]; 3],
            att_cov: [[0.01; 3]; 3],
            ess: out.ess as f64,
            valid: out.valid == 1,
        };
        e.mean.attitude.renormalize();
        e
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

// keep quat helpers referenced (used by the CPU path and future GPU covariance)
#[allow(dead_code)]
fn _unused(m: &Matrix4<f64>) -> nalgebra::UnitQuaternion<f64> {
    quat_mean(m)
}
