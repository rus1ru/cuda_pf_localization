//! Backend trait + filter driver + factory.

use nalgebra::Vector3;

use crate::config::{BackendKind, PfConfig};
use crate::cpu::CpuBackend;
use crate::landmark_map::LandmarkMap;
use crate::types::{Estimate, Observation, OdomDelta, Particle, Pose};

/// Abstract filter backend.
pub trait Backend: Send {
    fn name(&self) -> &'static str;

    /// Allocate for N particles.
    fn init(&mut self, n: usize, cfg: PfConfig);

    /// Overwrite the cloud with pose + Gaussian spread (the prior).
    fn reinit(&mut self, pose: &Pose, pos_std: &Vector3<f64>, rot_std: &Vector3<f64>);

    /// Apply an odometry delta + process noise.
    fn predict(&mut self, odom: &OdomDelta);

    /// Weight all particles against observations (replaces weights).
    fn weight(&mut self, obs: &[Observation], map: &LandmarkMap);

    /// Resample (systematic). Called by the driver when ESS is low.
    fn resample(&mut self, ess_ratio: f64, random_inject_ratio: f64);

    /// Weighted mean + covariances + ESS.
    fn estimate(&self) -> Estimate;

    /// Copy out the particle set (viz, tests).
    fn snapshot(&self) -> Vec<Particle>;
}

/// Create the backend requested by the config.
pub fn make_backend(kind: BackendKind) -> Result<Box<dyn Backend>, String> {
    match kind {
        BackendKind::Cpu => Ok(Box::new(CpuBackend::new())),
        BackendKind::Cuda => {
            #[cfg(feature = "cuda")]
            {
                match crate::cuda::CudaBackend::new() {
                    Ok(b) => Ok(Box::new(b)),
                    Err(e) => Err(e),
                }
            }
            #[cfg(not(feature = "cuda"))]
            {
                let _ = kind;
                Err("built without the `cuda` feature".to_string())
            }
        }
        BackendKind::Auto => {
            #[cfg(feature = "cuda")]
            {
                if crate::cuda::device_available() {
                    if let Ok(b) = crate::cuda::CudaBackend::new() {
                        return Ok(Box::new(b));
                    }
                }
            }
            Ok(Box::new(CpuBackend::new()))
        }
    }
}

/// High-level driver: owns config, map, backend; one-call update cycle.
pub struct ParticleFilter {
    cfg: PfConfig,
    map: LandmarkMap,
    backend: Box<dyn Backend>,
    last: Estimate,
}

impl ParticleFilter {
    pub fn new(cfg: PfConfig) -> Result<Self, String> {
        let mut backend = make_backend(cfg.backend)?;
        backend.init(cfg.particle_count, cfg.clone());
        Ok(Self {
            backend,
            cfg,
            map: LandmarkMap::new(),
            last: Estimate::default(),
        })
    }

    pub fn set_map(&mut self, map: LandmarkMap) {
        self.map = map;
    }

    pub fn map(&self) -> &LandmarkMap {
        &self.map
    }

    pub fn config(&self) -> &PfConfig {
        &self.cfg
    }

    pub fn backend_name(&self) -> &'static str {
        self.backend.name()
    }

    /// Re-seed the cloud around a pose (Gaussian prior).
    pub fn reinitialize(&mut self, pose: Pose, pos_std: Vector3<f64>, rot_std: Vector3<f64>) {
        self.backend.reinit(&pose, &pos_std, &rot_std);
        self.last = Estimate::default();
    }

    /// One full cycle: predict -> weight -> (maybe) resample -> estimate.
    pub fn update(&mut self, odom: &OdomDelta, obs: &[Observation]) -> Estimate {
        self.backend.predict(odom);
        self.backend.weight(obs, &self.map);

        let mut est = self.backend.estimate();
        let threshold = self.cfg.resample_ess_ratio * self.cfg.particle_count as f64;
        if self.cfg.resample_ess_ratio > 0.0 && est.ess < threshold {
            self.backend
                .resample(self.cfg.resample_ess_ratio, self.cfg.random_inject_ratio);
            est = self.backend.estimate();
        }
        self.last = est.clone();
        est
    }

    pub fn last_estimate(&self) -> &Estimate {
        &self.last
    }

    pub fn backend(&self) -> &dyn Backend {
        self.backend.as_ref()
    }
}
