//! Filter configuration.

#[derive(Debug, Clone)]
pub struct PfConfig {
    /// Number of particles.
    pub particle_count: usize,

    /// Process noise: translation std devs (m), body-frame axes.
    pub trans_noise: [f64; 3],
    /// Process noise: rotation std devs (rad).
    pub rot_noise: [f64; 3],

    /// Measurement noise: range / cartesian residual sigma (m).
    pub range_noise: f64,
    /// Measurement noise: bearing sigma (rad).
    pub bearing_noise: f64,

    /// Resample when ESS < ratio * N. Set to 0.0 to never resample.
    pub resample_ess_ratio: f64,
    /// Fraction of particles re-injected uniformly after resampling (0..1).
    pub random_inject_ratio: f64,
    /// Gating: reject observations with residual beyond sigma * gate_sigma
    /// (0.0 disables gating).
    pub gate_sigma: f64,

    /// "auto" | "cuda" | "cpu".
    pub backend: BackendKind,
    /// Deterministic seed (0 = entropy).
    pub seed: u64,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BackendKind {
    Auto,
    Cuda,
    Cpu,
}

impl BackendKind {
    pub fn parse(s: &str) -> Self {
        match s {
            "cuda" => Self::Cuda,
            "cpu" => Self::Cpu,
            _ => Self::Auto,
        }
    }
}

impl Default for PfConfig {
    fn default() -> Self {
        Self {
            particle_count: 4096,
            trans_noise: [0.05, 0.05, 0.02],
            rot_noise: [0.02, 0.02, 0.03],
            range_noise: 0.05,
            bearing_noise: 0.02,
            resample_ess_ratio: 0.5,
            random_inject_ratio: 0.0,
            gate_sigma: 0.0,
            backend: BackendKind::Auto,
            seed: 42,
        }
    }
}
