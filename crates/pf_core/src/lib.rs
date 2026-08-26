//! pf_core: 3D Monte Carlo localization from point landmarks.
//!
//! Backends:
//! - [`cpu`]: rayon-parallel CPU filter (always available)
//! - [`cuda`]: CUDA kernels via FFI (feature `cuda`, requires nvcc at build)
//! - [`global_loc`]: GPU global localization over occupancy grids
//!   (CBGL-style; feature `cuda`) with a rayon CPU twin for tests
//!
//! The filter tracks a full 6-DoF pose (position + quaternion attitude) and
//! accepts landmark observations as body-frame Cartesian offsets or
//! range+bearing.

pub mod config;
pub mod cpu;
pub mod landmark_map;
pub mod particle_filter;
pub mod sim;
pub mod types;

#[cfg(feature = "cuda")]
pub mod cuda;

pub mod global_loc;

pub use config::PfConfig;
pub use landmark_map::LandmarkMap;
pub use particle_filter::{Backend, ParticleFilter};
pub use types::{Estimate, Observation, OdomDelta, Particle, Pose};
