//! Shared types: pose, particles, observations, odometry deltas.

use nalgebra::{Quaternion, UnitQuaternion, Vector3};

/// Full 6-DoF pose.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Pose {
    pub position: Vector3<f64>,
    pub attitude: UnitQuaternion<f64>,
}

impl Default for Pose {
    fn default() -> Self {
        Self {
            position: Vector3::zeros(),
            attitude: UnitQuaternion::identity(),
        }
    }
}

impl Pose {
    /// Transform a body-frame vector into the world frame.
    pub fn transform(&self, v: &Vector3<f64>) -> Vector3<f64> {
        self.attitude.transform_vector(v) + self.position
    }

    /// Transform a world vector into the body frame.
    pub fn inverse_transform(&self, v: &Vector3<f64>) -> Vector3<f64> {
        self.attitude.inverse_transform_vector(&(v - self.position))
    }
}

/// One particle: pose + weight.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Particle {
    pub pose: Pose,
    pub weight: f64,
}

impl Default for Particle {
    fn default() -> Self {
        Self {
            pose: Pose::default(),
            weight: 0.0,
        }
    }
}

/// How an observation is expressed (always in the robot BODY frame).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ObsMode {
    /// (dx, dy, dz) offset to the landmark in body frame.
    Cartesian,
    /// Range in meters + unit bearing direction in body frame.
    RangeBearing,
}

/// A single landmark observation.
#[derive(Debug, Clone, Copy)]
pub struct Observation {
    pub landmark_id: i32,
    pub mode: ObsMode,
    /// Cartesian: the offset. RangeBearing: the unit direction.
    pub vector: Vector3<f64>,
    /// Range in meters (RangeBearing only).
    pub range: f64,
}

/// Odometry delta between two filter updates, expressed in the BODY frame of
/// the previous pose: displacement + rotation since then, with noise stds.
#[derive(Debug, Clone, Copy)]
pub struct OdomDelta {
    pub translation: Vector3<f64>,
    pub rotation: UnitQuaternion<f64>,
    /// Translation noise std devs (m) per axis, body frame.
    pub trans_noise: Vector3<f64>,
    /// Rotation noise std devs (rad), roll/pitch/yaw-ish tangent axes.
    pub rot_noise: Vector3<f64>,
}

impl Default for OdomDelta {
    fn default() -> Self {
        Self {
            translation: Vector3::zeros(),
            rotation: UnitQuaternion::identity(),
            trans_noise: Vector3::new(0.05, 0.05, 0.05),
            rot_noise: Vector3::new(0.02, 0.02, 0.03),
        }
    }
}

/// Filter estimate: weighted mean pose + covariances + effective sample size.
#[derive(Debug, Clone)]
pub struct Estimate {
    pub mean: Pose,
    /// Position covariance (3x3, world frame).
    pub pos_cov: [[f64; 3]; 3],
    /// Attitude covariance (3x3, tangent space at the mean).
    pub att_cov: [[f64; 3]; 3],
    /// Effective sample size (1..=N).
    pub ess: f64,
    pub valid: bool,
}

impl Default for Estimate {
    fn default() -> Self {
        Self {
            mean: Pose::default(),
            pos_cov: [[1.0; 3]; 3],
            att_cov: [[1.0; 3]; 3],
            ess: 0.0,
            valid: false,
        }
    }
}

/// Quaternion -> 4x4 moment matrix M = sum w q q^T (Markley mean).
pub(crate) fn quat_moment(quat: &Quaternion<f64>, w: f64, m: &mut nalgebra::Matrix4<f64>) {
    let q = [quat.w, quat.i, quat.j, quat.k];
    for r in 0..4 {
        for c in 0..4 {
            m[(r, c)] += w * q[r] * q[c];
        }
    }
}

/// Weighted quaternion mean via the largest-eigenvector method.
pub(crate) fn quat_mean(m: &nalgebra::Matrix4<f64>) -> UnitQuaternion<f64> {
    let es = nalgebra::linalg::SymmetricEigen::new(m.transpose() * 0.5 + m * 0.5);
    // SymmetricEigen of a symmetric matrix: eigenvalues ascending
    let mut idx = 0;
    for (i, &val) in es.eigenvalues.iter().enumerate() {
        if val > es.eigenvalues[idx] {
            idx = i;
        }
    }
    let col = es.eigenvectors.column(idx);
    let q = Quaternion::new(col[0], col[1], col[2], col[3]);
    let mut uq = UnitQuaternion::from_quaternion(q);
    if uq.quaternion().w < 0.0 {
        uq = UnitQuaternion::from_quaternion(-q);
    }
    uq
}
