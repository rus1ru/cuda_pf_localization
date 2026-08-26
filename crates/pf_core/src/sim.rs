//! Deterministic simulation fixtures shared by tests, examples and the
//! benchmark binary.
//!
//! The generators are FROZEN: their draw order defines the deterministic
//! scenarios that tests assert against. Change anything here and the
//! integration tests will move.

use nalgebra::{Unit, UnitQuaternion, Vector3};

use crate::landmark_map::LandmarkMap;
use crate::global_loc::OccGrid;
use crate::types::{Observation, OdomDelta, Pose};

/// Deterministic xorshift64 noise source.
pub struct Gauss(u64);

impl Gauss {
    pub fn new(seed: u64) -> Self {
        Self(seed)
    }

    /// One uniform draw in [0, 8) shifted by +1 (the frozen landmark-box draw).
    fn next_box(&mut self) -> f64 {
        self.0 ^= self.0 << 13;
        self.0 ^= self.0 >> 7;
        self.0 ^= self.0 << 17;
        (self.0 % 8000) as f64 / 1000.0 + 1.0
    }

    /// One uniform draw in [0, 1).
    fn next01(&mut self) -> f64 {
        self.0 ^= self.0 << 13;
        self.0 ^= self.0 >> 7;
        self.0 ^= self.0 << 17;
        (self.0 % 10000) as f64 / 10000.0
    }

    /// Approximate standard normal (sum of two uniforms, scaled).
    pub fn normal(&mut self) -> f64 {
        (self.next01() + self.next01() - 1.0) * 1.732
    }
}

/// `n` pseudo-random landmarks in an ~8 x 8 x 2.4 m box.
pub fn landmark_box(n: usize, seed: u64) -> LandmarkMap {
    let mut g = Gauss::new(seed);
    let mut map = LandmarkMap::new();
    for i in 0..n {
        map.add(i as i32, Vector3::new(g.next_box(), g.next_box(), g.next_box() * 0.3));
    }
    map
}

/// One trajectory step: noisy odometry + observations, plus the true pose
/// AFTER the motion.
pub struct Step {
    pub odom: OdomDelta,
    pub obs: Vec<Observation>,
    pub truth: Pose,
}

/// A complete deterministic localization scenario.
pub struct Scenario {
    pub map: LandmarkMap,
    pub prior: Pose,
    pub steps: Vec<Step>,
}

/// Circular 2D trajectory with constant climb (`dt` = 0.1 s), noisy
/// odometry, and Cartesian observations of landmarks within 5 m.
pub fn circle_scenario(steps_n: usize, seed: u64) -> Scenario {
    let map = landmark_box(12, 99);
    let prior = Pose {
        position: Vector3::new(5.0, 5.0, 0.5),
        attitude: UnitQuaternion::identity(),
    };

    let mut g = Gauss::new(seed);
    let dt = 0.1f64;
    let mut truth = prior;
    let mut steps = Vec::with_capacity(steps_n);
    for t in 0..steps_n {
        let vel = Vector3::new(
            -0.4 * (0.2 * t as f64 * dt).sin(),
            0.4 * (0.2 * t as f64 * dt).cos(),
            0.02,
        );
        let yaw = vel.y.atan2(vel.x);
        let z_ax = Unit::new_normalize(Vector3::<f64>::z());
        let att = UnitQuaternion::from_axis_angle(&z_ax, yaw);
        let new_pos = truth.position + vel * dt;

        let mut odo = OdomDelta {
            translation: truth.attitude.inverse_transform_vector(&(new_pos - truth.position)),
            rotation: truth.attitude.inverse() * att,
            ..Default::default()
        };
        odo.translation += Vector3::new(g.normal() * 0.05, g.normal() * 0.05, g.normal() * 0.05);

        truth.position = new_pos;
        truth.attitude = att;

        let mut obs = Vec::new();
        for i in 0..map.len_defined() as i32 {
            let body = truth.attitude.inverse_transform_vector(&(map.at(i) - truth.position));
            let d = body.norm();
            if d > 5.0 {
                continue;
            }
            let noisy = body * (1.0 + g.normal() * 0.05 / d.max(0.1));
            obs.push(Observation {
                landmark_id: i,
                mode: crate::types::ObsMode::Cartesian,
                vector: noisy,
                range: 0.0,
            });
        }
        steps.push(Step { odom: odo, obs, truth });
    }
    Scenario { map, prior, steps }
}

/// 10x10 m room at 0.05 m/cell (200x200 cells): border walls plus one
/// interior pillar centered at world (4, 5).
pub fn room_grid() -> OccGrid {
    let (w, h) = (200usize, 200usize);
    let mut cells = vec![0u8; w * h];
    for x in 0..w {
        cells[x] = 1; // bottom wall (y=0)
        cells[(h - 1) * w + x] = 1; // top wall
    }
    for y in 0..h {
        cells[y * w] = 1;
        cells[y * w + w - 1] = 1;
    }
    // pillar: x in [3.75, 4.30], y in [4.75, 5.30]
    let xr = ((4.0 - 0.25) / 0.05) as usize..=((4.0 + 0.25) / 0.05) as usize;
    let yr = ((5.0 - 0.25) / 0.05) as usize..=((5.0 + 0.25) / 0.05) as usize;
    for y in yr {
        for x in xr.clone() {
            cells[y * w + x] = 1;
        }
    }
    OccGrid {
        w,
        h,
        res: 0.05,
        origin: nalgebra::Vector2::new(0.0, 0.0),
        cells,
    }
}
