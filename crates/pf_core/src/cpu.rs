//! Rayon-parallel CPU backend.

use std::f64::consts::PI;

use nalgebra::{Matrix4, Quaternion, Unit, UnitQuaternion, Vector3};
use rand::prelude::*;
use rand::rngs::SmallRng;
use rand_distr::StandardNormal;
use rayon::prelude::*;

use crate::config::PfConfig;
use crate::landmark_map::LandmarkMap;
use crate::particle_filter::Backend;
use crate::types::{
    quat_mean, quat_moment, Estimate, Observation, OdomDelta, Particle, Pose,
};

/// CPU particle set with SoA layout for cache-friendly parallel passes.
pub struct CpuBackend {
    pub(crate) n: usize,
    pub(crate) cfg: PfConfig,

    pos: Vec<Vector3<f64>>,
    att: Vec<UnitQuaternion<f64>>,
    weight: Vec<f64>,
    cum: Vec<f64>,

    /// Per-particle RNG seeds (splitmix streams, deterministic).
    seeds: Vec<u64>,
    global_seed: u64,
}

impl CpuBackend {
    pub fn new() -> Self {
        Self {
            n: 0,
            cfg: PfConfig::default(),
            pos: Vec::new(),
            att: Vec::new(),
            weight: Vec::new(),
            cum: Vec::new(),
            seeds: Vec::new(),
            global_seed: 42,
        }
    }

    fn rng_for(&self, i: usize) -> SmallRng {
        // splitmix64 of (global_seed, particle index, generation)
        let mut z = self
            .global_seed
            .wrapping_mul(0x9E3779B97F4A7C15)
            .wrapping_add((i as u64).wrapping_mul(0xBF58476D1CE4E5B9));
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58476D1CE4E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D049BB133111EB);
        z ^= z >> 31;
        SmallRng::seed_from_u64(z)
    }
}

impl Backend for CpuBackend {
    fn name(&self) -> &'static str {
        "cpu"
    }

    fn init(&mut self, n: usize, cfg: PfConfig) {
        self.n = n;
        self.cfg = cfg.clone();
        self.pos = vec![Vector3::zeros(); n];
        self.att = vec![UnitQuaternion::identity(); n];
        self.weight = vec![1.0 / n as f64; n];
        self.cum = vec![0.0; n];
        self.global_seed = if cfg.seed == 0 { rand::random() } else { cfg.seed };
        self.seeds = (0..n).map(|i| i as u64).collect();
    }

    fn reinit(&mut self, pose: &Pose, pos_std: &Vector3<f64>, rot_std: &Vector3<f64>) {
        self.global_seed = self.global_seed.wrapping_add(0x9E37_7979_7979_7979);
        let pairs: Vec<(usize, (Vector3<f64>, UnitQuaternion<f64>))> = (0..self.n)
            .into_par_iter()
            .map(|i| {
                let mut rng = self.rng_for(i);
                let n3: [f64; 3] = [
                    StandardNormal.sample(&mut rng),
                    StandardNormal.sample(&mut rng),
                    StandardNormal.sample(&mut rng),
                ];
                let p = pose.position
                    + Vector3::new(
                        pos_std[0] * n3[0],
                        pos_std[1] * n3[1],
                        pos_std[2] * n3[2],
                    );
                let w3: [f64; 3] = [
                    StandardNormal.sample(&mut rng),
                    StandardNormal.sample(&mut rng),
                    StandardNormal.sample(&mut rng),
                ];
                let axis_ang = Vector3::new(
                    rot_std[0] * w3[0],
                    rot_std[1] * w3[1],
                    rot_std[2] * w3[2],
                );
                let angle = axis_ang.norm();
                let dq = if angle > 1e-12 {
                    let ax = Unit::new_normalize(axis_ang);
                    UnitQuaternion::from_axis_angle(&ax, angle)
                } else {
                    UnitQuaternion::identity()
                };
                let a = pose.attitude * dq;
                (i, (p, a))
            })
            .collect();
        for (i, (p, a)) in pairs {
            self.pos[i] = p;
            self.att[i] = a;
            self.weight[i] = 1.0 / self.n as f64;
        }
    }

    fn predict(&mut self, odom: &OdomDelta) {
        self.global_seed = self.global_seed.wrapping_add(0x9E37_7979_7979_7979);
        let updates: Vec<(usize, (Vector3<f64>, UnitQuaternion<f64>))> = (0..self.n)
            .into_par_iter()
            .map(|i| {
                let mut rng = self.rng_for(i);
                let n3: [f64; 3] = [
                    StandardNormal.sample(&mut rng),
                    StandardNormal.sample(&mut rng),
                    StandardNormal.sample(&mut rng),
                ];
                let w3: [f64; 3] = [
                    StandardNormal.sample(&mut rng),
                    StandardNormal.sample(&mut rng),
                    StandardNormal.sample(&mut rng),
                ];
                let body_disp = odom.translation
                    + Vector3::new(
                        odom.trans_noise[0] * n3[0],
                        odom.trans_noise[1] * n3[1],
                        odom.trans_noise[2] * n3[2],
                    );
                let world_disp = self.att[i].transform_vector(&body_disp);
                let p = self.pos[i] + world_disp;

                let axis_ang = Vector3::new(
                    odom.rot_noise[0] * w3[0],
                    odom.rot_noise[1] * w3[1],
                    odom.rot_noise[2] * w3[2],
                );
                let angle = axis_ang.norm();
                let dq = if angle > 1e-12 {
                    let ax = Unit::new_normalize(axis_ang);
                    UnitQuaternion::from_axis_angle(&ax, angle)
                } else {
                    UnitQuaternion::identity()
                };
                let a = self.att[i] * odom.rotation * dq;
                (i, (p, a))
            })
            .collect();
        for (i, (p, a)) in updates {
            self.pos[i] = p;
            self.att[i] = a;
        }
    }

    fn weight(&mut self, obs: &[Observation], map: &LandmarkMap) {
        if obs.is_empty() {
            let w = 1.0 / self.n as f64;
            self.weight.iter_mut().for_each(|x| *x = w);
            return;
        }
        let valid: Vec<&Observation> =
            obs.iter().filter(|o| map.has(o.landmark_id)).collect();
        if valid.is_empty() {
            let w = 1.0 / self.n as f64;
            self.weight.iter_mut().for_each(|x| *x = w);
            return;
        }
        let sr = self.cfg.range_noise.max(1e-6);
        let sb = self.cfg.bearing_noise.max(1e-6);
        let inv2rr = 1.0 / (2.0 * sr * sr);
        let inv2bb = 1.0 / (2.0 * sb * sb);
        let gate = self.cfg.gate_sigma;
        let cart_gate2 = gate * gate * 3.0 * sr * sr;
        let rng_gate2 = gate * gate * sr * sr;
        let brg_gate2 = gate * gate * sb * sb;

        // parallel log-likelihoods
        let lls: Vec<f64> = (0..self.n)
            .into_par_iter()
            .map(|i| {
                let mut ll = 0.0;
                for o in &valid {
                    let lm = map.at(o.landmark_id);
                    let pred = self.att[i].inverse_transform_vector(&(lm - self.pos[i]));
                    match o.mode {
                        crate::types::ObsMode::Cartesian => {
                            let e = pred - o.vector;
                            let r2 = e.norm_squared();
                            ll += if gate > 0.0 && r2 > cart_gate2 {
                                -0.5 * gate * gate * 3.0
                            } else {
                                -r2 * inv2rr
                            };
                        }
                        crate::types::ObsMode::RangeBearing => {
                            let rn = pred.norm().max(1e-9);
                            let er = rn - o.range;
                            let dir_pred = pred / rn;
                            let dot = (-dir_pred).dot(&o.vector).clamp(-1.0, 1.0);
                            let eb = dot.acos();
                            ll += if gate > 0.0 && er * er > rng_gate2 {
                                -0.5 * gate * gate
                            } else {
                                -er * er * inv2rr
                            };
                            ll += if gate > 0.0 && eb * eb > brg_gate2 {
                                -0.5 * gate * gate
                            } else {
                                -eb * eb * inv2bb
                            };
                        }
                    }
                }
                ll
            })
            .collect();

        // max-shift + normalize
        let max_ll = lls.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
        let sum: f64 = lls.iter().map(|&l| (l - max_ll).exp()).sum();
        if sum <= 0.0 || !sum.is_finite() {
            let w = 1.0 / self.n as f64;
            self.weight.iter_mut().for_each(|x| *x = w);
        } else {
            for i in 0..self.n {
                self.weight[i] = (lls[i] - max_ll).exp() / sum;
            }
        }
    }

    fn resample(&mut self, _ess_ratio: f64, random_inject_ratio: f64) {
        // systematic resampling
        self.cum[0] = self.weight[0];
        for i in 1..self.n {
            self.cum[i] = self.cum[i - 1] + self.weight[i];
        }
        let total = self.cum[self.n - 1];
        if total <= 0.0 || !total.is_finite() {
            return;
        }
        let mut rng = SmallRng::seed_from_u64(
            self.global_seed ^ 0xDEAD_BEEF_CAFE_F00D,
        );
        let u0: f64 = rng.gen::<f64>() * (total / self.n as f64);
        let step = total / self.n as f64;

        let mut out_pos = vec![Vector3::zeros(); self.n];
        let mut out_att = vec![UnitQuaternion::identity(); self.n];
        let mut j = 0usize;
        for i in 0..self.n {
            let t = u0 + i as f64 * step;
            while j < self.n - 1 && self.cum[j] < t {
                j += 1;
            }
            out_pos[i] = self.pos[j];
            out_att[i] = self.att[j];
        }

        // optional uniform re-injection (position jitter)
        if random_inject_ratio > 0.0 {
            let k = ((random_inject_ratio * self.n as f64) as usize).min(self.n);
            for _ in 0..k {
                let i = rng.gen_range(0..self.n);
                let jit = Vector3::new(
                    rng.gen_range(-0.5..0.5),
                    rng.gen_range(-0.5..0.5),
                    rng.gen_range(-0.5..0.5),
                );
                out_pos[i] += jit;
            }
        }
        self.pos = out_pos;
        self.att = out_att;
        let w = 1.0 / self.n as f64;
        self.weight.iter_mut().for_each(|x| *x = w);
    }

    fn estimate(&self) -> Estimate {
        // ESS (weights normalized)
        let s2: f64 = self.weight.iter().map(|w| w * w).sum();
        let ess = if s2 > 0.0 { 1.0 / s2 } else { 0.0 };

        let mut m = Matrix4::zeros();
        let mut mean_pos = Vector3::zeros();
        for i in 0..self.n {
            let w = self.weight[i];
            mean_pos += w * self.pos[i];
            quat_moment(self.att[i].quaternion(), w, &mut m);
        }
        let mean_att = quat_mean(&m);

        // covariances
        let mut pos_cov = [[0.0f64; 3]; 3];
        let mut att_cov = [[0.0f64; 3]; 3];
        for i in 0..self.n {
            let w = self.weight[i];
            let dp = self.pos[i] - mean_pos;
            for r in 0..3 {
                for c in 0..3 {
                    pos_cov[r][c] += w * dp[r] * dp[c];
                }
            }
            let dq = mean_att.inverse() * self.att[i];
            let q = dq.quaternion();
            let (sign, ang) = if q.w >= 0.0 {
                (1.0, 2.0 * q.w.acos().min(PI))
            } else {
                (-1.0, 2.0 * (-q.w).acos().min(PI))
            };
            let axis = Vector3::new(sign * q.i, sign * q.j, sign * q.k);
            let aav = if axis.norm() > 1e-9 {
                axis.normalize() * ang
            } else {
                Vector3::zeros()
            };
            for r in 0..3 {
                for c in 0..3 {
                    att_cov[r][c] += w * aav[r] * aav[c];
                }
            }
        }

        Estimate {
            mean: Pose {
                position: mean_pos,
                attitude: mean_att,
            },
            pos_cov,
            att_cov,
            ess,
            valid: true,
        }
    }

    fn snapshot(&self) -> Vec<Particle> {
        (0..self.n)
            .map(|i| Particle {
                pose: Pose {
                    position: self.pos[i],
                    attitude: self.att[i],
                },
                weight: self.weight[i],
            })
            .collect()
    }
}

// silence unused import warning when nalgebra Quaternion re-export unused
#[allow(unused)]
fn _t(_: Quaternion<f64>) {}
