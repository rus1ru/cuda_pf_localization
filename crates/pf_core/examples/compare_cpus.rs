//! Head-to-head CPU particle-filter comparison on the frozen benchmark
//! scenario (`pf_core::sim::circle_scenario`, identical to pf-bench).
//!
//! Competitors:
//!   1. `kalman_filters` crate — generic closure-based PF, sequential
//!   2. `pf_core` CpuBackend restricted to 1 rayon thread (--seq)
//!   3. `pf_core` CpuBackend, all threads
//!
//! Run:
//!   cargo run --release -p pf_core --example compare_cpus
//!   cargo run --release -p pf_core --example compare_cpus -- --seq

use std::cell::RefCell;
use std::time::Instant;

use nalgebra::{Unit, UnitQuaternion, Vector3};
use rand::rngs::SmallRng;
use rand::Rng;
use rand::SeedableRng;
use rand_distr::Uniform;

use pf_core::config::{BackendKind, PfConfig};
use pf_core::particle_filter::ParticleFilter;
use pf_core::sim;

const STEPS: usize = 100;

/// Gated Cartesian log-likelihood, same math as cpu.rs MeasModel.
fn cart_ll(pred: &Vector3<f64>, expected: &Vector3<f64>, range_noise: f64) -> f64 {
    let sr = range_noise.max(1e-6);
    let r2 = (pred - expected).norm_squared();
    -r2 / (2.0 * sr * sr)
}

/// The generic `kalman_filters` crate driven over the same scenario.
fn run_kalman_filters(n: usize, sc: &sim::Scenario) -> (f64, f64) {
    use kalman_filters::particle::filter::{ParticleFilter as KfPF, ResamplingStrategy};

    let dim = 7usize; // x y z qw qx qy qz
    let q = sc.prior.attitude.quaternion();
    let init_mean = vec![
        sc.prior.position.x,
        sc.prior.position.y,
        sc.prior.position.z,
        q.w,
        q.i,
        q.j,
        q.k,
    ];
    let mut pf = KfPF::initialize(
        dim,
        n,
        init_mean.clone(),
        vec![1.0; dim], // initial spread
        vec![0.0; dim], // noise added inside the motion model instead
        vec![0.05],     // unused by our likelihood closure
        0.1,
    )
    .unwrap();
    pf.resampling_strategy = ResamplingStrategy::Systematic;
    pf.ess_threshold = 0.5 * n as f64;

    let rng = RefCell::new(SmallRng::seed_from_u64(42));
    let u01 = Uniform::new(-0.5, 0.5);
    let nz = |r: &mut SmallRng| -> f64 { (r.sample(u01) + r.sample(u01)) * 1.732 };

    let mut err_sum = 0.0f64;
    let t0 = Instant::now();
    for step in &sc.steps {
        let odo = step.odom;

        pf.predict(|state: &[f64], _dt: f64| -> Vec<f64> {
            let mut r = rng.borrow_mut();
            let pos = Vector3::new(state[0], state[1], state[2]);
            let att = UnitQuaternion::from_quaternion(nalgebra::Quaternion::new(
                state[3], state[4], state[5], state[6],
            ));
            let body = odo.translation
                + Vector3::new(
                    odo.trans_noise[0] * nz(&mut r),
                    odo.trans_noise[1] * nz(&mut r),
                    odo.trans_noise[2] * nz(&mut r),
                );
            let p = pos + att.transform_vector(&body);
            let axis = Vector3::new(
                odo.rot_noise[0] * nz(&mut r),
                odo.rot_noise[1] * nz(&mut r),
                odo.rot_noise[2] * nz(&mut r),
            );
            let angle = axis.norm();
            let dq = if angle > 1e-12 {
                UnitQuaternion::from_axis_angle(&Unit::new_normalize(axis), angle)
            } else {
                UnitQuaternion::identity()
            };
            let a = att * odo.rotation * dq;
            let qq = a.quaternion();
            vec![p.x, p.y, p.z, qq.w, qq.i, qq.j, qq.k]
        });

        let obs = &step.obs;
        let map = &sc.map;
        pf.update(&[], |state: &[f64], _m: &[f64]| -> f64 {
            let pos = Vector3::new(state[0], state[1], state[2]);
            let att = UnitQuaternion::from_quaternion(nalgebra::Quaternion::new(
                state[3], state[4], state[5], state[6],
            ));
            let mut llsum = 0.0f64;
            for o in obs {
                if !map.has(o.landmark_id) {
                    continue;
                }
                let pred = att.inverse_transform_vector(&(map.at(o.landmark_id) - pos));
                // NOTE: this crate expects a LIKELIHOOD (takes .ln()
                // internally), not a log-likelihood.
                llsum += cart_ll(&pred, &o.vector, 0.05).exp();
            }
            llsum
        })
        .unwrap();

        // NOTE: update() already resamples when ESS < ess_threshold.

        let mean = pf.mean();
        let t = &step.truth.position;
        err_sum += ((mean[0] - t.x).powi(2)
            + (mean[1] - t.y).powi(2)
            + (mean[2] - t.z).powi(2))
        .sqrt();
    }
    (
        t0.elapsed().as_secs_f64() * 1000.0 / STEPS as f64,
        err_sum / STEPS as f64,
    )
}

/// Our backend through the standard driver.
fn run_ours(n: usize, kind: BackendKind) -> Option<(String, f64, f64)> {
    let sc = sim::circle_scenario(STEPS, 777);
    let cfg = PfConfig {
        particle_count: n,
        seed: 123,
        range_noise: 0.05,
        backend: kind,
        ..Default::default()
    };
    let mut pf = match ParticleFilter::new(cfg) {
        Ok(p) => p,
        Err(e) => return Some((format!("unavailable({e})"), f64::NAN, f64::NAN)),
    };
    pf.set_map(sc.map);
    pf.reinitialize(sc.prior, Vector3::new(1.0, 1.0, 1.0), Vector3::new(0.5, 0.5, 0.5));

    let s0 = &sc.steps[0];
    let _ = pf.update(&s0.odom, &s0.obs);

    let t0 = Instant::now();
    let mut err_sum = 0.0;
    for step in &sc.steps {
        let est = pf.update(&step.odom, &step.obs);
        err_sum += (est.mean.position - step.truth.position).norm();
    }
    Some((
        pf.backend_name().to_string(),
        t0.elapsed().as_secs_f64() * 1000.0 / STEPS as f64,
        err_sum / STEPS as f64,
    ))
}

fn main() {
    let seq_only = std::env::args().any(|a| a == "--seq");
    if seq_only {
        // must be set before rayon's global pool initializes
        std::env::set_var("RAYON_NUM_THREADS", "1");
    }

    println!(
        "{:>8} {:<30} {:>12} {:>14}",
        "N", "implementation", "ms/cycle", "track err"
    );
    println!("{}", "-".repeat(70));

    for &n in &[1_000usize, 10_000, 100_000] {
        let (ms, err) = run_kalman_filters(n, &sim::circle_scenario(STEPS, 777));
        println!("{0:>8} {1:<30} {2:>12.3} {3:>13.4} m", n, "kalman_filters (sequential)", ms, err);

        let (_, ms, err) = run_ours(n, BackendKind::Cpu).unwrap();
        let label = if seq_only {
            "pf_core cpu (1 thread)"
        } else {
            "pf_core cpu (rayon)"
        };
        println!("{0:>8} {1:<30} {2:>12.3} {3:>13.4} m", n, label, ms, err);

        #[cfg(feature = "cuda")]
        if !seq_only && pf_core::cuda::device_available() {
            if let Some((name, ms, err)) = run_ours(n, BackendKind::Cuda) {
                println!("{0:>8} {1:<30} {2:>12.3} {3:>13.4} m", n, name, ms, err);
            }
        }
        println!();
    }
}
