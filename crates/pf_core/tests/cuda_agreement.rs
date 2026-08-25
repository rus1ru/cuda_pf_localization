//! CPU-vs-CUDA agreement tests (feature `cuda`). Skipped when no GPU is
//! present or the crate is built without the `cuda` feature.

#![cfg(feature = "cuda")]

use approx::assert_abs_diff_eq;

use nalgebra::Vector3;
use pf_core::config::PfConfig;
use pf_core::cpu::CpuBackend;
use pf_core::cuda::CudaBackend;
use pf_core::particle_filter::{Backend, ParticleFilter};
use pf_core::types::{Observation, OdomDelta, Pose};

fn cuda() -> Option<CudaBackend> {
    match CudaBackend::new() {
        Ok(b) => Some(b),
        Err(_) => None,
    }
}

fn cfg(n: usize, seed: u64) -> PfConfig {
    PfConfig {
        particle_count: n,
        seed,
        ..Default::default()
    }
}

#[test]
fn cuda_covariances_match_gaussian_prior() {
    let Some(mut gpu) = cuda() else {
        eprintln!("skipping: no CUDA device");
        return;
    };
    let mut cpu = CpuBackend::new();

    // uniform-weight cloud right after a wide Gaussian reinit: both backends
    // must report covariances consistent with the injected spread.
    let pose = Pose::default();
    let pos_std = Vector3::new(1.0, 2.0, 0.5);
    let rot_std = Vector3::new(0.05, 0.07, 0.02);
    let n = 200_000;

    cpu.init(n, cfg(n, 7));
    cpu.reinit(&pose, &pos_std, &rot_std);
    gpu.init(n, cfg(n, 7));
    gpu.reinit(&pose, &pos_std, &rot_std);

    let ec = cpu.estimate();
    let eg = gpu.estimate();
    assert!(eg.valid);

    // means agree between backends
    for r in 0..3 {
        assert!(
            (ec.mean.position[r] - eg.mean.position[r]).abs() < 0.05,
            "mean mismatch axis {r}: {} vs {}",
            ec.mean.position[r],
            eg.mean.position[r]
        );
    }

    // diagonal variances track the prior on both backends
    for i in 0..3 {
        let want = pos_std[i] * pos_std[i];
        assert_abs_diff_eq!(ec.pos_cov[i][i], want, epsilon = 0.12 * want);
        assert_abs_diff_eq!(eg.pos_cov[i][i], want, epsilon = 0.12 * want);
        let wrot = rot_std[i] * rot_std[i];
        assert_abs_diff_eq!(ec.att_cov[i][i], wrot, epsilon = 0.15 * wrot);
        assert_abs_diff_eq!(eg.att_cov[i][i], wrot, epsilon = 0.15 * wrot);
        // off-diagonals ~ 0
        for j in 0..3 {
            if i != j {
                assert!(eg.pos_cov[i][j].abs() < 0.1 * want);
                assert!(eg.att_cov[i][j].abs() < 0.1 * wrot);
            }
        }
    }
}

#[test]
fn cuda_empty_obs_resets_weights_like_cpu() {
    let Some(mut gpu) = cuda() else {
        eprintln!("skipping: no CUDA device");
        return;
    };
    let n = 8192;
    let mut c = cfg(n, 11);
    c.resample_ess_ratio = 0.0; // never resample: inspect raw weights

    let mut map = pf_core::landmark_map::LandmarkMap::new();
    map.add(0, nalgebra::Vector3::new(5.0, 5.0, 1.0));
    let p0 = Pose {
        position: nalgebra::Vector3::new(5.0, 5.0, 1.0),
        attitude: nalgebra::UnitQuaternion::identity(),
    };

    let mut gpf = ParticleFilter::new(c.clone()).unwrap();
    gpf.set_map(map.clone());
    gpf.reinitialize(p0, Vector3::new(3.0, 3.0, 3.0), Vector3::new(0.1, 0.1, 0.1));

    let obs = vec![Observation {
        landmark_id: 0,
        mode: pf_core::types::ObsMode::Cartesian,
        vector: Vector3::zeros(),
        range: 0.0,
    }];

    // informative obs collapses ESS ...
    let e1 = gpf.update(&OdomDelta::default(), &obs);
    assert!(e1.ess < (n as f64) * 0.5, "ESS did not collapse: {}", e1.ess);

    // ... then a coasting cycle (no obs) must restore uniform weights
    let e2 = gpf.update(&OdomDelta::default(), &[]);
    assert!(
        e2.ess > (n as f64) * 0.95,
        "ESS not restored after empty obs: {}",
        e2.ess
    );

    // unknown landmark ids are ignored (weights stay uniform), like the CPU
    let ghost = vec![Observation {
        landmark_id: 42,
        mode: pf_core::types::ObsMode::Cartesian,
        vector: Vector3::new(1.0, 0.0, 0.0),
        range: 0.0,
    }];
    let e3 = gpf.update(&OdomDelta::default(), &ghost);
    assert!(e3.ess > (n as f64) * 0.95);
}

#[test]
fn cuda_random_injection_tracks_trajectory() {
    let Some(gpu) = cuda() else {
        eprintln!("skipping: no CUDA device");
        return;
    };
    let _ = gpu;

    let n = 4096;
    let mut c = cfg(n, 13);
    c.resample_ess_ratio = 0.5;
    c.random_inject_ratio = 0.1; // exercise the injection path

    let mut pf = ParticleFilter::new(c).unwrap();
    assert_eq!(pf.backend_name(), "cuda");

    let start = Vector3::new(0.0, 0.0, 0.0);
    let p0 = Pose {
        position: start,
        attitude: nalgebra::UnitQuaternion::identity(),
    };
    pf.reinitialize(p0, Vector3::new(0.5, 0.5, 0.5), Vector3::new(0.01, 0.01, 0.01));

    let odo = OdomDelta {
        translation: Vector3::new(0.5, 0.0, 0.0),
        trans_noise: Vector3::new(0.02, 0.02, 0.02),
        rot_noise: Vector3::new(0.005, 0.005, 0.005),
        ..Default::default()
    };
    let mut est = pf.update(&odo, &[]);
    for k in 1..20 {
        est = pf.update(&odo, &[]);
        let expect_x = 0.5 * (k + 1) as f64;
        assert!(
            (est.mean.position.x - expect_x).abs() < 0.6,
            "drifted off trajectory with injection: x={} want ~{}",
            est.mean.position.x,
            expect_x
        );
        assert!(est.valid && est.ess.is_finite());
    }
}
