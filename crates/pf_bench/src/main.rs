//! CPU (rayon) vs CUDA particle-filter benchmark.
//!
//! Identical scenario for both backends: landmark map, noisy odometry
//! trajectory, Cartesian landmark observations. Measures the full
//! update() cycle (predict + weight + resample-if-ess-low + estimate)
//! and reports mean wall time per cycle plus tracking error.
//!
//! Run:  cargo run --release -p pf-bench [-- --csv]
//!
//! Linking note: with the `cuda` feature, pf_core expects the symbols
//! from libpf_kernels (built by CMake into ../../build/). See
//! crates/pf_core/build.rs.

use std::time::Instant;

use nalgebra::Vector3;

use pf_core::config::{BackendKind, PfConfig};
use pf_core::particle_filter::ParticleFilter;

fn run_backend(kind: BackendKind, n: usize, steps_n: usize, seed: u64) -> (String, f64, f64, usize) {
    let sc = pf_core::sim::circle_scenario(steps_n, seed);
    let cfg = PfConfig {
        particle_count: n,
        seed: 123,
        range_noise: 0.05,
        ..Default::default()
    };
    // make_backend may fail if CUDA requested but unavailable
    let mut pf = match ParticleFilter::new(PfConfig { backend: kind, ..cfg }) {
        Ok(p) => p,
        Err(e) => {
            return (format!("unavailable({e})"), f64::NAN, f64::NAN, 0);
        }
    };
    pf.set_map(sc.map);
    pf.reinitialize(
        sc.prior,
        Vector3::new(1.0, 1.0, 1.0),
        Vector3::new(0.5, 0.5, 0.5),
    );

    // warmup (allocations, first-touch)
    let step0 = &sc.steps[0];
    let _ = pf.update(&step0.odom, &step0.obs);

    let t0 = Instant::now();
    let mut err_sum = 0.0;
    for step in &sc.steps {
        let est = pf.update(&step.odom, &step.obs);
        err_sum += (est.mean.position - step.truth.position).norm();
    }
    let per_cycle_ms = t0.elapsed().as_secs_f64() * 1000.0 / steps_n as f64;
    (
        pf.backend_name().to_string(),
        per_cycle_ms,
        err_sum / steps_n as f64,
        steps_n,
    )
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let csv = args.iter().any(|a| a == "--csv");

    let sizes = [1_000usize, 10_000, 100_000, 500_000];
    let steps = 100usize;

    if csv {
        println!("n,backend,ms_per_cycle,mean_track_err_m");
    } else {
        println!(
            "{:>9} {:<8} {:>14} {:>16}",
            "N", "backend", "ms/cycle", "mean track err"
        );
        println!("{}", "-".repeat(52));
    }

    for &n in &sizes {
        for &kind in &[BackendKind::Cpu, BackendKind::Cuda] {
            let (name, ms, err, _s) = run_backend(kind, n, steps, 777);
            if name.starts_with("unavailable") {
                if !csv {
                    println!("{n:>9} {name}");
                }
                continue;
            }
            if csv {
                println!("{n},{name},{ms:.4},{err:.4}");
            } else {
                println!("{n:>9} {name:<8} {ms:>14.3} {err:>15.4} m");
            }
        }
    }

    // ---- GPU global localization benchmark (--gl) ----
    if args.iter().any(|a| a == "--gl") {
        println!("\n== Global localization (CBGL-style, 10x10 m grid @0.05 m) ==");
        println!(
            "{:>10} {:<8} {:>12} {:>12} {:>10}",
            "hyps", "backend", "ms/localize", "pos err m", "success"
        );
        let grid = pf_core::sim::room_grid();
        // truth scan
        let truth = (6.0f64, 7.0f64, 0.8f64);
        let scan64 =
            grid.map_scan(truth.0, truth.1, truth.2, 360, std::f64::consts::TAU, 10.0);
        let scan32: Vec<f32> = scan64.iter().map(|v| *v as f32).collect();

        for &nh in &[5_000usize, 50_000, 200_000] {
            let params = pf_core::global_loc::GlParams {
                hypotheses: nh,
                ..Default::default()
            };
            // CPU timing
            let cpu = pf_core::global_loc::CpuGlobalLoc::new(grid.clone(), params.clone());
            let t0 = Instant::now();
            let runs = 3;
            let mut ok = 0;
            for _ in 0..runs {
                let top = cpu.localize(&scan64);
                let b = &top[0];
                if ((b.x - truth.0).powi(2) + (b.y - truth.1).powi(2)).sqrt() < 0.5 {
                    ok += 1;
                }
            }
            let cpu_ms = t0.elapsed().as_secs_f64() * 1000.0 / runs as f64;
            let cpu_top = cpu.localize(&scan64);
            let b = &cpu_top[0];
            let err = ((b.x - truth.0).powi(2) + (b.y - truth.1).powi(2)).sqrt();
            println!("{nh:>10} {:<8} {cpu_ms:>12.1} {err:>12.3} {:>9}/3", "cpu", ok);

            // CUDA timing
            if pf_core::cuda::device_available() {
                if let Ok(mut gpu) =
                    pf_core::global_loc::CudaGlobalLoc::new(&grid, params.clone())
                {
                    let _ = gpu.localize(&scan32); // warmup
                    let t0 = Instant::now();
                    let mut gok = 0;
                    let gruns = 20;
                    for i in 0..gruns {
                        gpu.params.seed = 100 + i as u64;
                        let top = match gpu.localize(&scan32) {
                            Ok(t) => t,
                            Err(_) => break,
                        };
                        if let Some(b) = top.first() {
                            if ((b.x - truth.0).powi(2)
                                + (b.y - truth.1).powi(2))
                                .sqrt()
                                < 0.5
                            {
                                gok += 1;
                            }
                        }
                    }
                    let gpu_ms =
                        t0.elapsed().as_secs_f64() * 1000.0 / gruns as f64;
                    let gpu_top = gpu.localize(&scan32).unwrap_or_default();
                    let gerr = gpu_top
                        .first()
                        .map(|b| {
                            ((b.x - truth.0).powi(2)
                                + (b.y - truth.1).powi(2))
                                .sqrt()
                        })
                        .unwrap_or(f64::NAN);
                    println!(
                        "{nh:>10} {:<8} {gpu_ms:>12.1} {gerr:>12.3} {gok:>8}/{gruns}",
                        "cuda"
                    );
                }
            }
        }
    }

}
