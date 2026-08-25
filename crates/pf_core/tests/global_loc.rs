//! Global localization tests: DDA raycast correctness + CPU/GPU agreement.

use pf_core::global_loc::{CpuGlobalLoc, GlParams, OccGrid};

/// 10x10 m room at 0.05 m/cell = 200x200 cells, walls on the border
/// plus one interior pillar.
fn test_grid() -> OccGrid {
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
    // pillar centered at world (4,5): x in [3.75,4.30], y in [4.75,5.30]
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

#[test]
fn raycast_hits_walls_at_expected_distance() {
    let g = test_grid();
    // from center (5,5) facing +x: wall at x=10 -> ~4.75 m to wall surface
    let d = g.cast_ray(5.0, 5.0, 1.0, 0.0, 20.0);
    assert!(
        (d - 4.95).abs() < 0.15,
        "expected ~4.95 m to east wall, got {d}"
    );
    // facing +y from center: also ~4.95 to north wall (pillar not in path)
    let d2 = g.cast_ray(5.0, 5.0, 0.0, 1.0, 20.0);
    assert!((d2 - 4.95).abs() < 0.15);
}

#[test]
fn raycast_hits_interior_pillar() {
    let g = test_grid();
    // from (2.5,5) facing +x: pillar front face at x=3.75 -> ~1.25 m
    let d = g.cast_ray(2.5, 5.0, 1.0, 0.0, 20.0);
    assert!(
        (d - 1.25).abs() < 0.12,
        "expected ~1.25 m to pillar, got {d}"
    );
}

#[test]
fn raycast_respects_rmax() {
    let g = test_grid();
    let d = g.cast_ray(5.0, 5.0, 1.0, 0.0, 2.0);
    assert_eq!(d, 2.0, "no hit within rmax should return rmax");
}

#[test]
fn cpu_gl_recovers_pose_in_synthetic_room() {
    let g = test_grid();
    // synthetic truth pose + perfect scan
    let truth = (6.0, 7.0, 0.8f64);
    let scan = g.map_scan(truth.0, truth.1, truth.2, 360, std::f64::consts::TAU, 10.0);

    let cpu = CpuGlobalLoc::new(g.clone(), GlParams {
        hypotheses: 20_000,
        rays: 360,
        rmax: 10.0,
        ang_span: std::f64::consts::TAU,
        topk: 10,
        seed: 7,
    });
    let top = cpu.localize(&scan);
    assert!(!top.is_empty());
    let best = &top[0];
    let dpos = ((best.x - truth.0).powi(2) + (best.y - truth.1).powi(2)).sqrt();
    assert!(
        dpos < 0.35,
        "CPU GL did not recover pose: best=({:.2},{:.2},{:.2}) err={dpos:.3} m",
        best.x, best.y, best.theta
    );
}

#[cfg(feature = "cuda")]
mod cuda_gl {
    use super::*;
    use pf_core::global_loc::{CudaGlobalLoc, PoseHypothesis};
    use pf_core::particle_filter::Backend;

    #[test]
    fn cuda_matches_cpu_ranking() {
        if !pf_core::cuda::device_available() {
            eprintln!("skipping: no CUDA device");
            return;
        }
        let g = test_grid();
        let truth = (6.0, 7.0, 0.8f64);
        let scan_f64 =
            g.map_scan(truth.0, truth.1, truth.2, 360, std::f64::consts::TAU, 10.0);
        let scan_f32: Vec<f32> = scan_f64.iter().map(|v| *v as f32).collect();

        let params = GlParams {
            hypotheses: 30_000,
            ..Default::default()
        };

        let mut gpu = CudaGlobalLoc::new(&g, params.clone()).unwrap();
        assert!(gpu.free_cells() > 0);
        let gpu_top = gpu.localize(&scan_f32).unwrap();

        let cpu = CpuGlobalLoc::new(g.clone(), params.clone());
        let cpu_top = cpu.localize(&scan_f64);

        // GPU winners must land near a CPU winner (same optimum region).
        // RNGs differ, so compare against the CPU's best position error.
        let mut best_gpu_err = f64::INFINITY;
        for gp in &gpu_top {
            for cp in &cpu_top {
                let e =
                    ((gp.x - cp.x).powi(2) + (gp.y - cp.y).powi(2)).sqrt();
                best_gpu_err = best_gpu_err.min(e);
            }
        }
        assert!(
            best_gpu_err < 0.5,
            "GPU and CPU optima disagree: {best_gpu_err:.3} m"
        );

        // GPU winner must be near ground truth too.
        let b = &gpu_top[0];
        let derr = ((b.x - truth.0).powi(2) + (b.y - truth.1).powi(2)).sqrt();
        assert!(derr < 0.5, "GPU best not near truth: {derr:.3} m");

        // silence unused warnings from trait import used by pf path
        let _ = pf_core::config::BackendKind::Auto;
    }

    // keep the Backend import referenced (used in the pf benchmark paths)
    #[allow(unused)]
    fn _t() {
        fn assert_backend<T: pf_core::particle_filter::Backend>(_: T) {}
        let _ = assert_backend::<pf_core::cpu::CpuBackend>;
    }
}
