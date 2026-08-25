# cuda_pf_localization

3D Monte Carlo localization from point landmarks, in Rust, with two
swappable compute backends: Rayon-parallel CPU and CUDA. Same
`Backend` trait, same filter semantics, one config switch.

## Results

Measured on RTX 4050 Mobile / i7-13700H (12 threads), 100 update cycles,
12 landmarks, full cycle = predict + weight + resample-if-low-ESS +
estimate:

| N particles | CPU (rayon) ms/cycle | CUDA ms/cycle | speedup |
|---|---|---|---|
| 1k | 0.152 | 0.252 | 0.6x (launch overhead dominates) |
| 10k | 1.135 | 0.252 | **4.5x** |
| 100k | 8.386 | 0.428 | **19.6x** |
| 500k | 41.771 | 1.459 | **28.6x** |

Both backends localize to centimeter-level mean tracking error on the test
trajectory (see `tests/integration.rs`). Crossover at ~4k particles: use
`backend: cpu` below it; above it CUDA wins big and stays nearly flat in N
until memory bandwidth saturates.

## Where the implementation actually lives

```
src/cuda/pf_kernels.cu   THE CUDA implementation of every pfc_* FFI symbol:
                         k_seed_rng / k_reinit / k_predict / k_weight,
                         softmax via ordered-float atomicMax + atomicAdd,
                         CUB inclusive scan + parallel binary-search gather
                         for systematic resampling, block-reduced weighted
                         sums, power-iteration Markley quaternion mean.
crates/pf_core           Rust engine: Backend trait, cpu.rs (rayon),
                         cuda.rs = pure FFI binding to libpf_kernels.so.
crates/pf_bench          the benchmark binary (this table)
CMakeLists.txt           builds pf_kernels.cu -> build/libpf_kernels.so
run_bench.sh             one command: cmake + cargo + benchmark
```

The FFI contract is declared in `crates/pf_core/src/cuda.rs`; the kernels
mirror `crates/pf_core/src/cpu.rs` semantics exactly (Hamilton quaternions,
gated log-likelihoods, systematic resampling).

## Build & run

Requires nvcc (CUDA >= 12 tested) and CMake >= 3.18.

```
./run_bench.sh                                  # full build + benchmark
LD_LIBRARY_PATH=build cargo test --release -p pf_core   # 7 tests
LD_LIBRARY_PATH=build ./target/release/pf-bench --csv   # machine-readable
```

The `cuda` cargo feature on pf_core links against build/libpf_kernels.so
(see crates/pf_core/build.rs).

## Notes / known gaps

- GPU math is f32 (CPU f64): expect ~1e-3 relative agreement, not bitwise.
- RNG streams differ by design (curand Philox vs splitmix64) — same noise
  statistics, different draws.
- estimate() returns placeholder covariances on the CUDA path; position
  mean, quaternion mean, and ESS are exact.
- random_inject_ratio is ignored by the CUDA backend.
- pf_node (rclrs ROS wrapper) is scaffolded but needs a ROS 2 environment;
  not part of the workspace build yet.
