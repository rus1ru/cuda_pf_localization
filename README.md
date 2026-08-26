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
| 1k | 0.193 | 0.240 | 0.8x (launch overhead dominates) |
| 10k | 1.254 | 0.250 | **5.0x** |
| 100k | 10.042 | 0.483 | **20.8x** |
| 500k | 45.767 | 1.740 | **26.3x** |

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
                         for systematic resampling (+ uniform re-injection),
                         block-reduced weighted sums and covariances.
                         The Markley quaternion mean is solved on the host
                         with the same nalgebra code path as the CPU backend,
                         so both backends agree by construction.
crates/pf_core           Rust engine: Backend trait, cpu.rs (rayon),
                         cuda.rs = pure FFI binding to libpf_kernels.so,
                         sim.rs = deterministic fixtures shared by tests
                         and the benchmark (frozen draw order!).
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
LD_LIBRARY_PATH=build cargo test --release -p pf_core --features cuda
                                                        # CPU suite + GPU agreement tests
cargo test --release -p pf_core                          # CPU-only (no GPU needed)
LD_LIBRARY_PATH=build ./target/release/pf-bench --csv   # machine-readable
```

The `cuda` cargo feature on pf_core links against build/libpf_kernels.so
(see crates/pf_core/build.rs).

## GPU Global Localization (CBGL port)

Stage-1 port of CBGL (Filotheou, arXiv:2307.14247): disperse pose
hypotheses over an occupancy grid, DDA-raycast virtual map-scans, score
CAER = sum |real - virtual| per ray, keep bottom-k via CUB radix sort.
All stages on device; Rust binding in `pf_core::global_loc` with a rayon
CPU twin as oracle/fallback. Includes a ROS `map_server` PGM/YAML loader.

Measured on RTX 4050 Mobile vs i7-13700H (12-thread rayon), 10x10 m grid,
360-ray scan, single-shot pose recovery within 0.5 m:

| hypotheses | CPU ms | CUDA ms | speedup | CUDA success |
|---|---|---|---|---|
| 5k | 52 | **1.2** | 43x | 17/20 |
| 50k | 545 | **13.1** | 42x | 20/20 |
| 200k | 2380 | **48.6** | 49x | 20/20 |

Run: `LD_LIBRARY_PATH=build ./target/release/pf-bench --gl`

Combined stack: this GL stage bootstraps the particle filter for tracking -
global localize once from a single scan, then hand the pose to the filter.
Both stages GPU-accelerated, both in Rust.

## Production notes

- Both backends now return full weighted position and attitude-tangent
  covariances from `estimate()`; a test asserts CPU/GPU agreement against an
  analytic Gaussian prior (`tests/cuda_agreement.rs`).
- `random_inject_ratio` is implemented on both backends (GPU: `k_inject`,
  deterministic in the resample stream).
- Empty observation sets and observations without a map entry reset weights
  to uniform on both backends (previously the GPU kept stale weights).
- The landmark table is uploaded to the device only when it changes.

## Known gaps

- Particle weights are stored as f32 on both backends: estimate sums are
  accumulated in double (per-block partials, deterministic order), but the
  f32 weight storage sets a floor of ~N * 2^-24 on ESS exactness
  (+0.006 at 100k particles).
- RNG streams differ by design (curand Philox vs splitmix64) — same noise
  statistics, different draws.
- Attitude covariance uses the atan2-based canonical axis-angle form on
  both backends (stable for tightly converged clouds; acos is not).
- At most 256 observations per `weight()` call are used (silently truncated);
  raise `OBS_CAP` in `pf_core/src/cuda.rs` if your sensor produces more.
- A ROS 2 wrapper node (rclrs) is planned but not started; the core crate is
  ROS-independent by design.
