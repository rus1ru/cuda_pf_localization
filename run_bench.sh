#!/usr/bin/env bash
# Build everything and run the CPU-vs-CUDA particle-filter benchmark.
set -e
cd "$(dirname "$0")"
cmake -B build -DCMAKE_BUILD_TYPE=Release > /dev/null
cmake --build build -j"$(nproc)"
cargo build --release -p pf-bench
export LD_LIBRARY_PATH="$PWD/build:$LD_LIBRARY_PATH"
exec ./target/release/pf-bench "$@"
