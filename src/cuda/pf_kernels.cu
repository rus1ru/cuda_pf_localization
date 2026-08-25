// pf_kernels.cu — CUDA backend for cuda_pf_localization.
//
// Implements the C ABI declared in crates/pf_core/src/cuda.rs:
//   pfc_create / pfc_destroy / pfc_init / pfc_upload_landmarks /
//   pfc_reinit / pfc_predict / pfc_weight / pfc_resample(h, inject) /
//   pfc_est_sums + pfc_covs (mean/covariance passes; quaternion mean is
//   solved on the host with the same nalgebra path as the CPU backend)
//   pfc_snapshot / pfc_device_available
//
// Semantics mirror crates/pf_core/src/cpu.rs exactly (Hamilton quaternions,
// gated log-likelihoods, systematic resampling, Markley quaternion mean via
// dominant eigenvector of M = sum_i w_i q_i q_i^T), computed in f32.
//
// Kernel map:
//   k_seed_rng       curand Philox states, re-seeded per generation
//   k_reinit         Gaussian prior around a pose
//   k_predict        odometry delta + process noise
//   k_weight         O(N*M) landmark scoring -> log-likelihoods
//   k_ll_max/expsum/normalize   softmax over log-likelihoods
//   (resample)       CUB inclusive scan + parallel binary-search gather
//   k_inject         uniform re-injection jitter after resampling
//   k_est_sums       weighted position, per-particle moment matrix,
//                    sum(w^2) for ESS  -> 14 partials, block-reduced
//   k_est_covs       weighted position + attitude-tangent covariance
//                    upper triangles around given means -> 12 partials

#include <cuda_runtime.h>
#include <curand_kernel.h>
#include <cub/cub.cuh>
#include <climits>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>

#define CK(x)                                                              \
    do {                                                                   \
        cudaError_t e_ = (x);                                              \
        if (e_ != cudaSuccess) {                                           \
            fprintf(stderr, "pf_cuda %s:%d %s: %s\n", __FILE__, __LINE__,  \
                    #x, cudaGetErrorString(e_));                           \
            return -1;                                                     \
        }                                                                  \
    } while (0)

#define CK_ALLOC(x)                                                        \
    do {                                                                   \
        cudaError_t e_ = (x);                                              \
        if (e_ != cudaSuccess) {                                           \
            fprintf(stderr, "pf_cuda alloc %s: %s\n", #x,                  \
                    cudaGetErrorString(e_));                               \
            return nullptr;                                                \
        }                                                                  \
    } while (0)

// Launch-and-check: kernel launches are expressions of type void.
#define KL(x)                                                              \
    do {                                                                   \
        (x);                                                               \
        cudaError_t e_ = cudaGetLastError();                               \
        if (e_ != cudaSuccess) {                                           \
            fprintf(stderr, "pf_cuda %s:%d launch failed: %s\n",           \
                    __FILE__, __LINE__, cudaGetErrorString(e_));           \
            return -1;                                                     \
        }                                                                  \
    } while (0)

namespace {

constexpr int EST_N = 14;  // 3 mean-pos + 10 M-entries + 1 sum-w2
constexpr int COV_N = 12;  // 6 pos-cov + 6 att-cov upper-triangle entries

struct alignas(16) QuatF {
    float w, x, y, z;
};

struct ObsF {
    int id;
    int mode;          // 0 cartesian, 1 range-bearing
    float dx, dy, dz;  // cartesian offset / bearing unit dir
    float range;
};

struct PfState {
    int n = 0;
    uint64_t seed = 42;
    uint64_t gen = 0;             // bumped per reinit/predict

    float* d_pos = nullptr;       // 3n
    QuatF*  d_att = nullptr;      // n
    float*  d_w = nullptr;        // n (normalized)
    float*  d_ll = nullptr;       // n scratch
    float*  d_pos2 = nullptr;     // resample double buffer
    QuatF*  d_att2 = nullptr;
    double* d_cum = nullptr;      // inclusive scan of weights
    void*   d_scan_tmp = nullptr;
    size_t  scan_tmp_bytes = 0;

    int*   d_max = nullptr;       // ordered-float atomicMax target
    float* d_sumexp = nullptr;    // atomicAdd accumulator
    float* d_est = nullptr;       // EST_N estimate partials

    float* d_lm = nullptr;        // 3*lm_cap, dense by id
    int    lm_cap = 0;

    ObsF*  d_obs = nullptr;
    int    obs_cap = 0;

    curandStatePhilox4_32_10_t* d_rng = nullptr;
    float* d_cov = nullptr;       // COV_N covariance partials

    unsigned long long lcg = 0x243F6A8885A308D3ull;  // resample u0 stream
};

__device__ inline float3 q_rot(const QuatF& q, float3 v) {
    // v' = v + 2*w*(qv x v) + 2*qv x (qv x v)
    float3 qv = make_float3(q.x, q.y, q.z);
    float3 t = make_float3(2.f * (qv.y * v.z - qv.z * v.y),
                           2.f * (qv.z * v.x - qv.x * v.z),
                           2.f * (qv.x * v.y - qv.y * v.x));
    float3 c = make_float3(qv.y * t.z - qv.z * t.y,
                           qv.z * t.x - qv.x * t.z,
                           qv.x * t.y - qv.y * t.x);
    return make_float3(v.x + q.w * t.x + c.x,
                       v.y + q.w * t.y + c.y,
                       v.z + q.w * t.z + c.z);
}

__device__ inline QuatF q_conj(const QuatF& q) {
    return {q.w, -q.x, -q.y, -q.z};
}

__device__ inline QuatF q_mul(const QuatF& a, const QuatF& b) {
    return {a.w * b.w - a.x * b.x - a.y * b.y - a.z * b.z,
            a.w * b.x + a.x * b.w + a.y * b.z - a.z * b.y,
            a.w * b.y - a.x * b.z + a.y * b.w + a.z * b.x,
            a.w * b.z + a.x * b.y - a.y * b.x + a.z * b.w};
}

__device__ inline QuatF axis_angle(float3 axis, float angle) {
    float n = sqrtf(axis.x * axis.x + axis.y * axis.y + axis.z * axis.z);
    if (!(angle > 1e-12f) || !(n > 1e-12f)) return {1.f, 0.f, 0.f, 0.f};
    float s = sinf(0.5f * angle) / n;
    return {cosf(0.5f * angle), axis.x * s, axis.y * s, axis.z * s};
}

__host__ __device__ inline int float_as_ordered(float f) {
    int i;
    memcpy(&i, &f, sizeof(int));
    return (i >= 0) ? i : i ^ 0x7FFFFFFF;
}

__host__ __device__ inline float ordered_as_float(int i) {
    int j = (i >= 0) ? i : i ^ 0x7FFFFFFF;
    float f;
    memcpy(&f, &j, sizeof(float));
    return f;
}

__global__ void k_seed_rng(curandStatePhilox4_32_10_t* st, uint64_t seed,
                           uint64_t gen, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    curand_init(static_cast<unsigned long long>(seed),
                static_cast<unsigned long long>(gen) * 1000003ull +
                    static_cast<unsigned long long>(i),
                0, &st[i]);
}

__global__ void k_reinit(curandStatePhilox4_32_10_t* rng, float* pos,
                         QuatF* att, float* w, int n, float3 p0, QuatF q0,
                         float3 pos_std, float3 rot_std) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    auto& st = rng[i];
    float4 nn = curand_normal4(&st);
    float4 ww = curand_normal4(&st);
    float3 n3 = make_float3(nn.x, nn.y, nn.z);
    float3 w3 = make_float3(ww.x, ww.y, ww.z);

    float3 p = make_float3(p0.x + pos_std.x * n3.x, p0.y + pos_std.y * n3.y,
                           p0.z + pos_std.z * n3.z);
    float3 axis = make_float3(rot_std.x * w3.x, rot_std.y * w3.y,
                              rot_std.z * w3.z);
    float ang = sqrtf(axis.x * axis.x + axis.y * axis.y + axis.z * axis.z);
    QuatF dq = axis_angle(axis, ang);
    att[i] = q_mul(q0, dq);
    pos[3 * i] = p.x;
    pos[3 * i + 1] = p.y;
    pos[3 * i + 2] = p.z;
    w[i] = 1.0f / static_cast<float>(n);
}

__global__ void k_predict(curandStatePhilox4_32_10_t* rng, float* pos,
                          QuatF* att, int n, float3 t_body, QuatF q_odo,
                          float3 tstd, float3 rstd) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    auto& st = rng[i];
    float4 nn = curand_normal4(&st);
    float4 ww = curand_normal4(&st);
    float3 n3 = make_float3(nn.x, nn.y, nn.z);
    float3 w3 = make_float3(ww.x, ww.y, ww.z);

    QuatF q = att[i];
    float3 body = make_float3(t_body.x + tstd.x * n3.x,
                              t_body.y + tstd.y * n3.y,
                              t_body.z + tstd.z * n3.z);
    float3 world = q_rot(q, body);
    pos[3 * i] += world.x;
    pos[3 * i + 1] += world.y;
    pos[3 * i + 2] += world.z;

    float3 axis = make_float3(rstd.x * w3.x, rstd.y * w3.y, rstd.z * w3.z);
    float ang = sqrtf(axis.x * axis.x + axis.y * axis.y + axis.z * axis.z);
    QuatF dq = axis_angle(axis, ang);
    att[i] = q_mul(q_mul(q, q_odo), dq);
}

__global__ void k_weight(const float* pos, const QuatF* att, const float* lm,
                         int lm_cap, const ObsF* obs, int nobs, float sigma_r,
                         float sigma_b, float gate, float* ll, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    float inv2rr = 1.0f / (2.0f * sigma_r * sigma_r);
    float inv2bb = 1.0f / (2.0f * sigma_b * sigma_b);
    float g2 = gate * gate;
    float cart_gate2 = g2 * 3.0f * sigma_r * sigma_r;
    float rng_gate2 = g2 * sigma_r * sigma_r;
    float brg_gate2 = g2 * sigma_b * sigma_b;

    float3 p = make_float3(pos[3 * i], pos[3 * i + 1], pos[3 * i + 2]);
    QuatF q = att[i];

    float acc = 0.0f;
    for (int k = 0; k < nobs; ++k) {
        const ObsF o = obs[k];
        if (o.id < 0 || o.id >= lm_cap) continue;
        float3 lmv = make_float3(lm[3 * o.id], lm[3 * o.id + 1],
                                 lm[3 * o.id + 2]);
        float3 d = make_float3(lmv.x - p.x, lmv.y - p.y, lmv.z - p.z);
        float3 pred = q_rot(q_conj(q), d);  // landmark into body frame
        if (o.mode == 0) {
            float ex = pred.x - o.dx, ey = pred.y - o.dy, ez = pred.z - o.dz;
            float r2 = ex * ex + ey * ey + ez * ez;
            acc += (gate > 0.0f && r2 > cart_gate2)
                       ? -0.5f * g2 * 3.0f
                       : -r2 * inv2rr;
        } else {
            float rn = fmaxf(sqrtf(pred.x * pred.x + pred.y * pred.y +
                                   pred.z * pred.z),
                             1e-9f);
            float er = rn - o.range;
            // dot((-dir_pred), obs_dir)
            float dot = -((pred.x * o.dx + pred.y * o.dy + pred.z * o.dz) / rn);
            dot = fminf(fmaxf(dot, -1.0f), 1.0f);
            float eb = acosf(dot);
            acc += (gate > 0.0f && er * er > rng_gate2)
                       ? -0.5f * g2
                       : -er * er * inv2rr;
            acc += (gate > 0.0f && eb * eb > brg_gate2)
                       ? -0.5f * g2
                       : -eb * eb * inv2bb;
        }
    }
    ll[i] = acc;
}

__global__ void k_ll_max(const float* ll, int n, int* out_max) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int local = INT_MIN;
    for (; i < n; i += gridDim.x * blockDim.x) {
        local = max(local, float_as_ordered(ll[i]));
    }
    if (local != INT_MIN) atomicMax(out_max, local);
}

__global__ void k_ll_expsum(const float* ll, int n, float maxll,
                            float* out_sum) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    float local = 0.0f;
    for (; i < n; i += gridDim.x * blockDim.x) {
        local += expf(ll[i] - maxll);
    }
    atomicAdd(out_sum, local);
}

__global__ void k_ll_normalize(const float* ll, int n, float maxll,
                               float total, float* w) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    w[i] = expf(ll[i] - maxll) / total;
}

__global__ void k_reset_uniform(float* w, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) w[i] = 1.0f / static_cast<float>(n);
}

__global__ void k_resample_pick(const float* pos, const QuatF* att,
                                const double* cum, int n, double u0,
                                double total, float* pos2, QuatF* att2) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    double step = total / static_cast<double>(n);
    double t = u0 + static_cast<double>(i) * step;
    int lo = 0, hi = n - 1;
    while (lo < hi) {
        int mid = (lo + hi) >> 1;
        if (cum[mid] < t) lo = mid + 1;
        else hi = mid;
    }
    pos2[3 * i] = pos[3 * lo];
    pos2[3 * i + 1] = pos[3 * lo + 1];
    pos2[3 * i + 2] = pos[3 * lo + 2];
    att2[i] = att[lo];
}

__device__ inline uint64_t sm64(uint64_t* x) {
    uint64_t z = (*x += 0x9E3779B97F4A7C15ull);
    z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ull;
    z = (z ^ (z >> 27)) * 0x94D049BB133111EBull;
    return z ^ (z >> 31);
}

// Uniform re-injection: mirrors cpu.rs resample() — pick k random particle
// indices and add U(-0.5,0.5)^3 jitter to their positions. Deterministic in
// the host LCG stream + generation.
__global__ void k_inject(float* pos, int n, int k, unsigned long long lcg,
                         unsigned long long gen) {
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    if (j >= k) return;
    uint64_t z = lcg ^ (gen * 0x9E3779B97F4A7C15ull) ^
                 (static_cast<uint64_t>(j) + 1) * 0xBF58476D1CE4E5B9ull;
    if (z == 0) z = 0x243F6A8885A308D3ull;
    auto u01 = [&z]() {
        return static_cast<double>(sm64(&z) >> 11) *
               (1.0 / 9007199254740992.0);
    };
    int idx = static_cast<int>(u01() * n);
    if (idx >= n) idx = n - 1;
    float jx = static_cast<float>(u01() - 0.5);
    float jy = static_cast<float>(u01() - 0.5);
    float jz = static_cast<float>(u01() - 0.5);
    atomicAdd(&pos[3 * idx], jx);
    atomicAdd(&pos[3 * idx + 1], jy);
    atomicAdd(&pos[3 * idx + 2], jz);
}

// Block-reduce weighted position means, per-particle moment-matrix entries
// (upper triangle of sum w q q^T), and sum w^2.
// est layout: [0..2] = sum w*p ; [3..12] = M00,M01,M02,M03,M11,M12,M13,M22,M23,M33
//             [13] = sum w^2
__global__ void k_est_sums(const float* pos, const QuatF* att, const float* w,
                           int n, float* est) {
    __shared__ float sh[EST_N][33];
    int tid = threadIdx.x;
    int lane = tid & 31, warp = tid >> 5;
    float acc[EST_N];
#pragma unroll
    for (int k = 0; k < EST_N; ++k) acc[k] = 0.f;

    for (int i = blockIdx.x * blockDim.x + tid; i < n;
         i += gridDim.x * blockDim.x) {
        float wi = w[i];
        const QuatF qi = att[i];
        float q[4] = {qi.w, qi.x, qi.y, qi.z};
        acc[0] += wi * pos[3 * i];
        acc[1] += wi * pos[3 * i + 1];
        acc[2] += wi * pos[3 * i + 2];
        int e = 3;
#pragma unroll
        for (int r = 0; r < 4; ++r) {
#pragma unroll
            for (int c = r; c < 4; ++c) {
                acc[e] += wi * q[r] * q[c];
                ++e;
            }
        }
        acc[13] += wi * wi;
    }
#pragma unroll
    for (int k = 0; k < EST_N; ++k) {
        for (int off = 16; off > 0; off >>= 1)
            acc[k] += __shfl_down_sync(0xffffffffu, acc[k], off);
        if (lane == 0) sh[k][warp] = acc[k];
    }
    __syncthreads();
    int nwarp = (blockDim.x + 31) / 32;
    if (tid < EST_N) {
        float v = 0.f;
        for (int k = 0; k < nwarp; ++k) v += sh[tid][k];
        atomicAdd(&est[tid], v);
    }
}

// Block-reduce weighted upper triangles of the position covariance and the
// attitude tangent-space covariance around given means.
// cov layout: [0..5]   = Pxx,Pxy,Pxz,Pyy,Pyz,Pzz
//             [6..11]  = Axx,Axy,Axz,Ayy,Ayz,Azz
__global__ void k_est_covs(const float* pos, const QuatF* att, const float* w,
                           int n, float3 mean_p, QuatF qm, float* cov) {
    __shared__ float sh[COV_N][33];
    int tid = threadIdx.x;
    int lane = tid & 31, warp = tid >> 5;
    float acc[COV_N];
#pragma unroll
    for (int k = 0; k < COV_N; ++k) acc[k] = 0.f;

    for (int i = blockIdx.x * blockDim.x + tid; i < n;
         i += gridDim.x * blockDim.x) {
        float wi = w[i];
        // position residual
        float dp[3] = {pos[3 * i] - mean_p.x, pos[3 * i + 1] - mean_p.y,
                       pos[3 * i + 2] - mean_p.z};
        int e = 0;
#pragma unroll
        for (int r = 0; r < 3; ++r) {
#pragma unroll
            for (int c = r; c < 3; ++c) {
                acc[e] += wi * dp[r] * dp[c];
                ++e;
            }
        }
        // attitude residual dq = qm^-1 * qi as signed axis-angle
        QuatF dq = q_mul(q_conj(qm), att[i]);
        float sgn = dq.w >= 0.f ? 1.f : -1.f;
        float cw = fminf(fabsf(dq.w), 1.f);
        float ang = 2.f * acosf(cw);
        float axn[3] = {sgn * dq.x, sgn * dq.y, sgn * dq.z};
        float an = sqrtf(axn[0] * axn[0] + axn[1] * axn[1] +
                         axn[2] * axn[2]);
        float aav[3] = {0.f, 0.f, 0.f};
        if (an > 1e-9f) {
            aav[0] = axn[0] / an * ang;
            aav[1] = axn[1] / an * ang;
            aav[2] = axn[2] / an * ang;
        }
        e = 6;
#pragma unroll
        for (int r = 0; r < 3; ++r) {
#pragma unroll
            for (int c = r; c < 3; ++c) {
                acc[e] += wi * aav[r] * aav[c];
                ++e;
            }
        }
    }
#pragma unroll
    for (int k = 0; k < COV_N; ++k) {
        for (int off = 16; off > 0; off >>= 1)
            acc[k] += __shfl_down_sync(0xffffffffu, acc[k], off);
        if (lane == 0) sh[k][warp] = acc[k];
    }
    __syncthreads();
    int nwarp = (blockDim.x + 31) / 32;
    if (tid < COV_N) {
        float v = 0.f;
        for (int k = 0; k < nwarp; ++k) v += sh[tid][k];
        atomicAdd(&cov[tid], v);
    }
}

}  // namespace

extern "C" {

int pfc_device_available() {
    int ndev = 0;
    if (cudaGetDeviceCount(&ndev) != cudaSuccess || ndev < 1) return 0;
    cudaDeviceProp prop{};
    if (cudaGetDeviceProperties(&prop, 0) != cudaSuccess) return 0;
    return prop.major >= 3 ? 1 : 0;
}

void* pfc_create(int cap_obs) {
    auto* s = new (std::nothrow) PfState();
    if (!s) return nullptr;
    if (cap_obs < 1 || cap_obs > 4096) cap_obs = 64;
    s->obs_cap = cap_obs;
    CK_ALLOC(cudaMalloc((void**)&s->d_max, sizeof(int)));
    CK_ALLOC(cudaMalloc((void**)&s->d_sumexp, sizeof(float)));
    CK_ALLOC(cudaMalloc((void**)&s->d_est, EST_N * sizeof(float)));
    CK_ALLOC(cudaMalloc((void**)&s->d_cov, COV_N * sizeof(float)));
    CK_ALLOC(cudaMalloc((void**)&s->d_obs, cap_obs * sizeof(ObsF)));
    return reinterpret_cast<void*>(s);
fail:
    delete s;
    return nullptr;
}

int pfc_destroy(void* h) {
    auto* s = static_cast<PfState*>(h);
    if (!s) return 0;
    cudaFree(s->d_pos); cudaFree(s->d_att); cudaFree(s->d_w);
    cudaFree(s->d_ll); cudaFree(s->d_pos2); cudaFree(s->d_att2);
    cudaFree(s->d_cum); cudaFree(s->d_scan_tmp);
    cudaFree(s->d_lm); cudaFree(s->d_obs);
    cudaFree(s->d_rng); cudaFree(s->d_max); cudaFree(s->d_sumexp);
    cudaFree(s->d_est); cudaFree(s->d_cov);
    delete s;
    return 0;
}

int pfc_init(void* h, int n, unsigned int seed) {
    auto* s = static_cast<PfState*>(h);
    if (!s || n <= 0) return -1;
    s->n = n;
    s->seed = seed ? static_cast<uint64_t>(seed) : 42ull;
    s->gen = 0;
    size_t f3 = 3 * sizeof(float) * n, qn = sizeof(QuatF) * n,
           fn = sizeof(float) * n, dn = sizeof(double) * n;

#define PFC_REALLOC(ptr, bytes)                      \
    do {                                             \
        cudaFree(ptr);                               \
        ptr = nullptr;                               \
        CK(cudaMalloc((void**)&ptr, bytes));         \
    } while (0)

    PFC_REALLOC(s->d_pos, f3);
    PFC_REALLOC(s->d_att, qn);
    PFC_REALLOC(s->d_w, fn);
    PFC_REALLOC(s->d_ll, fn);
    PFC_REALLOC(s->d_pos2, f3);
    PFC_REALLOC(s->d_att2, qn);
    PFC_REALLOC(s->d_cum, dn);
    PFC_REALLOC(s->d_rng, n * sizeof(*s->d_rng));

    if (s->d_scan_tmp) { cudaFree(s->d_scan_tmp); s->d_scan_tmp = nullptr; }
    s->scan_tmp_bytes = 0;
    CK(cub::DeviceScan::InclusiveSum(nullptr, s->scan_tmp_bytes, s->d_w,
                                     s->d_cum, n));
    CK(cudaMalloc(&s->d_scan_tmp, s->scan_tmp_bytes));
#undef PFC_REALLOC
    return 0;
}

int pfc_upload_landmarks(void* h, const float* data, int count) {
    auto* s = static_cast<PfState*>(h);
    if (!s || !data || count < 0) return -1;
    if (count == 0) { s->lm_cap = 0; return 0; }
    if (count > s->lm_cap) {
        cudaFree(s->d_lm);
        CK(cudaMalloc((void**)&s->d_lm, 3 * sizeof(float) * count));
        s->lm_cap = count;
    }
    CK(cudaMemcpy(s->d_lm, data, 3 * sizeof(float) * count,
                  cudaMemcpyHostToDevice));
    return 0;
}

int pfc_reinit(void* h, float x, float y, float z, float qw, float qx,
               float qy, float qz, const float* pos_std,
               const float* rot_std) {
    auto* s = static_cast<PfState*>(h);
    if (!s || s->n <= 0) return -1;
    s->gen++;
    KL((k_seed_rng<<<(s->n + 255) / 256, 256>>>(s->d_rng, s->seed, s->gen,
                                               s->n)));
    float3 p0 = make_float3(x, y, z);
    float3 ps = make_float3(pos_std[0], pos_std[1], pos_std[2]);
    float3 rs = make_float3(rot_std[0], rot_std[1], rot_std[2]);
    QuatF q0{qw, qx, qy, qz};
    KL((k_reinit<<<(s->n + 255) / 256, 256>>>(s->d_rng, s->d_pos, s->d_att,
                                             s->d_w, s->n, p0, q0, ps, rs)));
    return cudaPeekAtLastError() == cudaSuccess ? 0 : -1;
}

int pfc_predict(void* h, float tx, float ty, float tz, float qw, float qx,
                float qy, float qz, const float* tstd, const float* rstd) {
    auto* s = static_cast<PfState*>(h);
    if (!s || s->n <= 0) return -1;
    s->gen++;
    KL((k_seed_rng<<<(s->n + 255) / 256, 256>>>(s->d_rng, s->seed, s->gen,
                                               s->n)));
    float3 tb = make_float3(tx, ty, tz);
    float3 ts = make_float3(tstd[0], tstd[1], tstd[2]);
    float3 rs = make_float3(rstd[0], rstd[1], rstd[2]);
    QuatF qo{qw, qx, qy, qz};
    KL((k_predict<<<(s->n + 255) / 256, 256>>>(s->d_rng, s->d_pos, s->d_att,
                                              s->n, tb, qo, ts, rs)));
    return cudaPeekAtLastError() == cudaSuccess ? 0 : -1;
}

int pfc_weight(void* h, const void* obs, int nobs, float sigma_r,
               float sigma_b, float gate) {
    auto* s = static_cast<PfState*>(h);
    if (!s || s->n <= 0) return -1;
    nobs = nobs > s->obs_cap ? s->obs_cap : nobs;
    if (nobs <= 0) {
        KL((k_reset_uniform<<<(s->n + 255) / 256, 256>>>(s->d_w, s->n)));
        return 0;
    }
    CK(cudaMemcpy(s->d_obs, obs, static_cast<size_t>(nobs) * sizeof(ObsF),
                  cudaMemcpyHostToDevice));
    KL((k_weight<<<(s->n + 255) / 256, 256>>>(
        s->d_pos, s->d_att, s->d_lm, s->lm_cap, s->d_obs, nobs, sigma_r,
        sigma_b, gate, s->d_ll, s->n)));

    int minus_inf_ord = float_as_ordered(-INFINITY);
    CK(cudaMemcpy(s->d_max, &minus_inf_ord, sizeof(int),
                  cudaMemcpyHostToDevice));
    KL((k_ll_max<<<256, 256>>>(s->d_ll, s->n, s->d_max)));
    int max_ord = 0;
    CK(cudaMemcpy(&max_ord, s->d_max, sizeof(int), cudaMemcpyDeviceToHost));
    float maxll = ordered_as_float(max_ord);

    float zero = 0.f;
    CK(cudaMemcpy(s->d_sumexp, &zero, sizeof(float), cudaMemcpyHostToDevice));
    KL((k_ll_expsum<<<256, 256>>>(s->d_ll, s->n, maxll, s->d_sumexp)));
    float total = 0.f;
    CK(cudaMemcpy(&total, s->d_sumexp, sizeof(float), cudaMemcpyDeviceToHost));

    if (!(total > 0.f) || !isfinite(total)) {
        KL((k_reset_uniform<<<(s->n + 255) / 256, 256>>>(s->d_w, s->n)));
    } else {
        KL((k_ll_normalize<<<(s->n + 255) / 256, 256>>>(s->d_ll, s->n, maxll,
                                                       total, s->d_w)));
    }
    return 0;
}

int pfc_resample(void* h, float random_inject_ratio) {
    auto* s = static_cast<PfState*>(h);
    if (!s || s->n <= 0) return -1;

    CK(cub::DeviceScan::InclusiveSum(s->d_scan_tmp, s->scan_tmp_bytes, s->d_w,
                                     s->d_cum, s->n));
    double total = 0.0;
    CK(cudaMemcpy(&total, s->d_cum + (s->n - 1), sizeof(double),
                  cudaMemcpyDeviceToHost));
    if (!(total > 0.0) || !isfinite(total)) {
        KL((k_reset_uniform<<<(s->n + 255) / 256, 256>>>(s->d_w, s->n)));
        return 0;
    }

    // deterministic u0 stream on host (LCG, mirrors cpu.rs's SmallRng role)
    s->lcg = s->lcg * 6364136223846793005ull + 1442695040888963407ull;
    double u01 = static_cast<double>(s->lcg >> 11) *
                 (1.0 / 9007199254740992.0);  // [0,1)
    double step = total / static_cast<double>(s->n);
    double u0 = u01 * step;

    KL((k_resample_pick<<<(s->n + 255) / 256, 256>>>(
        s->d_pos, s->d_att, s->d_cum, s->n, u0, total, s->d_pos2, s->d_att2)));
    CK(cudaMemcpy(s->d_pos, s->d_pos2, 3 * sizeof(float) * s->n,
                  cudaMemcpyDeviceToDevice));
    CK(cudaMemcpy(s->d_att, s->d_att2, sizeof(QuatF) * s->n,
                  cudaMemcpyDeviceToDevice));
    KL((k_reset_uniform<<<(s->n + 255) / 256, 256>>>(s->d_w, s->n)));

    int k = (random_inject_ratio > 0.f)
                ? static_cast<int>(random_inject_ratio * s->n)
                : 0;
    if (k > 0) {
        if (k > s->n) k = s->n;
        KL((k_inject<<<(k + 255) / 256, 256>>>(s->d_pos, s->n, k, s->lcg,
                                               s->gen)));
    }
    return 0;
}

// Pass 1: weighted sums — [0..2] mean position, [3..12] packed upper
// triangle of the quaternion moment matrix, [13] sum w^2. The Markley
// quaternion mean is solved on the HOST (identical nalgebra path as the
// CPU backend); pass the solved mean back via pfc_covs.
int pfc_est_sums(void* h, float* out) {
    auto* s = static_cast<PfState*>(h);
    if (!s || s->n <= 0 || !out) return -1;
    float zero[EST_N] = {0.f};
    CK(cudaMemcpy(s->d_est, zero, sizeof(zero), cudaMemcpyHostToDevice));
    int blocks = std::min(1024, (s->n + 255) / 256);
    KL((k_est_sums<<<blocks, 256>>>(s->d_pos, s->d_att, s->d_w, s->n,
                                   s->d_est)));
    CK(cudaMemcpy(out, s->d_est, EST_N * sizeof(float),
                  cudaMemcpyDeviceToHost));
    return 0;
}

// Pass 2: weighted position + attitude-tangent covariance packed
// upper triangles around the given means.
int pfc_covs(void* h, float mx, float my, float mz, float qw, float qx,
             float qy, float qz, float* out) {
    auto* s = static_cast<PfState*>(h);
    if (!s || s->n <= 0 || !out) return -1;
    float zero[COV_N] = {0.f};
    CK(cudaMemcpy(s->d_cov, zero, sizeof(zero), cudaMemcpyHostToDevice));
    int blocks = std::min(1024, (s->n + 255) / 256);
    float3 mp = make_float3(mx, my, mz);
    QuatF qm{qw, qx, qy, qz};
    KL((k_est_covs<<<blocks, 256>>>(s->d_pos, s->d_att, s->d_w, s->n, mp, qm,
                                    s->d_cov)));
    CK(cudaMemcpy(out, s->d_cov, COV_N * sizeof(float),
                  cudaMemcpyDeviceToHost));
    return 0;
}

int pfc_snapshot(void* h, float* pos, float* quat) {
    auto* s = static_cast<PfState*>(h);
    if (!s || s->n <= 0 || !pos || !quat) return -1;
    CK(cudaMemcpy(pos, s->d_pos, 3 * sizeof(float) * s->n,
                  cudaMemcpyDeviceToHost));
    CK(cudaMemcpy(quat, s->d_att, sizeof(QuatF) * s->n,
                  cudaMemcpyDeviceToHost));
    return 0;

}  // extern "C"

}  // extern "C"

// ====================================================================
//  GPU global localization (CBGL-style): pfgl_* API
//
//  Pipeline (Filotheou, arXiv:2307.14247):
//    1. disperse pose hypotheses over the free space of an occupancy grid
//    2. raycast a virtual map-scan from each hypothesis (2D DDA)
//    3. score CAER = sum |real_scan - map_scan| per ray   (O(N_rays))
//    4. keep the bottom-k hypotheses by CAER (CUB radix sort)
//
//  All stages run on device; the host reads back only k winners.

namespace {

struct GlState {
    int   W = 0, H = 0;            // grid dims (cells)
    float res = 0.05f;             // meters per cell
    float ox = 0.f, oy = 0.f;      // world coords of cell (0,0) corner

    unsigned char* d_grid = nullptr;  // W*H, 1 = occupied
    int2*  d_free = nullptr;          // free-cell list
    int    nfree = 0;
    int    cap_hyp = 0;

    // hypothesis batch
    int      nh = 0;
    float*   d_hyp = nullptr;         // nh * {x,y,theta}
    float*   d_caer = nullptr;        // nh scores
    int*     d_keys = nullptr;        // ordered-float CAER bits
    int*     d_keys2 = nullptr;
    int*     d_vals = nullptr;        // hypothesis indices
    int*     d_vals2 = nullptr;
    void*    d_sort_tmp = nullptr;
    size_t   sort_tmp_bytes = 0;

    float* d_out = nullptr;           // top-k poses, k*3
};

__device__ inline bool gl_world_to_cell(const GlState& g, float wx, float wy,
                                        int& cx, int& cy) {
    const float fx = (wx - g.ox) / g.res;
    const float fy = (wy - g.oy) / g.res;
    cx = static_cast<int>(floorf(fx));
    cy = static_cast<int>(floorf(fy));
    return cx >= 0 && cy >= 0 && cx < g.W && cy < g.H;
}

// 2D DDA: distance to first occupied cell along a unit direction.
__device__ inline float gl_cast_ray(const GlState& g, float px, float py,
                                    float dirx, float diry, float rmax) {
    int cx, cy;
    if (!gl_world_to_cell(g, px, py, cx, cy)) return rmax;
    const int sx = dirx > 0 ? 1 : -1, sy = diry > 0 ? 1 : -1;
    const float adx = fabsf(dirx), ady = fabsf(diry);
    const float tdx = adx > 1e-9f ? g.res / adx : INFINITY;  // per cell step
    const float tdy = ady > 1e-9f ? g.res / ady : INFINITY;
    const float frx = (px - g.ox) / g.res - cx;  // in-cell fraction [0,1)
    const float fry = (py - g.oy) / g.res - cy;
    float tmx = adx > 1e-9f ? ((dirx > 0 ? 1.f - frx : frx) * tdx) : INFINITY;
    float tmy = ady > 1e-9f ? ((diry > 0 ? 1.f - fry : fry) * tdy) : INFINITY;
    float dist = 0.f;
    for (int step = 0; step < 8192; ++step) {
        if (tmx <= tmy) {
            dist = tmx; tmx += tdx; cx += sx;
        } else {
            dist = tmy; tmy += tdy; cy += sy;
        }
        if (dist > rmax) return rmax;
        if (cx < 0 || cy < 0 || cx >= g.W || cy >= g.H) return rmax;
        if (g.d_grid[cy * g.W + cx]) return dist;
    }
    return rmax;
}

// Philox-based uniform dispersion over free cells.
__global__ void k_gl_gen(GlState g, int nfree, unsigned long long seed) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= g.cap_hyp) return;
    curandStatePhilox4_32_10_t st;
    curand_init(seed, static_cast<unsigned long long>(i), 0, &st);
    const float4 u = curand_uniform4(&st);
    const int c =
        min(static_cast<int>(u.w * static_cast<float>(nfree)), nfree - 1);
    const int2 cell = g.d_free[c];
    g.d_hyp[3 * i] = g.ox + (static_cast<float>(cell.x) + u.x) * g.res;
    g.d_hyp[3 * i + 1] = g.oy + (static_cast<float>(cell.y) + u.y) * g.res;
    g.d_hyp[3 * i + 2] = -3.14159265f + 6.28318530f * u.z;
}

// One block per hypothesis: cast all rays cooperatively, reduce |diff|.
__global__ void k_gl_score(GlState g, float rmax, int nrays, float ang_span,
                           const float* real_scan, float* caer) {
    const int h = blockIdx.x;
    if (h >= g.nh || blockDim.x < 32) return;
    __shared__ float sh[33];
    const int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5;
    float acc = 0.f;

    const float x = g.d_hyp[3 * h], y = g.d_hyp[3 * h + 1],
                th = g.d_hyp[3 * h + 2];
    for (int r = tid; r < nrays; r += blockDim.x) {
        const float a =
            th + (nrays > 1 ? ang_span * static_cast<float>(r) /
                                    static_cast<float>(nrays - 1)
                            : 0.f);
        float dx, dy;
        sincosf(a, &dy, &dx);
        const float vm = gl_cast_ray(g, x, y, dx, dy, rmax);
        acc += fabsf(real_scan[r] - vm);
    }
#pragma unroll
    for (int off = 16; off > 0; off >>= 1)
        acc += __shfl_down_sync(0xffffffffu, acc, off);
    if (lane == 0) sh[warp] = acc;
    __syncthreads();
    const int nwarp = (blockDim.x + 31) / 32;
    if (tid == 0) {
        float v = 0.f;
        for (int k2 = 0; k2 < nwarp; ++k2) v += sh[k2];
        caer[h] = v;
    }
}

// Stream compaction of free cells (one-time map setup).
__global__ void k_gl_compact(GlState g, int* count) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= g.W * g.H) return;
    if (!g.d_grid[i]) {
        const int slot = atomicAdd(count, 1);
        g.d_free[slot] = make_int2(i % g.W, i / g.W);
    }
}

__global__ void k_gl_iota(int* v, int n) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) v[i] = i;
}

// Ordered-float key packing: integer compare == float compare.
__global__ void k_gl_pack(const float* caer, int* keys, int n) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) keys[i] = float_as_ordered(caer[i]);
}

// Gather of the top-k winning poses.
__global__ void k_gl_gather(const int* vals_sorted, const float* hyp,
                            float* out, int* k_io) {
    const int k = k_io[0];
    for (int j = 0; j < k; ++j) {
        const int src = vals_sorted[j];
        out[3 * j] = hyp[3 * src];
        out[3 * j + 1] = hyp[3 * src + 1];
        out[3 * j + 2] = hyp[3 * src + 2];
    }
    k_io[1] = k;
}

}  // namespace

extern "C" {

// occ: row-major W*H bytes, 1 = occupied. Returns NULL on failure.
void* pfgl_create(int W, int H, float res, float ox, float oy,
                  const unsigned char* occ, int cap_hyp) {
    auto* s = new (std::nothrow) GlState();
    int* d_count = nullptr;
    size_t tn = 0;
    if (!s || W <= 0 || H <= 0 || cap_hyp <= 0) {
        delete s;
        return nullptr;
    }
    s->W = W; s->H = H; s->res = res; s->ox = ox; s->oy = oy;
    s->cap_hyp = cap_hyp;
    CK_ALLOC(cudaMalloc((void**)&s->d_grid, static_cast<size_t>(W) * H));
    CK_ALLOC(cudaMalloc((void**)&s->d_free,
                        static_cast<size_t>(W) * H * sizeof(int2)));
    CK_ALLOC(cudaMalloc((void**)&s->d_hyp,
                        static_cast<size_t>(cap_hyp) * 3 * sizeof(float)));
    CK_ALLOC(cudaMalloc((void**)&s->d_caer,
                        static_cast<size_t>(cap_hyp) * sizeof(float)));
    CK_ALLOC(cudaMalloc((void**)&s->d_keys,
                        static_cast<size_t>(cap_hyp) * sizeof(int)));
    CK_ALLOC(cudaMalloc((void**)&s->d_keys2,
                        static_cast<size_t>(cap_hyp) * sizeof(int)));
    CK_ALLOC(cudaMalloc((void**)&s->d_vals,
                        static_cast<size_t>(cap_hyp) * sizeof(int)));
    CK_ALLOC(cudaMalloc((void**)&s->d_vals2,
                        static_cast<size_t>(cap_hyp) * sizeof(int)));
    CK_ALLOC(cudaMalloc((void**)&s->d_out,
                        static_cast<size_t>(cap_hyp) * 3 * sizeof(float)));

    CK_ALLOC(cudaMemcpy(s->d_grid, occ, static_cast<size_t>(W) * H,
                  cudaMemcpyHostToDevice));
    CK_ALLOC(cudaMalloc((void**)&d_count, sizeof(int)));
    CK_ALLOC(cudaMemset(d_count, 0, sizeof(int)));
    do {
        k_gl_compact<<<(W * H + 255) / 256, 256>>>(*s, d_count);
        cudaError_t e_ = cudaGetLastError();
        if (e_ != cudaSuccess) {
            fprintf(stderr, "pf_cuda k_gl_compact: %s\n",
                    cudaGetErrorString(e_));
            goto fail;
        }
    } while (0);
    CK_ALLOC(cudaMemcpy(&s->nfree, d_count, sizeof(int), cudaMemcpyDeviceToHost));
    cudaFree(d_count);
    d_count = nullptr;

    CK_ALLOC(cub::DeviceRadixSort::SortPairs(nullptr, tn, s->d_keys, s->d_keys2,
                                       s->d_vals, s->d_vals2, cap_hyp));
    CK_ALLOC(cudaMalloc(&s->d_sort_tmp, tn));
    s->sort_tmp_bytes = tn;
    return reinterpret_cast<void*>(s);
fail:
    delete s;
    return nullptr;
}

int pfgl_destroy(void* h) {
    auto* s = static_cast<GlState*>(h);
    if (!s) return 0;
    cudaFree(s->d_grid); cudaFree(s->d_free); cudaFree(s->d_hyp);
    cudaFree(s->d_caer); cudaFree(s->d_keys); cudaFree(s->d_keys2);
    cudaFree(s->d_vals); cudaFree(s->d_vals2); cudaFree(s->d_sort_tmp);
    cudaFree(s->d_out);
    delete s;
    return 0;
}

int pfgl_free_cells(void* h) {
    auto* s = static_cast<GlState*>(h);
    return s ? s->nfree : -1;
}

// Disperse n hypotheses uniformly over free cells (device RNG).
int pfgl_generate(void* h, int n, unsigned long long seed) {
    auto* s = static_cast<GlState*>(h);
    if (!s || s->nfree <= 0) return -1;
    if (n <= 0 || n > s->cap_hyp) return -1;
    s->nh = n;
    KL((k_gl_gen<<<(n + 255) / 256, 256>>>(*s, s->nfree, seed)));
    KL((k_gl_iota<<<(n + 255) / 256, 256>>>(s->d_vals, n)));
    return 0;
}

// Score current hypotheses against a real scan.
int pfgl_score(void* h, const float* real_scan, int nrays, float rmax,
               float ang_span) {
    auto* s = static_cast<GlState*>(h);
    if (!s || s->nh <= 0 || !real_scan || nrays <= 0) return -1;
    float* d_rs = nullptr;
    CK(cudaMalloc((void**)&d_rs, static_cast<size_t>(nrays) * sizeof(float)));
    CK(cudaMemcpy(d_rs, real_scan, static_cast<size_t>(nrays) * sizeof(float),
                  cudaMemcpyHostToDevice));
    KL((k_gl_score<<<s->nh, 128>>>(*s, rmax, nrays, ang_span, d_rs, s->d_caer)));
    cudaFree(d_rs);
    return 0;
}

// Sort by CAER ascending, write top-k poses to out (k*3). Returns k.
int pfgl_topk(void* h, int k, float* out) {
    auto* s = static_cast<GlState*>(h);
    if (!s || !out || k <= 0 || s->nh <= 0) return -1;
    if (k > s->nh) k = s->nh;
    KL((k_gl_pack<<<(s->nh + 255) / 256, 256>>>(s->d_caer, s->d_keys, s->nh)));
    size_t tn = 0;
    CK(cub::DeviceRadixSort::SortPairs(nullptr, tn, s->d_keys, s->d_keys2,
                                       s->d_vals, s->d_vals2, s->nh));
    if (tn > s->sort_tmp_bytes) {
        cudaFree(s->d_sort_tmp);
        CK(cudaMalloc(&s->d_sort_tmp, tn));
        s->sort_tmp_bytes = tn;
    }
    CK(cub::DeviceRadixSort::SortPairs(s->d_sort_tmp, s->sort_tmp_bytes,
                                       s->d_keys, s->d_keys2, s->d_vals,
                                       s->d_vals2, s->nh));
    int kd[2] = {k, 0};
    int* d_k = nullptr;
    CK(cudaMalloc((void**)&d_k, sizeof(kd)));
    CK(cudaMemcpy(d_k, kd, sizeof(kd), cudaMemcpyHostToDevice));
    KL((k_gl_gather<<<1, 1>>>(s->d_vals2, s->d_hyp, s->d_out, d_k)));
    CK(cudaMemcpy(&kd[1], d_k + 1, sizeof(int), cudaMemcpyDeviceToHost));
    CK(cudaMemcpy(out, s->d_out,
                  static_cast<size_t>(kd[1]) * 3 * sizeof(float),
                  cudaMemcpyDeviceToHost));
    cudaFree(d_k);
    return kd[1];
}

}  // extern "C"
