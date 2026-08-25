// pf_kernels.cu — CUDA backend for cuda_pf_localization.
//
// Implements the C ABI declared in crates/pf_core/src/cuda.rs:
//   pfc_create / pfc_destroy / pfc_init / pfc_upload_landmarks /
//   pfc_reinit / pfc_predict / pfc_weight / pfc_resample / pfc_estimate /
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
//   k_est_sums       weighted position, per-particle moment matrix,
//                    sum(w^2) for ESS  -> 14 partials, block-reduced
//   k_quat_mean      power iteration on the 4x4 moment matrix

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
    QuatF* d_qmean = nullptr;

    float* d_lm = nullptr;        // 3*lm_cap, dense by id
    int    lm_cap = 0;

    ObsF*  d_obs = nullptr;
    int    obs_cap = 0;

    curandStatePhilox4_32_10_t* d_rng = nullptr;

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

// Dominant eigenvector of the symmetric 4x4 moment matrix (row-major full).
__global__ void k_quat_mean(const float* M, QuatF* out) {
    float4 v = make_float4(1.f, 0.f, 0.f, 0.f);
    for (int it = 0; it < 48; ++it) {
        float4 r;
        r.x = M[0] * v.x + M[1] * v.y + M[2] * v.z + M[3] * v.w;
        r.y = M[4] * v.x + M[5] * v.y + M[6] * v.z + M[7] * v.w;
        r.z = M[8] * v.x + M[9] * v.y + M[10] * v.z + M[11] * v.w;
        r.w = M[12] * v.x + M[13] * v.y + M[14] * v.z + M[15] * v.w;
        float nr = sqrtf(r.x * r.x + r.y * r.y + r.z * r.z + r.w * r.w);
        if (nr < 1e-20f) break;
        v = make_float4(r.x / nr, r.y / nr, r.z / nr, r.w / nr);
    }
    if (v.w < 0.f) v = make_float4(-v.w, -v.x, -v.y, -v.z);
    *out = {v.w, v.x, v.y, v.z};
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
    CK_ALLOC(cudaMalloc((void**)&s->d_qmean, sizeof(QuatF)));
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
    cudaFree(s->d_est); cudaFree(s->d_qmean);
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

int pfc_resample(void* h) {
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
    return 0;
}

int pfc_estimate(void* h, void* out) {
    struct CEstimate {
        float x, y, z, qw, qx, qy, qz, ess;
        int valid;
    };
    auto* s = static_cast<PfState*>(h);
    auto* e = static_cast<CEstimate*>(out);
    if (!s || s->n <= 0 || !e) return -1;

    float zero[EST_N] = {0.f};
    CK(cudaMemcpy(s->d_est, zero, sizeof(zero), cudaMemcpyHostToDevice));
    int blocks = std::min(1024, (s->n + 255) / 256);
    KL((k_est_sums<<<blocks, 256>>>(s->d_pos, s->d_att, s->d_w, s->n,
                                   s->d_est)));

    float est[EST_N];
    CK(cudaMemcpy(est, s->d_est, sizeof(est), cudaMemcpyDeviceToHost));

    // symmetric moment matrix from upper-triangle partials
    float Mfull[16];
    int eidx = 3;
    for (int r = 0; r < 4; ++r) {
        for (int c = 0; c < 4; ++c) {
            Mfull[4 * r + c] =
                (c >= r) ? est[eidx + (r * (r + 1)) / 2 + (c - r)] : 0.f;
        }
    }
    for (int r = 0; r < 4; ++r)
        for (int c = 0; c < r; ++c) Mfull[4 * r + c] = Mfull[4 * c + r];

    float* d_M = nullptr;
    CK(cudaMalloc((void**)&d_M, sizeof(Mfull)));
    CK(cudaMemcpy(d_M, Mfull, sizeof(Mfull), cudaMemcpyHostToDevice));
    KL((k_quat_mean<<<1, 1>>>(d_M, s->d_qmean)));
    QuatF qm;
    CK(cudaMemcpy(&qm, s->d_qmean, sizeof(QuatF), cudaMemcpyDeviceToHost));
    CK(cudaFree(d_M));

    double ess = (est[13] > 0.f) ? 1.0 / static_cast<double>(est[13]) : 0.0;

    e->x = est[0]; e->y = est[1]; e->z = est[2];
    e->qw = qm.w; e->qx = qm.x; e->qy = qm.y; e->qz = qm.z;
    e->ess = static_cast<float>(ess);
    e->valid = 1;
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
}

}  // extern "C"
