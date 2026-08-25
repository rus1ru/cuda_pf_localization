
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
    int* d_count = nullptr;
    CK_ALLOC(cudaMalloc((void**)&d_count, sizeof(int)));
    CK_ALLOC(cudaMemset(d_count, 0, sizeof(int)));
    KL((k_gl_compact<<<(W * H + 255) / 256, 256>>>(*s, d_count)));
    CK_ALLOC(cudaMemcpy(&s->nfree, d_count, sizeof(int), cudaMemcpyDeviceToHost));
    cudaFree(d_count);

    size_t tn = 0;
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
