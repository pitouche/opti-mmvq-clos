// (c) 2026 guile — MIT. Microbench kernel mmvq (mul_mat_vec_q) Q6_K/Q8_0 T=1/4 — 23/10.
//
// mb_mmvq.cu — harnais standalone des kernels mmvq de la V100, SANS rebuild du .so prod.
// Meme principe que mb_fa : dlopen du .so prod + libggml-base.so.0, pilotage de
//   ggml_cuda_mul_mat_vec_q(ctx, src0, src1, ids=nullptr, dst, fusion=nullptr)
// (symbole exporté, c'est exactement le dispatch que fait la prod pour le verify MTP).
//
// Geometries REELLES du modele prod (Qwen3.8-27B, dense) :
//   - FFN gate/up :  src0 [K=5120,  rows=17408]
//   - FFN down    :  src0 [K=17408, rows=5120]
//   - types : Q6_K (36,9 % du temps kernel prod) et Q8_0
//   - T = ncols_dst = 1 (decode) et 4 (verify MTP)
// src1 est F32 [K, T] (le kernel quantise F32->Q8_1 lui-meme, comme en prod).
//
// Validation : dequant CPU de la 1ere ligne (reference) vs sortie device, max err.
//
// Usage : ./mb_mmvq [iters]     (defaut 50)
// Sortie : JSONL stdout, 1 ligne par (type, K, rows, T).

#include <dlfcn.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <cstdint>
#include <cmath>
#include <vector>
#include <algorithm>

#define CK(x) do{ cudaError_t e_=(x); if(e_){fprintf(stderr,"CUDA err @%d: %s\n",__LINE__,cudaGetErrorString(e_));exit(1);} }while(0)

// ---------------------------------------------------------------- types ggml minimaux
// Layout EXACT de ggml-base .so.0 (identique a mb_fa, valide).
struct ggml_tensor {
    int32_t  type;
    void    *buffer;
    int64_t  ne[4];
    size_t   nb[4];
    int32_t  op;
    int32_t  op_params[16];
    int32_t  flags;
    ggml_tensor *src[10];
    ggml_tensor *view_src;
    size_t   view_offs;
    void    *data;
    char     name[64];
    void    *extra;
    char     padding[8];
};
static_assert(sizeof(ggml_tensor) % 8 == 0, "align 8");
struct ggml_context;
typedef struct ggml_backend_buffer ggml_backend_buffer;
typedef ggml_backend_buffer * ggml_backend_buffer_t;

// enum ggml_type (fork ggml, identique prod)
#define GGML_TYPE_F32 0
#define GGML_TYPE_Q8_0 8
#define GGML_TYPE_Q5_K 13
#define GGML_TYPE_Q6_K 14

// ggml_block_q6_K (ggml-common.h, QK_K=256) : 256 octets
struct block_q6_K {
    uint8_t  ql[128];      // quants, lower 4 bits
    uint8_t  qh[64];       // quants, upper 2 bits
    int8_t   scales[16];   // scales, 8 bits
    __half   d;            // super-block scale
};
static_assert(sizeof(block_q6_K) == 210, "q6_K block 210 = 2 + 256/16 + 3*256/4");
struct block_q8_0 { __half d; int8_t qs[32]; };
static_assert(sizeof(block_q8_0) == 34, "q8_0 block 34");

// ggml_block_q5_K (ggml-common.h L344, QK_K=256) : 174 octets
// dm = {d, dmin} (half2), scales[12] (6-bit packs sc/m), qh[32], qs[128]
struct block_q5_K {
    __half2  dm;
    uint8_t  scales[12];
    uint8_t  qh[32];
    uint8_t  qs[128];
};
static_assert(sizeof(block_q5_K) == 176, "q5_K block 176 = 4 + 12 + 32 + 128");

struct ggml_init_params { size_t mem_size; void * mem_buffer; bool no_alloc; };

// ---------------------------------------------------------------- API ggml (ptrs)
typedef struct ggml_context * (*f_ggml_init)(ggml_init_params);
typedef ggml_backend_buffer_t  (*f_buft_alloc_buffer)(void *, size_t);
typedef void *                 (*f_buf_get_base)(ggml_backend_buffer_t);
typedef void                   (*f_buf_free)(ggml_backend_buffer_t);
typedef void                   (*f_backend_free)(void *);
typedef void (*f_mmvecq)(void *, const ggml_tensor *, const ggml_tensor *,
                         const ggml_tensor *, ggml_tensor *, const void *);

static const char * SO_CUDA() {
    static const char * p = getenv("MB_MMVQ_SO") ? getenv("MB_MMVQ_SO")
        : "/home/guile/llama-experiments/ollama-v100/lib/ollama/cuda_v12/libggml-cuda.so";
    return p;
}
static const char * SO_BASE = "/home/guile/llama-experiments/ollama-v100/lib/ollama/libggml-base.so.0";

struct Env {
    void * hbase, * hcuda;
    f_ggml_init        ggml_init;
    f_buf_free         buf_free;
    f_backend_free     backend_free;
    f_mmvecq           mmvecq;
    void * backend = nullptr;
    void * ctx_cuda = nullptr;
    void * buft = nullptr;            // ggml_backend_buffer_type *
    f_buft_alloc_buffer alloc = nullptr;
    ggml_context * gctx = nullptr;
};

static void *must_dlsym(void *h, const char *name) {
    void *p = dlsym(h, name);
    if (!p) { fprintf(stderr, "dlsym %s manquant\n", name); exit(1); }
    return p;
}

// Retourne le buffer_type device et sa vtable->alloc_buffer (1er slot apres vtable).
static void get_buft(Env & E, f_buft_alloc_buffer & alloc_out, void * & buft_out) {
    void * dev = *(void **)((char *)E.backend + 136);
    void * get_buft = *(void **)((char *)dev + 6 * sizeof(void *));
    void * buft = ((void *(*)(void *))get_buft)(dev);
    alloc_out = (f_buft_alloc_buffer)(*(void **)((char *)buft + sizeof(void *)));
    buft_out = buft;
}

static void env_open(Env & E) {
    // Meme env que la prod pour le mmvq (W4R4 Q6_K T=4 actif).
    setenv("GGML_CUDA_VOLTA_Q6_W4R4", "1", 1);
    setenv("CUDA_VISIBLE_DEVICES", "0", 1);
    E.hbase = dlopen(SO_BASE, RTLD_NOW | RTLD_LOCAL);
    E.hcuda = dlopen(SO_CUDA(), RTLD_NOW | RTLD_LOCAL);
    if (!E.hbase || !E.hcuda) { fprintf(stderr, "dlopen echoué (%s) : %s\n", SO_CUDA(), dlerror()); exit(1); }
    E.ggml_init         = (f_ggml_init)must_dlsym(E.hbase, "ggml_init");
    E.buf_free          = (f_buf_free)must_dlsym(E.hbase, "ggml_backend_buffer_free");
    E.backend_free      = (f_backend_free)must_dlsym(E.hbase, "ggml_backend_free");
    E.mmvecq            = (f_mmvecq)must_dlsym(E.hcuda,
        "_Z23ggml_cuda_mul_mat_vec_qR25ggml_backend_cuda_contextPK11ggml_tensorS3_S3_PS1_PK29ggml_cuda_mm_fusion_args_host");
    void *init_p = must_dlsym(E.hcuda, "ggml_backend_cuda_init");
    E.backend = ((ggml_backend_buffer_t(*)(int))init_p)(0);
    if (!E.backend) { fprintf(stderr, "ggml_backend_cuda_init(0) echoué\n"); exit(1); }
    // ggml_backend = { guid(8), iface(128B), device(8B)@136, context(8B)@144 }
    E.ctx_cuda = *(void **)((char *)E.backend + 144);
    if (!E.ctx_cuda) { fprintf(stderr, "ctx_cuda null\n"); exit(1); }
    get_buft(E, E.alloc, E.buft);
    if (!E.alloc) { fprintf(stderr, "alloc null\n"); exit(1); }
    fprintf(stderr, "[mb] ok (backend=%p ctx=%p so=%s)\n", E.backend, E.ctx_cuda, SO_CUDA());
}

static void env_close(Env & E) {
    if (E.gctx)      ((void(*)(ggml_context*))must_dlsym(E.hbase, "ggml_free"))(E.gctx);
    if (E.backend)   E.backend_free(E.backend);
    if (E.hcuda)     dlclose(E.hcuda);
    if (E.hbase)     dlclose(E.hbase);
}

// ---------------------------------------------------------------- reference CPU
static void dequant_q6_K(const block_q6_K * x, float * y) {
    const float d = __half2float(x->d);
    for (int n = 0; n < 256; n += 128) {
        for (int l = 0; l < 32; ++l) {
            int is = l / 16;
            const int8_t q1 = (int8_t)((x->ql[l +  0] & 0xF) | (((x->qh[l] >> 0) & 3) << 4)) - 32;
            const int8_t q2 = (int8_t)((x->ql[l + 32] & 0xF) | (((x->qh[l] >> 2) & 3) << 4)) - 32;
            const int8_t q3 = (int8_t)((x->ql[l +  0] >> 4) | (((x->qh[l] >> 4) & 3) << 4)) - 32;
            const int8_t q4 = (int8_t)((x->ql[l + 32] >> 4) | (((x->qh[l] >> 6) & 3) << 4)) - 32;
            y[n + l +  0] = d * x->scales[is + 0] * q1;
            y[n + l + 32] = d * x->scales[is + 2] * q2;
            y[n + l + 64] = d * x->scales[is + 4] * q3;
            y[n + l + 96] = d * x->scales[is + 6] * q4;
        }
    }
}
static void dequant_q8_0(const block_q8_0 * x, float * y) {
    const float d = __half2float(x->d);
    for (int i = 0; i < 32; ++i) y[i] = d * x->qs[i];
}
// get_scale_min_k4 (ggml-quants.c L880) : decodage 6-bit des scales/mins packes
static inline void get_scale_min_k4(int j, const uint8_t * q, uint8_t * d, uint8_t * m) {
    if (j < 4) { *d = q[j] & 63; *m = q[j + 4] & 63; }
    else       { *d = (q[j+4] & 0xF) | ((q[j-4] >> 6) << 4);
                 *m = (q[j+4] >>  4) | ((q[j-0] >> 6) << 4); }
}
static void dequant_q5_K(const block_q5_K * x, float * y) {
    const float d   = __half2float(x->dm.x);
    const float min = __half2float(x->dm.y);
    const uint8_t * ql = x->qs;
    const uint8_t * qh = x->qh;
    int is = 0;
    uint8_t sc, m;
    uint8_t u1 = 1, u2 = 2;
    for (int j = 0; j < 256; j += 64) {
        get_scale_min_k4(is + 0, x->scales, &sc, &m);
        const float d1 = d * sc; const float m1 = min * m;
        get_scale_min_k4(is + 1, x->scales, &sc, &m);
        const float d2 = d * sc; const float m2 = min * m;
        for (int l = 0; l < 32; ++l) y[j + l +  0] = d1 * ((ql[l] & 0xF) + (qh[l] & u1 ? 16 : 0)) - m1;
        for (int l = 0; l < 32; ++l) y[j + l + 32] = d2 * ((ql[l] >> 4) + (qh[l] & u2 ? 16 : 0)) - m2;
        ql += 32; is += 2;
        u1 <<= 2; u2 <<= 2;
    }
}

struct Shape { const char * name; int type; int K; int rows; };

int main(int argc, char ** argv) {
    int iters = argc > 1 ? atoi(argv[1]) : 50;
    const char * shfilter = argc > 2 ? argv[2] : nullptr;   // sous-chaîne de la shape
    int Tfilter = argc > 3 ? atoi(argv[3]) : 0;            // 0 = les deux
    Shape shapes[] = {
        {"q6_f17408x5120",  GGML_TYPE_Q6_K,  5120, 17408},
        {"q8_f17408x5120",  GGML_TYPE_Q8_0,  5120, 17408},
        {"q5_f17408x5120",  GGML_TYPE_Q5_K,  5120, 17408},
        {"q6_f5120x17408",  GGML_TYPE_Q6_K,  17408, 5120},
        {"q8_f5120x17408",  GGML_TYPE_Q8_0,  17408, 5120},
        {"q5_f5120x17408",  GGML_TYPE_Q5_K,  17408, 5120},
    };
    int Ts[] = {1, 4};

    CK(cudaSetDevice(0));

    Env E;
    env_open(E);
    ggml_init_params p { 0, nullptr, true };
    E.gctx = E.ggml_init(p);
    if (!E.gctx) { fprintf(stderr, "ggml_init echoué\n"); return 1; }

    for (auto & sh : shapes) {
        if (shfilter && !strstr(sh.name, shfilter)) continue;
        const int nb = sh.K / 256;                    // nb de blocks par ligne
        const size_t ts = (sh.type == GGML_TYPE_Q6_K) ? sizeof(block_q6_K)
                         : (sh.type == GGML_TYPE_Q5_K) ? sizeof(block_q5_K)
                         : sizeof(block_q8_0);
        const size_t nsrc0 = (size_t)nb * sh.rows * ts;
        const int TMAX = 4;
        const size_t nsrc1 = (size_t)sh.K * TMAX * 4;
        const size_t ndst  = (size_t)sh.rows * TMAX * 4;
        size_t total = nsrc0 + nsrc1 + ndst + (1u<<20);

        ggml_backend_buffer_t buf = E.alloc(E.buft, total);
        if (!buf) { fprintf(stderr, "alloc echoué (%zu oct) %s\n", total, sh.name); return 1; }
        void * base = ((f_buf_get_base)(*(void **)((char *)buf + sizeof(void *))))(buf);
        if (!base) { fprintf(stderr, "get_base null\n"); return 1; }

        // --- donnees host
        std::vector<char> hsrc0(nsrc0);
        srand(987654 + sh.K);
        if (sh.type == GGML_TYPE_Q6_K) {
            block_q6_K * bq = (block_q6_K *)hsrc0.data();
            for (size_t i = 0; i < nsrc0/sizeof(block_q6_K); ++i) {
                for (int j = 0; j < 128; ++j) bq[i].ql[j] = (uint8_t)(rand() & 0xFF);
                for (int j = 0; j < 64; ++j)  bq[i].qh[j] = (uint8_t)(rand() & 0xFF);
                for (int j = 0; j < 16; ++j)  bq[i].scales[j] = (int8_t)(rand() % 161 - 80);
                bq[i].d = __float2half((rand()/(float)RAND_MAX - 0.5f) * 0.1f);
            }
        } else if (sh.type == GGML_TYPE_Q5_K) {
            block_q5_K * bq = (block_q5_K *)hsrc0.data();
            for (size_t i = 0; i < nsrc0/sizeof(block_q5_K); ++i) {
                bq[i].dm.x = __float2half((rand()/(float)RAND_MAX - 0.5f) * 0.1f);
                bq[i].dm.y = __float2half((rand()/(float)RAND_MAX - 0.5f) * 0.1f);
                for (int j = 0; j < 12;  ++j) bq[i].scales[j] = (uint8_t)(rand() & 0xFF);
                for (int j = 0; j < 32;  ++j) bq[i].qh[j] = (uint8_t)(rand() & 0xFF);
                for (int j = 0; j < 128; ++j) bq[i].qs[j] = (uint8_t)(rand() & 0xFF);
            }
        } else {
            block_q8_0 * bq = (block_q8_0 *)hsrc0.data();
            for (size_t i = 0; i < nsrc0/sizeof(block_q8_0); ++i) {
                bq[i].d = __float2half((rand()/(float)RAND_MAX - 0.5f) * 0.1f);
                for (int j = 0; j < 32; ++j) bq[i].qs[j] = (int8_t)(rand() % 201 - 100);
            }
        }
        std::vector<float> hsrc1(nsrc1), hdst(ndst);
        // LAYOUT DEVICE : element (k,t) est lu a l'offset 4*(k + t*K) => hsrc1[k + t*K].
        for (size_t i = 0; i < nsrc1; ++i) hsrc1[i] = (rand()/(float)RAND_MAX - 0.5f);
        CK(cudaMemcpy(base, hsrc0.data(), nsrc0, cudaMemcpyHostToDevice));
        CK(cudaMemcpy((char*)base + nsrc0, hsrc1.data(), nsrc1, cudaMemcpyHostToDevice));

        // --- reference CPU : 1ere ligne de src0 x colonnes de src1
        std::vector<float> wref(sh.K);
        if (sh.type == GGML_TYPE_Q6_K) {
            const block_q6_K * bq = (const block_q6_K *)hsrc0.data();
            for (int b = 0; b < nb; ++b) dequant_q6_K(&bq[b], &wref[b*256]);
        } else if (sh.type == GGML_TYPE_Q5_K) {
            const block_q5_K * bq = (const block_q5_K *)hsrc0.data();
            for (int b = 0; b < nb; ++b) dequant_q5_K(&bq[b], &wref[b*256]);
        } else {
            const block_q8_0 * bq = (const block_q8_0 *)hsrc0.data();
            for (int b = 0; b < nb; ++b) dequant_q8_0(&bq[b], &wref[b*256]);
        }

        for (int ti = 0; ti < 2; ++ti) {
            const int T = Ts[ti];
            if (Tfilter && T != Tfilter) continue;
            ggml_tensor * s0 = (ggml_tensor *)calloc(1, sizeof(ggml_tensor));
            ggml_tensor * s1 = (ggml_tensor *)calloc(1, sizeof(ggml_tensor));
            ggml_tensor * ds = (ggml_tensor *)calloc(1, sizeof(ggml_tensor));
            // src0 : [K, rows] quantized
            s0->type = sh.type; s0->buffer = buf;
            s0->ne[0] = sh.K; s0->ne[1] = sh.rows; s0->ne[2] = 1; s0->ne[3] = 1;
            s0->nb[0] = ts; s0->nb[1] = (size_t)nb*ts; s0->nb[2] = (size_t)nb*ts*sh.rows; s0->nb[3] = s0->nb[2];
            s0->data = base;
            // src1 : [K, T] f32
            s1->type = GGML_TYPE_F32; s1->buffer = buf;
            s1->ne[0] = sh.K; s1->ne[1] = T; s1->ne[2] = 1; s1->ne[3] = 1;
            s1->nb[0] = 4; s1->nb[1] = (size_t)sh.K*4; s1->nb[2] = s1->nb[1]*T; s1->nb[3] = s1->nb[2];
            s1->data = (char*)base + nsrc0;
            // dst : [rows, T] f32
            ds->type = GGML_TYPE_F32; ds->buffer = buf;
            ds->ne[0] = sh.rows; ds->ne[1] = T; ds->ne[2] = 1; ds->ne[3] = 1;
            ds->nb[0] = 4; ds->nb[1] = (size_t)sh.rows*4; ds->nb[2] = ds->nb[1]*T; ds->nb[3] = ds->nb[2];
            ds->data = (char*)base + nsrc0 + nsrc1;

            // warmup (cree le stream ggml lazy + cache)
            for (int w = 0; w < 5; ++w) E.mmvecq(E.ctx_cuda, s0, s1, nullptr, ds, nullptr);
            CK(cudaGetLastError());
            CK(cudaDeviceSynchronize());
            cudaStream_t gst = *(cudaStream_t *)((char *)E.ctx_cuda + 48);
            if (!gst) { fprintf(stderr, "stream ggml null apres warmup\n"); return 1; }
            { cudaEvent_t se; CK(cudaEventCreate(&se)); CK(cudaEventRecord(se, gst)); CK(cudaEventSynchronize(se)); CK(cudaEventDestroy(se)); }

            // --- validation : 1ere ligne
            float outref[4] = {0,0,0,0};
            for (int t = 0; t < T; ++t)
                for (int k = 0; k < sh.K; ++k) outref[t] += wref[k] * hsrc1[(size_t)k + (size_t)t*sh.K];
            float hout[TMAX];
            // dst[row][t] = base + row*4 + t*(rows*4) ; on valide row=0, chaque t.
            for (int t = 0; t < T; ++t)
                CK(cudaMemcpy(&hout[t], (char*)ds->data + (size_t)t*sh.rows*4, 4, cudaMemcpyDeviceToHost));
            float maxerr = 0;
            for (int t = 0; t < T; ++t) maxerr = fmaxf(maxerr, fabsf(hout[t] - outref[t]));
            float rel = maxerr / (fabsf(outref[0]) + 1e-6f);
            // dump hex des 16 premiers floats dst (row0) pour compare bit-exact prod vs H3
            std::vector<float> hdump(16);
            CK(cudaMemcpy(hdump.data(), ds->data, sizeof(float)*16, cudaMemcpyDeviceToHost));
            std::string hexd;
            for (int i = 0; i < 16; ++i) { unsigned int u; memcpy(&u, &hdump[i], 4); char b[10]; snprintf(b, 9, "%08x%s", u, i<15?" ":" "); hexd += b; }

            // --- mesure
            cudaEvent_t e0, e1;
            CK(cudaEventCreate(&e0)); CK(cudaEventCreate(&e1));
            std::vector<float> samples;
            for (int it = 0; it < iters; ++it) {
                CK(cudaEventRecord(e0, gst));
                E.mmvecq(E.ctx_cuda, s0, s1, nullptr, ds, nullptr);
                CK(cudaEventRecord(e1, gst));
                CK(cudaEventSynchronize(e1));
                CK(cudaGetLastError());
                float ms; CK(cudaEventElapsedTime(&ms, e0, e1));
                samples.push_back(ms * 1000.0f);
            }
            std::sort(samples.begin(), samples.end());
            float us_med = samples[samples.size()/2];
            float us_min = samples.front();
            float us_max = samples.back();
            printf("{\"shape\":\"%s\",\"T\":%d,\"K\":%d,\"rows\":%d,\"us_med\":%.1f,\"us_min\":%.1f,\"us_max\":%.1f,\"val_maxerr\":%.4f,\"val_rel\":%.4f,\"dump\":\"%s\"}\n",
                   sh.name, T, sh.K, sh.rows, us_med, us_min, us_max, maxerr, rel, hexd.c_str());
            fflush(stdout);
            free(s0); free(s1); free(ds);
            CK(cudaEventDestroy(e0)); CK(cudaEventDestroy(e1));
        }
        E.buf_free(buf);
    }
    env_close(E);
    return 0;
}
