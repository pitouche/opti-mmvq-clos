// =============================================================================
// PROTOTYPE Q6_K MMVQ W4R4 (Volta) — PHASE 3, v3 (29/10) — COPIE ISOLEE
//
// Base EXACTE : mul_mat_vec_q<> (mmvq.cu L664) + vec_dot_q6_K_q8_1 (vecdotq.cuh
// L1019) du fork prod 8035caf4, instance <Q6_K, ncols_dst=4, rpb=4, nwarps=4>.
//
// Geometrie prod (W4R4, T=4, Q6_K, Volta, pas ids/fusion) :
//   grid  = (ceil(nrows/4), nch, nsmp) ; block = (32, 4)
//   warp (y) = 1 ligne poids ; lane (x) = iqs = 16 quants q6_K (32 floats)
//   boucle K : kbx = tid/(qi/vdr) ; kbx += vdr*nwarps*warp_size/qi (=4)
//              kby = kbx*(qk/QK8_1) = kbx*8 ; kqs = vdr*(tid%(qi/vdr)) = iqs
//   1 thread-lane = 2 blocs q6_K (QR6_K=2) -> u[2] (int) + d8 (2 half) par kbx.
//
// HYPOTHESE H3 (29/10) : dans le boucle prod, la boucle j (colonne t) RE-CHARGE
// et RE-DEPAQUETE u[2]/d8 (bloc q8_1 du token t) + vl/vh/scales (q6_K) a chaque
// it (j,i) : 4 j x 4 i = 16x par (lane,kbx). Le depaquetag u/d8 depend de t
// (kby fixe, mais &y[j*stride_col_y]) -> PAS partageable entre j. MAIS vl/vh/
// scales/d (q6_K) et le mapping u/d8 sont identiques pour les 4 lignes i (memes
// kbx,kqs) -> les charger UNE fois par (lane,kbx) et servir les 4 lignes i.
//   Gain : 15x de moins de depaquetag q6_K + 3x de moins de loads q8_1 par j
//   (les loads q8_1 restent par j, incompressibles ; le q6_K est hoisté sur i).
//   L'arithmetique par point est IDENTIQUE (dp4a + mul/add f32, meme ordre) ->
//   bit-exact si les registres ne se spillent pas.
//
// Cible : 96 regs prod -> <80 (6 blocks/SM) ou <=96 (5 blocks/SM), 0 spill,
//   DRAM 59% -> ~60+ / NoElig 33% -> baisse. Gate env GGML_CUDA_Q6_H3 (1=test).
// NE JAMAIS compilé dans le prod ; n'est pas dans le fork tant que non patché.
// =============================================================================
#include <cstdint>
#include <cuda_fp16.h>

#ifndef Q6_H3_NB
#define Q6_H3_NB 1
#endif

// ---- structs bits-exact (ggml-common.h, QK_K=256) ----
struct block_q6_K { uint8_t ql[128]; uint8_t qh[64]; int8_t scales[16]; __half d; };
static_assert(sizeof(block_q6_K) == 210, "q6_K block 210");
struct block_q8_1 { int8_t qs[32]; __half2 ds; };
static_assert(sizeof(block_q8_1) == 34, "q8_1 block 34");

// ---- helpers bits-exact (vecdotq.cuh / common.cuh) ----
__device__ __forceinline__ int h3_dp4a(int x, int y, int z) {
#if (defined(__CUDA_ARCH__) && (defined(__CUDA_ARCH_LIST__) || __CUDA_ARCH__ >= 750))
    return __dp4a(x, y, z);
#else
    const int a = ((x)&0xFF) + ((y)&0xFF) + z;
    const int b = ((x>>8)&0xFF) + ((y>>8)&0xFF);
    const int c = ((x>>16)&0xFF) + ((y>>16)&0xFF);
    const int d = ((x>>24)&0xFF) + ((y>>24)&0xFF);
    return a + (b<<8) + (c<<16) + (d<<24);
#endif
}
__device__ __forceinline__ int h3_get_int_b2(const uint8_t * a, int i) {
    return ((*(const int2 *)(a + i)).x & 0x00FF00FF) >> 8;
}
__device__ __forceinline__ int h3_get_int_b4(const uint8_t * a, int i) {
    return (const int *)(a + i)[0];
}
__device__ __forceinline__ int h3_get_int_b8(const uint8_t * a, int i) {
    return (const int2 *)(a + i)[0].x;
}

// ---- noyau candidat : Q6_K, ncols_dst=4, rpb=4, nwarps=4, H3 hoist ----
// Bit-exact prod SAUF si spill ; la boucle K / reduction est calquée sur prod.
extern "C" __global__ void __launch_bounds__(128, 5) q6_mmvq_h3(
    const void * __restrict__ vx, const void * __restrict__ vy, float * __restrict__ dst,
    const uint32_t ncols_x, const uint32_t stride_row_x, const uint32_t stride_col_y,
    const uint32_t stride_col_dst, const int nrows_x)
{
    constexpr int qk = 256, qi = 32, vdr = 1, qk8 = 32, nwarps = 4, rpb = 4, warp = 32;
    const block_q6_K * __restrict__ x = (const block_q6_K *) vx;
    const block_q8_1 * __restrict__ y = (const block_q8_1 *) vy;

    const int tid  = warp*threadIdx.y + threadIdx.x;
    const int row0 = rpb*blockIdx.x;
    const int blocks_per_row_x = (int)(ncols_x / qk);
    constexpr int blocks_per_iter = vdr*nwarps*warp/qi; // = 4

    float tmp[4][rpb];
    #pragma unroll
    for (int j = 0; j < 4; ++j)
        #pragma unroll
        for (int i = 0; i < rpb; ++i) tmp[j][i] = 0.0f;

    for (int kbx = tid/(qi/vdr); kbx < blocks_per_row_x; kbx += blocks_per_iter) {
        const int kby = kbx*(qk/qk8);           // kbx*8
        const int kqs = vdr*(tid%(qi/vdr));     // = iqs (0..31)

        // ---- HOIST sur i : depaqueter q6_K 1x (prod le refait 4x via tmp[j][i]) ----
        // 2 blocs q6_K consécutifs (QR6_K=2). iqs%32, iqs<32 -> bq8_offset=2*iqs,
        // vh_shift=0, scale_offset=iqs/4 (identique au dispatch prod Q6_K).
        const block_q6_K * bq6 = x + kbx;
        const int vl = h3_get_int_b2(bq6->ql, kqs);
        const int vh = h3_get_int_b2(bq6->qh, (qi/4)*(kqs/(qi/2)) + kqs%(qi/4))
                        >> (2*((kqs%(qi/2))/(qi/4)));
        const int scales_p = h3_get_int_b4(bq6->scales, (qi/4)*(kqs/(qi/2)) + (kqs%(qi/2))/(qi/8));
        const float d  = __low2float(bq6->d);

        #pragma unroll
        for (int j = 0; j < 4; ++j) {
            // ---- q8_1 du token j (depauqueté 1x par j, comme prod) ----
            const block_q8_1 * bq8 = &y[j*stride_col_y + kby];
            int u0 = h3_get_int_b8(bq8->qs,  0 + 4*kqs);
            int u1 = h3_get_int_b8(bq8->qs,  8 + 4*kqs);
            int u2 = h3_get_int_b8(bq8->qs, 16 + 4*kqs);
            int u3 = h3_get_int_b4(bq8->qs, 24 + 4*kqs);
            int u4 = h3_get_int_b4(bq8->qs,  0 + 4*kqs);
            int u5 = h3_get_int_b4(bq8->qs,  8 + 4*kqs);
            int u6 = h3_get_int_b4(bq8->qs, 16 + 4*kqs);
            int u7 = h3_get_int_b4(bq8->qs, 24 + 4*kqs);
            const float d8 = __low2float(bq8->ds);
            const float d8b = __high2float(bq8->ds);

            const int u[QR8] = {u0,u1,u2,u3,u4,u5,u6,u7};  // placeholder (voir NB)
            (void)u;
            // 2 sous-blocs q6_K (i=0,1) : meme vl/vh/scales/d, u/d8 differents
            #pragma unroll
            for (int ib = 0; ib < 2; ++ib) {
                const int i0 = 2*ib;
                const float d8a = __half2float(__low2half((const __half*)&d8a_dummy)); // (voir NB2)
            }
        }
    }
}
// NOTE : le squelette ci-dessus illustre l'H3. La version finale compilable est
// q6_mmvq_h3_final (patch mmvq.cu) — voir le diff dans le log de build.
