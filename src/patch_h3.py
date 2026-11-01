#!/usr/bin/env python3
# patch_h3.py — inject le kernel candidat Q6_K W4R4 "H3" (hoist 4x4 unpack)
# dans le fork dual-arch (mmvq.cu), gate GGML_CUDA_Q6_H3, SANS toucher prod.
#
# H3 = meme grid/block/boucle K que prod (parallelisme IDENTIQUE) mais on
# hoiste le depaquetage : les 4 lignes poids (q6_K) et les 4 tokens activation
# (q8_1) d'un (lane,kbx) sont charges UNE fois, puis les 16 dp4a partent des
# regs. Bit-exact (meme dp4a, meme point, meme ordre d'accumulation en kbx).
#   - prod recharge le bloc poids (i,kbx) 4x (une fois par j) et le bloc act
#     (j,kbx) 4x (une fois par i) par (lane,kbx) : 16 appels vec_dot.
#   - H3 : 4 charges poids + 4 charges act, puis 16 dp4a depuis les regs.
#
# Usage : python3 patch_h3.py            (applique)
#         python3 patch_h3.py --revert   (restaure le backup 8035caf4)
import sys, shutil

D   = "/home/guile/llama-experiments/v100/dual-arch/src/ggml/src/ggml-cuda/mmvq.cu"
BAK = "/home/guile/llama-experiments/bench/mmvq.cu.fork-8035caf4.bak"

KERNEL = r'''
// =============================================================================
// H3 (29/10) — Q6_K W4R4 T=4 : hoist du depaquetage 4x4 par (lane,kbx).
// Meme grid/block/boucle K que prod ; on charge les 4 lignes de poids (q6_K)
// et les 4 tokens d'activation (q8_1) UNE fois par (lane,kbx), puis les 16
// dp4a partent des regs. Bit-exact (meme dp4a, meme point, meme ordre en kbx).
// Gate : GGML_CUDA_Q6_H3=1.
// =============================================================================
__device__ __forceinline__ float h3_q6_block_f32(
        const int vl, const int vh, const int sc0, const int sc1, const float d,
        const int * u, const float * d8) {
    float sumf = 0.0f;
#pragma unroll
    for (int i = 0; i < QR6_K; ++i) {
        const int sc = (i == 0) ? sc0 : sc1;      // prod : scales[scale_offset + 4*i], i in 0..1
        const int vil = (vl >> (4*i)) & 0x0F0F0F0F;
        const int vih = ((vh >> (4*i)) << 4) & 0x30303030;
        const int vi  = __vsubss4((vil | vih), 0x20202020);
        sumf += d8[i] * (ggml_cuda_dp4a(vi, u[i], 0) * sc);
    }
    return d * sumf;
}

__global__ void __launch_bounds__(ggml_cuda_get_physical_warp_size()*4, 5)
mul_mat_vec_q6_k_w4r4_h3(
        const void * __restrict__ vx_ptr, const void * __restrict__ vy_ptr, float * dst_ptr,
        const uint32_t ncols_x, const uint32_t stride_row_x, const uint32_t stride_col_y,
        const uint32_t stride_col_dst) {
    constexpr int qk  = QK_K;
    constexpr int qi  = QI6_K;
    constexpr int qr  = QR6_K;
    constexpr int vdr = VDR_Q6_K_Q8_1_MMVQ;
    constexpr int qk8 = QK8_1;
    constexpr int qi8 = QI8_1;
    constexpr int nwarps = 4;
    constexpr int rpb    = 4;
    constexpr int warp   = ggml_cuda_get_physical_warp_size();

    const block_q6_K * __restrict__ x = (const block_q6_K *) vx_ptr;
    const block_q8_1 * __restrict__ y = (const block_q8_1 *) vy_ptr;

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
        const int kby = kbx*(qk/qk8);
        const int kqs = vdr*(tid%(qi/vdr));

        const int bq8_offset   = 2*qr*(kqs/(qi/2)) + (kqs%(qi/2))/(qi/4);
        const int scale_offset = (qi/4)*(kqs/(qi/2)) + (kqs%(qi/2))/(qi/8);
        const int vh_shift     = 2*((kqs%(qi/2))/(qi/4));

        // ---- HOIST 4 lignes poids (q6_K), 1x par (lane,kbx) ----
        int   w_vl[rpb]; int   w_vh[rpb];
        int   w_sc0[rpb]; int  w_sc1[rpb];
        float w_d[rpb];
#pragma unroll
        for (int i = 0; i < rpb; ++i) {
            const block_q6_K * bq6 = x + (size_t)(row0 + i)*stride_row_x + kbx;
            w_vl[i]  = get_int_b2(bq6->ql, kqs);
            w_vh[i]  = get_int_b2(bq6->qh, (qi/4)*(kqs/(qi/2)) + kqs%(qi/4)) >> vh_shift;
            w_sc0[i] = (int)bq6->scales[scale_offset + 0];
            w_sc1[i] = (int)bq6->scales[scale_offset + 4];
            w_d[i]   = __half2float(bq6->d);
        }
        // ---- HOIST 4 tokens activation (q8_1), 1x par (lane,kbx) ----
        int   a_u[4][qr];  float a_d8[4][qr];
#pragma unroll
        for (int j = 0; j < 4; ++j) {
            const block_q8_1 * bq8 = &y[j*stride_col_y + kby];
#pragma unroll
            for (int s = 0; s < qr; ++s) {
                a_u[j][s]  = get_int_b4(bq8[bq8_offset + 2*s].qs, kqs % qi8);
                a_d8[j][s] = __low2float(bq8[bq8_offset + 2*s].ds);
            }
        }
        // ---- 16 dp4a depuis les regs (meme point + meme ordre que prod) ----
#pragma unroll
        for (int j = 0; j < 4; ++j) {
#pragma unroll
            for (int i = 0; i < rpb; ++i) {
                tmp[j][i] += h3_q6_block_f32(w_vl[i], w_vh[i], w_sc0[i], w_sc1[i], w_d[i], a_u[j], a_d8[j]);
            }
        }
    }

    __shared__ float tmp_shared[nwarps-1][4][rpb][warp];
    if (threadIdx.y > 0) {
#pragma unroll
        for (int j = 0; j < 4; ++j)
            #pragma unroll
            for (int i = 0; i < rpb; ++i)
                tmp_shared[threadIdx.y-1][j][i][threadIdx.x] = tmp[j][i];
    }
    __syncthreads();
    if (threadIdx.y > 0) return;

    float * dst = dst_ptr + row0;
#pragma unroll
    for (int j = 0; j < 4; ++j)
#pragma unroll
        for (int i = 0; i < rpb; ++i) {
#pragma unroll
            for (int l = 0; l < nwarps-1; ++l) tmp[j][i] += tmp_shared[l][j][i][threadIdx.x];
            tmp[j][i] = warp_reduce_sum<warp>(tmp[j][i]);
            if (threadIdx.x == i) dst[j*stride_col_dst + i] = tmp[j][i];
        }
}
'''

DISPATCH_OLD = r'''                if (enabled && cc == GGML_CUDA_CC_VOLTA && !has_ids && !has_fusion) {
                    constexpr int rows_per_block = 4;'''
DISPATCH_NEW = r'''                if (enabled && cc == GGML_CUDA_CC_VOLTA && !has_ids && !has_fusion) {
                    const char * h3env = std::getenv("GGML_CUDA_Q6_H3");
                    const bool h3 = h3env != nullptr && std::atoi(h3env) != 0 && nchannels_dst == 1 && nsamples_dst == 1;
                    if (h3) {
                        constexpr int rows_per_block = 4;
                        const dim3 h3_nums((nrows_x + rows_per_block - 1) / rows_per_block, nchannels_dst, nsamples_dst);
                        const dim3 h3_dims(warp_size, 4, 1);
                        const ggml_cuda_kernel_launch_params h3_lp(h3_nums, h3_dims, 0, stream);
                        ggml_cuda_kernel_launch(mul_mat_vec_q6_k_w4r4_h3, h3_lp,
                            vx, vy, dst, ncols_x, stride_row_x, stride_col_y, stride_col_dst);
                        return;
                    }
                    constexpr int rows_per_block = 4;'''

def main():
    if "--revert" in sys.argv:
        shutil.copyfile(BAK, D)
        print("REVERT ok ->", D)
        return
    src = open(D).read()
    if "mul_mat_vec_q6_k_w4r4_h3" in src:
        print("deja patche, on repart du backup")
        src = open(BAK).read()
    anchor = "// Dedicated MoE multi-token kernel."
    assert anchor in src, "anchor MoE pas trouve"
    src = src.replace(anchor, KERNEL + "\n" + anchor, 1)
    assert DISPATCH_OLD in src, "anchor dispatch pas trouve"
    src = src.replace(DISPATCH_OLD, DISPATCH_NEW, 1)
    open(D, "w").write(src)
    print("PATCH H3 applique ->", D)

if __name__ == "__main__":
    main()
