#!/usr/bin/env python3
"""Patch W2R4 : Q6_K W4R4 T=4, nwarps 4->2 (gate GGML_CUDA_Q6_W2R4).

Idempotent. --revert restaure le backup clean 8035caf4.
3 modifications dans mmvq.cu :
  1. template param nwarps_override = 0 (apres rows_per_block_override)
  2. nwarps = nwarps_override>0 ? nwarps_override : calc_nwarps(...)
  3. branch W2R4 dans le dispatch case 4 (Q6_K), avant le W4R4
"""
import sys, shutil, hashlib

SRC = "/home/guile/llama-experiments/v100/dual-arch/src/ggml/src/ggml-cuda/mmvq.cu"
BAK = "/home/guile/llama-experiments/bench/mmvq.cu.fork-w2-8035caf4.bak"

def md5(p):
    return hashlib.md5(open(p,'rb').read()).hexdigest()

def revert():
    shutil.copyfile(BAK, SRC)
    print(f"REVERT ok -> {SRC} (md5 {md5(SRC)[:9]})")
    return 0

def patch():
    s = open(SRC).read()
    n = 0

    # 1. template param
    old1 = "template <ggml_type type, int ncols_dst, bool has_fusion, bool small_k = false, bool halve_iters = false, bool q5_x4 = false, int rows_per_block_override = 0>"
    new1 = "template <ggml_type type, int ncols_dst, bool has_fusion, bool small_k = false, bool halve_iters = false, bool q5_x4 = false, int rows_per_block_override = 0, int nwarps_override = 0>"
    if new1 in s:
        print("  [skip 1] deja patche (template param present)")
    elif old1 not in s:
        print("  [ERREUR] ancre 1 introuvable"); return 1
    else:
        s = s.replace(old1, new1, 1); n += 1; print("  [1] template param nwarps_override")

    # 2. nwarps override (unique : table_id)
    old2 = "    constexpr int nwarps = calc_nwarps(type, ncols_dst, table_id, small_k, halve_iters);"
    new2 = "    constexpr int nwarps = nwarps_override > 0 ? nwarps_override : calc_nwarps(type, ncols_dst, table_id, small_k, halve_iters);"
    if new2 in s:
        print("  [skip 2] deja patche (nwarps override present)")
    elif old2 not in s:
        print("  [ERREUR] ancres 2 introuvable(s)"); return 1
    else:
        s = s.replace(old2, new2, 1); n += 1; print("  [2] nwarps = nwarps_override ? ... : calc_nwarps")

    # 3. branch W2R4 (marqueur = le nom du gate, introuvable si clean)
    marker = "GGML_CUDA_Q6_W2R4"
    if marker in s:
        print("  [skip 3] deja patche (gate W2R4 present)")
    else:
        old3 = """            if constexpr (type == GGML_TYPE_Q6_K) {
                const bool has_fusion = fusion.gate != nullptr || fusion.x_bias != nullptr || fusion.gate_bias != nullptr ||
                                        fusion.x_scale != nullptr || fusion.gate_scale != nullptr;
                const char * env = std::getenv("GGML_CUDA_VOLTA_Q6_W4R4");
                const bool enabled = env == nullptr || std::atoi(env) != 0;
                if (enabled && cc == GGML_CUDA_CC_VOLTA && !has_ids && !has_fusion) {"""
        new3 = """            if constexpr (type == GGML_TYPE_Q6_K) {
                const bool has_fusion = fusion.gate != nullptr || fusion.x_bias != nullptr || fusion.gate_bias != nullptr ||
                                        fusion.x_scale != nullptr || fusion.gate_scale != nullptr;
                const char * env_w2 = std::getenv("GGML_CUDA_Q6_W2R4");
                const bool w2_enabled = env_w2 != nullptr && std::atoi(env_w2) != 0;
                if (w2_enabled && cc == GGML_CUDA_CC_VOLTA && !has_ids && !has_fusion) {
                    constexpr int rows_per_block_w2 = 4;
                    constexpr int nwarps_w2 = 2;
                    const dim3 block_nums((nrows_x + rows_per_block_w2 - 1) / rows_per_block_w2, nchannels_dst, nsamples_dst);
                    const dim3 block_dims(warp_size, nwarps_w2, 1);
                    const ggml_cuda_kernel_launch_params launch_params(block_nums, block_dims, 0, stream);
                    ggml_cuda_kernel_launch(
                        mul_mat_vec_q<type, c_ncols_dst, false, false, false, false, rows_per_block_w2, nwarps_w2>, launch_params,
                        vx, vy, ids, fusion, dst, ncols_x, nchannels_y_fd, stride_row_x, stride_col_y, stride_col_dst,
                        channel_ratio_fd, stride_channel_x, stride_channel_y, stride_channel_dst,
                        sample_ratio_fd, stride_sample_x, stride_sample_y, stride_sample_dst, ids_stride);
                    return;
                }
                const char * env = std::getenv("GGML_CUDA_VOLTA_Q6_W4R4");
                const bool enabled = env == nullptr || std::atoi(env) != 0;
                if (enabled && cc == GGML_CUDA_CC_VOLTA && !has_ids && !has_fusion) {"""
        if old3 not in s:
            print("  [ERREUR] ancre 3 (dispatch W4R4) introuvable"); return 1
        s = s.replace(old3, new3, 1); n += 1; print("  [3] branch W2R4 dans dispatch case 4")

    open(SRC, 'w').write(s)
    print(f"PATCH ok : {n} modification(s) appliquee(s), md5 {md5(SRC)[:9]}")
    return 0

if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--revert":
        sys.exit(revert())
    sys.exit(patch())
