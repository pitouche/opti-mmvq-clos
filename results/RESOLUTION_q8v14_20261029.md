## RESOLUTION: q8v14partial identity (29/10)

ggml_q8v14partial_kernel  ==  ggml_q8v::partial_kernel  (mangled: 8=ggml_q8v + 14=partial_kernel)
- SOURCE: fattn-q8-volta.cuh:205 (namespace ggml_q8v), launched via ggml_q8v::launch (fattn.cu:946)
- ROUTE:  BEST_FATTN_KERNEL_VOLTA_Q8_W4 (H2, Q->ne[1]==4)  = the 4-row MTP VERIFY ATTENTION
- READS:  q (queries f32) + K,V (block_q8_0 = KV cache) + mask  -> FLASH ATTENTION, KV reads
- PAIR:   ggml_q8v13reduce_kernel == ggml_q8v::reduce_kernel (fattn-q8-volta.cuh reduce_kernel)
- NOT a weight-reading GEMV. grep "q8v14partial" never matches source because the name is fused by mangling.

=> Per user constraint "Cible mémoire : lecture des poids, pas du KV", q8v14partial (30.1%) is
   OUT OF SCOPE (it reads KV, not weights). It is the H2 flash-attention kernel.

CORRECTED in-scope (weight-reading) verify GEMV = 48.4% of GPU kernel:
   mul_mat_vec_q type14 (q6-k)  28.9%  <- DOMINANT, in-scope target
   mul_mat_vec_q type8  (q8   )  11.2%
   mul_mat_vec_q type13 (q6_k)   8.3%
ATTENTION (KV, out of scope) = 34.2%: q8v14partial 30.1 + q8v13reduce 1.7 + volta_small 2.4
CONFIRMS Q6_K (type14, 28.9%) as the dominant in-scope weight-reading GEMV -> Phase 3 target.
