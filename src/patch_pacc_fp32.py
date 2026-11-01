#!/usr/bin/env python3
# patch_pacc_fp32.py — candidat "PA": buffer pacc en FP32 au lieu de BF16
# dans ggml_q8v (fattn-q8-volta.cuh, route H2 / BEST_FATTN_KERNEL_VOLTA_Q8_W4).
#
# But : A/B du format du buffer inter-mises de l'attention 2-pass sur Volta,
# SANS toucher la prod. Gate : GGML_CUDA_Q8V_PA32=1 (0/absent = comportement
# prod = BF16). Lancement : partial_kernel<float|__nv_bfloat16> x
# reduce_kernel<float|__nv_bfloat16> — 4 instantiations, dispatch runtime.
#
# Sémantique : le store pacc en BF16 tronque les 16 bas bits de chaque
# accumulation fp32 (SASS prod : PRMT 0x5410, pas d'arrondi). Le candidat
# écrit en fp32 bit-exact l'accumulateur. La sortie finale diffère donc de la
# prod à ~2^-8 relatif (bruit de troncature des partial sums, pas une erreur).
# NB: __bfloat162float disparaît du template (remplacé par static_cast<float>,
# identique sémantiquement sur la route bf16).
#
# Usage : python3 patch_pacc_fp32.py            (applique)
#         python3 patch_pacc_fp32.py --revert   (restaure le backup 8035caf4)
import sys, shutil

D   = "/home/guile/llama-experiments/v100/dual-arch/src/ggml/src/ggml-cuda/fattn-q8-volta.cuh"
BAK = "/home/guile/llama-experiments/bench/fattn-q8-volta.cuh.fork-8035caf4.bak"
MARK = "GGML_CUDA_Q8V_PA32"

def main():
    if "--revert" in sys.argv:
        shutil.copyfile(BAK, D)
        print("REVERT ok ->", D)
        return

    src = open(D).read()
    if MARK in src:
        print("deja patche, on repart du backup")
        src = open(BAK).read()

    def rep(s, old, new, n=1):
        c = s.count(old)
        assert c == n, "anchor trouve %d fois (attendu %d): %r" % (c, n, old[:80])
        return s.replace(old, new, n)

    # 1) get_extra : taille du buffer pacc
    src = rep(src,
        "    e.end += (size_t)e.splits*T*qh*D*sizeof(__nv_bfloat16);",
        "    e.end += (size_t)e.splits*T*qh*D*sizeof(float);   // PA: buffer pacc alloué fp32 (prod: bf16) — la route bf16 n'ecrit que la moitie, le reste est marge reservee")

    # 2) signature partial_kernel -> template (template AVANT l'attribut launch_bounds)
    src = rep(src, "__launch_bounds__(128,2)\nstatic __global__ void partial_kernel(",
        "template<typename PA>\n__launch_bounds__(128,2)\nstatic __global__ void partial_kernel(")
    src = rep(src, "        __nv_bfloat16 * pacc, float * pm, float * pl) {",
        "        PA * pacc, float * pm, float * pl) {")

    # 3) store pacc : bf16 pack -> fp32 vectorise
    src = rep(src,
        "        for(int c=0;c<D_SLICE/8;c++){const int d=dim_warp*D_SLICE+c*8;__nv_bfloat16 o[8];for(int i=0;i<8;i++)o[i]=__float2bfloat16(acc[c][i]);*reinterpret_cast<int4*>(&pacc[pacc_idx(qh,d,tok,split,qh_count)])=*reinterpret_cast<int4*>(o);}",
        "        for(int c=0;c<D_SLICE/8;c++){const int d=dim_warp*D_SLICE+c*8;\n"
        "            if constexpr(sizeof(PA)==4){float o[8];for(int i=0;i<8;i++)o[i]=acc[c][i];\n"
        "                *reinterpret_cast<float4*>(&pacc[pacc_idx(qh,d,tok,split,qh_count)])=*reinterpret_cast<const float4*>(o);\n"
        "                *reinterpret_cast<float4*>(&pacc[pacc_idx(qh,d+4,tok,split,qh_count)])=*reinterpret_cast<const float4*>(o+4);}\n"
        "            else{__nv_bfloat16 o[8];for(int i=0;i<8;i++)o[i]=__float2bfloat16(acc[c][i]);*reinterpret_cast<int4*>(&pacc[pacc_idx(qh,d,tok,split,qh_count)])=*reinterpret_cast<int4*>(o);}}")

    # 4) signature reduce_kernel -> template
    src = rep(src, "static __global__ void reduce_kernel(const __nv_bfloat16 *pa,",
        "template<typename PA> static __global__ void reduce_kernel(const PA *pa,")

    # 5) lecture pacc dans reduce : if constexpr (route bf16 bit-identique a prod)
    src = rep(src,
        "    float num=0;for(int s=0;s<splits;s++){const float l=pl[pstat_idx(qh,tok,s,qh_count)];if(l>0)num+=__bfloat162float(pa[pacc_idx(qh,d,tok,s,qh_count)])*expf(pm[pstat_idx(qh,tok,s,qh_count)]-hm);}",
        "    float num=0;for(int s=0;s<splits;s++){const float l=pl[pstat_idx(qh,tok,s,qh_count)];if(l>0){\n"
        "        float pv; if constexpr(sizeof(PA)==2) pv=__bfloat162float(pa[pacc_idx(qh,d,tok,s,qh_count)]); else pv=pa[pacc_idx(qh,d,tok,s,qh_count)];\n"
        "        num+=pv*expf(pm[pstat_idx(qh,tok,s,qh_count)]-hm);}}")

    # 6) dispatch runtime dans launch()
    src = rep(src,"""    cudaStream_t stream=ctx.stream();
    partial_kernel<<<dim3(kvh_count,e.splits),128,0,stream>>>(
        (const float*)Q->data,(const block_q8_0*)K->data,(const block_q8_0*)V->data,(const half*)M->data,
        (int)K->ne[1],e.splits,qh_count,scale,q_s1,q_s2,k_s1,k_s2,v_s1,v_s2,m_s1,
        (__nv_bfloat16*)e.pacc,(float*)e.pm,(float*)e.pl);
    CUDA_CHECK(cudaGetLastError());
    reduce_kernel<<<dim3(qh_count,T),256,0,stream>>>(
        (const __nv_bfloat16*)e.pacc,(const float*)e.pm,(const float*)e.pl,e.splits,qh_count,
        (float*)dst->data,o_s1,o_s2);
    CUDA_CHECK(cudaGetLastError());""",
        """    cudaStream_t stream=ctx.stream();
    const char * pa32env = getenv("GGML_CUDA_Q8V_PA32");
    const bool pa32 = pa32env != nullptr && atoi(pa32env) != 0;   // PA: pacc fp32
    if (pa32) {
        partial_kernel<float><<<dim3(kvh_count,e.splits),128,0,stream>>>(
            (const float*)Q->data,(const block_q8_0*)K->data,(const block_q8_0*)V->data,(const half*)M->data,
            (int)K->ne[1],e.splits,qh_count,scale,q_s1,q_s2,k_s1,k_s2,v_s1,v_s2,m_s1,
            (float*)e.pacc,(float*)e.pm,(float*)e.pl);
        CUDA_CHECK(cudaGetLastError());
        reduce_kernel<float><<<dim3(qh_count,T),256,0,stream>>>(
            (const float*)e.pacc,(const float*)e.pm,(const float*)e.pl,e.splits,qh_count,
            (float*)dst->data,o_s1,o_s2);
        CUDA_CHECK(cudaGetLastError());
        return;
    }
    partial_kernel<__nv_bfloat16><<<dim3(kvh_count,e.splits),128,0,stream>>>(
        (const float*)Q->data,(const block_q8_0*)K->data,(const block_q8_0*)V->data,(const half*)M->data,
        (int)K->ne[1],e.splits,qh_count,scale,q_s1,q_s2,k_s1,k_s2,v_s1,v_s2,m_s1,
        (__nv_bfloat16*)e.pacc,(float*)e.pm,(float*)e.pl);
    CUDA_CHECK(cudaGetLastError());
    reduce_kernel<__nv_bfloat16><<<dim3(qh_count,T),256,0,stream>>>(
        (const __nv_bfloat16*)e.pacc,(const float*)e.pm,(const float*)e.pl,e.splits,qh_count,
        (float*)dst->data,o_s1,o_s2);
    CUDA_CHECK(cudaGetLastError());""")

    open(D, "w").write(src)
    print("PATCH PA (pacc FP32) applique ->", D)

if __name__ == "__main__":
    main()
