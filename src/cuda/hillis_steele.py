import numpy as np
from numba import cuda, int32

TAMANHO_BLOCO = 8

# ==========================================
# HILLIS-STEELE SCAN INCLUSIVO
# ==========================================

@cuda.jit
def hillis_steele_kernel(v_in, v_out):

    temp = cuda.shared.array(shape=(TAMANHO_BLOCO,), dtype=int32)

    tid = cuda.threadIdx.x

    if tid < v_in.size:
        temp[tid] = v_in[tid]
    else:
        temp[tid] = 0

    cuda.syncthreads()

    stride = 1

    while stride < cuda.blockDim.x:

        valor = 0

        if tid >= stride:
            valor = temp[tid - stride]

        cuda.syncthreads()

        if tid >= stride:
            temp[tid] += valor

        cuda.syncthreads()

        stride *= 2

    if tid < v_out.size:
        v_out[tid] = temp[tid]


# ==========================================
# MAIN
# ==========================================

v = np.array([1, 2, 3, 4, 5, 6, 7, 8], dtype=np.int32)

out = np.zeros_like(v)

d_v = cuda.to_device(v)
d_out = cuda.to_device(out)

hillis_steele_kernel[1, TAMANHO_BLOCO](d_v, d_out)

resultado = d_out.copy_to_host()

print("Entrada:", v)
print("Scan Inclusivo:", resultado)