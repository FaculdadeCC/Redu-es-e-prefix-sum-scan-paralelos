import numpy as np
from numba import cuda, int32

TAMANHO_BLOCO = 32

@cuda.jit
def soma_otimizada_kernel(v_in, v_out):

    sdata = cuda.shared.array(shape=(TAMANHO_BLOCO,), dtype=int32)

    tid = cuda.threadIdx.x

    if tid < v_in.size:
        sdata[tid] = v_in[tid]
    else:
        sdata[tid] = 0

    cuda.syncthreads()

    stride = cuda.blockDim.x // 2

    while stride > 0:

        if tid < stride:
            sdata[tid] += sdata[tid + stride]

        cuda.syncthreads()

        stride //= 2

    if tid == 0:
        v_out[0] = sdata[0]

@cuda.jit
def maximo_otimizado_kernel(v_in, v_out):

    sdata = cuda.shared.array(shape=(TAMANHO_BLOCO,), dtype=int32)

    tid = cuda.threadIdx.x

    if tid < v_in.size:
        sdata[tid] = v_in[tid]
    else:
        sdata[tid] = -999999

    cuda.syncthreads()

    stride = cuda.blockDim.x // 2

    while stride > 0:

        if tid < stride:
            sdata[tid] = max(sdata[tid], sdata[tid + stride])

        cuda.syncthreads()

        stride //= 2

    if tid == 0:
        v_out[0] = sdata[0]

@cuda.jit
def minimo_otimizado_kernel(v_in, v_out):

    sdata = cuda.shared.array(shape=(TAMANHO_BLOCO,), dtype=int32)

    tid = cuda.threadIdx.x

    if tid < v_in.size:
        sdata[tid] = v_in[tid]
    else:
        sdata[tid] = 999999

    cuda.syncthreads()

    stride = cuda.blockDim.x // 2

    while stride > 0:

        if tid < stride:
            sdata[tid] = min(sdata[tid], sdata[tid + stride])

        cuda.syncthreads()

        stride //= 2

    if tid == 0:
        v_out[0] = sdata[0]

v = np.array([1, 2, 3, 4, 5], dtype=np.int32)

out_soma = np.zeros(1, dtype=np.int32)
out_max = np.zeros(1, dtype=np.int32)
out_min = np.zeros(1, dtype=np.int32)

d_v = cuda.to_device(v)

d_out_soma = cuda.to_device(out_soma)
d_out_max = cuda.to_device(out_max)
d_out_min = cuda.to_device(out_min)

soma_otimizada_kernel[1, TAMANHO_BLOCO](d_v, d_out_soma)

maximo_otimizado_kernel[1, TAMANHO_BLOCO](d_v, d_out_max)

minimo_otimizado_kernel[1, TAMANHO_BLOCO](d_v, d_out_min)

resultado_soma = d_out_soma.copy_to_host()
resultado_max = d_out_max.copy_to_host()
resultado_min = d_out_min.copy_to_host()

print("Vetor:", v)

print("\n===== GPU OTIMIZADA =====")

print("Soma GPU:", resultado_soma[0])
print("Máximo GPU:", resultado_max[0])
print("Mínimo GPU:", resultado_min[0])