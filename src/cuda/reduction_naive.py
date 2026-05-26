import numpy as np
from numba import cuda, int32

TAMANHO_BLOCO = 32




@cuda.jit
def soma_ingenua_kernel(v_in, v_out):

    
    sdata = cuda.shared.array(shape=(TAMANHO_BLOCO,), dtype=int32)

    tid = cuda.threadIdx.x

    
    if tid < v_in.size:
        sdata[tid] = v_in[tid]
    else:
        sdata[tid] = 0

    cuda.syncthreads()

    
    stride = 1

    while stride < cuda.blockDim.x:

        if tid % (2 * stride) == 0 and tid + stride < cuda.blockDim.x:

            sdata[tid] += sdata[tid + stride]

        stride *= 2

        cuda.syncthreads()

    if tid == 0:
        v_out[0] = sdata[0]

@cuda.jit
def maximo_ingenuo_kernel(v_in, v_out):

    sdata = cuda.shared.array(shape=(TAMANHO_BLOCO,), dtype=int32)

    tid = cuda.threadIdx.x

    if tid < v_in.size:
        sdata[tid] = v_in[tid]
    else:
        sdata[tid] = -999999

    cuda.syncthreads()

    stride = 1

    while stride < cuda.blockDim.x:

        if tid % (2 * stride) == 0 and tid + stride < cuda.blockDim.x:

            sdata[tid] = max(sdata[tid], sdata[tid + stride])

        stride *= 2

        cuda.syncthreads()

    if tid == 0:
        v_out[0] = sdata[0]

@cuda.jit
def minimo_ingenuo_kernel(v_in, v_out):

    sdata = cuda.shared.array(shape=(TAMANHO_BLOCO,), dtype=int32)

    tid = cuda.threadIdx.x

    if tid < v_in.size:
        sdata[tid] = v_in[tid]
    else:
        sdata[tid] = 999999

    cuda.syncthreads()

    stride = 1

    while stride < cuda.blockDim.x:

        if tid % (2 * stride) == 0 and tid + stride < cuda.blockDim.x:

            sdata[tid] = min(sdata[tid], sdata[tid + stride])

        stride *= 2

        cuda.syncthreads()

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

soma_ingenua_kernel[1, TAMANHO_BLOCO](d_v, d_out_soma)

maximo_ingenuo_kernel[1, TAMANHO_BLOCO](d_v, d_out_max)

minimo_ingenuo_kernel[1, TAMANHO_BLOCO](d_v, d_out_min)

resultado_soma = d_out_soma.copy_to_host()

resultado_max = d_out_max.copy_to_host()

resultado_min = d_out_min.copy_to_host()

print("Vetor:", v)

print("\n===== GPU =====")

print("Soma GPU:", resultado_soma[0])
print("Máximo GPU:", resultado_max[0])
print("Mínimo GPU:", resultado_min[0])