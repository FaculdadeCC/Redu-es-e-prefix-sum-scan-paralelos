import numpy as np
from numba import cuda, int32


TAMANHO_BLOCO = 32

@cuda.jit
def blelloch_scan_kernel(v_in, v_out):
  
    sdata = cuda.shared.array(shape=(TAMANHO_BLOCO,), dtype=int32)
    tid = cuda.threadIdx.x
  
    if tid < v_in.size:
        sdata[tid] = v_in[tid]
    else:
        sdata[tid] = 0
        
    cuda.syncthreads()
  
    offset = 1
    d = TAMANHO_BLOCO // 2
    
    while d > 0:
        cuda.syncthreads()
        if tid < d:
            ai = offset * (2 * tid + 1) - 1
            bi = offset * (2 * tid + 2) - 1
            sdata[bi] += sdata[ai]
            
        offset *= 2
        d //= 2
        
    cuda.syncthreads()
  
    if tid == 0:
        sdata[TAMANHO_BLOCO - 1] = 0
        
    d = 1
    while d < TAMANHO_BLOCO:
        offset //= 2
        cuda.syncthreads()
        
        if tid < d:
            ai = offset * (2 * tid + 1) - 1
            bi = offset * (2 * tid + 2) - 1
            
            temp = sdata[ai]
            sdata[ai] = sdata[bi]
            sdata[bi] += temp
            
        d *= 2
        
    cuda.syncthreads()
   
    if tid < v_out.size:
        v_out[tid] = sdata[tid]

if __name__ == '__main__':
    v = np.array([1, 2, 3, 4, 5], dtype=np.int32)
    out = np.zeros_like(v)

    d_v = cuda.to_device(v)
    d_out = cuda.to_device(out)

    blelloch_scan_kernel[1, TAMANHO_BLOCO](d_v, d_out)

    resultado = d_out.copy_to_host()

    print("Entrada:", v)
    print("Saída Exclusiva:", resultado)
