import numpy as np

def soma_cpu(v):
    soma = 0

    for i in range(len(v)):
        soma += v[i]

    return soma

v = np.array([1, 2, 3, 4, 5])

print("Vetor:", v)

print("Soma:", soma_cpu(v))
