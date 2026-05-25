import numpy as np

def somaCpu(v):
    soma = 0

    for i in range(len(v)):
        soma += v[i]

    return soma

def maximoCpu(v):
    maior  = v[0]

    for i in range(1, len(v)):
        if v[i] > maior:
            maior = v[i]

    return maior

v = np.array([1, 2, 3, 4, 5])

print("Vetor:", v)

print("Soma:", somaCpu(v))
print("Máximo:", maximoCpu(v))