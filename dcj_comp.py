import numpy as np
from genome_file import encodeAdj

def neighbor(A, num):
    index = np.where(A == num)[0][0]
    if index % 2 == 0:
        index += 1
    else:
        index -= 1
    return A[index]

def findPath(A, B, num, visited, path):
    if num in visited:
        return True
    if num == -1:
        return False
    
    visited.append(num)
    nb = neighbor(B, num)
    path.append((num, nb))
    return findPath(B, A, nb, visited, path)

def dcj_dist(A, B):
    if A[0] != -1 or A[-1] != -1:
        A = encodeAdj(A)
    if B[0] != -1 or B[-1] != -1:
        B = encodeAdj(B)
        
    visited = []
    cycle, oddp = 0, 0
    
    for i in range(1, len(A) - 1):
        if A[i] in visited:
            continue
        path = []
        if findPath(A, B, A[i], visited, path):
            cycle += 1
            continue
            
        nb = neighbor(A, A[i])
        if nb not in visited:
            findPath(A, B, nb, visited, path)
        path = [(nb, A[i])] + path
        if len(path) % 2 == 0:
            oddp += 1
        
    return cycle, oddp, (len(A) - 2)//2 - (cycle + oddp // 2)