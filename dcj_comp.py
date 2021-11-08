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
#     if num == -1:
#         return False
    
    visited.append(num)
    nb = neighbor(B, num)
    path.append((num, nb))
    return findPath(B, A, nb, visited, path)

def dcj_dist_old(A, B):
    A, B = encodeAdj(A), encodeAdj(B)
    
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
        
    return cycle, oddp, len(A)//2 - (cycle + oddp // 2) # (len(A) - 2)//2 - (cycle + oddp // 2)


def dcj_dist(A, B):
    A, B = encodeAdj(A), encodeAdj(B)
    
    A_visited = {x:False for x in A}
    A_nb, B_nb = {}, {}
    for i in range(len(A)):
        t = i + 1 if i % 2 == 0 else i - 1
        A_nb[A[i]] = A[t]
        B_nb[B[i]] = B[t]
        
    cycle, oddp, evenp = 0, 0, 0
    
    for i in range(len(A)):
        p = A[i]
        if A_visited[p]:
            continue
            
        while True:
            A_visited[p] = True
            p_next = B_nb[p]
#             if p_next == None:
#                 oddp += 1
#                 break
                
            A_visited[p_next] = True
            p = A_nb[p_next]
#             if p == None:
#                 evenp += 1
#                 break
            if A_visited[p]:
                cycle += 1
                break
                
    return cycle, oddp, len(A)//2 - (cycle + oddp // 2)