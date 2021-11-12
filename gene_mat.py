import numpy as np
import random

import torch as th

from scipy.io import savemat, loadmat

from dcj_comp import dcj_dist

device = th.device('cuda')

# def revers_mat(size, p1, p2):
def revers_mat(size, *p):
    p1, p2 = p[:2]
    if p1 > p2:
        p1, p2 = p2, p1
    res = np.identity(size) #np.diag(np.repeat(1,size))
    res[:, p1:p2] = np.fliplr(res[:, p1:p2]) * -1
    return res

# def circle_mat(size, p1, p2):
#     if p1 > p2:
#         p1, p2 = p2, p1
#     res = np.diag(np.repeat(1,size))
#     return res[:, np.r_[0:p1, p2:size]], res[:,np.r_[p1:p2, p1]]

# def trans_mat(size, p1, p2, p3, p4):
def trans_mat(size, *p):    
    p1,p2,p3,p4 = sorted(p[:4]) # sorted([p1,p2,p3,p4])
    res = np.identity(size) #np.diag(np.repeat(1,size))
    return res[:, np.r_[0:p1, p3:p4, p2:p3, p1:p2, p4:size]]

# def trans_rev(size, p1, p2, p3, p4):
def trans_rev(size, *p):
    p1, p2, p3, p4 = p[:4] # sorted(p[:4]) # sorted([p1, p2, p3, p4])
    if p1 > p2:
        p1,p2 = p2,p1
    res = np.identity(size) # np.diag(np.repeat(1,size))
    left = res[:, np.r_[0:p1, p2:size]]
    right = res[:, np.r_[p1:p2]]
    lsize, rsize = left.shape[-1], right.shape[-1]
    p3 = p3 % (lsize + 1)
    p4 = p4 % (rsize + 1)
    right = np.fliplr(right[:, np.r_[p4:rsize, 0:p4]]) * -1
    
    return np.concatenate((left[:, 0:p3], right, left[:, p3:lsize]), axis = 1)
#     res = res[:, np.r_[0:p1, p3:p4, p2:p3, p1:p2, p4:size]]
#     end = p1 + p4 - p3 + p3 - p2
#     res[:, p1:end] = np.fliplr(res[:, p1:end]) * -1
#     return res


def rand_param(size, op_type):    
    p = np.random.randint(0,size + 1, size=4).tolist() #[random.randint(0,size) for _ in range(4)]
    if op_type == 2: # reverse operation
        while p[0] == p[1]:
            p = np.random.randint(0,size + 1, size=4).tolist() #[random.randint(0,size) for _ in range(4)]
    elif op_type == 0: # trans reverse
        while p[0] == p[1] or \
        (min(p[0], p[1]) == p[2] and (p[3] == 0 or p[3] % abs(p[0] - p[1]) == 0)) or \
        p[2] % (size - abs(p[0] - p[1]) + 1) == min(p[0], p[1]):
            p = np.random.randint(0,size + 1, size=4).tolist() #[random.randint(0,size) for _ in range(4)]
    elif op_type == 1: # trans op
        p = sorted(p)
        while (p[0] == p[1] and p[2] == p[3]) or \
        (p[0] == p[1] and p[1] == p[2]) or \
        (p[1] == p[2] and p[2] == p[3]):
            p = sorted(np.random.randint(0,size + 1, size=4).tolist()) #sorted([random.randint(0,size) for _ in range(4)])
            
    return p


def generate_seq(gene):
    if len(gene) == 1:
        return [[gene[0]], [-gene[0]]]
    res = []
    for i in range(len(gene)):
        g = gene[i]
        rest = gene[0:i] + gene[i+1:]
        tmp = generate_seq(rest)
        res += [[g] + x for x in tmp] + [[-g] + x for x in tmp]
    return res

def gen_random_seq(l, n, r_n):
    res = np.zeros((n, l), dtype = np.int32)
    for i in range(n):
        s = np.random.permutation(l) + 1
        s[np.random.randint(l, size = r_n)] *=-1
        res[i] = s
    return res

def gen_seqs(l,n):
    genes = np.zeros((n, l), dtype = np.int32)
    for i in range(n):
#         genes[i] = gen_random_seq(l,1,np.randint(0,l+1))
        genes[i] = np.random.permutation(l) + 1
        genes[i, np.random.randint(l, size = np.random.randint(0,l+1))] *= -1
#     np.random.shuffle(genes)
    return np.reshape(genes, (-1,1,l))

def gen_op_mat(l, n):
#     t_dist = [0 for _ in range(n)]
    
    mat_op_list = [trans_rev, trans_mat, revers_mat]
    rand_op = np.random.randint(0,3,size = n) # [random.randint(0,2) for _ in range(n)]
    param_op = [rand_param(l, op_type) for op_type in rand_op] #[[random.randint(0,n) for _ in range(4)] for _ in range(vol)]
#     t_dist = np.add(t_dist, [1 if x == 2 else 2 for x in rand_op])
    t_dist = [1 if x == 2 else 2 for x in rand_op]

    op_list = np.array([mat_op_list[op](l, *param) 
           for op, param in zip(rand_op, param_op)])
    return op_list, t_dist


def gen_dataset(l,n, repeat = 1):
    s = np.zeros((n, repeat, l))
    t = np.zeros((n, repeat))
    
    s[:,0:1] = gen_seqs(l,n)
    t[:, 0] = 0
    new_seq = th.tensor(s[:,0:1], dtype = th.float, device = device)
    
    o = np.repeat(np.expand_dims(np.identity(l), (0,1)), n, axis = 0)
    
    for i in range(1, repeat):
#     t = np.expand_dims(np.repeat(0, n), axis = 1)
#     for _ in range(repeat):
        new_o, new_t = gen_op_mat(s.shape[-1], s.shape[0])
#         new_seq = np.matmul(new_seq, new_o)
        new_seq = th.matmul(new_seq, th.tensor(new_o, dtype = th.float, device = device))
#         s = np.concatenate((s, new_seq.cpu().numpy()), axis = 1)
        s[:, i:(i+1)] = new_seq.cpu().numpy()
#         o = np.concatenate((o, np.expand_dims(new_o, 1)), axis = 1)
#         t = np.concatenate((t, np.expand_dims(new_t, 1)), axis = 1)
        t[:, i] = new_t
    return s, o, t


def gen_data_file(l,n,repeat,filename):
    s, o, t = gen_dataset(l, n, repeat)
    d = np.array([[dcj_dist(a[0], x)[-1] for x in a] for a in s])
    
    savemat(filename, {'s':s, 'o':o, 't':t, 'd':d}, do_compression = True)
