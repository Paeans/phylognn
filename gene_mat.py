import numpy as np
import torch as th

from scipy.io import savemat, loadmat

from dcj_comp import dcj_dist
from genome_file import encodeAdj

from multiprocessing import Pool

# device = th.device('cuda' if th.cuda.is_available() else 'cpu')

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

def trans_mat(size, *p):    
    p1,p2,p3,p4 = sorted(p[:4]) # sorted([p1,p2,p3,p4])
    res = np.identity(size) #np.diag(np.repeat(1,size))
    return res[:, np.r_[0:p1, p3:p4, p2:p3, p1:p2, p4:size]]

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
    p = np.random.randint(0,size + 1, size=4).tolist() 
    if op_type == 2: # reverse operation
        while p[0] == p[1]:
            p = np.random.randint(0,size + 1, size=4).tolist() 
        return p[:2] 
    elif op_type == 0: # trans reverse
        while p[0] == p[1] or \
        (min(p[0], p[1]) == p[2] and (p[3] == 0 or p[3] % abs(p[0] - p[1]) == 0)) or \
        p[2] % (size - abs(p[0] - p[1]) + 1) == min(p[0], p[1]):
            p = np.random.randint(0,size + 1, size=4).tolist() 
    elif op_type == 1: # trans op
        p = sorted(p)
        while (p[0] == p[1] and p[2] == p[3]) or \
        (p[0] == p[1] and p[1] == p[2]) or \
        (p[1] == p[2] and p[2] == p[3]):
            p = sorted(np.random.randint(0,size + 1, size=4).tolist())
            
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

def gen_op_mat(l, n, rand_op = None):
    
    mat_op_list = [trans_rev, trans_mat, revers_mat]
    if rand_op == None:
        rand_op = np.random.randint(0,3,size = n) 
    else:
        rand_op = np.repeat(rand_op, n)
    param_op = [rand_param(l, op_type) for op_type in rand_op]
    t_dist = [1 if x == 2 else 2 for x in rand_op]
    
    op_list = np.array([mat_op_list[op](l, *param) 
           for op, param in zip(rand_op, param_op)])
    
    # with Pool(22) as p:
        # op_list = p.starmap(op_mat, [(op, l, param) for op, param in zip(rand_op,param_op)])
    
    return np.array(op_list), t_dist

def gen_op_mat_wb(l, n, rand_op = None):
    
    mat_op_list = [trans_rev, trans_mat, revers_mat]
    if rand_op == None:
        rand_op = np.random.randint(0,3,size = n) 
    else:
        rand_op = np.repeat(rand_op, n)
    param_op = [rand_param(l, op_type) for op_type in rand_op]
    t_dist = [1 if x == 2 else 2 for x in rand_op]

    op_list = np.array([mat_op_list[op](l, *param) 
           for op, param in zip(rand_op, param_op)])
    # with Pool(22) as p:
        # op_list = p.starmap(op_mat, [(op, l, param) for op, param in zip(rand_op,param_op)])
        
    return np.array(op_list), t_dist, param_op

# def op_mat(op, l, param):
#     mat_op_list = [trans_rev, trans_mat, revers_mat]
#     return mat_op_list[op](l, *param)

# def gen_op_mat_multi(l, n, rand_op = None):
    
#     mat_op_list = [trans_rev, trans_mat, revers_mat]
#     if rand_op == None:
#         rand_op = np.random.randint(0,3,size = n) 
#     else:
#         rand_op = np.repeat(rand_op, n)
#     # param_op = [rand_param(l, op_type) for op_type in rand_op]

#     with Pool(22) as p:
#         param_op = p.starmap(rand_param, [(l,x) for x in rand_op])
#         op_list = p.starmap(op_mat, [(op, l, param) for op, param in zip(rand_op,param_op)])

#     # op_list = np.array([mat_op_list[op](l, *param) 
#     #        for op, param in zip(rand_op, param_op)])
    
#     op_list = np.array(op_list)
#     t_dist = [1 if x == 2 else 2 for x in rand_op]
#     return op_list, t_dist


def gen_dataset(l,n, repeat = 1):
    s = np.zeros((n, repeat, l))
    t = np.zeros((n, repeat))
    
    s[:,0:1] = gen_seqs(l,n)
    t[:, 0] = 0
    # new_seq = th.tensor(s[:,0:1], dtype = th.float, device = device)
    new_seq = s[:, 0:1]
    # o = np.repeat(np.expand_dims(np.identity(l), (0,1)), n, axis = 0)
    
    for i in range(1, repeat):
#     t = np.expand_dims(np.repeat(0, n), axis = 1)
#     for _ in range(repeat):
        new_o, new_t = gen_op_mat(s.shape[-1], s.shape[0])
        new_seq = np.matmul(new_seq, new_o)
        # new_seq = th.matmul(new_seq, th.tensor(new_o, dtype = th.float, device = device))
#         s = np.concatenate((s, new_seq.cpu().numpy()), axis = 1)
        s[:, i:(i+1)] = new_seq # .cpu().numpy()
#         o = np.concatenate((o, np.expand_dims(new_o, 1)), axis = 1)
#         t = np.concatenate((t, np.expand_dims(new_t, 1)), axis = 1)
        t[:, i] = new_t
    return s, None, t

def gen_dataset_wt(l,n, step = 1, op_type = None):
    s = np.zeros((n, step, l))
    t = np.zeros((n, step))
    
    s[:,0:1] = gen_seqs(l,n)
    t[:, 0] = 0
    # new_seq = th.tensor(s[:,0:1], dtype = th.float, device = device)
    new_seq = s[:, 0:1]
    # o = np.repeat(np.expand_dims(np.identity(l), (0,1)), n, axis = 0)
    
    for i in range(1, step):
        new_o, new_t = gen_op_mat(s.shape[-1], s.shape[0], op_type)
        # new_seq = th.matmul(new_seq, th.tensor(new_o, dtype = th.float, device = device))
        new_seq = np.matmul(new_seq, new_o)
        s[:, i:(i+1)] = new_seq # .cpu().numpy()
        t[:, i] = new_t
    return s, None, t

def gen_bp(seq, bp):
    b = np.zeros(len(seq)*2)
    
    bp = [x * 2 for x in bp]
    bp += [x + 1 for x in bp]
    seq_index = encodeAdj(seq)[bp]
    seq_index = seq_index[seq_index >= 0]
    
    b[seq_index] = 1
    
    return b

def gen_dataset_wb(l,n, step = 1, op_type = None): # , device = None):
    s = np.zeros((n, step, l))
    b = np.zeros((n, step, l * 2))
    
    t = np.zeros((n, step))
    
    s[:,0:1] = gen_seqs(l,n)
    t[:, 0] = 0
    # new_seq = th.tensor(s[:,0:1], dtype = th.float, device = device)
    new_seq = s[:, 0:1]
    # o = np.repeat(np.expand_dims(np.identity(l), (0,1)), n, axis = 0)
    
    for i in range(1, step):
        new_o, new_t, bp_list = gen_op_mat_wb(s.shape[-1], s.shape[0], op_type)
        
        with Pool(22) as p:
            b[:, i] = p.starmap(gen_bp, [(seq[0], bp) 
                                             for seq, bp in 
                                         zip(new_seq, bp_list)])
                                             # zip(new_seq.cpu().numpy(), bp_list)])
        # new_seq = th.matmul(new_seq, th.tensor(new_o, dtype = th.float, device = device))
        new_seq = np.matmul(new_seq, new_o)
        s[:, i:(i+1)] = new_seq # .cpu().numpy()
        t[:, i] = new_t
    return s, None, t, b


def gen_data_file(l,n,repeat,filename):
    s, o, t = gen_dataset(l, n, repeat)
    d = np.array([[dcj_dist(a[0], x)[-1] for x in a] for a in s])
    
    savemat(filename, {'s':s, 'o':o, 't':t, 'd':d}, do_compression = True)

def check_dcj(x):
    a,b,c = dcj_dist(x[0], x[1])[-1], dcj_dist(x[-1], x[1])[-1], dcj_dist(x[0], x[-1])[-1]
    if a != b or (a+b) != c:
        return False
    return True

def gen_g2g_data(gene_len, graph_num, step, op_type):
    l = 0
    res = np.zeros((graph_num, 3, gene_len), dtype = np.int32)
    while True:
        s,o,t = gen_dataset_wt(gene_len, graph_num * 2, 2*step + 1, op_type)
        s = s[:, (0, step, -1)]

        with Pool(10) as p:
            tags = p.map(check_dcj, list(s))
        s =  s[tags]
        size = min(s.shape[0], graph_num - l)

        res[l: (l + size)] = s[:size]
        l += size
        if l>=graph_num:
            return res
        
def mid_dcj(s, p):
    dlist = [dcj_dist(s, x)[-1] for x in p]
    cand_len = len(p)
    dists = [dcj_dist(p[i], p[j])[-1] for i in range(cand_len) for j in range(i + 1, cand_len)]
    # a, b, c = p
    return sum(dlist) == np.ceil(sum(dists)/2)

def gene_mid(s, cand_seqs, mid_num = 3):
    cand_len = len(cand_seqs)
    cnum = np.random.choice(cand_len, mid_num, replace = False)
    
    while not mid_dcj(s, cand_seqs[cnum].squeeze()):
        cnum = np.random.choice(cand_len, mid_num, replace = False)
    return cand_seqs[cnum]

def gen_m3g_data_old(gene_len, graph_num, step, op_type, mid_num = 3, k = 10):
    seq = gen_seqs(gene_len, graph_num)
    
    new_seq = np.zeros((graph_num, k * step, 1, gene_len)) # graph_num, k * step, 1, gene_len
    tmp_seq = np.expand_dims(seq, axis = 1) # graph_num, 1, 1, gene_len
    # tmp_seq = th.tensor(np.expand_dims(seq, axis = 1), dtype = th.float, device = device)
    for i in range(step):
        op = gen_op_mat(gene_len, graph_num * k, 
                        op_type)[0].reshape(graph_num, k, 
                                            gene_len, gene_len) # graph_num, k, gene_len, gene_len
        tmp_seq = np.matmul(tmp_seq, op) # graph_num, k, 1, gene_len
        # tmp_seq = th.matmul(tmp_seq, th.tensor(op, dtype = th.float, device = device))
        new_seq[:, i * k:(i+1) * k] = tmp_seq # .cpu().numpy() # tmp_seq
       
    # with Pool(10) as p:
        # mid_seq = p.starmap(gene_mid, [(s[0], ns, mid_num) for s, ns in zip(seq, new_seq)])
        # graph_num, mid_num, 1, gene_len
    mid_seq = [gene_mid(s[0], ns, mid_num) for s, ns in zip(seq, new_seq)]
    
    return np.array(mid_seq).squeeze(axis = -2), seq

def gen_m3g_data(gene_len, graph_num, step, op_type, mid_num = 3, k = 10):
    seq = gen_seqs(gene_len, graph_num)
    
    new_seq = np.zeros((graph_num, k * step, 1, gene_len)) # graph_num, k * step, 1, gene_len
    tmp_seq = np.expand_dims(seq, axis = 1) # graph_num, 1, 1, gene_len
    # tmp_seq = th.tensor(np.expand_dims(seq, axis = 1), dtype = th.float, device = device)
    for i in range(step):
        op = np.expand_dims(gen_op_mat(gene_len, k, op_type)[0], axis = 0)
        tmp_seq = np.matmul(tmp_seq, op) # graph_num, k, 1, gene_len
        # tmp_seq = th.matmul(tmp_seq, th.tensor(op, dtype = th.float, device = device))
        new_seq[:, i * k:(i+1) * k] = tmp_seq # .cpu().numpy() # tmp_seq
       
    # with Pool(10) as p:
        # mid_seq = p.starmap(gene_mid, [(s[0], ns, mid_num) for s, ns in zip(seq, new_seq)])
        # graph_num, mid_num, 1, gene_len
    mid_seq = [gene_mid(s[0], ns, mid_num) for s, ns in zip(seq, new_seq)]
    
    return np.array(mid_seq).squeeze(axis = -2), seq