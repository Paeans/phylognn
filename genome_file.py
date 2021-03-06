import numpy as np
import random


def write_genome(gene_groups, fname = 'genome.txt'):
    with open(fname, 'w') as gfile:
        for i in range(len(gene_groups)):
            gfile.write('>g' + str(i) + '\n')
            for g in gene_groups[i]:
                if g[0] == g[-1] and len(g) >= 2:
                    gfile.write('C: ' + ' '.join([str(x) for x in g[:-1]]) + '\n')
                else:
                    gfile.write('L: ' + ' '.join([str(x) for x in g]) + '\n')
            gfile.write('\n')

def write_genome_notag(gene_groups, fname = 'genome.txt'):
    with open(fname, 'w') as gfile:
        for i in range(len(gene_groups)):
            gfile.write('>g' + str(i) + '\n')
            for g in gene_groups[i]:
                if g[0] == g[-1] and len(g) >= 2:
                    gfile.write(' '.join([str(x) for x in g[:-1]]) + '\n')
                else:
                    gfile.write(' '.join([str(x) for x in g]) + '\n')
            gfile.write('\n')
            
def read_genome(fname = 'genome.txt'):
    gene_groups = []
    with open(fname, 'r') as gfile:        
        gene = []
        tag = False
        
        for gline in gfile:
            if gline.startswith('>'):
                if tag and gene:
                    gene_groups.append(gene)
                tag = True
                gene = []
                continue
            
            genes = gline.strip().split()
            if genes == []:
                continue
            if genes[0] == 'C:':
                genes.append(genes[1])
            elif genes[0] != 'L:':
                continue
            gene.append([int(x) for x in genes[1:]])
        if gene:
            gene_groups.append(gene)
    return gene_groups

def trans_circle(g, p1, p2, rand_op = None):
    if rand_op == None:
        rand_op = random.random()
    
    g = g[:-1]
    p1 %= len(g)
    p2 %= len(g)
    
    if p1 == p2:
        return [g[p1:] + g[:p1]]
    if p2 < p1:
        p1, p2 = p2, p1
        
    if rand_op < 0.5:
        c0 = g[0:p1] + [-x for x in reversed(g[p1:p2])] + g[p2:]
        return [c0 + c0[0:1]]
    else:
        c1 = g[0:p1] + g[p2:]
        c2 = g[p1:p2]
        return [c1 + c1[0:1], c2 + c2[0:1]]
    
def trans_linear(g, p1, p2, rand_op = None):
    if rand_op == None:
        rand_op = random.random()
    size = len(g) + 1
    p1 %= size
    p2 %= size
    
    if p1 == p2:        
        return [g]
    if p2 < p1:
        p1, p2 = p2, p1
        
    if p1 == 0 and p2 == len(g):
        if rand_op < 0.5:
            return [[-x for x in reversed(g)]]
        else:
            return [g + g[0:1]]
    
    if rand_op < 0.5:
        return [g[0:p1] + [-x for x in reversed(g[p1:p2])] + g[p2:]]
    else:
        return [g[0:p1] + g[p2:], g[p1:p2] + g[p1:p1+1]]

def trans(g, p1, p2, rand_op = None):
    if g[0] == g[-1] and len(g) >= 2:
        return trans_circle(g, p1, p2, rand_op)
    return trans_linear(g, p1, p2, rand_op)

def trans_cross(g1, g2, p1, p2, rand_op = None):
    if rand_op == None:
        rand_op = random.random()
        
    if g1[0] == g1[-1] and g2[0] == g2[-1] and len(g1) >= 2 and len(g2) >= 2:
        # circle and circle
        if rand_op < 0.5:
            res = g1[:p1] + g2[p2:-1] + g2[:p2] + g1[p1:-1]          
        else:
            res = g1[:p1] + [-x for x in reversed(g2[:p2])] + \
                    [-x for x in reversed(g2[p2:-1])] + g1[p1:-1]
        return [res + res[0:1]]
        
    if (g1[0] != g1[-1] or len(g1) == 1) and (g2[0] != g2[-1] or len(g2) == 1):
        # linear and linear
        if rand_op < 0.5:
            r1 = g1[:p1] + g2[p2:]
            r2 = g2[:p2] + g1[p1:]
        else:
            r1 = g1[:p1] + [-x for x in reversed(g2[:p2])]
            r2 = [-x for x in reversed(g2[p2:])] + g1[p1:]
        return [x for x in [r1, r2] if x!=[] ]
        
    # linear and circle
    if g1[0] == g1[-1] and len(g1) >= 2:
        c, l = g1, g2
        cp, lp = p1, p2
    else:
        c, l = g2, g1
        cp, lp = p2, p1
    if rand_op < 0.5:
        return [l[:lp] + c[cp:-1] + c[:cp] + l[lp:]]
    else:
        return [l[:lp] + [-x for x in reversed(c[:cp])] + 
                [-x for x in reversed(c[cp:-1])] + l[lp:]]
    
def trans_op(g, p1, p2):
    size_list = []
    for gene in g:
        size = len(gene)
        if gene[0] == gene[-1] and size >=2 :
            size -= 2
        size_list.append(size)
    p1 %= sum(size_list) + len(size_list)
    p2 %= sum(size_list) + len(size_list)
        
    t1, t2 = 0, 0
    if p1 > p2:
        p2, p1 = p1, p2
    for t1 in range(len(size_list)):
        if p1 <= size_list[t1]:
            break
        p1 -= size_list[t1] + 1
    for t2 in range(len(size_list)):
        if p2 <= size_list[t2]:
            break
        p2 -= size_list[t2] + 1
    # print(t1, p1, t2, p2)
    
    if t1 == t2:
        res = trans(g[t1], p1, p2)
        return g[:t1] + res + g[t1+1:]
    
    res = trans_cross(g[t1], g[t2], p1, p2)
    return g[:t1] + g[t1 + 1:t2] + g[t2 + 1:] + res

def is_circle(g):
    if len(g) >= 2 and g[0] == g[-1]:
        return True
    return False

def rev_trans(g, p1 = None, p2 = None):
    g0_range = len(g[0]) - (2 if is_circle(g[0]) else 0)
    if p1 == None:
        p1 = random.randint(0, g0_range)
    if p2 == None:
        start = 0 if len(g) == 1 else (g0_range + 1)
        end = g0_range if len(g) == 1 else (start + len(g[1]) - (2 if is_circle(g[1]) else 0))
        p2 = random.randint(start, end)
    return trans_op(g, p1, p2)

def encodeAdj_old(genome):
    l = len(genome)
    adjacency = np.zeros(l*2 + 2, dtype=np.int32)
    adjacency[0] = -1

    for i in range(l):
        genin = genome[i] * 2
        if genome[i] > 0:
            adjacency[i*2 + 1] = genin - 2
            adjacency[i*2 + 2] = genin - 1
        else:
            adjacency[i*2 + 1] = -genin - 1
            adjacency[i*2 + 2] = -genin - 2

    adjacency[l * 2 + 1] = -2
    if l>= 2 and genome[0] == genome[-1]:
        # cicle
        return np.delete(adjacency, -2)
    return adjacency

def encodeAdj(genome):
    l = len(genome)
    adjacency = np.zeros(l*2 + 2, dtype = np.int32)
    adjH = np.abs(genome) * 2 - 1
    
    t = np.zeros(l)
    t[np.array(genome) > 0] = 1
    
    adjacency[1:-1] = np.stack((adjH - t, adjH + t - 1)).flatten('F')
    
    adjacency[0] = -1
    adjacency[-1] = -2
    if l>= 2 and genome[0] == genome[-1]:
        # cicle
        return np.delete(adjacency, -2)
    return adjacency

def test_cycle(adj_dict, x, y):
    while True:
        x = x + (1 if x % 2 == 0 else -1)
        if x == y:
            return True
        if x not in adj_dict.keys():
            return False
        x = adj_dict[x]

def dict2adj(adj_dict, start):
    res = []
    while True:
        end = start + (1 if start % 2 == 0 else -1)
        res += [start, end]
        if end not in adj_dict.keys():
            break
        start = adj_dict[end]
        
    return [(res[i]//2 + 1) * (-1 if res[i] > res[i+1] else 1) for i in range(0, len(res), 2)]
        

# def mat2adj(res_mat):
#     gen_len = res_mat.shape[0]
#     tmp = np.copy(res_mat)
#     for i in range(0, gen_len, 2):
#         tmp[i, i:i+2] = 0
#         tmp[i + 1, i:i+2] = 0
#     adj_dict = {}
#     while True:
#         r,c = np.unravel_index(np.argmax(tmp, axis = None), tmp.shape)
                
#         # test cycles
#         if test_cycle(adj_dict, r, c):
#             tmp[r, c] = 0
#             tmp[c, r] = 0
#             continue
            
#         tmp[(r,c), :] = 0
#         tmp[:, (r,c)] = 0
#         adj_dict[r] = c
#         adj_dict[c] = r
#         if len(adj_dict)//2 == (gen_len//2 - 1):
#             break
            
#     start = list(set(range(gen_len)) - set(adj_dict.keys()))
    
#     return [dict2adj(adj_dict, a) for a in start]

def mat2adj(res_mat):
    gen_len = res_mat.shape[0]
    tmp = np.copy(res_mat)
    for i in range(0, gen_len, 2):
        tmp[i, i:i+2] = 0
        tmp[i + 1, i:i+2] = 0
    adj_dict = {}
    p_list = []
    while True:
        r,c = np.unravel_index(np.argmax(tmp, axis = None), tmp.shape)
        if r == c:
            print(adj_dict)
            break
                
        # test cycles
        if test_cycle(adj_dict, r, c):
            tmp[r, c] = 0
            tmp[c, r] = 0
            continue
        p_list.append(tmp[r,c])    
        tmp[(r,c), :] = 0
        tmp[:, (r,c)] = 0
        adj_dict[r] = c
        adj_dict[c] = r
        if len(adj_dict)//2 == (gen_len//2 - 1):
            break
    
    start = list(set(range(gen_len)) - set(adj_dict.keys()))
    
    return [dict2adj(adj_dict, a) for a in start], -np.log(p_list).sum()

def pre_set_mat(res_mat):
    tmp_mat = np.copy(res_mat)
    for i in range(0, tmp_mat.shape[0], 2):
        tmp_mat[i, i:i+2] = 0
        tmp_mat[i + 1, i:i+2] = 0
        
    return tmp_mat

def dist_eval(res, steps = 10, rate = 0.9):
    pred_list, prob_list = [], []

    res_mat = pre_set_mat(res)
    pred, prob_ = mat2adj(res_mat)
    pred_list.append(pred)
    prob_list.append(prob_)

    for _ in range(1, steps):
        r,c = np.unravel_index(np.argmax(res_mat, axis = None), res_mat.shape)
        if r == c:
            break
        if res_mat[r,c] == 1.0:
            redv = 0.95
        else:
            redv = res_mat[r,c] * rate # res_mat[r,c]
        res_mat[r,c] = redv
        res_mat[c,r] = redv
        pred, prob_ = mat2adj(res_mat)
        
        pred_list.append(pred)
        prob_list.append(prob_)
        
    return pred_list, prob_list

def pred_pair(res, val_seqs, steps = 10, rate = 0.9):
    
    from dcj_comp import dcj_dist
    
    pred_list, prob_list = dist_eval(res, steps, rate)
    result = []
    for pred, prob_ in zip(pred_list, prob_list):
        tmp_dist = [sum([dcj_dist(p, s)[-1] for s in val_seqs]) for p in pred]
        t = np.argmin(tmp_dist)
        # p_seq = pred[np.argmin(tmp_dist)]
        result.append((pred[t], prob_, tmp_dist[t]))
    return result