import numpy as np

# def revers_mat(size, p1, p2):
def revers_mat(size, *p):
    p1, p2 = p[:2]
    if p1 > p2:
        p1, p2 = p2, p1
    res = np.diag(np.repeat(1,size))
    res[:, p1:p2] = np.fliplr(res[:, p1:p2]) * -1
    return res

def circle_mat(size, p1, p2):
    if p1 > p2:
        p1, p2 = p2, p1
    res = np.diag(np.repeat(1,size))
    return res[:, np.r_[0:p1, p2:size]], res[:,np.r_[p1:p2, p1]]

# def trans_mat(size, p1, p2, p3, p4):
def trans_mat(size, *p):    
    p1,p2,p3,p4 = sorted(p[:4]) # sorted([p1,p2,p3,p4])
    res = np.diag(np.repeat(1,size))
    return res[:, np.r_[0:p1, p3:p4, p2:p3, p1:p2, p4:size]]

# def trans_rev(size, p1, p2, p3, p4):
def trans_rev(size, *p):
    p1, p2, p3, p4 = p[:4] # sorted(p[:4]) # sorted([p1, p2, p3, p4])
    if p1 > p2:
        p1,p2 = p2,p1
    res = np.diag(np.repeat(1,size))
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