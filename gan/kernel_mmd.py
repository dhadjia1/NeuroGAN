# Calculte the kernel mean maximum discrepency between two sets of images.


import numpy as np

def distance(x1, x2):
    return np.linalg.norm(x1-x2)

def gkernel(x1, x2, sigma):
    x1 = x1.reshape(-1,)
    x2 = x2.reshape(-1,)

    d = distance(x1, x2)
    
    num = d  ** 2
    denom = 2 * (sigma ** 2)
    evaluation = np.exp(-num/denom)
    return evaluation
    

def mmd(imgs1, imgs2, sigma):
    
    N1 = imgs1.shape[0]
    N2 = imgs2.shape[0]

    imgs1.reshape(N1,-1)
    imgs2.reshape(N2,-1)
    
    
    statistic = 0.0
    count = 0
    for i in range(N1):
        xi, yi = imgs1[i], imgs2[i]
        for j in range(N2):
            if (i == j):
                continue
            count += 1
            xj, yj = imgs1[j], imgs2[j]
            
            xij = gkernel(xi, xj, sigma)
            yij = gkernel(yi, yj, sigma)
            xiyj = gkernel(xi, yj, sigma)
            xjyi = gkernel(xj, yi, sigma)
            iter_val = xij + yij - xiyj - xjyi
            statistic += iter_val
    statistic /= count
    if statistic >= 0.0:
        return np.sqrt(statistic)
    else:
        return 0.0
 