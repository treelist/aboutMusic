
import numpy as np

def get_mean(ndarr2d):
    return ndarr2d.var(axis = 0).mean()

def find_best(ndarr1d, start, end):
    max_value = ndarr1d.max()
    cutoff = max_value * 0.05
    ndarr1d[ndarr1d < cutoff] = 0.0001
    #ndarr1d = ndarr1d ** .5
    ndarr1d = ndarr1d + 1
    ndarr1d = np.log(ndarr1d)

    min = 1000000
    ndarr2d = None
    lst = []
    size = ndarr1d.shape[0]
    for intrv in range(start, end + 1):
        col_num = size // intrv
        rest = size % intrv
        
        if (rest == 0):
            temp2d = ndarr1d.reshape(-1, intrv)
            #temp2d = np.divide(temp2d.transpose(), temp2d.max(axis = 1)).transpose()
            temp = get_mean(temp2d)
        else:
            temp2d = ndarr1d[:-rest].reshape(-1, intrv)
            #temp2d = np.divide(temp2d.transpose(), temp2d.max(axis = 1)).transpose()
            temp = get_mean(temp2d)
        
        if min > temp:
            min = temp
            ndarr2d = temp2d
        
        lst.append((intrv, temp))
    return np.array(lst)
    #arr[np.argsort(arr[:, 1])]