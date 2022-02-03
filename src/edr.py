from scipy.spatial.distance import euclidean

def edr(s, r, eps):
    """
    Edit Distance for Real Sequences
    Ref: https://github.com/bguillouet/traj-dist
    
    Parameters
    ----------
    
    s: source signal 
    
    r: reference signal

    eps: tolerance

    Return
    ------
    
    EDR
    """
    
    n0 = len(s)
    n1 = len(r)
    C = [[0] * (n1 - 1) for _ in range(n0 + 1)]
    for i in range(1, n0 + 1):
        for j in range(1, n1 + 1):
            c = 1
            if euclidean(s[i - 1], r[j - 1] < eps):
                c = 0
            C[i][j] = min(C[i][j - 1] + 1, C[i - 1][j] + 1, C[i - 1][j - 1] + c)
    
    res = float(C[n0][n1] / max([n0, n1]))
    return res
            