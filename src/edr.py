import numpy as np
import pandas as pd


def edr(df_mod: pd.DataFrame, df_ots: pd.DataFrame, eps):
    """
    Edit Distance for Real Sequences
    Ref: https://github.com/bguillouet/traj-dist
    Ref: MATLAB edr.m

    Parameters
    ----------

    s: source signal

    r: reference signal

    eps: tolerance

    Return
    ------

    EDR
    """

    M = df_mod.size
    N = df_ots.size

    # compute cumulative distance
    print("Computing Cumulative Distance...")
    C = np.zeros(shape=(M + 1, N + 1))
    for i in range(1, M + 1):
        for j in range(1, N + 1):
            if np.linalg.norm(df_mod[i - 1] - df_ots[j - 1]) < eps:
                print("LESS THAN TOL!")
                c = 0
            else:
                c = 1

            C[i][j] = min(C[i][j - 1] + 1, C[i - 1][j] + 1, C[i - 1][j - 1] + c)

    # retrieve traceback
    print("Retreiving Traceback...")
    ix = np.zeros(shape=(M + N, 1))
    iy = np.zeros(shape=(M + N, 1))
    i, j, k = M, N, 1
    while i > 1 or j > 1:
        if j == 1:
            i -= 1
        elif i == 1:
            j -= 1
        else:
            cij = C[i - 1, j - 1]
            ci = C[i - 1, j]
            cj = C[i, j - 1]
            i = i - int((ci <= cj) or (cij <= cj) or (cj != cj))
            j = j - int((cj < ci) or (cij <= ci) or (ci != ci))
        k += 1
        ix[k] = i
        iy[k] = j

    ix = ix[k::-1]
    iy = iy[k::-1]
    dist = float(C[M][N] / max([M, N]))
    return dist, ix, iy
