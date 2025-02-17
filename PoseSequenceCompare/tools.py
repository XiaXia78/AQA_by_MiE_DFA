
import numpy as np

from numba import njit, prange
from pyts.metrics import dtw

@njit()
def cosine_similarity(A, B):
    A_norm, B_norm = np.linalg.norm(A), np.linalg.norm(B)
    if A_norm == 0 or B_norm == 0:
        return 1.0
    return 1.0 - np.dot(A, B) / (A_norm * B_norm)


def get_cost_matrix(A, B):
    n, d1, _ = A.shape
    m, d2, _ = B.shape
    cost_matrix = np.array([[np.mean([cosine_similarity(A[i, k], B[j, k])
                                      for k in range(d1)]) for j in range(m)] for i in range(n)])
    return cost_matrix


def DTW(seq1, seq2,method,options):
    dis,cost_mat,path=dtw(seq1, seq2,dist='precomputed',method=method,options=options,precomputed_cost=get_cost_matrix(seq1, seq2),return_cost=True,return_path=True)
    print(dis,len(path[0]),dis/len(path[0]))
    score=int(100*(1 - dis / len(path[0])))
    return score


