'''
A Probabilistic Distance Measure for Hidden
            Markov Models
AT&T Technical Journal
Vol. 64, No.2, February 1985
Printed in U.S.A.


'''

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sb


def get_index(arr):
    cum = np.cumsum(arr)
    val = np.random.rand()
    return np.searchsorted(cum, val)


def fast_pow(B, e):
    ans = np.eye(B.shape[0])
    while (e > 0):
        if (e & 1):
            ans = ans.dot(B)
        B = B.dot(B)
        e >>= 1
    return ans


class HMM:
    def __init__(self, N, M, A, B, u):
        self.A = A
        self.B = B
        self.u = u
        self.a = np.array([])  # Stationary distribution matrix
        self.N = N
        self.M = M
        self.find_st_dist()

    def show(self):
        print (self.N, self.M)
        print (self.A)
        print (self.B)
        print (self.u)

    def gen_rw(self, T):
        cur_state = get_index(self.a)
        rw = [[cur_state, get_index(self.B[cur_state])]]
        for i in range(T - 1):
            cur_state = get_index(self.A[cur_state])
            rw.append([cur_state, get_index(self.B[cur_state])])
        return rw

    def find_st_dist(self):
        M = fast_pow(self.A, 10 ** 18)
        self.a = self.u.dot(M)
        self.a /= np.sum(self.a)
        b = self.a.dot(self.A)
        assert(np.allclose(self.a, b))

    def mu(self, sequence):
        T = len(sequence)
        dp = np.zeros([T, self.N])

        for i in range(self.N):
            dp[T - 1][i] = self.B[i][sequence[T - 1]]

        for j in range(T - 2, -1, -1):
            for i in range(self.N):
                for k in range(self.N):
                    dp[j][i] += dp[j + 1][k] * self.A[i][k]
                dp[j][i] *= self.B[i][sequence[j]]
        ans = 0
        for i in range(self.N):
            ans += dp[0][i] * self.a[i]
        return ans

    def H(self, sequence):
        T = len(sequence)
        return (1.0 / T) * np.log(self.mu(sequence))


class KLUpperBoundDist:
    def __init__(self, hmm1, hmm2):
        N, M = hmm1.emissionprob_.shape
        self.M1 = HMM(N, M, hmm1.transmat_, hmm1.emissionprob_, hmm1.startprob_)
        self.M2 = HMM(N, M, hmm2.transmat_, hmm2.emissionprob_, hmm2.startprob_)

    def dist(self):

        return 1.0


class ProbDist:
    def __init__(self, hmm1, hmm2):
        N, M = hmm1.emissionprob_.shape
        self.M1 = HMM(N, M, hmm1.transmat_, hmm1.emissionprob_, hmm1.startprob_)
        self.M2 = HMM(N, M, hmm2.transmat_, hmm2.emissionprob_, hmm2.startprob_)

    def dist(self, T):
        rw1 = self.M1.gen_rw(T)
        y = [i[1] for i in rw1]
        lp1 = self.M1.mu(y)
        lp2 = self.M2.mu(y)
        if lp1 == 0:
            lp1 = 1e-20
        if lp2 == 0:
            lp2 = 1e-20
        lp1 = np.log(lp1)
        lp2 = np.log(lp2)
        return (1.0 / T) * (lp1 - lp2), lp1, lp2


class ProbDistVit:
    def __init__(self, hmm1, hmm2, e=1e-15):
        N, M = hmm1.emissionprob_.shape
        self.eps = e
        self.M1 = HMM(N, M, hmm1.transmat_, hmm1.emissionprob_, hmm1.startprob_)
        self.M2 = HMM(N, M, hmm2.transmat_, hmm2.emissionprob_, hmm2.startprob_)

    def dec(self):
        ans = 0.0
        for i in range(self.M1.N):
            tmp = self.M1.B[i].dot(self.M2.B[i])
            ans += tmp
        ans /= self.M1.N
        return np.sqrt(ans)

    def dist(self):
        ans = 0.0
        for i in range(self.M1.N):
            for j in range(self.M1.N):
                tmp = np.log(self.M2.A[i][j]+self.eps) - np.log(self.M1.A[i][j]+self.eps)
                ans += self.M1.A[i][j] * self.M1.a[i] * tmp
            for k in range(self.M1.M):
                tmp = np.log(self.M2.B[i][k]+self.eps) - np.log(self.M1.B[i][k]+self.eps)
                ans += self.M1.B[i][k] * self.M1.a[i] * tmp
        return ans
