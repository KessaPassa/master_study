import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.neighbors import NearestNeighbors


class Common:
    def __init__(self, train_data, test_data):
        self.train_data = train_data
        self.test_data = test_data

    def k_neighborhood(self, width=100, nk=1):
        train = self.embed(self.train_data, width)
        test = self.embed(self.test_data, width)

        neigh = NearestNeighbors(n_neighbors=nk)
        neigh.fit(train)
        d = neigh.kneighbors(test)[0]

        # 距離をmax1にするデータ整形
        mx = np.max(d)
        d = d / mx

        # プロット
        test_for_plot = self.test_data
        fig = plt.figure(figsize=(30, 10), dpi=50)
        ax1 = fig.add_subplot(111)
        ax2 = ax1.twinx()

        p1, = ax1.plot(d, '-b')
        ax1.set_ylabel('distance')
        # ax1.set_ylim(0, 1.2)
        p2, = ax2.plot(test_for_plot, '-g')
        ax2.set_ylabel('original')
        # ax2.set_ylim(0, 12.0)
        plt.title("Nearest Neighbors")
        ax1.legend([p1, p2], ["distance", "original"])
        # plt.savefig('./results/knn.png')
        plt.show()

    def ssa(self, width=100):
        m = 2
        k = width // 2
        L = k // 2  # lag
        Tt = len(self.test_data)
        score = np.zeros(Tt)

        for t in range(width + k, Tt - L + 1 + 1):
            tstart = t - width - k + 1
            tend = t - 1
            X1 = self.embed(self.test_data[tstart:tend], width).T[::-1, :]  # trajectory matrix
            X2 = self.embed(self.test_data[(tstart + L):(tend + L)], width).T[::-1, :]  # test matrix

            U1, s1, V1 = np.linalg.svd(X1, full_matrices=True)
            U1 = U1[:, 0:m]
            U2, s2, V2 = np.linalg.svd(X2, full_matrices=True)
            U2 = U2[:, 0:m]

            U, s, V = np.linalg.svd(U1.T.dot(U2), full_matrices=True)
            sig1 = s[0]
            score[t] = 1 - np.square(sig1)

        # 変化度をmax1にするデータ整形
        mx = np.max(score)
        score = score / mx

        # プロット
        test_for_plot = self.test_data
        fig = plt.figure(figsize=(30, 10), dpi=50)
        ax1 = fig.add_subplot(111)
        ax2 = ax1.twinx()

        p1, = ax1.plot(score, '-b')
        ax1.set_ylabel('degree of change')
        # ax1.set_ylim(0, 1.2)
        p2, = ax2.plot(test_for_plot, '-g')
        ax2.set_ylabel('original')
        # ax2.set_ylim(0, 12.0)
        plt.title("Singular Spectrum Transformation")
        ax1.legend([p1, p2], ["degree of change", "original"])
        # plt.savefig('./results/sst.png')
        plt.show()

    def embed(self, lst, dim):
        emb = np.empty((0, dim), float)
        for i in range(len(lst) - dim + 1):
            tmp = np.array(lst[i:i + dim])[::-1].reshape((1, -1))
            emb = np.append(emb, tmp, axis=0)
        return emb
