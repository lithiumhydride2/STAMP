import numpy as np
import math
from utils.tsp_controller import TSPSolver


class VTSPGaussian:
    """
    生成多个高斯函数的分布,并沿随机生成的旅行商问题(TSP)的路径移动它们
    """

    def __init__(self, n_targets=2):
        """
        生成 n_target 个高斯函数
        """
        self.n_targets = n_targets
        self.n_tsp_nodes = 50
        self.tsp_coord = self.get_tsp_nodes()  # in shape(n_targets,n_tsp_nodes,2)
        self.tsp_idx = [
            0
        ] * self.n_targets  # 存储高斯函数当前所在 TSP 节点的索引，初始为 0
        self.mean = self.tsp_coord[:, 0, :]  # 初始值为 每个 TSP 路径第一个坐标点
        self.sigma = np.array([0.1] * self.n_targets)
        self.max_value = 1 / (2 * np.pi * self.sigma**2)
        self.trajectories = [
            self.mean.copy()
        ]  # 存储了每个高斯函数的移动轨迹，初始值为 [self.mean.copy()]

    def get_tsp_nodes(self):
        """随机初始化一些点，并用 TSP 求解器求解其连接顺序"""
        tsp_solver = TSPSolver()
        coord = np.random.rand(self.n_targets, self.n_tsp_nodes, 2)
        for i in range(self.n_targets):
            index = tsp_solver.run_solver(coord[i])
            coord[i] = coord[i][index]
        return coord

    def step(self, steplen):
        """
        根据 steplen 在 tsp 的节点之前移动
        """
        for i in range(self.n_targets):
            d = np.linalg.norm(self.tsp_coord[i, self.tsp_idx[i] + 1, :] - self.mean[i])
            next_len = steplen
            if d > next_len:
                pt = (
                    self.tsp_coord[i, self.tsp_idx[i] + 1, :] - self.mean[i]
                ) * next_len / d + self.mean[i]
            else:
                while True:
                    self.tsp_idx[i] += 1
                    next_len -= d
                    d = np.linalg.norm(
                        self.tsp_coord[i, self.tsp_idx[i] + 1, :]
                        - self.tsp_coord[i, self.tsp_idx[i], :]
                    )
                    if d > next_len:
                        pt = (
                            self.tsp_coord[i, self.tsp_idx[i] + 1, :]
                            - self.tsp_coord[i, self.tsp_idx[i], :]
                        ) * next_len / d + self.mean[i]
                        break
            self.mean[i] = pt
        self.trajectories += [self.mean.copy()]
        return self.mean

    def fn(self, X):
        """
        # description : 计算一个二维数组 X 中每个点的函数值 y
         ---------------
        # param :
         - X : in shape (grid_size^2,2)
         ---------------
        # returns :
        - y : 表示每个坐标对应的每个高斯函数的函数值
         ---------------
        """
        y = np.zeros((X.shape[0], self.n_targets))
        row_mat, col_mat = X[:, 0], X[:, 1]
        for target_id in range(self.n_targets):
            gaussian_mean = self.mean[target_id]
            sigma_x1 = sigma_x2 = self.sigma[target_id]
            covariance = 0
            r = covariance / (sigma_x1 * sigma_x2)
            coefficients = 1 / (
                2 * math.pi * sigma_x1 * sigma_x2 * np.sqrt(1 - math.pow(r, 2))
            )
            p1 = -1 / (2 * (1 - math.pow(r, 2)))
            px = np.power((row_mat - gaussian_mean[0]) / sigma_x1, 2)
            py = np.power((col_mat - gaussian_mean[1]) / sigma_x2, 2)
            pxy = (
                2
                * r
                * (row_mat - gaussian_mean[0])
                * (col_mat - gaussian_mean[1])
                / (sigma_x1 * sigma_x2)
            )
            distribution_matrix = coefficients * np.exp(p1 * (px - pxy + py))
            y[:, target_id] += distribution_matrix
        y /= self.max_value
        return y


if __name__ == "__main__":
    pass
