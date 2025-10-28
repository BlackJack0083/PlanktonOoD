from typing import Any

import numpy as np
import torch
import torch.nn as nn
from numpy.linalg import norm, pinv
from sklearn.covariance import EmpiricalCovariance
from tqdm import tqdm

from .base_postprocessor import BasePostprocessor


class ResidualPostprocessor(BasePostprocessor):
    def __init__(self, config):
        super().__init__(config)
        self.args = self.config.postprocessor.postprocessor_args
        self.dim = self.args.dim
        # self.dim = 256  # 针对 ResNet18

    def setup(self, net: nn.Module, id_loader_dict, ood_loader_dict):
        net.eval()

        with torch.no_grad():
            self.w, self.b = net.get_fc()
            print('Extracting id training feature')
            feature_id_train = []
            for batch in tqdm(id_loader_dict['val'],
                              desc='Eval: ',
                              position=0,
                              leave=True):
                data = batch['data'].cuda()
                data = data.float()
                _, feature = net(data, return_feature=True)
                feature_id_train.append(feature.cpu().numpy())
            feature_id_train = np.concatenate(feature_id_train, axis=0)

            print('Extracting id testing feature')
            feature_id_val = []
            for batch in tqdm(id_loader_dict['test'],
                              desc='Eval: ',
                              position=0,
                              leave=True):
                data = batch['data'].cuda()
                data = data.float()
                _, feature = net(data, return_feature=True)
                feature_id_val.append(feature.cpu().numpy())
            feature_id_val = np.concatenate(feature_id_val, axis=0)

        self.u = -np.matmul(pinv(self.w), self.b)
        ec = EmpiricalCovariance(assume_centered=True)
        ec.fit(feature_id_train - self.u)
        eig_vals, eigen_vectors = np.linalg.eig(ec.covariance_)
        self.NS = np.ascontiguousarray(
            (eigen_vectors.T[np.argsort(eig_vals * -1)[self.dim:]]).T)

        self.score_id = -norm(np.matmul(feature_id_val - self.u, self.NS),
                              axis=-1)
        
        print(f"w shape: {self.w.shape}")
        print(f"b shape: {self.b.shape}")
        print(f"id_train feature shape: {feature_id_train.shape}")

        # 检查 pinv 的结果
        w_pinv = pinv(self.w)
        print(f"pinv(w) shape: {w_pinv.shape}")

        # 检查 u 的值
        self.u = -np.matmul(w_pinv, self.b)
        print(f"u shape: {self.u.shape}")
        print(f"u values: {self.u[:5]}") # 打印前几个值，检查是否有 nan 或 inf

        # 检查协方差矩阵和特征值
        ec = EmpiricalCovariance(assume_centered=True)
        ec.fit(feature_id_train - self.u)
        eig_vals, eigen_vectors = np.linalg.eig(ec.covariance_)
        print(f"eig_vals: {eig_vals[:5]}") # 检查特征值，确保没有非数值

        # 检查 NS 的维度
        sorted_indices = np.argsort(eig_vals * -1)
        if self.dim >= len(eig_vals):
            print(f"Error: self.dim ({self.dim}) is too large for eigen values ({len(eig_vals)})")
            # 你的程序很可能在这里崩溃

        self.NS = np.ascontiguousarray((eigen_vectors.T[sorted_indices[self.dim:]]).T)
        print(f"NS shape: {self.NS.shape}")

    @torch.no_grad()
    def postprocess(self, net: nn.Module, data: Any):
        _, feature_ood = net(data, return_feature=True)
        logit_ood = feature_ood.cpu() @ self.w.T + self.b
        _, pred = torch.max(logit_ood, dim=1)
        score_ood = -norm(np.matmul(feature_ood.cpu() - self.u, self.NS),
                          axis=-1)
        return pred, torch.from_numpy(score_ood)
