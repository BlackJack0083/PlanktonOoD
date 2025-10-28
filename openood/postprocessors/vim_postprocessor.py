# from typing import Any

# import numpy as np
# import torch
# import torch.nn as nn
# from numpy.linalg import norm, pinv
# from scipy.special import logsumexp
# from sklearn.covariance import EmpiricalCovariance
# from tqdm import tqdm

# from .base_postprocessor import BasePostprocessor


# class VIMPostprocessor(BasePostprocessor):
#     def __init__(self, config):
#         super().__init__(config)
#         self.args = self.config.postprocessor.postprocessor_args
#         self.args_dict = self.config.postprocessor.postprocessor_sweep
#         self.dim = self.args.dim
#         self.setup_flag = False

#     def setup(self, net: nn.Module, id_loader_dict, ood_loader_dict):
#         # print(self.config)

#         if not self.setup_flag:
#             net.eval()


#             # 收集所有批次的特征，最后用np.concatenate拼一个形状为(N, D)大批量，其中N是样本总数，D是特征维度。
#             with torch.no_grad():
#                 self.w, self.b = net.get_fc()
#                 print('Extracting id training feature')
#                 feature_id_train = []
#                 for batch in tqdm(id_loader_dict['train'],
#                                   desc='Setup: ',
#                                   position=0,
#                                   leave=True):
#                     data = batch['data'].cuda()
#                     data = data.float()
#                     _, feature = net(data, return_feature=True)
#                     feature_id_train.append(feature.cpu().numpy())
#                 feature_id_train = np.concatenate(feature_id_train, axis=0)
#                 logit_id_train = feature_id_train @ self.w.T + self.b


#             # 计算特征空间的“中心”向量， 通过 w u + b = 0，求解 u = -w⁺ b，这里 pinv(self.w) 是 w 的 Moore-Penrose 伪逆。
#             self.u = -np.matmul(pinv(self.w), self.b)

#             # 估计 ID 特征的协方差矩阵，并求其主成分
#             ec = EmpiricalCovariance(assume_centered=True)
#             ec.fit(feature_id_train - self.u)
#             eig_vals, eigen_vectors = np.linalg.eig(ec.covariance_)
#             #print("特征值：", sorted(eig_vals, reverse=True))
#             #print("维数：", len(eig_vals))

#             # 构造零空间，对特征值按从大到小排序，取前 self.dim 个方向为主成分，剩下的方向即零空间基向量。
#             print("vim维数dim：", self.dim)
#             self.NS = np.ascontiguousarray(
#                 (eigen_vectors.T[np.argsort(eig_vals * -1)[self.dim:]]).T)

#             # 计算在零空间上的投影模长，该范数越大，说明样本更偏离主成分空间
#             vlogit_id_train = norm(np.matmul(feature_id_train - self.u,
#                                              self.NS),
#                                    axis=-1)
            
#             # 计算比例系数 α：每个样本在所有类上的最高打分（最自信的那个类）的平均值 / 在零空间上的平均投影模长
#             self.alpha = logit_id_train.max(
#                 axis=-1).mean() / vlogit_id_train.mean()
#             print(f'{self.alpha=:.4f}')

#             #不能阻止之后的setup!超参数调要用
#             self.setup_flag = True
#         else:
#             pass

#     @torch.no_grad()
#     def postprocess(self, net: nn.Module, data: Any):
#         _, feature_ood = net.forward(data, return_feature=True)
#         feature_ood = feature_ood.cpu()
#         # print('feature_ood: ', feature_ood[:1])
#         logit_ood = feature_ood @ self.w.T + self.b
#         # print('logit_ood:', logit_ood[:1])

#         # 找出每个样本最大 logit 值对应的类别，即模型的预测类别。
#         _, pred = torch.max(logit_ood, dim=1)

#         # 计算 能量分数（energy），对OOD数据，模型通常会输出一个 logit 特别大的类别，其他很小，所以求和式主要贡献来自一个大值，energy 比较低。
#         energy_ood = logsumexp(logit_ood.numpy(), axis=-1)
        
#         vlogit_ood = norm(np.matmul(feature_ood.numpy() - self.u, self.NS),
#                           axis=-1) * self.alpha
#         score_ood = -vlogit_ood + energy_ood

#         # print('score_ood:', score_ood[:1])

#         return pred, torch.from_numpy(score_ood)

#     def set_hyperparam(self, hyperparam: list):
#         self.dim = hyperparam[0]
#         #加这个让每次调超参数的时候都能重新setup
#         self.setup_flag = False

#     def get_hyperparam(self):
#         return self.dim


from typing import Any

import numpy as np
import torch
import torch.nn as nn
from numpy.linalg import norm, pinv
from scipy.special import logsumexp
from sklearn.covariance import EmpiricalCovariance
from tqdm import tqdm

from .base_postprocessor import BasePostprocessor


class VIMPostprocessor(BasePostprocessor):
    def __init__(self, config):
        super().__init__(config)
        self.args = self.config.postprocessor.postprocessor_args
        self.args_dict = self.config.postprocessor.postprocessor_sweep
        self.dim = self.args.dim
        self.setup_flag = False

    def setup(self, net: nn.Module, id_loader_dict, ood_loader_dict):
        if not self.setup_flag:
            net.eval()

            with torch.no_grad():
                self.w, self.b = net.get_fc()
                print('Extracting id training feature')
                feature_id_train = []
                for batch in tqdm(id_loader_dict['train'],
                                  desc='Setup: ',
                                  position=0,
                                  leave=True):
                    data = batch['data'].cuda()
                    data = data.float()
                    _, feature = net(data, return_feature=True)
                    feature_id_train.append(feature.cpu().numpy())
                feature_id_train = np.concatenate(feature_id_train, axis=0)
                self.logit_id_train = feature_id_train @ self.w.T + self.b

            self.u = -np.matmul(pinv(self.w), self.b)
            ec = EmpiricalCovariance(assume_centered=True)
            ec.fit(feature_id_train - self.u)
            eig_vals, eigen_vectors = np.linalg.eig(ec.covariance_)
            
            self.feature_id_train = feature_id_train
            self.eigen_vectors = eigen_vectors
            self.eig_vals = eig_vals
            
            self._compute_NS_and_alpha()

            self.setup_flag = True
        else:
            pass
        
    def _compute_NS_and_alpha(self):
        """根据当前的dim计算NS和alpha"""
        self.NS = np.ascontiguousarray(
            (self.eigen_vectors.T[np.argsort(self.eig_vals * -1)[self.dim:]]).T)

        vlogit_id_train = norm(np.matmul(self.feature_id_train - self.u,
                                         self.NS),
                               axis=-1)
        self.alpha = self.logit_id_train.max(
            axis=-1).mean() / vlogit_id_train.mean()
        print(f'dim={self.dim}, {self.alpha=:.4f}')

    @torch.no_grad()
    def postprocess(self, net: nn.Module, data: Any):
        _, feature_ood = net.forward(data, return_feature=True)
        feature_ood = feature_ood.cpu()
        logit_ood = feature_ood @ self.w.T + self.b
        _, pred = torch.max(logit_ood, dim=1)
        energy_ood = logsumexp(logit_ood.numpy(), axis=-1)
        vlogit_ood = norm(np.matmul(feature_ood.numpy() - self.u, self.NS),
                          axis=-1) * self.alpha
        score_ood = -vlogit_ood + energy_ood
        return pred, torch.from_numpy(score_ood)

    def set_hyperparam(self, hyperparam: list):
        self.dim = hyperparam[0]
        if self.setup_flag:  
            self._compute_NS_and_alpha()

    def get_hyperparam(self):
        return self.dim
