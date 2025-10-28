from typing import Any
import os
import pickle
import numpy as np
import torch
import torch.nn as nn
from numpy.linalg import norm, pinv
from scipy.special import logsumexp
from sklearn.covariance import EmpiricalCovariance
from tqdm import tqdm

from .base_postprocessor import BasePostprocessor

def mkdir_or_exist(dir_path):
    if not os.path.exists(dir_path):
        os.makedirs(dir_path, exist_ok=True)

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
                
                # 保存 w 和 b 到 pkl 文件
                mkdir_or_exist('outputs')  # 确保目录存在
                
                fc_path = os.path.join('outputs', 'fc.pkl')
                with open(fc_path, 'wb') as f:
                    pickle.dump({'w': self.w, 'b': self.b}, f)
                print(f"FC层参数已保存到: {fc_path}")
                
                
                print('Extracting id training feature')
                feature_id_train = []
                for batch in tqdm(id_loader_dict['train'],
                                  desc='Setup: ',
                                  position=0,
                                  leave=True):
                    data = batch['data'].cuda()
                    data = data.float()
                    
                    # print("dataset[0]:", data[0])  # 更推荐的方式：用逗号分隔，自动转换类型                    
                    # print("输入数据 shape:", data.shape)
                    # print(type(data))
                    
                    # 将数据保存为 .npz 格式
                    # data_file = os.path.join('outputs', 'input_data.npz')  # 输出文件路径

                    # # 将 data 转换为 numpy 数组并保存为 .npz 文件
                    # np.savez(data_file, data=data.cpu().numpy())  # 将张量移到 CPU 并转换为 numpy 数组
                    # print(f"输入数据已保存到: {data_file}")
                    
                    logits, feature = net(data, return_feature=True)

                    # print("print(net):", net)
                    
                    # print("net.forward(data):", feature)
                    # print("第一个样本特征 shape:", feature.shape)
                    # print("特征向量（前10个值）:", feature[:10])

                    # 将特征向量保存为 .npz 格式
                    # mkdir_or_exist('outputs')  # 确保输出文件夹存在
                    # output_file = os.path.join('outputs', 'feature_vector.npz')  # 输出文件路径
                    # np.savez(output_file, feature=feature.cpu().numpy())  # 将张量移到 CPU 并转换为 numpy 数组
                    # # 保存特征向量为 .npz 格式
                    # print(f"特征向量已保存到: {output_file}")
                    
                    feature_id_train.append(feature.cpu().numpy())
                
                feature_id_train = np.concatenate(feature_id_train, axis=0)
                print("第一个样本特征 shape:", feature_id_train.shape)
                logit_id_train = feature_id_train @ self.w.T + self.b
                print("logit_id_train:", logit_id_train[:1])

            self.u = -np.matmul(pinv(self.w), self.b)
            ec = EmpiricalCovariance(assume_centered=True)
            ec.fit(feature_id_train - self.u)
            eig_vals, eigen_vectors = np.linalg.eig(ec.covariance_)
            self.NS = np.ascontiguousarray(
                (eigen_vectors.T[np.argsort(eig_vals * -1)[self.dim:]]).T)

            vlogit_id_train = norm(np.matmul(feature_id_train - self.u,
                                             self.NS),
                                   axis=-1)
            self.alpha = logit_id_train.max(
                axis=-1).mean() / vlogit_id_train.mean()
            print("alpha = ")
            print(f'{self.alpha=:.4f}')

            self.setup_flag = True
        else:
            pass

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

    def get_hyperparam(self):
        return self.dim