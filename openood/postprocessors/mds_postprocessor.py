from typing import Any
from copy import deepcopy

import numpy as np
import torch
import torch.nn as nn
import sklearn.covariance
from tqdm import tqdm

from .base_postprocessor import BasePostprocessor
from .info import num_classes_dict


class MDSPostprocessor(BasePostprocessor):
    def __init__(self, config):
        self.config = config
        self.num_classes = num_classes_dict[self.config.dataset.name]
        self.setup_flag = False

    def setup(self, net: nn.Module, id_loader_dict, ood_loader_dict):
        if not self.setup_flag:
            # estimate mean and variance from training set
            print('\n Estimating mean and variance from training set...')
            all_feats = []
            all_labels = []
            all_preds = []
            with torch.no_grad():
                for batch in tqdm(id_loader_dict['train'],
                                  desc='Setup: ',
                                  position=0,
                                  leave=True):
                    data, labels = batch['data'].cuda(), batch['label']
                    logits, features = net(data, return_feature=True)
                    all_feats.append(features.cpu())
                    all_labels.append(deepcopy(labels))
                    all_preds.append(logits.argmax(1).cpu())

            all_feats = torch.cat(all_feats)
            all_labels = torch.cat(all_labels)
            all_preds = torch.cat(all_preds)
            # sanity check on train acc
            train_acc = all_preds.eq(all_labels).float().mean()
            print(f' Train acc: {train_acc:.2%}')

            # compute class-conditional statistics
            self.class_mean = []
            centered_data = []
            for c in range(self.num_classes):
                class_samples = all_feats[all_labels.eq(c)].data
                    # ===== 新增检查点 =====

                self.class_mean.append(class_samples.mean(0))
                centered_data.append(class_samples -
                                     self.class_mean[c].view(1, -1))

            self.class_mean = torch.stack(
                self.class_mean)  # shape [#classes, feature dim]

            group_lasso = sklearn.covariance.EmpiricalCovariance(
                assume_centered=False)
            group_lasso.fit(
                torch.cat(centered_data).cpu().numpy().astype(np.float32))
            # inverse of covariance
            # inverse of covariance
            print('\n--- 协方差矩阵检查 ---')
            cov_matrix = group_lasso.covariance_
            print("协方差矩阵行列式:", np.linalg.det(cov_matrix))
            print("协方差矩阵包含NaN:", np.isnan(cov_matrix).any())
            print("协方差矩阵包含inf:", np.isinf(cov_matrix).any())
            
            print('\n--- 精度矩阵检查 ---')
            precision = group_lasso.precision_
            print("精度矩阵最大值:", precision.max())
            print("精度矩阵最小值:", precision.min())
            print("精度矩阵包含NaN:", np.isnan(precision).any())
            print("精度矩阵包含inf:", np.isinf(precision).any())
            
            # 注入正则化（可选）
            # reg_param = 1e-5
            # cov_matrix_reg = cov_matrix + reg_param * np.eye(cov_matrix.shape[0])
            # precision_reg = np.linalg.inv(cov_matrix_reg)
            # print('\n--- 正则化后检查 ---')
            # print("正则化后协方差矩阵行列式:", np.linalg.det(cov_matrix_reg))
            # print("正则化后精度矩阵最大值:", precision_reg.max())
            
            self.precision = torch.from_numpy(group_lasso.precision_).float()  # 使用正则化后的精度矩阵
            
            self.setup_flag = True
        else:
            pass

    @torch.no_grad()
    def postprocess(self, net: nn.Module, data: Any):
        logits, features = net(data, return_feature=True)
        pred = logits.argmax(1)

        class_scores = torch.zeros((logits.shape[0], self.num_classes))
        for c in range(self.num_classes):
            tensor = features.cpu() - self.class_mean[c].view(1, -1)
            
            # ===== 新增检查点 =====
            if torch.isnan(tensor).any() or torch.isinf(tensor).any():
                print(f"\n!!! 类别 {c} 差值异常 !!!")
                print("差值张量统计:", tensor.min().item(), tensor.mean().item(), tensor.max().item())
            
            temp = torch.matmul(tensor, self.precision)
            if torch.isnan(temp).any() or torch.isinf(temp).any():
                print(f"\n!!! 类别 {c} 矩阵乘法异常 !!!")
                print("中间结果统计:", temp.min().item(), temp.mean().item(), temp.max().item())
            
            score = -torch.matmul(temp, tensor.t()).diag()
            if torch.isnan(score).any() or torch.isinf(score).any():
                print(f"\n!!! 类别 {c} 得分异常 !!!")
                print("得分统计:", score.min().item(), score.mean().item(), score.max().item())
            # ======================
            
            class_scores[:, c] = score

        conf = torch.max(class_scores, dim=1)[0]
        return pred, conf
