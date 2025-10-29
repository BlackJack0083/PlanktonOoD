from typing import Callable, List, Type

import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
import csv

from openood.evaluators.metrics import compute_all_metrics
from openood.postprocessors import BasePostprocessor
from openood.networks.ash_net import ASHNet
from openood.networks.react_net import ReactNet
from openood.networks.scale_net import ScaleNet

from .datasets import DATA_INFO, data_setup, get_id_ood_dataloader
from .postprocessor import get_postprocessor
from .preprocessor import get_default_preprocessor


class Evaluator:
    def __init__(
        self,
        net: nn.Module,
        id_name: str,  # ID数据集的名称
        data_root: str = './data',  # 数据根目录
        config_root: str = './configs',  # 配置根目录
        preprocessor: Callable = None,  # 预处理函数
        postprocessor_name: str = None, #  后处理器名称
        postprocessor: Type[BasePostprocessor] = None,  # 后处理器类
        batch_size: int = 200,
        shuffle: bool = False,
        num_workers: int = 4,
    ) -> None:
        """A unified, easy-to-use API for evaluating (most) discriminative OOD
        detection methods.

        Args:
            net (nn.Module):
                The base classifier.
            id_name (str):
                The name of the in-distribution dataset.
            data_root (str, optional):
                The path of the data folder. Defaults to './data'.
            config_root (str, optional):
                The path of the config folder. Defaults to './configs'.
            preprocessor (Callable, optional):
                The preprocessor of input images.
                Passing None will use the default preprocessor
                following convention. Defaults to None.
            postprocessor_name (str, optional):
                The name of the postprocessor that obtains OOD score.
                Ignored if an actual postprocessor is passed.
                Defaults to None.
            postprocessor (Type[BasePostprocessor], optional):
                An actual postprocessor instance which inherits
                OpenOOD's BasePostprocessor. Defaults to None.
            batch_size (int, optional):
                The batch size of samples. Defaults to 200.
            shuffle (bool, optional):
                Whether shuffling samples. Defaults to False.
            num_workers (int, optional):
                The num_workers argument that will be passed to
                data loaders. Defaults to 4.

        Raises:
            ValueError:
                If both postprocessor_name and postprocessor are None.
            ValueError:
                If the specified ID dataset {id_name} is not supported.
            TypeError:
                If the passed postprocessor does not inherit BasePostprocessor.
        """
        # check the arguments
        # 检查参数
        if postprocessor_name is None and postprocessor is None:
            raise ValueError('Please pass postprocessor_name or postprocessor')
        
        # 检查是否同时传递了postprocessor_name和postprocessor
        if postprocessor_name is not None and postprocessor is not None:
            print(
                'Postprocessor_name is ignored because postprocessor is passed'
            )

        # 检查id_name是否在支持的数据集列表中
        if id_name not in DATA_INFO:
            raise ValueError(f'Dataset [{id_name}] is not supported')

        # get data preprocessor
        if preprocessor is None:
            preprocessor = get_default_preprocessor(id_name)

        print("Using preprocessor:", preprocessor)

        # set up config root
        if config_root is None:
            filepath = os.path.dirname(os.path.abspath(__file__))
            config_root = os.path.join(*filepath.split('/')[:-2], 'configs')

        # get postprocessor
        if postprocessor is None:
            postprocessor = get_postprocessor(config_root, postprocessor_name,
                                              id_name)
        if not isinstance(postprocessor, BasePostprocessor):
            raise TypeError(
                'postprocessor should inherit BasePostprocessor in OpenOOD')

        # load data
        # 加载数据集
        data_setup(data_root, id_name)
        loader_kwargs = {
            'batch_size': batch_size,
            'shuffle': shuffle,
            'num_workers': num_workers
        }
        # 获取ID和OOD数据集的数据加载器
        dataloader_dict = get_id_ood_dataloader(id_name, data_root,
                                                preprocessor, **loader_kwargs)

        # wrap base model to work with certain postprocessors
        # 包装基本模型以适应某些后处理器
        if postprocessor_name == 'react':
            net = ReactNet(net)
        elif postprocessor_name == 'ash':
            net = ASHNet(net)
        elif postprocessor_name == 'scale':
            net = ScaleNet(net)

        # postprocessor setup
        postprocessor.setup(net, dataloader_dict['id'], dataloader_dict['ood'])
        # 加了这一个 
        self.postprocessor_name = postprocessor_name

        self.id_name = id_name
        self.net = net
        self.preprocessor = preprocessor
        self.postprocessor = postprocessor
        self.dataloader_dict = dataloader_dict
        self.metrics = {
            'id_acc': None,
            'csid_acc': None,
            'ood': None,
            'fsood': None
        }
        self.scores = {
            'id': {
                'train': None,
                'val': None,
                'test': None
            },
            'csid': {k: None
                     for k in dataloader_dict['csid'].keys()},
            'ood': {
                'val': None,
                'near':
                {k: None
                 for k in dataloader_dict['ood']['near'].keys()},
                'far': {k: None
                        for k in dataloader_dict['ood']['far'].keys()},
            },
            'id_preds': None,
            'id_labels': None,
            'csid_preds': {k: None
                           for k in dataloader_dict['csid'].keys()},
            'csid_labels': {k: None
                            for k in dataloader_dict['csid'].keys()},
        }
        # perform hyperparameter search if have not done so
        if (self.postprocessor.APS_mode
                and not self.postprocessor.hyperparam_search_done):
            self.hyperparam_search()

        self.net.eval()

        # how to ensure the postprocessors can work with
        # models whose definition doesn't align with OpenOOD

    def _classifier_inference(self,
                              data_loader: DataLoader,
                              msg: str = 'Acc Eval',
                              progress: bool = True):
        self.net.eval()

        all_preds = []
        all_labels = []
        with torch.no_grad():
            for batch in tqdm(data_loader, desc=msg, disable=not progress):
                data = batch['data'].cuda()
                #print("[DEBUG]data:", data)
                logits = self.net(data)
                preds = logits.argmax(1)
                all_preds.append(preds.cpu())
                all_labels.append(batch['label'])

        all_preds = torch.cat(all_preds)
        all_labels = torch.cat(all_labels)
        return all_preds, all_labels

    def eval_acc(self, data_name: str = 'id') -> float:
        if data_name == 'id':
            if self.metrics['id_acc'] is not None:
                return self.metrics['id_acc']
            else:
                if self.scores['id_preds'] is None:
                    all_preds, all_labels = self._classifier_inference(
                        self.dataloader_dict['id']['test'], 'ID Acc Eval')
                    self.scores['id_preds'] = all_preds
                    self.scores['id_labels'] = all_labels
                else:
                    all_preds = self.scores['id_preds']
                    all_labels = self.scores['id_labels']

                assert len(all_preds) == len(all_labels)
                correct = (all_preds == all_labels).sum().item()
                acc = correct / len(all_labels) * 100
                self.metrics['id_acc'] = acc
                return acc
        elif data_name == 'csid':
            if self.metrics['csid_acc'] is not None:
                return self.metrics['csid_acc']
            else:
                correct, total = 0, 0
                for _, (dataname, dataloader) in enumerate(
                        self.dataloader_dict['csid'].items()):
                    if self.scores['csid_preds'][dataname] is None:
                        all_preds, all_labels = self._classifier_inference(
                            dataloader, f'CSID {dataname} Acc Eval')
                        self.scores['csid_preds'][dataname] = all_preds
                        self.scores['csid_labels'][dataname] = all_labels
                    else:
                        all_preds = self.scores['csid_preds'][dataname]
                        all_labels = self.scores['csid_labels'][dataname]

                    assert len(all_preds) == len(all_labels)
                    c = (all_preds == all_labels).sum().item()
                    t = len(all_labels)
                    correct += c
                    total += t

                if self.scores['id_preds'] is None:
                    all_preds, all_labels = self._classifier_inference(
                        self.dataloader_dict['id']['test'], 'ID Acc Eval')
                    self.scores['id_preds'] = all_preds
                    self.scores['id_labels'] = all_labels
                else:
                    all_preds = self.scores['id_preds']
                    all_labels = self.scores['id_labels']

                correct += (all_preds == all_labels).sum().item()
                total += len(all_labels)

                acc = correct / total * 100
                self.metrics['csid_acc'] = acc
                return acc
        else:
            raise ValueError(f'Unknown data name {data_name}')

    def eval_ood(self, fsood: bool = False, progress: bool = True):
        # fsood: 是否使用 FSOOD（Few-Shot OOD）评估。若为 True，ID 数据将包括 CSID 子集
        # 如果是普通 OOD 任务，就使用 ID 数据和 'ood' 任务名。
        # 如果是 few-shot OOD 任务，就使用 CSID 数据和 'fsood' 任务名。
        id_name = 'id' if not fsood else 'csid'
        task = 'ood' if not fsood else 'fsood'
        # 如果已经计算过该任务的指标，就直接返回。
        if self.metrics[task] is None:
            self.net.eval()

            # id score
            if self.scores['id']['test'] is None:  # 如果还没有计算 ID 类别的分数，就开始对 ID 数据集进行推断。推断后保存预测结果、置信度和真实标签：
                print(f'Performing inference on {self.id_name} test set...',
                      flush=True)
                # 获得id数据的预测信息
                id_pred, id_conf, id_gt = self.postprocessor.inference(self.net, self.dataloader_dict['id']['test'], progress)
                # print('score_id:', id_conf[:1])
                # print('id_conf_type:',id_conf.dtype)
                # print('id_conf_shape:', id_conf.shape)

                self.scores['id']['test'] = [id_pred, id_conf, id_gt]
            else:
                id_pred, id_conf, id_gt = self.scores['id']['test']

            # 如果是 FSOOD，则拼接上 CSID 数据
            if fsood:
                csid_pred, csid_conf, csid_gt = [], [], []
                for i, dataset_name in enumerate(self.scores['csid'].keys()):
                    if self.scores['csid'][dataset_name] is None:
                        print(
                            f'Performing inference on {self.id_name} '
                            f'(cs) test set [{i+1}]: {dataset_name}...',
                            flush=True)
                        temp_pred, temp_conf, temp_gt = \
                            self.postprocessor.inference(
                                self.net,
                                self.dataloader_dict['csid'][dataset_name],
                                progress)
                        self.scores['csid'][dataset_name] = [
                            temp_pred, temp_conf, temp_gt
                        ]

                    csid_pred.append(self.scores['csid'][dataset_name][0])
                    csid_conf.append(self.scores['csid'][dataset_name][1])
                    csid_gt.append(self.scores['csid'][dataset_name][2])

                csid_pred = np.concatenate(csid_pred)
                csid_conf = np.concatenate(csid_conf)
                csid_gt = np.concatenate(csid_gt)

                id_pred = np.concatenate((id_pred, csid_pred))
                id_conf = np.concatenate((id_conf, csid_conf))
                id_gt = np.concatenate((id_gt, csid_gt))

            # load nearood data and compute ood metrics
            # 评估nearood数据和计算ood指标
            near_metrics = self._eval_ood([id_pred, id_conf, id_gt],
                                          ood_split='near',
                                          progress=progress)  # 分别计算near和far的指标
            # load farood data and compute ood metrics
            far_metrics = self._eval_ood([id_pred, id_conf, id_gt],
                                         ood_split='far',
                                         progress=progress)
            # 计算准确率
            if self.metrics[f'{id_name}_acc'] is None:
                self.eval_acc(id_name)  # 计算ID的准确率

            near_metrics[:, -1] = np.array([self.metrics[f'{id_name}_acc']] * len(near_metrics))  # 将ID的准确率添加到near_metrics的最后一列
            far_metrics[:, -1] = np.array([self.metrics[f'{id_name}_acc']] * len(far_metrics))

            self.metrics[task] = pd.DataFrame(
                np.concatenate([near_metrics, far_metrics], axis=0),
                index=list(self.dataloader_dict['ood']['near'].keys()) +
                ['nearood'] + list(self.dataloader_dict['ood']['far'].keys()) +
                ['farood'],
                columns=[
                    'FPR@99', 'FPR@95', 'AUROC', 'AUPR_IN', 'AUPR_OUT',  'ACC'
                ],
            )  # 将near和far的指标合并，并创建一个DataFrame
        else:
            print('Evaluation has already been done!')

        with pd.option_context(
                'display.max_rows', None, 'display.max_columns', None,
                'display.float_format',
                '{:,.2f}'.format):  # more options can be specified also
            print(self.metrics[task])

        return self.metrics[task]

    def _eval_ood(self,
                  id_list: List[np.ndarray],
                  ood_split: str = 'near',
                  progress: bool = True):
        # 用于评估一组 OOD 数据集（near 或 far）在 OOD 检测任务上的性能。
        
        print(f'Processing {ood_split} ood...', flush=True)
        
        [id_pred, id_conf, id_gt] = id_list
        
        metrics_list = []
        
        for dataset_name, ood_dl in self.dataloader_dict['ood'][ood_split].items():
            # 如果还没对该 OOD 子数据集推理，则推理并保存结果
            if self.scores['ood'][ood_split][dataset_name] is None:
                print(f'Performing inference on {dataset_name} dataset...',
                      flush=True)
                # ood_pred是预测标签标签，ood_conf是置信度，ood_gt是真实标签
                ood_pred, ood_conf, ood_gt = self.postprocessor.inference(
                    self.net, ood_dl, progress)
                self.scores['ood'][ood_split][dataset_name] = [
                    ood_pred, ood_conf, ood_gt
                ]
                # print('score_ood:', ood_conf[:1])
                # print('score_ood_dtype:', ood_conf.dtype)
                # print('score_ood_shape:', ood_conf.shape)
            else:
                print(
                    'Inference has been performed on '
                    f'{dataset_name} dataset...',
                    flush=True)
                [ood_pred, ood_conf, ood_gt] = self.scores['ood'][ood_split][dataset_name]

            # 合并 ID 和 OOD 数据
            # 将ID和OOD数据合并，将ID的真实标签设为0，OOD的真实标签设为-1
            ood_gt = -1 * np.ones_like(ood_gt)  # hard set to -1 as ood
            pred = np.concatenate([id_pred, ood_pred])
            conf = np.concatenate([id_conf, ood_conf])
            label = np.concatenate([id_gt, ood_gt])
            
            # print('id_conf(前10个值):', id_conf[:10])
            # print('ood_conf(前10个值):', ood_conf[:10])
            # print('ind_conf_shape', id_conf.shape)
            # print('ood_conf_shape:', ood_conf.shape)

            # # 确保outputs文件夹存在
            # os.makedirs('outputs', exist_ok=True)
            
            # # 保存数组
            # np.save('outputs/ind_conf', id_conf)
            # np.save('outputs/ood_conf', ood_conf)
            # # print("已保存 id_conf 和 ood_conf 到 outputs 文件夹")

            # np.save('outputs/ind_pred', id_pred)
            # np.save('outputs/ood_pred', ood_pred)

            # np.save('outputs/ind_gt', id_gt)
            # np.save('outputs/ood_gt', ood_gt)

            # print(f'Computing metrics on {dataset_name} dataset...')


#debug__________________________________________________________________________________________________________
            from sklearn.metrics import roc_curve

            # —— Debug 插入开始 —— 
            # 构造 labels：0 表示 ID，1 表示 OOD
            dbg_labels = np.concatenate([np.zeros_like(id_conf), np.ones_like(ood_conf)])
            # dbg_scores 取负，使得“越大越倾向 OOD”
            dbg_scores = np.concatenate([-id_conf, -ood_conf])

            # 1. 计算 ROC 曲线（fpr: P(将 ID 误判为 OOD)，tpr: P(将 OOD 正确判为 OOD)）
            fpr, tpr, thresholds = roc_curve(dbg_labels, dbg_scores)

            # 2. 找到最接近 FPR=5%（即 ID 正确率 TNR=95%）的那个点
            target_fpr = 0.05
            idx = np.argmin(np.abs(fpr - target_fpr))

            # 3. 读出对应的 TPR（也就是 OOD 检测正确率）
            tpr_at_fpr5 = tpr[idx]
            print(f'[DEBUG] {dataset_name}  OOD_TPR@ID_FPR5 (ID TNR=95%) = {tpr_at_fpr5:.2%}')
            # —— Debug 插入结束 —— 

            from sklearn.metrics import roc_curve

            # —— Debug 插入：反转正负类 —— 
            # 现在把 ID 当正类（label=1），OOD 当负类（label=0）
            dbg_labels = np.concatenate([np.ones_like(id_conf), np.zeros_like(ood_conf)])
            # dbg_scores 保持不取负：分数越大越倾向“预测为 ID”
            dbg_scores = np.concatenate([id_conf, ood_conf])

            # 1. 计算 ROC 曲线
            #    fpr = P(将负类（OOD）误判为正类（ID])) 
            #    tpr = P(将正类（ID）正确判为正类（ID]))
            fpr, tpr, thresholds = roc_curve(dbg_labels, dbg_scores)

            # 2. 找到最接近 FPR=5%（即 OOD 错报率 5%）的那个点
            target_fpr = 0.05
            idx = np.argmin(np.abs(fpr - target_fpr))

            # 3. 读出对应的 TPR（也就是在 OOD 错报率 5% 时，ID 的召回率）
            tpr_at_fpr5 = tpr[idx]
            print(f'[DEBUG] {dataset_name}  ID_TPR@OOD_FPR5 = {tpr_at_fpr5:.2%}')
# —— Debug 结束 —— 


#debug——————————————————————————————————————————————————————————————————————————————————————
            
            # 计算指标
            ood_metrics = compute_all_metrics(conf, label, pred)
            metrics_list.append(ood_metrics)
            self._print_metrics(ood_metrics)

        print('Computing mean metrics...', flush=True)
        metrics_list = np.array(metrics_list)
        metrics_mean = np.mean(metrics_list, axis=0, keepdims=True)
        self._print_metrics(list(metrics_mean[0]))
        return np.concatenate([metrics_list, metrics_mean], axis=0) * 100

    def _print_metrics(self, metrics):
        [fpr99, fpr, auroc, aupr_in, aupr_out, _] = metrics

        # print ood metric results
        print('FPR@99:{:.2f}, FPR@95: {:.2f}, AUROC: {:.2f}'.format(100 * fpr99, 100 * fpr, 100 * auroc),
              end=' ',
              flush=True)
        print('AUPR_IN: {:.2f}, AUPR_OUT: {:.2f}'.format(
            100 * aupr_in, 100 * aupr_out),
              flush=True)
        print(u'\u2500' * 70, flush=True)
        print('', flush=True)

    def hyperparam_search(self):
        print('Starting automatic parameter search...')
        max_auroc = 0
        hyperparam_names = []
        hyperparam_list = []
        count = 0

        for name in self.postprocessor.args_dict.keys():
            hyperparam_names.append(name)
            count += 1

        for name in hyperparam_names:
            hyperparam_list.append(self.postprocessor.args_dict[name])

        hyperparam_combination = self.recursive_generator(
            hyperparam_list, count)

        final_index = None

        results = []

        for i, hyperparam in enumerate(hyperparam_combination):
            self.postprocessor.set_hyperparam(hyperparam)

            id_pred, id_conf, id_gt = self.postprocessor.inference(
                self.net, self.dataloader_dict['id']['val'])
            ood_pred, ood_conf, ood_gt = self.postprocessor.inference(
                self.net, self.dataloader_dict['ood']['val'])

            ood_gt = -1 * np.ones_like(ood_gt)  # hard set to -1 as ood
            pred = np.concatenate([id_pred, ood_pred])
            conf = np.concatenate([id_conf, ood_conf])
            label = np.concatenate([id_gt, ood_gt])
            ood_metrics = compute_all_metrics(conf, label, pred)
            auroc = ood_metrics[2]

            print('Hyperparam: {}, auroc: {}'.format(hyperparam, auroc))
            print('Hyperparam: {}, fpr95: {}'.format(hyperparam, ood_metrics[1]))
            print('Hyperparam: {}, fpr99: {}'.format(hyperparam, ood_metrics[0]))
            results.append(dict(dim=hyperparam, auroc=auroc, fpr95 = ood_metrics[1], fpr99 = ood_metrics[0]))
            if auroc > max_auroc:
                final_index = i
                max_auroc = auroc

        self.postprocessor.set_hyperparam(hyperparam_combination[final_index])
        print('Final hyperparam: {}'.format(
           self.postprocessor.get_hyperparam()))
        self.postprocessor.hyperparam_search_done = True
        
        # df_all = pd.DataFrame(results)
        # df_all.to_csv('./results/vim_dim_sweep_results.csv', index=False)
        # print('所有结果已保存至 ./results/vim_dim_sweep_results.csv')


    def recursive_generator(self, list, n):
        if n == 1:
            results = []
            for x in list[0]:
                k = []
                k.append(x)
                results.append(k)
            return results
        else:
            results = []
            temp = self.recursive_generator(list, n - 1)
            for x in list[n - 1]:
                for y in temp:
                    k = y.copy()
                    k.append(x)
                    results.append(k)
            return results
