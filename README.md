# Benchmarking Out-of-Distribution Detection for Plankton Recognition

![License](https://img.shields.io/badge/license-MIT-blue.svg)  
![Python](https://img.shields.io/badge/python-3.8%2B-blue)  

This repository contains the official implementation of the paper:

> **Benchmarking Out-of-Distribution Detection for Plankton Recognition: A Systematic Evaluation of Advanced Methods in Marine Ecological Monitoring**  
> Yingzi Han*, Jiakai He*, Chuanlong Xie+, Jianping Li  
> *Beijing Normal University, Shenzhen Institutes of Advanced Technology, Chinese Academy of Sciences*  
> Published in ICCV Workshop 2025  
> [Paper](https://openaccess.thecvf.com/content/ICCV2025W/CVAUI%20&%20AAMVEM/papers/Han_Benchmarking_Out-of-Distribution_Detection_for_Plankton_Recognition_A_Systematic_Evaluation_of_ICCVW_2025_paper.pdf) | [Project Page](https://github.com/BlackJack0083/PlanktonOoD)

---

## 📌 Overview

Automated plankton recognition models face significant challenges in real-world deployment due to **distribution shifts (Out-of-Distribution, OoD)** between training and test data. This work presents the **first large-scale systematic evaluation** of OoD detection methods in plankton recognition, establishing a comprehensive benchmark based on the **DYB-PlanktonNet** dataset.

We evaluate **22 post-hoc OoD detection methods** across **9 network architectures**, covering **distance-based**, **classification-based**, and **density-based** approaches, under both **Near-OoD** and **Far-OoD** scenarios.

---

## 🚀 Features

- ✅ **22 OoD detection methods** systematically evaluated
- ✅ **9 network backbones** (ResNet, DenseNet, SE-ResNeXt, ViT)
- ✅ **Stratified OoD benchmark**: ID, Near-OoD, Far-OoD (Bubbles & Particles), Far-OoD (General)
- ✅ **Reproducible training & evaluation** pipeline based on [OpenOOD v1.5](https://github.com/Jingkang50/OpenOOD)
- ✅ **Hyperparameter search** and **three repeated runs** for robust results

---

## 📊 Dataset

We use the **DYB-PlanktonNet** dataset, publicly available at:  
🔗 [IEEE DataPort - DYB-PlanktonNet](https://ieee-dataport.org/documents/dyb-planktonnet)

### Dataset Splits

We partition the original 92 categories into:

| Split | Description | Example Classes |
|-------|-------------|-----------------|
| **ID** | Ecologically significant species | `Jellyfish`, `Creseis acicula` |
| **Near-OoD** | Biologically related but less frequent | `Hydroid`, `Ostracoda` |
| **Far-OoD (Bubbles & Particles)** | Non-biological noise | `Bubbles`, `Fish eggs` |
| **Far-OoD (General)** | External datasets | CIFAR-10, SVHN, MNIST, etc. |

Use the provided script to reproduce the split:

```bash
python split_dataset_new_class.py --data_dir ./data/DYB-PlanktonNet
```

---

## 🧩 Methods & Architectures

### OoD Detection Methods

We evaluate the following post-hoc methods (grouped by category):

| Category | Methods |
|----------|---------|
| **Distance-based** | Mahalanobis, RMDS, KNN, fDBD |
| **Classification-based** | ViM, Residual, ODIN, OpenMax, Relation, TempScale, GEN, MSP, MCDropout, MLS, KL Matching, ReAct, ASH, SHE, RankFeat, GradNorm |
| **Density-based** | Energy, DICE |

### Network Architectures

We test the following backbones:

- ResNet-18, ResNet-50, ResNet-101, ResNet-152
- DenseNet-121, DenseNet-169, DenseNet-201
- SE-ResNeXt-50
- Vision Transformer (ViT)

---

## 📈 Results

### Key Findings

- **ViM** achieves the best overall performance across both Near-OoD and Far-OoD benchmarks.
- **Distance-based methods** (e.g., Mahalanobis, KNN) excel in Far-OoD detection.
- **Density-based methods** (e.g., Energy, DICE) perform well on Near-OoD tasks.

### Detailed Results

We provide full experimental results (AUROC, FPR95, FPR99) for all method-backbone pairs, along with hyperparameter settings and model checkpoints:  
🔗 [Google Drive - Results & Models](https://drive.google.com/drive/folders/1Co6qMhBsL9BVBcjPmC8AKMbyGwh3V4NI?usp=sharing)



<!-- ## 📄 Citation

If you find this work useful, please cite our paper:

```bibtex
@inproceedings{han2025benchmarking,
  title={Benchmarking Out-of-Distribution Detection for Plankton Recognition: A Systematic Evaluation of Advanced Methods in Marine Ecological Monitoring},
  author={Han, Yingzi and He, Jiakai and Xie, Chuanlong and Li, Jianping},
  booktitle={Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV)},
  year={2025}
}
``` -->

---

## 🙏 Acknowledgments

- This work was supported by the National Natural Science Foundation of China (No. 12201048, 42476218).
- We thank the Interdisciplinary Intelligence Super Computer Center of Beijing Normal University at Zhuhai.
- Our implementation is built upon the [OpenOOD](https://github.com/Jingkang50/OpenOOD) framework.

---

## 📮 Contact

For questions or issues, please open a GitHub Issue or contact:  
- Yingzi Han: hanyingzi@mail.bnu.edu.cn  
- Jiakai He: hejiakai@mail.bnu.edu.cn

---

## 📜 License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

---

