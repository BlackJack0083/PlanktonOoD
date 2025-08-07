#!/usr/bin/env python
import os
import argparse
import random
from collections import defaultdict
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np
from tqdm import tqdm

'''
运行方式:
python split_dataset_new_class.py \
  --data_dir ./data/DYB-PlanktonNet
  # --output_dir ./data/benchmark_imglist
'''

def is_image_file(filename):
    """
    判断文件是否为图片，支持 jpg、jpeg、png、bmp、gif 等格式
    """
    return filename.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif'))

# 一、ID_CLASSES（核心监测物种，54类）
ID_CLASSES = [
    '001_Polychaeta_most with eggs',     # 多毛类(带卵) 1772 ▶高繁殖风险
    '003_Polychaeta_Type A',            # 多毛类A型 436
    '004_Polychaeta_Type B',            # 多毛类B型 64
    '005_Polychaeta_Type C',            # 多毛类C型 72
    '006_Polychaeta_Type D',            # 多毛类D型 153
    '007_Polychaeta_Type E',            # 多毛类E型 235
    '008_Polychaeta_Type F',            # 多毛类F型 233
    '009_Penilia avirostris',           # 鸟喙莹虾 505 ▶滤网堵塞主力
    '010_Evadne tergestina',            # 捷氏桅足虫 70
    '011_Acartia sp.A',                 # 纺锤水蚤A型 75
    '012_Acartia sp.B',                 # 纺锤水蚤B型 1823 ▶优势种
    '013_Acartia sp.C',                 # 纺锤水蚤C型 4124 ▶绝对优势种
    '014_Calanopia sp',                 # 拟哲水蚤 78
    '015_Labidocera sp',                # 唇角水蚤 148
    '016_Tortanus gracilis',            # 瘦尾筒角水蚤 1423
    '017_Calanoid with egg',            # 带卵哲水蚤 633
    '019_Calanoid_Type A',              # 哲水蚤A型 469
    '020_Calanoid_Type B',              # 哲水蚤B型 321
    '024_Oithona sp.B with egg',        # 带卵奥氏水蚤B型 253
    '025_Cyclopoid_Type A_with egg',    # 带卵剑水蚤A型 389
    '027_Harpacticoid_mating',          # 交配猛水蚤 71
    '029_Microsetella sp',              # 小毛猛水蚤 585
    '033_Caligus sp',                   # 鱼虱 893 ▶寄生虫监测
    '034_Copepod_Type A',               # 桡足类A型 53
    '035_Caprella sp',                  # 麦秆虫 352 ▶附着生物
    '036_Amphipoda_Type A',             # 端足类A型 110
    '037_Amphipoda_Type B',             # 端足类B型 199
    '038_Amphipoda_Type C',             # 端足类C型 86
    '039_Gammarids_Type A',             # 钩虾A型 175
    '040_Gammarids_Type B',             # 钩虾B型 155
    '041_Gammarids_Type C',             # 钩虾C型 777
    '042_Cymodoce sp',                  # 浪漂水虱 80
    '043_Lucifer sp',                   # 莹虾 943 ▶趋光堵塞
    '044_Macrura larvae',               # 长尾类幼体 624
    '046_Megalopa larva_Phase 1_Type B',# 大眼幼体B型 627
    '047_Megalopa larva_Phase 1_Type C',# 大眼幼体C型 139
    '048_Megalopa larva_Phase 1_Type D',# 大眼幼体D型 761
    '049_Megalopa larva_Phase 2',       # 大眼幼体二期 61
    '050_Porcrellanidae larva',         # 瓷蟹幼体 742
    '051_Shrimp-like larva_Type A',     # 类虾幼体A型 98
    '052_Shrimp-like larva_Type B',     # 类虾幼体B型 564
    '053_Shrimp-like_Type A',           # 类虾A型 153
    '054_Shrimp-like_Type B',           # 类虾B型 75
    '056_Shrimp-like_Type D',           # 类虾D型 63
    '058_Shrimp-like_Type F',           # 类虾F型 386
    '060_Cumacea_Type A',               # 涟虫A型 719
    '061_Cumacea_Type B',               # 涟虫B型 68
    '062_Chaetognatha',                 # 毛颚动物 658
    '063_Oikopleura sp. parts',         # 住囊虫残体 332
    '065_Tunicata_Type A',              # 被囊动物A型 385
    '068_Jellyfish',                    # 水母 414 ▶历史堵塞主因
    '071_Creseis acicula',              # 尖笔帽螺 （大亚湾大爆发） 3762
    '082_Noctiluca scintillans',        # 夜光虫 91
    '091_Phaeocystis globosa',          # 棕囊藻 481
]

# 二、NEAROOD_CLASSES（次级关联物种，28类）
NEAROOD_CLASSES = [
    '002_Polychaeta larva',             # 多毛类幼虫 44
    '018_Calanoid Nauplii',             # 哲水蚤无节幼体 7
    '021_Calanoid_Type C',              # 哲水蚤C型 9
    '022_Calanoid_Type D',              # 哲水蚤D型 15
    '023_Oithona sp.A with egg',        # 带卵奥氏水蚤A型 7
    '026_Cyclopoid_Type A',             # 剑水蚤A型 14
    '028_Harpacticoid',                 # 猛水蚤 7
    '030_Monstrilla sp.A',              # 怪水蚤A型 40
    '031_Monstrilla sp.B',              # 怪水蚤B型 25
    '045_Megalopa larva_Phase 1_Type A',# 大眼幼体A型 21
    '055_Shrimp-like_Type C',           # 类虾C型 21
    '057_Shrimp-like_Type E',           # 类虾E型 12
    '059_Ostracoda',                    # 介形虫 41
    '064_Oikopleura sp',                # 住囊虫 14
    '066_Actiniaria larva',             # 海葵幼虫 28
    '067_Hydroid',                      # 水螅虫 37
    '069_Jelly-like',                   # 胶状物 45
    '070_Bryozoan larva',               # 苔藓虫幼体 50
    '072_Gelatinous Zooplankton',       # 胶质浮游动物 50
    '073_Unknown_Type A',               # 未知物A型 384
    '074_Unknown_Type B',               # 未知物B型 83
    '075_Unknown_Type C',               # 未知物C型 11
    '076_Unknown_Type D',               # 未知物D型 13
    '077_Balanomorpha exuviate',        # 藤壶蜕壳 97
    '081_Fish Larvae',                  # 鱼类幼体 353
    '032_Monstrilloid',                 # 怪水蚤 2
]

# 三、FAROOD_CLASSES（无关干扰项，10类）
FAROOD_CLASSES = [
    '078_Crustacean limb_Type A',       # 甲壳残肢A型 88
    '079_Crustacean limb_Type B',       # 甲壳残肢B型 76
    '080_Fish egg',                     # 鱼卵 25
    '083_Particle_filamentous_Type A',  # 丝状颗粒A型 66
    '084_Particle_filamentous_Type B',  # 丝状颗粒B型 1257
    '085_Particle_bluish',              # 蓝色颗粒 168
    '086_Particle_molts',               # 蜕壳碎片 50
    '087_Particle_translucent flocs',   # 半透明絮状物 673
    '088_Particle_yellowish flocs',     # 黄色絮状颗粒 3063
    '089_Particle_yellowish rods',      # 黄色棒状颗粒 3671
    '090_Bubbles',                      # 气泡 7760 ▶光学干扰
    '092_Fish tail',                    # 鱼尾 134
]

def calculate_mean_std_incremental(train_items, data_dir):
    """
    使用逐步算法计算训练集的均值和标准差，节约内存
    """
    count = 0
    mean = np.zeros(3, dtype=np.float64)
    M2 = np.zeros(3, dtype=np.float64)

    for path, _ in tqdm(train_items, desc="Calculating Mean/Std (Incremental)"):
        try:
            img_path = os.path.join(data_dir, path)
            img = Image.open(img_path).convert('RGB')
            img_array = np.array(img, dtype=np.float64) / 255.0
            pixels = img_array.reshape(-1, 3)

            for pixel in pixels:
                count += 1
                delta = pixel - mean
                mean += delta / count
                delta2 = pixel - mean
                M2 += delta * delta2
        except Exception as e:
            print(f"Error loading image: {img_path} - {e}")

    if count > 1:
        variance = M2 / (count - 1)
        std = np.sqrt(variance)
        return mean.tolist(), std.tolist()
    elif count == 1:
        return mean.tolist(), np.zeros(3).tolist()
    else:
        return [], []
def split_dataset(data_dir,
                    output_dir,
                    train_ratio=0.7,
                    test_ratio=0.2,
                    ood_val_ratio=0.2,
                    seed=42,
                    use_relative=True,
                    keep_ood_label=False):
    """
    按手动定义的 ID/nearOOD/farOOD 类别划分数据集并生成列表
    这次正确处理 ID 标签的连续分配
    """
    if train_ratio + test_ratio >= 1.0:
        raise ValueError("train_ratio + test_ratio 必须小于 1.0！")

    classes = sorted([
        d for d in os.listdir(data_dir)
        if os.path.isdir(os.path.join(data_dir, d))
    ])
    if not classes:
        print(f"在 {data_dir} 下未找到子文件夹，请检查数据组织结构！")
        return

    # 创建 ID 类别到连续索引的映射
    id_class_to_contiguous_index = {cls: idx for idx, cls in enumerate(ID_CLASSES)}
    contiguous_index_to_id_class = {idx: cls for cls, idx in id_class_to_contiguous_index.items()}

    
    print("ID 类别与连续索引的映射：")
    for id_class, contiguous_index in id_class_to_contiguous_index.items():
        print(f"  ID 类别: {id_class}, 连续索引: {contiguous_index}")


    print("共有", len(ID_CLASSES), "个ID类，", len(NEAROOD_CLASSES), "个NearOOD类， ", len(FAROOD_CLASSES), "个FarOOD类。")

    id_items, near_items, far_items = [], [], []
    for cls in classes:
        cls_dir = os.path.join(data_dir, cls)
        for root, _, fnames in os.walk(cls_dir):
            for f in fnames:
                if not is_image_file(f):
                    continue
                full = os.path.join(root, f)
                path = (os.path.normpath(os.path.relpath(full, data_dir))
                        if use_relative else os.path.abspath(full))
                if cls in ID_CLASSES:
                    contiguous_index = id_class_to_contiguous_index[cls]
                    id_items.append((path, contiguous_index))
                elif cls in NEAROOD_CLASSES:
                    # nearOOD 保留原始类别名作为标签 (如果 keep_ood_label 为 True)
                    near_items.append((path, cls))
                elif cls in FAROOD_CLASSES:
                    # farOOD 保留原始类别名作为标签 (如果 keep_ood_label 为 True)
                    far_items.append((path, cls))
                else:
                    # 未列出类别默认为 farOOD
                    far_items.append((path, cls))

    print(f"共 ID: {len(id_items)} 张, nearOOD: {len(near_items)} 张, farOOD: {len(far_items)} 张")

    # 划分 ID - 改为逐类划分以便统计每个类的 train/val/test 数量
    per_class_counts = []  # 用于保存每类的划分统计信息

    id_train, id_val, id_test = [], [], []

    min_total = 1e9

    for cls in ID_CLASSES:
        cls_items = [(p, id_class_to_contiguous_index[cls]) for p, label in id_items if label == id_class_to_contiguous_index[cls]]
        random.seed(seed)
        random.shuffle(cls_items)
        n_total = len(cls_items)
        n_train = int(n_total * train_ratio)
        n_test = int(n_total * test_ratio)
        n_val = n_total - n_train - n_test

        cls_train = cls_items[:n_train]
        cls_val = cls_items[n_train:n_train+n_val]
        cls_test = cls_items[n_train+n_val:]

        id_train.extend(cls_train)
        id_val.extend(cls_val)
        id_test.extend(cls_test)

        min_total = min(min_total, n_total)

        per_class_counts.append({
            "Class": cls,
            "Train": len(cls_train),
            "Val": len(cls_val),
            "Test": len(cls_test),
            "Total": n_total
        })

    # 打印并保存每个 ID 类别的划分统计信息
    print("\n每个 ID 类别的划分统计：")
    print("{:<40} {:>6} {:>6} {:>6} {:>6}".format("Class", "Train", "Val", "Test", "Total"))
    for row in per_class_counts:
        print("{:<40} {:>6} {:>6} {:>6} {:>6}".format(
            row["Class"], row["Train"], row["Val"], row["Test"], row["Total"]
        ))

    print("最少的一类有", min_total, "个样本。")

    # 保存为 CSV
    os.makedirs(output_dir, exist_ok=True)
    id_class_split_csv = os.path.join(output_dir, 'id_class_split_stats.csv')
    pd.DataFrame(per_class_counts).to_csv(id_class_split_csv, index=False, encoding='utf-8')
    print(f"每类 ID 样本划分已保存至: {id_class_split_csv}")


    # # 计算训练集均值和标准差 (使用逐步算法)
    # train_mean, train_std = calculate_mean_std_incremental(
    #     [(path, contiguous_index_to_id_class[label]) for path, label in id_train], data_dir
    # )
    # if train_mean and train_std:
    #     mean_std_output_path = os.path.join(output_dir, 'train_mean_std.txt')
    #     with open(mean_std_output_path, 'w', encoding='utf-8') as f:
    #         f.write(str([train_mean, train_std]))
    #     print(f"训练集均值和标准差已保存至: {mean_std_output_path} (逐步算法)")
    # else:
    #     print("未能计算训练集的均值和标准差，请检查图像加载。")

    # 划分 nearOOD
    random.seed(seed)
    random.shuffle(near_items)
    n_near = len(near_items)
    n_near_val = int(n_near * ood_val_ratio)
    near_val, near_test = near_items[:n_near_val], near_items[n_near_val:]

    # 划分 farOOD
    random.seed(seed)
    random.shuffle(far_items)
    n_far = len(far_items)
    n_far_val = int(n_far * ood_val_ratio)
    far_val, far_test = far_items[:n_far_val], far_items[n_far_val:]

    if keep_ood_label:
        # nearOOD 类别划分统计
        near_class_counts = defaultdict(lambda: {"val": 0, "test": 0})
        for p, lbl in near_val:
            near_class_counts[lbl]["val"] += 1
        for p, lbl in near_test:
            near_class_counts[lbl]["test"] += 1

        print("\nnearOOD 每类样本划分统计：")
        print("{:<40} {:>6} {:>6} {:>6}".format("Class", "Val", "Test", "Total"))
        near_ood_rows = []
        for cls, cnts in near_class_counts.items():
            total = cnts["val"] + cnts["test"]
            print("{:<40} {:>6} {:>6} {:>6}".format(cls, cnts["val"], cnts["test"], total))
            near_ood_rows.append({
                "Class": cls,
                "Val": cnts["val"],
                "Test": cnts["test"],
                "Total": total
            })
        near_csv = os.path.join(output_dir, "nearOOD_class_split_stats.csv")
        pd.DataFrame(near_ood_rows).to_csv(near_csv, index=False, encoding="utf-8")
        print(f"nearOOD 类别划分统计保存至: {near_csv}")

        # farOOD 类别划分统计
        far_class_counts = defaultdict(lambda: {"val": 0, "test": 0})
        for p, lbl in far_val:
            far_class_counts[lbl]["val"] += 1
        for p, lbl in far_test:
            far_class_counts[lbl]["test"] += 1

        print("\nfarOOD 每类样本划分统计：")
        print("{:<40} {:>6} {:>6} {:>6}".format("Class", "Val", "Test", "Total"))
        far_ood_rows = []
        for cls, cnts in far_class_counts.items():
            total = cnts["val"] + cnts["test"]
            print("{:<40} {:>6} {:>6} {:>6}".format(cls, cnts["val"], cnts["test"], total))
            far_ood_rows.append({
                "Class": cls,
                "Val": cnts["val"],
                "Test": cnts["test"],
                "Total": total
            })
        far_csv = os.path.join(output_dir, "farOOD_class_split_stats.csv")
        pd.DataFrame(far_ood_rows).to_csv(far_csv, index=False, encoding="utf-8")
        print(f"farOOD 类别划分统计保存至: {far_csv}")


    # 输出文件
    os.makedirs(output_dir, exist_ok=True)
    splits = {
        'ID_train': (id_train, True), 'ID_val': (id_val, True), 'ID_test': (id_test, True),
        'plankton_near_val': (near_val, keep_ood_label), 'plankton_near_test': (near_test, keep_ood_label),
        'plankton_far_val': (far_val, keep_ood_label),   'plankton_far_test': (far_test, keep_ood_label)
    }
    for name, (items, keep_lbl) in splits.items():
        fname = f"{name}.txt"
        fp = os.path.join(output_dir, fname)
        with open(fp, 'w', encoding='utf-8') as f:
            for p, lbl in items:
                if name.startswith('ID'):
                    f.write(f"{p} {lbl}\n")
                else:
                    f.write(f"{p} {lbl if keep_lbl else -1}\n")
        print(f"写入 {name}: {fp} (共 {len(items)} 张)")

    # 统计保存 CSV
    counts = {k: defaultdict(int) for k in splits}
    for p, lbl in id_train:   counts['ID_train'][contiguous_index_to_id_class[lbl]] += 1
    for p, lbl in id_val:     counts['ID_val'][contiguous_index_to_id_class[lbl]]   += 1
    for p, lbl in id_test:    counts['ID_test'][contiguous_index_to_id_class[lbl]]  += 1
    for p, lbl in near_val:   counts['plankton_near_val'][lbl if keep_ood_label else -1] += 1
    for p, lbl in near_test:  counts['plankton_near_test'][lbl if keep_ood_label else -1] += 1
    for p, lbl in far_val:    counts['plankton_far_val'][lbl if keep_ood_label else -1]   += 1
    for p, lbl in far_test:   counts['plankton_far_test'][lbl if keep_ood_label else -1]   += 1

    all_labels = sorted(list(set(str(l) for counts_dict in counts.values() for l in counts_dict)))
    df = pd.DataFrame({s: [counts[s].get(l if isinstance(l, str) else str(l), 0) for l in all_labels] for s in counts},
                  index=all_labels)

    df.index.name = 'Class'
    out_csv = os.path.join(output_dir, 'split_counts.csv')
    df.to_csv(out_csv, encoding='utf-8')
    print(f"统计表保存至 {out_csv}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="按手动分类标准将数据集划分为 ID/nearOOD/farOOD 并生成列表文件"
    )
    parser.add_argument('--data_dir',     type=str, default='./data/DYB-PlanktonNet',
                        help="数据集根目录，如 './data/DYB-PlanktonNet'" )
    parser.add_argument('--output_dir',   type=str, default="./data/benchmark_imglist/plankton54",
                        help="输出列表文件目录" )
    parser.add_argument('--train_ratio',  type=float, default=0.7, help="ID 训练集比例" )
    parser.add_argument('--test_ratio',   type=float, default=0.1, help="ID 测试集比例" )
    parser.add_argument('--ood_val_ratio', type=float, default=0.2, help="OOD 验证集比例" )
    parser.add_argument('--seed',         type=int,   default=42,   help="随机种子" )
    parser.add_argument('--absolute',     action='store_true',     help="写入绝对路径" )
    parser.add_argument('--keep_ood_label', action='store_true',   help="保留 OOD 原标签" )
    args = parser.parse_args()
    split_dataset(
        args.data_dir, args.output_dir,
        args.train_ratio, args.test_ratio, args.ood_val_ratio,
        args.seed, use_relative=not args.absolute,
        keep_ood_label=args.keep_ood_label
    )