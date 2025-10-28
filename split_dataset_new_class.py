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
Usage:
python split_dataset_new_class.py \
  --data_dir ./data/DYB-PlanktonNet
  # --output_dir ./data/benchmark_imglist
'''

def is_image_file(filename):
    """
    Check whether a file is an image, supports jpg, jpeg, png, bmp, gif, etc.
    """
    return filename.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif'))


ID_CLASSES = [
    '001_Polychaeta_most with eggs',     
    '003_Polychaeta_Type A',            
    '004_Polychaeta_Type B',            
    '005_Polychaeta_Type C',            
    '006_Polychaeta_Type D',            
    '007_Polychaeta_Type E',            
    '008_Polychaeta_Type F',            
    '009_Penilia avirostris',           
    '010_Evadne tergestina',            
    '011_Acartia sp.A',                 
    '012_Acartia sp.B',                 
    '013_Acartia sp.C',                 
    '014_Calanopia sp',                 
    '015_Labidocera sp',                
    '016_Tortanus gracilis',            
    '017_Calanoid with egg',            
    '019_Calanoid_Type A',              
    '020_Calanoid_Type B',              
    '024_Oithona sp.B with egg',        
    '025_Cyclopoid_Type A_with egg',    
    '027_Harpacticoid_mating',          
    '029_Microsetella sp',              
    '033_Caligus sp',                   
    '034_Copepod_Type A',               
    '035_Caprella sp',                  
    '036_Amphipoda_Type A',             
    '037_Amphipoda_Type B',             
    '038_Amphipoda_Type C',             
    '039_Gammarids_Type A',             
    '040_Gammarids_Type B',             
    '041_Gammarids_Type C',             
    '042_Cymodoce sp',                  
    '043_Lucifer sp',                   
    '044_Macrura larvae',               
    '046_Megalopa larva_Phase 1_Type B',
    '047_Megalopa larva_Phase 1_Type C',
    '048_Megalopa larva_Phase 1_Type D',
    '049_Megalopa larva_Phase 2',       
    '050_Porcrellanidae larva',         
    '051_Shrimp-like larva_Type A',     
    '052_Shrimp-like larva_Type B',     
    '053_Shrimp-like_Type A',           
    '054_Shrimp-like_Type B',           
    '056_Shrimp-like_Type D',           
    '058_Shrimp-like_Type F',           
    '060_Cumacea_Type A',               
    '061_Cumacea_Type B',               
    '062_Chaetognatha',                 
    '063_Oikopleura sp. parts',         
    '065_Tunicata_Type A',              
    '068_Jellyfish',                    
    '071_Creseis acicula',              
    '082_Noctiluca scintillans',        
    '091_Phaeocystis globosa',          
]


NEAROOD_CLASSES = [
    '002_Polychaeta larva',            
    '018_Calanoid Nauplii',             
    '021_Calanoid_Type C',              
    '022_Calanoid_Type D',              
    '023_Oithona sp.A with egg',       
    '026_Cyclopoid_Type A',             
    '028_Harpacticoid',                 
    '030_Monstrilla sp.A',             
    '031_Monstrilla sp.B',              
    '045_Megalopa larva_Phase 1_Type A',
    '055_Shrimp-like_Type C',           
    '057_Shrimp-like_Type E',           
    '059_Ostracoda',                    
    '064_Oikopleura sp',                
    '066_Actiniaria larva',             
    '067_Hydroid',                     
    '069_Jelly-like',                   
    '070_Bryozoan larva',               
    '072_Gelatinous Zooplankton',       
    '073_Unknown_Type A',               
    '074_Unknown_Type B',               
    '075_Unknown_Type C',               
    '076_Unknown_Type D',               
    '077_Balanomorpha exuviate',        
    '081_Fish Larvae',                  
    '032_Monstrilloid',                
]


FAROOD_CLASSES = [
    '078_Crustacean limb_Type A',      
    '079_Crustacean limb_Type B',       
    '080_Fish egg',                     
    '083_Particle_filamentous_Type A',  
    '084_Particle_filamentous_Type B',  
    '085_Particle_bluish',              
    '086_Particle_molts',               
    '087_Particle_translucent flocs',   
    '088_Particle_yellowish flocs',     
    '089_Particle_yellowish rods',      
    '090_Bubbles',                      
    '092_Fish tail',                    
]

def calculate_mean_std_incremental(train_items, data_dir):
    """
    Incrementally calculate mean and std for training set to save memory
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
    if train_ratio + test_ratio >= 1.0:
        raise ValueError("train_ratio + test_ratio must be less than 1.0!")

    classes = sorted([
        d for d in os.listdir(data_dir)
        if os.path.isdir(os.path.join(data_dir, d))
    ])
    if not classes:
        print(f"No subfolders found under {data_dir}, please check the data structure!")
        return

    # 创建 ID 类别到连续索引的映射
    id_class_to_contiguous_index = {cls: idx for idx, cls in enumerate(ID_CLASSES)}
    contiguous_index_to_id_class = {idx: cls for cls, idx in id_class_to_contiguous_index.items()}

    
    print("Mapping of ID Classes to Contiguous Indices:")
    for id_class, contiguous_index in id_class_to_contiguous_index.items():
        print(f"  ID Class: {id_class}, Contiguous Index: {contiguous_index}")


    print("There are", len(ID_CLASSES), "ID classes,", len(NEAROOD_CLASSES), "NearOOD classes, and ", len(FAROOD_CLASSES), "FarOOD classes.")

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
                    near_items.append((path, cls))
                elif cls in FAROOD_CLASSES:
                    far_items.append((path, cls))
                else:
                    far_items.append((path, cls))

    print(f"Total ID: {len(id_items)} images, nearOOD: {len(near_items)} images, farOOD: {len(far_items)} images")


    per_class_counts = []  # Used to store split statistics for each class

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

# Print and save split statistics for each ID class
    print("\nSplit Statistics for Each ID Class:")
    print("{:<40} {:>6} {:>6} {:>6} {:>6}".format("Class", "Train", "Val", "Test", "Total"))
    for row in per_class_counts:
        print("{:<40} {:>6} {:>6} {:>6} {:>6}".format(
            row["Class"], row["Train"], row["Val"], row["Test"], row["Total"]
        ))

    print("The smallest class has", min_total, "samples.")

    # Save to CSV
    os.makedirs(output_dir, exist_ok=True)
    id_class_split_csv = os.path.join(output_dir, 'id_class_split_stats.csv')
    pd.DataFrame(per_class_counts).to_csv(id_class_split_csv, index=False, encoding='utf-8')
    print(f"ID sample split per class saved to:{id_class_split_csv}")


    # Calculate training set mean and standard deviation (using incremental algorithm)
    # train_mean, train_std = calculate_mean_std_incremental(
    #     [(path, contiguous_index_to_id_class[label]) for path, label in id_train], data_dir
    # )
    # if train_mean and train_std:
    #     mean_std_output_path = os.path.join(output_dir, 'train_mean_std.txt')
    #     with open(mean_std_output_path, 'w', encoding='utf-8') as f:
    #         f.write(str([train_mean, train_std]))
    #     print(f"Training set mean and std saved to: {mean_std_output_path} (Incremental Algorithm)")
    # else:
    #     print("Failed to calculate mean and std for the training set, please check image loading.")

    # Split Near-OoD
    random.seed(seed)
    random.shuffle(near_items)
    n_near = len(near_items)
    n_near_val = int(n_near * ood_val_ratio)
    near_val, near_test = near_items[:n_near_val], near_items[n_near_val:]

    # Split Far-OoD
    random.seed(seed)
    random.shuffle(far_items)
    n_far = len(far_items)
    n_far_val = int(n_far * ood_val_ratio)
    far_val, far_test = far_items[:n_far_val], far_items[n_far_val:]

    if keep_ood_label:
        # Near-OoD class split statistics
        near_class_counts = defaultdict(lambda: {"val": 0, "test": 0})
        for p, lbl in near_val:
            near_class_counts[lbl]["val"] += 1
        for p, lbl in near_test:
            near_class_counts[lbl]["test"] += 1

        print("\nNear-OoD Sample Split Statistics per Class:")
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
        near_csv = os.path.join(output_dir, "Near-OoD_class_split_stats.csv")
        pd.DataFrame(near_ood_rows).to_csv(near_csv, index=False, encoding="utf-8")
        print(f"Near-OoD class split statistics saved to: {near_csv}")

        # Far-OoD 类别划分统计
        far_class_counts = defaultdict(lambda: {"val": 0, "test": 0})
        for p, lbl in far_val:
            far_class_counts[lbl]["val"] += 1
        for p, lbl in far_test:
            far_class_counts[lbl]["test"] += 1

        print("\nFar-OoD Sample Split Statistics per Class:")
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
        print(f"Far-OoD class split statistics saved to:: {far_csv}")


    # Output files
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
        print(f"Writing {name}: {fp} (Total {len(items)} images)")

    # Save statistics CSV
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
    print(f"Statistics table saved to {out_csv}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="Divide dataset into ID/nearOOD/farOOD based on manual classification criteria and generate list files"
    )
    parser.add_argument('--data_dir',     type=str, default='./data/DYB-PlanktonNet',
                        help="Dataset root directory, e.g., './data/DYB-PlanktonNet'" )
    parser.add_argument('--output_dir',   type=str, default="./data/benchmark_imglist/plankton54",
                        help="Output list file directory" )
    parser.add_argument('--train_ratio',  type=float, default=0.7, help="ID Train set ratio" )
    parser.add_argument('--test_ratio',   type=float, default=0.1, help="ID Test set ratio" )
    parser.add_argument('--ood_val_ratio', type=float, default=0.2, help="OOD Validation set ratio" )
    parser.add_argument('--seed',         type=int,   default=42,   help="Random seed" )
    parser.add_argument('--absolute',     action='store_true',     help="Write absolute paths" )
    parser.add_argument('--keep_ood_label', action='store_true',   help="Keep OOD original labels" )
    args = parser.parse_args()
    split_dataset(
        args.data_dir, args.output_dir,
        args.train_ratio, args.test_ratio, args.ood_val_ratio,
        args.seed, use_relative=not args.absolute,
        keep_ood_label=args.keep_ood_label
    )