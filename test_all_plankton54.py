import subprocess
import os
import glob
import pandas as pd
import argparse
# 放在你的脚本最上面
import sys
import logging

# ====== 设置 logging 日志 ======
logging.basicConfig(
    level=logging.INFO,
    format='[%(asctime)s] %(levelname)s: %(message)s',
    filename='output.log',
    filemode='w'
)

# ====== 重定向 print() 到终端 + 日志 ======
class Tee:
    def __init__(self, stream, logger_func):
        self.stream = stream
        self.logger_func = logger_func

    def write(self, message):
        if message.strip() != '':
            self.logger_func(message.strip())
        self.stream.write(message)

    def flush(self):
        self.stream.flush()

sys.stdout = Tee(sys.__stdout__, logging.info)
sys.stderr = Tee(sys.__stderr__, logging.error)

def run_and_log(command):
    print(f"\n====== 运行命令: {command} ======")
    result = subprocess.run(command, shell=True, capture_output=True, text=True)
    
    if result.stdout:
        for line in result.stdout.strip().splitlines():
            print(line)
    if result.stderr:
        for line in result.stderr.strip().splitlines():
            print(line, file=sys.stderr)

# -------------------------------
# 1. 定义所有方法的 alias（均使用小写，直接参照 Wiki 页面“Alias”列）
# 提取自页面内容，共 40 个有 alias 的方法
# -------------------------------
aliases = [
    "msp",    # post-hoc
    "vim",
    "ash",   # post-hoc
    # "cider",  # training
    # "conf_branch",  # training
    "ebo",   # post-hoc, energy
    "odin",  # post-hoc
    "mds",   # post-hoc
    # "mds_ensemble",  # post-hoc
    # "npos",  # training 
    "rmds",  # post-hoc
    #"gmm",  # post-hoc
    #"patchcore",  # post-hoc
    "openmax", # post-hoc
    "react",  # post-hoc
    "gradnorm",  # post-hoc
    # "godin",  # training
    # "gram",  # post-hoc
    # "cutpaste",  # ?
    "mls",  # post-hoc
    "residual",  # Deep Residual Flow for Out of Distribution Detection
    "klm",   # post-hoc
    "temperature_scaling",  # post-hoc
    # "ensemble",  
    "dropout",  # post-hoc
    # "draem",   # ?
    # "dsvdd",   # ?
    # "mos",  # training
    # "mcd",  # training
    # "opengan",  # post-hoc
    "knn",    # post-hoc
    "dice",    # post-hoc
    # "ssd",   # SSD: A Unified Framework for Self-Supervised Outlier Detection
    "she",   # post-hoc
    # "rd4ad",   # posthoc
    # "rotpred",  # training
    "rankfeat",  # post-hoc
    "temp_scaling",  # post-hoc
    # 补全缺失项（注释可自行补充）
    "fdbd",    # post-hoc
    # "rts",     # post-hoc
    "gen",     # post-hoc
    "relation",# post-hoc
    # "t2fnorm", # post-hoc
]


# -------------------------------
# 2. 评测参数设置
# -------------------------------
id_data = "plankton54"
# 基础结果目录：为避免不同方法输出文件互相覆盖，每个方法将输出到 base_root/<alias> 子目录
base_root = "results/plankton54_resnet152_base_e100_lr0.1_default"
# 公共参数
common_args = f"--id-data {id_data} --save-score --save-csv"

# -------------------------------
# 3. 遍历所有 alias 进行评测
# -------------------------------
# for alias in aliases:
#     # 为每个方法创建独立的结果目录（目录名为 alias）
#     result_dir = base_root  # os.path.join(base_root, alias)
#     # os.makedirs(result_dir, exist_ok=True)

#     # 拼接评测命令，注意 alias 均为小写，若 alias 中含空格或特殊字符，用双引号包裹
#     command = f'python scripts/eval_ood.py {common_args} --root {result_dir} --postprocessor "{alias}"'
#     print("运行命令:", command)
    
#         # -------------------------------

import subprocess

# 3. 遍历所有 alias 进行评测
# -------------------------------
# for alias in aliases:
#     # 为每个方法创建独立的结果目录（目录名为 alias）
#     result_dir = base_root  # os.path.join(base_root, alias)
#     # os.makedirs(result_dir, exist_ok=True)

#     # 拼接评测命令，注意 alias 均为小写，若 alias 中含空格或特殊字符，用双引号包裹
#     command = f'python scripts/eval_ood.py {common_args} --root {result_dir} --postprocessor "{alias}"'
#     print("运行命令:", command)
    
#     # 执行命令（可根据需要添加异常处理、超时控制等）
#     subprocess.run(command, shell=True)

for alias in aliases:
    result_dir = base_root
    command = f'python scripts/eval_ood.py {common_args} --root {result_dir} --postprocessor "{alias}"'
    run_and_log(command)
# -------------------------------
# 4. 合并所有评测结果（CSV 文件）
# -------------------------------
# 假设每个方法评测后都会在对应目录下生成 CSV 文件
# 修改 csv_pattern 以直接匹配 ood 目录下的所有 CSV 文件
csv_pattern = os.path.join(base_root, "ood", "*.csv")
csv_files = glob.glob(csv_pattern)
dfs = []

for csv_file in csv_files:
    try:
        df = pd.read_csv(csv_file)
        # 提取文件名（不带扩展名）作为方法 alias，并添加一列标识
        method_name = os.path.splitext(os.path.basename(csv_file))[0]
        df["method"] = method_name
        dfs.append(df)
    except Exception as e:
        print(f"读取 {csv_file} 时发生错误: {e}")

if dfs:
    merged_df = pd.concat(dfs, ignore_index=True)
    merged_csv_path = os.path.join(base_root, "merged_results.csv")
    merged_df.to_csv(merged_csv_path, index=False)
    print("所有方法的评测结果已合并并保存到:", merged_csv_path)
else:
    print("未找到 CSV 文件，请检查各方法的评测结果。")
