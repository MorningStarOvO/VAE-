# ==================== 导入必要的包 ==================== #
# ----- 命令行交互 & 配置模型相关 ----- #
import yaml # 一种直观的、能被电脑识别的数据序列化格式
import argparse

# ----- 数据处理相关 ----- #
import numpy as np

# ----- 系统操作相关 ----- #
import os
import time

# ----- 模型构建相关 ----- #
import torch.backends.cudnn as cudnn
from pytorch_lightning import Trainer
from pytorch_lightning.logging import TestTubeLogger

# ----- 自定义的包 ----- #
from models import *
from experiment import VAEXperiment

# ==================== 开始运行程序 ==================== #
# ----- 开始计时 ----- #
T_Start = time.time()
print("程序开始运行 !")

# ----- 命令行交互相关 ----- #
parser = argparse.ArgumentParser(description='Generic runner for VAE models')
parser.add_argument('--config',  '-c',
                    dest="filename",
                    metavar='FILE',
                    help =  'path to the config file',
                    default='configs/vae.yaml')

args = parser.parse_args()

# 打开 config.yaml 文件
with open(args.filename, 'r') as file:
    try:
        config = yaml.safe_load(file)
    except yaml.YAMLError as exc:
        print(exc)

# ----- 建立用于 log 输出的东西 ----- #
# 暂时不清楚这是干什么用的。。。
tt_logger = TestTubeLogger(
    save_dir=config['logging_params']['save_dir'],
    name=config['logging_params']['name'],
    debug=False,
    create_git_tag=False,
)

# ----- 固定随机数种子 ----- #
# 为了再复现性
# For reproducibility
torch.manual_seed(config['logging_params']['manual_seed'])
np.random.seed(config['logging_params']['manual_seed'])
cudnn.deterministic = True
cudnn.benchmark = False

# ----- 定义对应类型的模型 ----- #
# 这里是「函数名」作为了「字典值」
model = vae_models[config['model_params']['name']](**config['model_params'])

# ----- 建立当前模型对应的各个实验函数 ----- #
# 传入的参数有：模型和实验参数
experiment = VAEXperiment(model,
                          config['exp_params'])


# ----- 使用过 pytorch_lightning, 定义训练器 ----- #
# 暂没看懂具体是如何使用的
runner = Trainer(default_save_path=f"{tt_logger.save_dir}",
                 min_nb_epochs=1,
                 logger=tt_logger,
                 log_save_interval=100,
                 train_percent_check=1.,
                 val_percent_check=1.,
                 num_sanity_val_steps=5,
                 early_stop_callback = False,
                 **config['trainer_params'])

# 输出当前正在运行的模型名称
print(f"======= Training {config['model_params']['name']} =======")

# ----- 开始训练 ----- #
runner.fit(experiment)


# ----- 结束计时 ----- #
T_End = time.time()
T_Sum = T_End  - T_Start
T_Hour = int(T_Sum/3600)
T_Minute = int((T_Sum%3600)/60)
T_Second = round((T_Sum%3600)%60, 2)
print("程序运行时间: {}时{}分{}秒".format(T_Hour, T_Minute, T_Second))


print("程序已结束 ！")