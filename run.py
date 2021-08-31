# -------------------- 导入必要的包 -------------------- #
import yaml
import argparse
import numpy as np


import time

from models import *
from experiment import VAEXperiment
import torch.backends.cudnn as cudnn
from pytorch_lightning import Trainer
from pytorch_lightning.logging import TestTubeLogger


# ----- 开始计时 ----- #
T_Start = time.time()
print("程序开始运行 !")

parser = argparse.ArgumentParser(description='Generic runner for VAE models')
parser.add_argument('--config',  '-c',
                    dest="filename",
                    metavar='FILE',
                    help =  'path to the config file',
                    default='configs/vae.yaml')

args = parser.parse_args()
with open(args.filename, 'r') as file:
    try:
        config = yaml.safe_load(file)
    except yaml.YAMLError as exc:
        print(exc)


tt_logger = TestTubeLogger(
    save_dir=config['logging_params']['save_dir'],
    name=config['logging_params']['name'],
    debug=False,
    create_git_tag=False,
)

# For reproducibility
torch.manual_seed(config['logging_params']['manual_seed'])
np.random.seed(config['logging_params']['manual_seed'])
cudnn.deterministic = True
cudnn.benchmark = False

model = vae_models[config['model_params']['name']](**config['model_params'])
experiment = VAEXperiment(model,
                          config['exp_params'])

runner = Trainer(default_save_path=f"{tt_logger.save_dir}",
                 min_nb_epochs=1,
                 logger=tt_logger,
                 log_save_interval=100,
                 train_percent_check=1.,
                 val_percent_check=1.,
                 num_sanity_val_steps=5,
                 early_stop_callback = False,
                 **config['trainer_params'])

print(f"======= Training {config['model_params']['name']} =======")
runner.fit(experiment)


# ----- 结束计时 ----- #
T_End = time.time()
T_Sum = T_End  - T_Start
T_Hour = int(T_Sum/3600)
T_Minute = int((T_Sum%3600)/60)
T_Second = round((T_Sum%3600)%60, 2)
print("程序运行时间: {}时{}分{}秒".format(T_Hour, T_Minute, T_Second))


print("程序已结束 ！")