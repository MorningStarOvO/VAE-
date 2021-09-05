"""
    本代码用于: 观察实验结果的损失值、重构损失、KLD 值曲线图
    创建时间: 2021 年 9 月 2 日
    创建人: MorningStar
    最后一次修改时间: 2021 年 9 月 2 日
    具体步骤:
        step1: 绘制损失值图像
        step2: 绘制重构损失值图像
        step3: 绘制 KLD 值图像
        step4: 绘制三条曲线在一张图的图像
"""
# -------------------- 导入必要的包 -------------------- #
# ----- 系统操作相关 ----- #
import time
import os


# ----- 文件读取相关 ----- #
import csv 


# ----- 数据处理相关 ----- #
import numpy as np
import scipy.io as scio


# ----- 绘制图片相关 ----- #
import matplotlib 
import matplotlib.pyplot as plt


# ----- 创建字体对象 ----- #
from matplotlib.font_manager import FontProperties
font = FontProperties(fname='SimHei.ttf', size=16)

# -------------------- 设置常量参数 -------------------- #
# ----- 读取文件相关 ----- #
PATH_LOSS = "logs/VanillaVAE/" + "version_11/" + "metrics.csv"


# ----- 保存文件相关 ----- #
PATH_SAVE_PIC = "output_qxy/VanillaVAE/" + "version_11/"


# -------------------- 函数实现 -------------------- #


# -------------------- 主函数运行 -------------------- #
if __name__ == '__main__':
    # ----- 开始计时 ----- #
    T_Start = time.time()
    print("程序开始运行 !")

    # ---------- step0: 绘图前的准备 ---------- #
    # ----- 读取文件 ----- #
    csv_reader = csv.reader(open(PATH_LOSS))
    
    # ----- 保存各个 loss ----- #
    list_loss = []
    list_recon = []
    list_kld = []
    list_iter = []
    i = 0
    j = 0
    for line in csv_reader:
        # ----- 获取各个值 ----- #
        temp_epoch = line[4]

        i += 1

        # ----- 记录数据 ----- #
        if i > 4 and temp_epoch == '':

            # 获取他们的值        
            temp_loss = float(line[0])
            temp_recon_loss = float(line[1])
            temp_kld = float(line[2])

            # print(temp_loss) 
            list_loss.append(temp_loss)
            list_recon.append(temp_recon_loss)
            list_kld.append(temp_kld)
            list_iter.append(j)

            j += 1

    # print(list_loss)


    # ----- 创建保存图片的路径 ----- #
    if not os.path.exists(PATH_SAVE_PIC):
        os.makedirs(PATH_SAVE_PIC)

    # ---------- step1: 绘制损失值图像 ---------- #
    # 暂不画    
 
    # ---------- step2: 绘制重构损失值图像 ---------- #
    # 暂不画

    # ---------- step3: 绘制 KLD 值图像 ---------- #
    plt.figure()
    plt.plot(list_kld[0:3000], color='skyblue', label="KLD") 

    plt.xlabel("iter", fontproperties=font)
    plt.ylabel("kld 值", fontproperties=font)
    plt.title("KLD Analysis", fontproperties=font)
    plt.legend() # 显示图例
    plt.savefig(os.path.join(PATH_SAVE_PIC, "KLD Analysis.png"))
    

    # ---------- step4: 绘制三条曲线在一张图的图像 ---------- #
    plt.figure()
    plt.plot(list_loss[0:3000], color='red', label="loss")
    plt.plot(list_recon[0:3000], color='green', label="recon loss")
    # plt.plot(list_iter, list_kld, color='skyblue', label="KLD/10") 

    plt.xlabel("iter", fontproperties=font)
    plt.ylabel("loss", fontproperties=font)
    plt.title("Result Analysis", fontproperties=font)
    plt.legend() # 显示图例
    plt.savefig(os.path.join(PATH_SAVE_PIC, "Result Analysis.png"))
    # plt.show()

    # ----- 结束计时 ----- #
    T_End = time.time()
    T_Sum = T_End  - T_Start
    T_Hour = int(T_Sum/3600)
    T_Minute = int((T_Sum%3600)/60)
    T_Second = round((T_Sum%3600)%60, 2)
    print("程序运行时间: {}时{}分{}秒".format(T_Hour, T_Minute, T_Second))

    
    print("程序已结束 ！")
    