"""
    本代码用于: 观察 celebA 数据集提供的标注信息
    创建时间: 2021 年 8 月 30 日
    创建人: MorningStar
    最后一次修改时间: 2021 年 8 月 30 日
    具体步骤:
        step1: 观察 list_landmarks_align_celeba.txt
        step2: 观察 list_landmarks_celeba.txt
        step3: 观察 identity_CelebA.txt
        step4: 观察 list_attr_celeba.txt
        step5: 观察 list_bbox_celeba.txt
        step6: 查看 bbox 所代表的信息
        step7: 观察 img_align_celeba_png 的图片及其相关信息
        step8: 观察 img_align_celeba 的图片及相关信息
"""
# -------------------- 导入必要的包 -------------------- #
# ----- 系统操作相关 ----- #
import time
import os


# ----- 文件读取相关 ----- #
from PIL import Image, ImageDraw, ImageFont 


# ----- 数据处理相关 ----- #



# -------------------- 设置常量参数 -------------------- #
# ----- 读取文件相关 ----- #
PATH_ANN = "data/celebA/Anno"

PATH_PIC_ALIGN = "data/celebA/celeba/img_align_celeba_png"
PATH_PIC = "data/celebA/celeba/img_celeba"
PATH_PIC_SMALL = "data/celebA/celeba/img_align_celeba"

# ----- 保存文件相关 ----- #
PATH_SAVE_BBOX = "output_qxy"


# -------------------- 函数实现 -------------------- #
# ----- 画图片的矩形框函数实现 ----- #
def Draw_Pic_Rect(path_read, bbox, path_save):
    """
    仅保存一个 bbox 的矩形框
    @path_read: 读取图片的路径
    @bbox: bbox 的值，为 [x, y, w, d]
    @path_save: 存储图片的路径
    """
    # ----- 读取图片 ----- #
    temp_img = Image.open(path_read)
    img_draw = ImageDraw.ImageDraw(temp_img)
        
    # ----- 绘制矩形框 ----- #
    x = bbox[0]
    y = bbox[1]
    w = bbox[2]
    d = bbox[3]
    img_draw.rectangle((x, y, x+w, y+d), fill=None, outline="red", width=5)

    temp_img.save(path_save)


# -------------------- 主函数运行 -------------------- #
if __name__ == '__main__':
    # ----- 开始计时 ----- #
    T_Start = time.time()
    print("程序开始运行 !")


    # ---------- step1: 观察 list_landmarks_align_celeba.txt ---------- #
    # ----- 读取并输出 TXT 文件信息 ----- #
    f = open(os.path.join(PATH_ANN, "list_landmarks_align_celeba.txt"), "r")
    lines = f.readlines()

    print("================================================")
    print("list_landmarks_align_celeba.txt:")
    print("================================================")

    i = 0
    for temp_line in lines:
        print(temp_line)

        i += 1
        if i > 2:
            break


    # ---------- step2: 观察 list_landmarks_celeba.txt ---------- #
    # ----- 读取并输出 TXT 文件信息 ----- #
    f = open(os.path.join(PATH_ANN, "list_landmarks_celeba.txt"), "r")
    lines = f.readlines()

    print("================================================")
    print("list_landmarks_celeba.txt")
    print("================================================")

    i = 0
    for temp_line in lines:
        print(temp_line)

        i += 1
        if i > 2:
            break

    



    # ---------- step3: 观察 identity_CelebA.txt ---------- #
    # ----- 读取并输出 TXT 文件信息 ----- #
    f = open(os.path.join(PATH_ANN, "identity_CelebA.txt"), "r")
    lines = f.readlines()

    print("================================================")
    print("identity_CelebA.txt")
    print("================================================")

    i = 0
    for temp_line in lines:
        print(temp_line)

        i += 1
        if i > 2:
            break


    # ---------- step4: 观察 list_attr_celeba.txt ---------- #
    # ----- 读取并输出 TXT 文件信息 ----- #
    f = open(os.path.join(PATH_ANN, "list_attr_celeba.txt"), "r")
    lines = f.readlines()

    print("================================================")
    print("list_attr_celeba.txt")
    print("================================================")

    i = 0
    for temp_line in lines:
        print(temp_line)

        i += 1
        if i > 2:
            break


    
    # ---------- step5: 观察 list_bbox_celeba.txt ---------- #
    # ----- 读取并输出 TXT 文件信息 ----- #
    f = open(os.path.join(PATH_ANN, "list_bbox_celeba.txt"), "r")
    lines = f.readlines()

    print("================================================")
    print("list_bbox_celeba.txt")
    print("================================================")

    i = 0
    for temp_line in lines:
        print(temp_line)

        i += 1
        if i > 3:
            break


    # ---------- step6: 查看 bbox 所代表的信息 ---------- #
    # 标签信息：000001.jpg  95  71  226  313
    # 标签信息：000002.jpg  72  94  221  306
    # ----- 得到基本信息 ----- #
    # path_read = os.path.join(PATH_PIC, "000001.jpg")
    # bbox = [95, 71, 226, 313]
    # path_save = os.path.join(PATH_SAVE_BBOX, "000001-bbox.jpg")

    path_read = os.path.join(PATH_PIC, "000002.jpg")
    bbox = [72, 94, 221, 306]
    path_save = os.path.join(PATH_SAVE_BBOX, "000002-bbox.jpg")

    # ----- 绘图 ----- #
    Draw_Pic_Rect(path_read, bbox, path_save)


    # ---------- step7: 观察 img_align_celeba_png 的图片及其相关信息 ---------- #
    img_align = Image.open(os.path.join(PATH_PIC_ALIGN, "000001.png")) 
    img_align.save(os.path.join(PATH_SAVE_BBOX, "000001.png"))
    print("----------------------------------------")
    print("图片的尺寸为: ", img_align.size) # (178, 218)


    # ---------- step8: 观察 img_align_celeba 的图片及相关信息 ---------- #
    img_small = Image.open(os.path.join(PATH_PIC_SMALL, "000001.jpg"))
    img_small.save(os.path.join(PATH_SAVE_BBOX, "000001_small.jpg"))
    print("----------------------------------------")
    print("图片的尺寸为: ", img_small.size) # (178, 218)


    # ----- 结束计时 ----- #
    T_End = time.time()
    T_Sum = T_End  - T_Start
    T_Hour = int(T_Sum/3600)
    T_Minute = int((T_Sum%3600)/60)
    T_Second = round((T_Sum%3600)%60, 2)
    print("程序运行时间: {}时{}分{}秒".format(T_Hour, T_Minute, T_Second))

    
    print("程序已结束 ！")