from tqdm import tqdm
import numpy as np
import torch
import torch.nn
from PIL import Image
import torchvision.transforms as transforms
import math
from infer import infer_main
from infer_patch import infer_patch
from single_im_conf import single_im_conf
import shutil
import os
from DEfunction import mutation, crossover, selection, initiation
from swinfuse import swinfuse, single_swinfuse
from densefuse import densefuse, single_densefuse

toTensor = transforms.ToTensor()
toPIL = transforms.ToPILImage() #这个函数可以将张量转为PIL图片，由小数转为0-255之间的像素值

def draw_single_im(file, xxyy, patchnum, rate, h, col, prate, save_dir):

    predtxt = open("./infer/m3fd/DE/meta/pred.txt",
                   "w")  # “w”表示写入 “w+”表示读取和写入
    predtxt.write(file)
    predtxt.close()
    file = file.replace('.jpg', '')
    for choose in range(2):
        if choose == 0:
            file_name = 'vi/' + file
        if choose == 1:
            file_name = 'ir/' + file

        img_path = "./infer/base/" + file_name + ".jpg"
        img = Image.open(img_path)
        img_tensor = toTensor(img)

        width = float(xxyy[1]) - float(xxyy[0])
        height = float(xxyy[3]) - float(xxyy[2])


        num = 20
        maxxs = num/prate
        # 正常width>67像素时
        if width >= maxxs:
            w_block_size = prate * width / num
            lnum = num
        # width<67像素时，以width长度像素作为补丁长度
        else:
            w_block_size = 1
            lnum = int(width * prate)
        for pn in range(patchnum):
            for i in range(lnum):
                for j in range(lnum):
                    xmin = math.floor(
                        i * w_block_size + float(xxyy[0]) + float(rate[pn * 2 + 0]) * float(width))
                    ymin = math.floor(
                        j * w_block_size + float(xxyy[2]) + float(rate[pn * 2 + 1]) * float(height))
                    xmax = math.floor(i * w_block_size + float(xxyy[0]) + float(rate[pn * 2 + 0]) * float(
                        width) + w_block_size)
                    ymax = math.floor(j * w_block_size + float(xxyy[2]) + float(rate[pn * 2 + 1]) * float(
                        height) + w_block_size)
                    if choose == 1:
                        color = math.floor(h[i * num + j:i * num + j + 1] * 255) + 1
                        # color = z[i:i + 1, j:j + 1, 0:1]*255 + 1
                        img_tensor[:, ymin:ymax, xmin:xmax] = color
                    else:
                        color = col[0:1, i:i + 1, j:j + 1]
                        img_tensor[0:1, ymin:ymax, xmin:xmax] = color
                        color = col[1:2, i:i + 1, j:j + 1]
                        img_tensor[1:2, ymin:ymax, xmin:xmax] = color
                        color = col[2:3, i:i + 1, j:j + 1]
                        img_tensor[2:3, ymin:ymax, xmin:xmax] = color
        img_PIL = toPIL(img_tensor)  # 张量tensor转换为图片
        img_PIL.save(
            save_dir + file_name + ".jpg")  # 保存图片；img_PIL.show()可以直接显示图片



def DEimprove(image_path, epoch, POP_num, patchnum, prate, h, col):
    save_dir = './infer/m3fd/DE/'
    Rm = 0.5  # 变异率
    Rc = 0.6  # 交叉率
    size = 6*2
    for file in tqdm(os.listdir(image_path)):
        source_path = './infer/m3fd/DE/labels/' + file.replace('.jpg', '.txt')
        destination_path = './runs/tmp/attacktarLLVIP/DE/labels'
        shutil.copy(source_path, destination_path)
        s = './runs/tmp/attacktarLLVIP/DE/labels/' + file.replace('.jpg', '.txt')
        d = './runs/tmp/attacktarLLVIP/DE/labels/DE.txt'
        os.rename(s, d)
        name = file.replace('.jpg', '')

        xy_path = "./infer/m3fd/img_xy/" + name + "_xy.txt"
        patch_rate = []
        with open(xy_path, "r") as xy_txt:

            for targetnum in xy_txt.readlines():
                xxyy = targetnum[:-1].split(' ')
                #best_conf = 1

                # 种群初始化
                POP = initiation(POP_num, size, xxyy[0], xxyy[2], xxyy[1], xxyy[3])  # 【100,7*3】
                POP_conf = np.ones(POP_num)  # 1行100列的二维数组，数组内容为1

                for i in range(POP_num):
                    print("轮数：",i)
                    #rate = torch.rand(patchnum*2)*(1-prate)
                    draw_single_im(file, xxyy, patchnum, POP[i], h, col, prate, save_dir)
                    infer_main()
                    #swinfuse()
                    #densefuse()
                    s = './runs/tmp/attacktarLLVIP/DE/images/' + file
                    d = './runs/tmp/attacktarLLVIP/DE/images/DE.jpg'
                    os.rename(s, d)
                    conf = single_im_conf()
                    POP_conf[i] = conf

                # 父代个体传参
                POP_f = initiation(POP_num, size, xxyy[0], xxyy[2], xxyy[1], xxyy[3])  # 父代个体特征
                POP_conf_f = np.ones(POP_num)  # 父代个体置信度
                for i in range(POP_num):
                    for j in range(size):
                        POP_f[i][j] = POP[i][j]
                for i in range(POP_num):
                    POP_conf_f[i] = POP_conf[i]

                # DE优化
                for step in range(epoch):  # 迭代次数
                    print("step:",step)
                    POP = mutation(POP, Rm)
                    print("第" + str(step) + "次突变",POP)
                    POP = crossover(POP, POP_f, Rc)
                    print("第" + str(step) + "次交叉",POP)

                    for i in range(POP_num):
                        print("pop_num：", i)
                        # rate = torch.rand(patchnum*2)*(1-prate)
                        draw_single_im(file, xxyy, patchnum, POP[i], h, col, prate, save_dir)
                        infer_main()
                        #swinfuse()
                        #densefuse()
                        s = './runs/tmp/attacktarLLVIP/DE/images/' + file
                        d = './runs/tmp/attacktarLLVIP/DE/images/DE.jpg'
                        os.rename(s, d)
                        conf = single_im_conf()
                        POP_conf[i] = conf

                    POP, POP_conf = selection(POP, POP_f, POP_conf, POP_conf_f)
                    print("第" + str(step) + "次选择")

                #最后一次优化结束，选出最优的该目标框pop
                a, b = POP.shape
                best_conf = 1
                best_pop = []
                for i in range(a):
                    if POP_conf[i] < best_conf:
                        best_pop = POP[i]
                        best_conf = POP_conf[i]
                patch_rate.append(best_pop)
        rate_txt = open(
            "./infer/m3fd/rate/" + name + "_rate.txt", "w")
        print("写入的：",str(patch_rate))
        rate_txt.write(str(patch_rate))
        rate_txt.close()


def read_rate(name):
    rate_txt = open(
        './infer/m3fd/rate/' + name + '_rate.txt', 'r')

    # 读取每一行数据并将其存储到data数组中
    data = []
    for line in rate_txt.readlines():
        line = line.replace('[', '').replace('(', '').replace(']', '').replace('),', '+').replace(' ', '').replace(
            'array', '').replace(')', '')
        #print("去除符号：", line)
        line = line.split('+')
        object_num = len(line)
        for i in range(len(line)):
            data.append(line[i])

    # 关闭txt文件
    rate_txt.close()

    item = np.zeros((object_num, 12))
    # 打印每一行数据
    for i in range(len(data)):
        temp = data[i].split(',')
        for j in range(len(temp)):
            item[i][j] = (temp[j])

    # a, b = item.shape
    # for i in range(a):
    #     for j in range(b):
    #         print("分行：", item[i][j])
    #print("item:",item)
    return item

def tf_read_rate(name):
    rate_txt = open(
        './infer/m3fd/tf_rate/' + name + '_rate.txt', 'r')

    total = rate_txt.read().replace('\n', '').replace('array([','').replace('])', '').replace('[', '').replace(']', '').replace(' ', '')
    #print(total)
    a_rate = total.split(',')
    object_num = int(len(a_rate)/12)
    #print("object_num",object_num)
    item = np.zeros((object_num, 12))
    for i in range(object_num):
        for j in range(12):
            item[i][j] = float(a_rate[i*12+j])
    # 读取每一行数据并将其存储到数组中

    # 关闭txt文件
    rate_txt.close()


    #打印每一行数据
    for i in range(object_num):
        for j in range(12):
            print("item:",item[i][j])
    return item


def rotate_coordinates(x, y, angle, center_x, center_y):
    """
    计算旋转后的坐标(x', y')，旋转角度为angle（弧度），旋转中心为(center_x, center_y)
    """
    # 旋转角度为弧度
    cos_angle = np.cos(angle)
    sin_angle = np.sin(angle)

    # 平移坐标系，使得旋转中心在原点
    x_translated = x - center_x
    y_translated = y - center_y

    # 应用旋转矩阵
    x_rot = cos_angle * x_translated - sin_angle * y_translated + center_x
    y_rot = sin_angle * x_translated + cos_angle * y_translated + center_y

    return x_rot, y_rot

def draw_patch(file, patchnum, h, col, prate, rate, save_dir):
    predtxt = open("./infer/m3fd/single/meta/pred.txt",
                   "w")  # “w”表示写入 “w+”表示读取和写入
    predtxt.write(file)
    predtxt.close()
    name = file.replace('.jpg', '')
    for choose in range(2):
        if choose == 0:
            file_name = 'vi/' + name
        if choose == 1:
            file_name = 'ir/' + name

        img_path = "./infer/base/" + file_name + ".jpg"

        # img = cv2.imread(img_path)
        img = Image.open(img_path)
        img_tensor = toTensor(img)

        img_w = './infer/m3fd/img_xy/' + name + '_xy.txt'

        # ======================计算检测框的长宽
        num = 20
        maxxs = num / prate
        on = 0

        with open(img_w, "r") as fp:
            # 以空格划分

            # 读取坐标xy，计算二维码大小以及位置
            for line in fp.readlines():
                xxyy = line[:-1].split(' ')
                # print(contline)
                width = float(xxyy[1]) - float(xxyy[0])
                height = float(xxyy[3]) - float(xxyy[2])

                # 正常width>67像素时
                if width >= maxxs:
                    w_block_size = prate * width / num
                    lnum = num
                # width<67像素时，以width长度像素作为补丁长度
                else:
                    w_block_size = 1
                    lnum = int(width * prate)

#EOT褶皱
#===============================================================================================================
                #正常的画法
                #====================================
                for pn in range(patchnum):
                    for i in range(lnum):
                        for j in range(lnum):
                            xmin = math.floor(
                                i * w_block_size + float(xxyy[0]) + float(rate[on][pn * 2 + 0]) * float(width))
                            ymin = math.floor(
                                j * w_block_size + float(xxyy[2]) + float(rate[on][pn * 2 + 1]) * float(height))
                            xmax = math.floor(i * w_block_size + float(xxyy[0]) + float(rate[on][pn * 2 + 0]) * float(
                                width) + w_block_size)
                            ymax = math.floor(j * w_block_size + float(xxyy[2]) + float(rate[on][pn * 2 + 1]) * float(
                                height) + w_block_size)
                #======================================
                # 褶皱,t为程度
                # ====================================
                # t = 6
                # for pn in range(patchnum):
                #     for i in range(lnum):
                #         if i % 2 == 1:
                #             eot1 = t
                #         else:
                #             eot1 = 0
                #         for j in range(lnum):
                #             xmin = math.floor(
                #                 i * w_block_size + float(xxyy[0]) + float(rate[on][pn * 2 + 0]) * float(
                #                     width))
                #             ymin = math.floor(
                #                 j * w_block_size + float(xxyy[2]) + float(rate[on][pn * 2 + 1]) * float(
                #                     height) - eot1)
                #             xmax = math.floor(
                #                 i * w_block_size + float(xxyy[0]) + float(rate[on][pn * 2 + 0]) * float(
                #                     width) + w_block_size)
                #             ymax = math.floor(
                #                 j * w_block_size + float(xxyy[2]) + float(rate[on][pn * 2 + 1]) * float(
                #                     height) + w_block_size - eot1)
                # ======================================
# ===============================================================================================================

#EOT角度
# ===============================================================================================================
#                 #45度
#                 for pn in range(patchnum):
#                     for i in range(lnum):
#                         for j in range(lnum):
#                             xmin = math.floor(
#                                 (i + j) * w_block_size + float(xxyy[0]) + float(rate[on][pn * 2 + 0]) * float(
#                                     width))
#                             ymin = math.floor(
#                                 (j-i) * w_block_size + float(xxyy[2]) + float(rate[on][pn * 2 + 1]) * float(
#                                     height))
#                             xmax = math.floor(
#                                 (i + j) * w_block_size + float(xxyy[0]) + float(rate[on][pn * 2 + 0]) * float(
#                                     width) + 2 * w_block_size)
#                             ymax = math.floor(
#                                 (j-i) * w_block_size + float(xxyy[2]) + float(rate[on][pn * 2 + 1]) * float(
#                                     height) + w_block_size)

                # 135度
                # for pn in range(patchnum):
                #     for i in range(lnum):
                #         for j in range(lnum):
                #             xmin = math.floor(
                #                 (i - j) * w_block_size + float(xxyy[0]) + float(
                #                     rate[on][pn * 2 + 0]) * float(
                #                     width))
                #             ymin = math.floor(
                #                 (j + i) * w_block_size + float(xxyy[2]) + float(
                #                     rate[on][pn * 2 + 1]) * float(
                #                     height))
                #             xmax = math.floor(
                #                 (i - j) * w_block_size + float(xxyy[0]) + float(
                #                     rate[on][pn * 2 + 0]) * float(
                #                     width) + 2 * w_block_size)
                #             ymax = math.floor(
                #                 (j + i) * w_block_size + float(xxyy[2]) + float(
                #                     rate[on][pn * 2 + 1]) * float(
                #                     height) + w_block_size)
# ===============================================================================================================

                            #EOT亮度
                            light = 0
                            if choose == 1:
                                color = math.floor(h[i * num + j:i * num + j + 1] * 255) + 1 - light
                                # color = z[i:i + 1, j:j + 1, 0:1]*255 + 1
                                img_tensor[:, ymin:ymax, xmin:xmax] = color
                            else:
                                color = col[0:1, i:i + 1, j:j + 1] - light
                                img_tensor[0:1, ymin:ymax, xmin:xmax] = color
                                color = col[1:2, i:i + 1, j:j + 1]- light
                                img_tensor[1:2, ymin:ymax, xmin:xmax] = color
                                color = col[2:3, i:i + 1, j:j + 1]- light
                                img_tensor[2:3, ymin:ymax, xmin:xmax] = color
                on = on+1
        img_PIL = toPIL(img_tensor)  # 张量tensor转换为图片
        img_PIL.save(save_dir + file_name + ".jpg")  # 保存图片；img_PIL.show()可以直接显示图片



def patch_draw(image_path, h ,col, patchnum, prate):
    save_dir = './infer/m3fd/single/'
    for file in tqdm(os.listdir(image_path)):
        name = file.replace('.jpg', '')
        print("名字:", name)
        rate = read_rate(name)
        #rate = tf_read_rate(name)
        draw_patch(file, patchnum, h, col, prate, rate, save_dir)
        infer_patch()
        #single_swinfuse(file)
        #single_densefuse()


#tf_read_rate('010004')
# image_path = '/data/Newdisk2/linxiuhao/old_project/project/yolov3-master-tar-updata/infer/m3fd/DE/img'
# h = torch.load('/data/Newdisk2/linxiuhao/old_project/project/yolov3-master-tar-updata/infer/m3fd/h_col/huidu_best.pt')  # best
# col = torch.load('/data/Newdisk2/linxiuhao/old_project/project/yolov3-master-tar-updata/infer/m3fd/h_col/col_best.pt')  # best
# patch_draw(image_path, h, col, 6, 0.25)















