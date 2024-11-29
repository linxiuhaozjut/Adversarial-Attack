import numpy as np
import cv2
import random
import math

# import os
# import matplotlib.pyplot as plt
# from mmdet.apis import init_detector, inference_detector
# import mmcv

# from detect import detection
#import pandas as pd

# 图像上绘制patch，并保存为新文件
def img_camera_sticker_pattern(img, path, R, POP, x1, x2):  # 图像，保存路径，比例系数，位置角度，区域范围,t=pop_num

    # 修改3*个数
    b = 21
    # b = POP.shape[0]  # 贴纸数量，b有3个量值，位置+角度
    # print('b = ', b)
    for i in range(int(b/3)):  # 循环遍历每个贴纸，0~b/3
        if int(R * (x2 - x1)) > 0:
            w = int(R * (x2 - x1))
        else:
            w = 1
        h = w

        # x,y,角度换弧度
        x, y, angel = int(POP[3*i+0]), int(POP[3*i+1]), int(POP[3*i+2])
        angelPi = (angel / 180) * math.pi

        # 四个顶点坐标
        X1 = x + (w / 2) * math.cos(angelPi) - (h / 2) * math.sin(angelPi)
        Y1 = y + (w / 2) * math.sin(angelPi) + (h / 2) * math.cos(angelPi)

        X2 = x + (w / 2) * math.cos(angelPi) + (h / 2) * math.sin(angelPi)
        Y2 = y + (w / 2) * math.sin(angelPi) - (h / 2) * math.cos(angelPi)

        X3 = x - (w / 2) * math.cos(angelPi) + (h / 2) * math.sin(angelPi)
        Y3 = y - (w / 2) * math.sin(angelPi) - (h / 2) * math.cos(angelPi)

        X4 = x - (w / 2) * math.cos(angelPi) - (h / 2) * math.sin(angelPi)
        Y4 = y - (w / 2) * math.sin(angelPi) + (h / 2) * math.cos(angelPi)

        pts = np.array([(X1, Y1), (X2, Y2), (X3, Y3), (X4, Y4)], np.int32)
        cv2.fillPoly(img, [pts], (0, 0, 0))

    cv2.imwrite(path, img)  # 保存路径


# 老版的 用直线代替矩形
def img_camera_sticker_pattern1(img, path, R, POP, X1, X2):

    b = POP.shape[0]
    # print('b = ', b)
    for i in range(int(b/3)):
        lenth = int(R * (X2 - X1)) if int(R * (X2 - X1)) > 0 else 1
        # thickness = int(lenth / 2) if int(lenth / 2) > 0 else 1  # 宽度
        thickness = lenth
        x1, y1, angle = int(POP[3*i+0]), int(POP[3*i+1]), int(POP[3*i+2])
        x2, y2 = int(x1 + lenth * (math.sin(math.radians(angle)))), int(y1 + lenth * (math.cos(math.radians(angle))))
        cv2.line(img, (x1, y1), (x2, y2), (0, 0, 0), thickness)

    cv2.imwrite(path, img)


# 白色直线起（x1,y1）→（x2,y2）,厚度thickness
def img_camera_sticker_visible(img, path, R, POP, X1, X2):

    b = POP.shape[0]
    # print('b = ', b)
    for i in range(int(b/3)):
        lenth = int(R * (X2 - X1)) if int(R * (X2 - X1)) > 0 else 1
        thickness = int(lenth / 2) if int(lenth / 2) > 0 else 1  # 宽度
        x1, y1, angle = int(POP[3*i+0]), int(POP[3*i+1]), int(POP[3*i+2])
        x2, y2 = int(x1 + lenth * (math.sin(math.radians(angle)))), int(y1 + lenth * (math.cos(math.radians(angle))))
        cv2.line(img, (x1, y1), (x2, y2), (255, 255, 255), thickness)

    cv2.imwrite(path, img)

# 随机生成位置角度
def img_camera_sticker_pattern_infrared_random(img, path, R, patch_number, x1, y1, x2, y2):

    # print('R, patch_number, x1, y1, x2, y2, R * (x2 - x1) = ', R, patch_number, x1, y1, x2, y2, R * (x2 - x1))

    for i in range(patch_number):
        w = int(R * (x2 - x1)) if int(R * (x2 - x1)) > 0 else 1
        h = w

        x, y, angel = random.randint(x1, x2), random.randint(y1, y2), random.randint(0, 180)
        angelPi = (angel / 180) * math.pi

        X1 = x + (w / 2) * math.cos(angelPi) - (h / 2) * math.sin(angelPi)
        Y1 = y + (w / 2) * math.sin(angelPi) + (h / 2) * math.cos(angelPi)

        X2 = x + (w / 2) * math.cos(angelPi) + (h / 2) * math.sin(angelPi)
        Y2 = y + (w / 2) * math.sin(angelPi) - (h / 2) * math.cos(angelPi)

        X3 = x - (w / 2) * math.cos(angelPi) + (h / 2) * math.sin(angelPi)
        Y3 = y - (w / 2) * math.sin(angelPi) - (h / 2) * math.cos(angelPi)

        X4 = x - (w / 2) * math.cos(angelPi) - (h / 2) * math.sin(angelPi)
        Y4 = y - (w / 2) * math.sin(angelPi) + (h / 2) * math.cos(angelPi)

        pts = np.array([(X1, Y1), (X2, Y2), (X3, Y3), (X4, Y4)], np.int32)
        cv2.fillPoly(img, [pts], (0, 0, 0))

    cv2.imwrite(path, img)


# 生成指定数量贴纸
def img_camera_sticker_pattern_visible(img, path, R, POP, X1, X2):

    b = POP.shape[0]
    # print('b = ', b)
    for i in range(int(b/3)):
        lenth = int(R * (X2 - X1)) if int(R * (X2 - X1)) > 0 else 1
        thickness = int(lenth / 2) if int(lenth / 2) > 0 else 1  # 宽度
        x1, y1, angle = int(POP[3*i+0]), int(POP[3*i+1]), int(POP[3*i+2])
        x2, y2 = int(x1 + lenth * (math.sin(math.radians(angle)))), int(y1 + lenth * (math.cos(math.radians(angle))))
        cv2.line(img, (x1, y1), (x2, y2), (255, 255, 255), thickness)

    cv2.imwrite(path, img)


# 初始化，随机生成位置和角度
def initiation(POP_num, size, x1, y1, x2, y2):
    POP = np.zeros((POP_num, size))         # 贴纸数量+贴纸信息大小，100行，21列（7个补丁的每个，x,y,角度 ）
    # height = y2 - y1
    # width = x2 - x1
    # length = 0.1 * width
    # body_height = 0.14 * height
    # leg_height = 0.2 * height
    # body_y1 = y1 + body_height
    # body_y2 = y2 - leg_height
    # left = x1 + length
    # right = x2 - length
    for i in range(0, POP_num):             # 遍历贴纸
        for j in range(0, size):
            if j % 2 == 0:
                POP[i][j] = str(random.uniform(0.05, 0.75))
            if j % 2 == 1:
                POP[i][j] = str(random.uniform(0.15, 0.75))
            # if j % 3 == 2:
            #     POP[i][j] = random.randint(0, 180)
    return POP

# 补丁重叠判断
def is_valid_position(new_pos, positions, min_distance):
    for pos in positions:
        distance = math.sqrt((new_pos[0] - pos[0]) ** 2 + (new_pos[1] - pos[1]) ** 2)
        if distance < min_distance:
            return False
    return True

# 优化生成，补丁限位设置
def initiation_up(POP_num, size, x1, y1, x2, y2, side_length):
    POP = np.zeros((POP_num, size))  # 贴纸数量+贴纸信息大小，100行，21列（7个补丁的每个，x,y,角度 ）
    height = y2 - y1
    width = x2 - x1
    length = 0.1 * width
    body_height = 0.14 * height
    leg_height = 0.2 * height
    body_y1 = y1 + body_height
    body_y2 = y2 - leg_height
    left = x1 + length
    right = x2 - length
    A = rate(side_length)
    min_distance = int(A * width * math.sqrt(2))

    for i in range(POP_num):  # 遍历贴纸
        positions = []
        for j in range(0, size, 3):
            while True:
                x = random.uniform(left, right)
                y = random.uniform(body_y1, body_y2)
                if is_valid_position((x, y), positions, min_distance):
                    positions.append((x, y))
                    POP[i][j] = x
                    POP[i][j + 1] = y
                    POP[i][j + 2] = random.randint(0, 180)
                    break

    return POP

# 初始化，可优化
def initia(POP_num, size, x1, y1, x2, y2):
    POP = np.zeros((POP_num, size))  # 贴纸数量+贴纸信息大小
    for i in range(0, POP_num):      # 遍历贴纸
        for j in range(0, size):
            if j % 3 == 0:
                POP[i][j] = random.randint(x1, x2)
            if j % 3 == 1:
                POP[i][j] = random.randint(y1, y2)
            if j % 3 == 2:
                POP[i][j] = random.randint(0, 180)
    return POP

# 突变
def mutation(POP, Rm): # Rm变异系数
    # a为种群个数，b为个体信息（7*3）
    a, b = POP.shape
    POP1 = np.zeros((a, b))
    for i in range(a):
        for j in range(b):
            POP1[i][j] = POP[i][j]
    x, y = 0, 0

    # 变异
    for i in range(a):
        for tag in range(0, 100):
            x, y = random.randint(0, a-1), random.randint(0, a-1)
            if x != i and y != i and x != y:
                break
        # print('i, x, y = ', i, x, y)

        # 计算差值得到变异量
        for j in range(b):
            POP[i][j] = POP1[i][j] + Rm * (POP1[x][j] - POP1[y][j])
    # print('POP = ', POP)

    # # 取整
    # for i in range(a):
    #     for j in range(b):
    #         POP[i][j] = int(POP[i][j])
    # # print('POP = ', POP)

    # Clip，修正
    for i in range(0, a):
        for j in range(0, b):
            if j % 2 == 0:
                if POP[i][j] < 0.05 or POP[i][j] > 0.75:
                    POP[i][j] = random.uniform(0.05, 0.75)
            if j % 2 == 1:
                if POP[i][j] < 0.15 or POP[i][j] > 0.75:
                    POP[i][j] = random.uniform(0.15, 0.75)
            # if j % 3 == 2:
            #     if POP[i][j] < 0 or POP[i][j] > 180:
            #         POP[i][j] = random.randint(0, 180)
    # print('POP = ', POP)

    return POP

# 交叉操作
def crossover(POP, POP_f, Rc):
    a, b = POP.shape  # 个体数量，基因数量
    for i in range(a):
        for j in range(b):
            tag = random.randint(1, 10)
            # print('i, j, tag = ', i, j, tag)
            if tag <= 10 * Rc:
                POP[i][j] = POP_f[i][j]
    return POP

# 选择适应度【位置，父代个体，位置适应度，父代个体适应度】
def selection(POP, POP_f, POP_conf, POP_conf_f):
    a, b = POP.shape
    for i in range(a):
        print("\n子代：" + str(POP_conf[i]) + "||父代：" + str(POP_conf_f[i]))
        if POP_conf[i] > POP_conf_f[i]:
            for j in range(b):
                POP[i][j] = POP_f[i][j]
                POP_conf[i] = POP_conf_f[i]
            print("父代")
        else:
            print("子代")
    return POP, POP_conf

# 选择策略+模拟退火策略
def selection_up1(POP, POP_f, POP_conf, POP_conf_f):
    a, b = POP.shape
    # 更新最优解
    for i in range(a):
        # 模拟退火:新个体与当前的差值
        delta_f = POP_conf_f[0][i] - POP_conf[0][i]
        print("\n父-子适应度差值：" + str(delta_f) + "||子代：" + str(POP_conf[0][i]) + "||父代：" + str(POP_conf_f[0][i]))
        # 子代高于父代且概率大于设定
        if POP_conf[0][i] > POP_conf_f[0][i]:
            print("选择父代")
            for j in range(b):
                POP[i][j] = POP_f[i][j]
        else:
            print("选择子代")
    return POP

def selection_up2(POP, POP_f, POP_conf, POP_conf_f, tem):
    a, b = POP.shape
    # 更新最优解
    for i in range(a):
        # 模拟退火:新个体与当前的差值
        delta_f = POP_conf[0][i] - POP_conf_f[0][i]
        randomp = np.random.rand()
        setp = np.exp(-delta_f / tem)
        print("\n子-父适应度差值：" + str(delta_f) + "||子代：" + str(POP_conf[0][i]) + "||父代：" + str(POP_conf_f[0][i]))
        print("条件概率：" + str(randomp) + "||门限概率：" + str(setp))
        # 新解比当前解更优，则接受新解；
        # 如果新解不如当前解，则以一定概率接受新解
        if POP_conf[0][i] < POP_conf_f[0][i] or randomp < setp:
            print("选择子代")
        else:
            print("选择父代")
            for j in range(b):
                POP[i][j] = POP_f[i][j]
    return POP

# 边框比例系数R
def rate(side_length):
    R = 0
    if side_length == 0:
        R = 0.06
    if side_length == 1:
        R = 0.08
    if side_length == 2:
        R = 0.1
    if side_length == 3:  # 原0.12
        R = 0.12
    if side_length == 4:
        R = 0.14
    if side_length == 5:
        R = 0.16

    return R