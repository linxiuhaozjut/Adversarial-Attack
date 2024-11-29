import os
from PIL import Image
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
from dense_net import DenseFuse
from tqdm import tqdm
import torch
from dense_config import Config
def densefuse():
    model_path = Config.model_path_gray
    ir_path = Config.test_Inf_img
    vi_path = Config.test_Vis_img
    save_dir = "./runs/tmp/attacktarLLVIP/DE/images"
    # 加载模型

    model = DenseFuse(Config.in_channel, Config.in_channel, fusion_strategy=Config.fusion_strategy)  # 这里需要你自行定义网络的参数
    model.load_state_dict(torch.load(model_path))

    if Config.cuda and torch.cuda.is_available():
        model = model.cuda()

    to_tensor = transforms.ToTensor()
    #ir_images = sorted(os.listdir(ir_path), key=lambda x: int(x.split('.')[0]))
    #vi_images = sorted(os.listdir(vi_path), key=lambda x: int(x.split('.')[0]))
    #=====================================
    # 打开txt文件
    file = open('./infer/m3fd/DE/meta/pred.txt', 'r')
    # 读取每一行数据并将其存储到data数组中
    data = []
    for line in file.readlines():
        data.append(line.strip())
    # 关闭txt文件
    file.close()
    ir_images = data
    vi_images = data
    #=====================================
    # 遍历测试数据
    for i, (ir_image_file, vi_image_file) in tqdm(enumerate(zip(ir_images, vi_images)), total=len(ir_images)):
        #print('Processing image pair {}/{}'.format(i+1, len(ir_images)))
        # 加载并转换图像
        ir_image = Image.open(os.path.join(ir_path, ir_image_file)).convert('L')
        vi_image = Image.open(os.path.join(vi_path, vi_image_file)).convert('L')
        ir_tensor = to_tensor(ir_image).unsqueeze(0)
        vi_tensor = to_tensor(vi_image).unsqueeze(0)

        if torch.cuda.is_available():
            ir_tensor = ir_tensor.cuda()
            vi_tensor = vi_tensor.cuda()

        # 前向传播
        outputs = model(ir_tensor, vi_tensor)

        # 将输出转换为图像并保存
        output_image = transforms.ToPILImage()(outputs.cpu().data[0])
        output_image.save(os.path.join(save_dir, ir_image_file))


def single_densefuse():
    model_path = Config.model_path_gray
    ir_path = Config.single_Inf_img
    vi_path = Config.single_Vis_img
    save_dir = "./runs/tmp/attacktarLLVIP/infer/images"
    # 加载模型

    model = DenseFuse(Config.in_channel, Config.in_channel, fusion_strategy=Config.fusion_strategy)  # 这里需要你自行定义网络的参数
    model.load_state_dict(torch.load(model_path))

    if Config.cuda and torch.cuda.is_available():
        model = model.cuda()

    to_tensor = transforms.ToTensor()
    #ir_images = sorted(os.listdir(ir_path), key=lambda x: int(x.split('.')[0]))
    #vi_images = sorted(os.listdir(vi_path), key=lambda x: int(x.split('.')[0]))
    #=====================================
    # 打开txt文件
    file = open('./infer/m3fd/single/meta/pred.txt', 'r')
    # 读取每一行数据并将其存储到data数组中
    data = []
    for line in file.readlines():
        data.append(line.strip())
    # 关闭txt文件
    file.close()
    ir_images = data
    vi_images = data
    #=====================================
    # 遍历测试数据
    for i, (ir_image_file, vi_image_file) in tqdm(enumerate(zip(ir_images, vi_images)), total=len(ir_images)):
        #print('Processing image pair {}/{}'.format(i+1, len(ir_images)))
        # 加载并转换图像
        ir_image = Image.open(os.path.join(ir_path, ir_image_file)).convert('L')
        vi_image = Image.open(os.path.join(vi_path, vi_image_file)).convert('L')
        ir_tensor = to_tensor(ir_image).unsqueeze(0)
        vi_tensor = to_tensor(vi_image).unsqueeze(0)

        if torch.cuda.is_available():
            ir_tensor = ir_tensor.cuda()
            vi_tensor = vi_tensor.cuda()

        # 前向传播
        outputs = model(ir_tensor, vi_tensor)

        # 将输出转换为图像并保存
        output_image = transforms.ToPILImage()(outputs.cpu().data[0])
        output_image.save(os.path.join(save_dir, ir_image_file))


# 使用模型

single_densefuse()