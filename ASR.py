import torch
import os
from torchvision import transforms
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np
from PIL import Image, ImageOps

# 加载YOLOv3模型
from models.experimental import attempt_load
from utils.general import non_max_suppression

# 自定义图像尺寸调整函数
def letterbox(img, new_shape=(640, 480), color=(114, 114, 114), auto=True, scaleFill=False, scaleup=True):
    shape = img.size  # current shape [width, height]
    if isinstance(new_shape, int):
        new_shape = (new_shape, new_shape)

    # Scale ratio (new / old)
    r = min(new_shape[1] / shape[1], new_shape[0] / shape[0])
    if not scaleup:  # only scale down, do not scale up (for better test mAP)
        r = min(r, 1.0)

    # Compute padding
    ratio = r, r  # width, height ratios
    new_unpad = int(round(shape[0] * r)), int(round(shape[1] * r))
    dw, dh = new_shape[0] - new_unpad[0], new_shape[1] - new_unpad[1]  # wh padding
    if auto:  # minimum rectangle
        dw, dh = np.mod(dw, 64), np.mod(dh, 64)  # wh padding
    elif scaleFill:  # stretch
        dw, dh = 0.0, 0.0
        new_unpad = new_shape
        ratio = new_shape[0] / shape[0], new_shape[1] / shape[1]  # width, height ratios

    dw /= 2  # divide padding into 2 sides
    dh /= 2

    if shape != new_unpad:  # resize
        img = img.resize(new_unpad, Image.BICUBIC)
    padding = (int(dw), int(dh), int(dw), int(dh))
    img = ImageOps.expand(img, padding, fill=color)  # add border
    return img, ratio, (dw, dh)

# 自定义数据集加载
class CustomImageDataset:
    def __init__(self, img_dir, transform=None):
        self.img_dir = img_dir
        self.img_names = os.listdir(img_dir)
        self.transform = transform
        self.to_tensor = transforms.ToTensor()

    def __len__(self):
        return len(self.img_names)

    def __getitem__(self, idx):
        img_path = os.path.join(self.img_dir, self.img_names[idx])
        image = Image.open(img_path).convert('RGB')
        if image is None:
            raise FileNotFoundError(f"Image at path {img_path} could not be loaded.")
        if self.transform:
            image = self.transform(image)
        return self.to_tensor(image), self.img_names[idx]

# 图像预处理函数
def preprocess_image(img, img_size=(640, 480)):
    img = letterbox(img, new_shape=img_size)[0]
    img = np.array(img).transpose(2, 0, 1)  # HWC to CHW
    img = np.ascontiguousarray(img, dtype=np.float32)  # uint8 to float32
    img /= 255.0  # 0 - 255 to 0.0 - 1.0
    return torch.from_numpy(img).unsqueeze(0)

# 定义检测攻击成功率的函数
def detect_attack_success(model, dataloader, device, conf_thresh=0.25, iou_thresh=0.45):
    model.eval()
    attack_success = 0
    total = 0

    with torch.no_grad():
        for images, img_names in tqdm(dataloader, desc='Detecting Attack Success Rate'):
            images = images.to(device)
            for img in images:
                pred = model(img.unsqueeze(0), augment=False)[0]
                pred = non_max_suppression(pred, conf_thresh, iou_thresh, classes=None, agnostic=False)
                if len(pred) == 0 or len(pred[0]) == 0:
                    attack_success += 1
                if len(pred) > 0 and len(pred[0]) > 0:
                    for det in pred[0]:
                        conf = det[4].item()  # 获取置信度
                        # if conf < 0.3:
                        #     attack_success += 1
                        print(f"Image: {img_names[0]}, Confidence: {conf:.2f}")

                # 判断是否有目标被检测到 (假设检测不到目标表示攻击成功)

                total += 1

    success_rate = (attack_success / total) * 100
    print(f'Attack Success Rate: {success_rate:.2f}%')
    return success_rate

# 主函数
if __name__ == "__main__":
    # 设置设备
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 加载YOLOv3权重
    weights_path = './runs/train/exp2/weights/best.pt'  # 替换为你的YOLOv3权重文件路径
    model = attempt_load(weights_path, device=device)  # 加载模型
    model.to(device).eval()

    # 数据集路径 (包含对抗攻击样本的数据集路径)
    #adv_img_dir = '/data/Newdisk2/linxiuhao/old_project/project/tardal/runs/tmp/images'
    adv_img_dir = ''  # 替换为你对抗样本数据集的路径

    # 创建数据集和数据加载器
    dataset = CustomImageDataset(adv_img_dir)
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False)

    # 检测对抗攻击成功率
    detect_attack_success(model, dataloader, device)
