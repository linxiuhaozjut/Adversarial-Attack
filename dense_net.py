import torch
import torch.nn as nn
import torch.nn.functional as F
#from torchsummary import summary
class Convlayer(nn.Module):
    def __init__(self, in_channel, out_channel):
        super().__init__()
        self.reflection_pad = nn.ReflectionPad2d(1)
        self.conv_layer = nn.Conv2d(
            in_channels=in_channel,  # 输入的通道数，
            out_channels=out_channel,  # 输出的通道数，也就是卷积核的数量
            kernel_size=3,  # 卷积核的大小
            stride=1,  # 卷积核的步长
        )
        self.batch_norm = nn.BatchNorm2d(out_channel)  # 添加Batch Normalization层

    def forward(self, x):
        x = self.reflection_pad(x)#在卷积前进行反射填充
        x = self.conv_layer(x)
        x = self.batch_norm(x)
        x = F.relu(x)
        return x

class DenseBlock(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = Convlayer(16,16)
        self.conv2 = Convlayer(32,16) 
        self.conv3 = Convlayer(48,16)  

    def forward(self, x):
        conv1_out = self.conv1(x)
        conv2_out = self.conv2(torch.cat([x, conv1_out], dim=1))
        conv3_out = self.conv3(torch.cat([x, conv1_out, conv2_out], dim=1))
        out = torch.cat([x, conv1_out, conv2_out, conv3_out], dim=1)  # cat along channel dimension
        return out

class Encoder(nn.Module):
    def __init__(self,in_channel):
        super().__init__()
        self.conv1 = Convlayer(in_channel,16)
        self.dense_block = DenseBlock()

    def forward(self, x):
        x = self.conv1(x)
        x = self.dense_block(x)
        return x

class Decoder(nn.Module):
    def __init__(self, out_channel):
        super().__init__()
        self.conv2 = Convlayer(64, 64)
        self.conv3 = Convlayer(64, 32)
        self.conv4 = Convlayer(32, 16)
        self.conv5 = Convlayer(16, out_channel)

    def forward(self, x):
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)
        return x

def addition_fusion(x1, x2):
    return x1 + x2
  
def l1_norm_fusion(x1, x2):
    # 计算活动水平图
    c1 = x1.norm(p=1, dim=1, keepdim=True)
    c2 = x2.norm(p=1, dim=1, keepdim=True)

    # 对活动水平图应用基于块的平均算子
    c1_avg = F.avg_pool2d(c1, kernel_size=3, stride=1, padding=1)
    c2_avg = F.avg_pool2d(c2, kernel_size=3, stride=1, padding=1)

    # 计算权重
    w1 = c1_avg / (c1_avg + c2_avg)
    w2 = c2_avg / (c1_avg + c2_avg)

    # 融合特征图
    f = w1 * x1 + w2 * x2

    return f

class DenseFuse(nn.Module):
    def __init__(self, in_channel, out_channel, fusion_strategy='addition'):
        super().__init__()
        self.encoder1 = Encoder(in_channel)
        self.encoder2 = Encoder(in_channel)
        
        self.decoder = Decoder(out_channel) 
        # 根据参数选择融合策略
        if fusion_strategy == 'addition':
            self.fusion = addition_fusion
        elif fusion_strategy == 'l1_norm':
            self.fusion = l1_norm_fusion
        else:
            raise ValueError(f'Unknown fusion_strategy: {fusion_strategy}')

    def forward(self, x1, x2):
        x1 = self.encoder1(x1)
        x2 = self.encoder2(x2)
        x = self.fusion(x1,x2)
        x = self.decoder(x)
        return x

'''model = DenseFuse(fusion_strategy='addition')
x1 = torch.randn(10, 1, 28, 28)
x2 = torch.randn(10, 1, 28, 28)

# 使用这个模型对输入x进行前向传播
output = model(x1, x2)
summary(model, input_size=(1, 28, 28))
for name, layer in model.named_children():
    print(f"Layer: {name}")
    
    # 遍历该层的每个参数
    for param_name, param in layer.named_parameters():
        #print(f"Parameter: {param_name}, Size: {param.size()}, Values: {param.detach().numpy()}")
        print(f"Parameter: {param_name}, Size: {param.size()}")'''
