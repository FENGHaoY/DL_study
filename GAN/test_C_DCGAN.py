import torch
import torch.nn as nn
from torchvision.utils import save_image
import os
from DCGAN_pro import DCgen_model  # 假设你把 Generator 定义放在 model.py 中
import argparse
from train_DCGAN_pro import get_one_hot
#torch.manual_seed(40)
# 参数设置
parser = argparse.ArgumentParser()
parser.add_argument('--checkpoint', type=str, default='./runs/C_DCgan/G_final.pth_40', help='路径：保存的生成器模型')
parser.add_argument('--n_classes', type=int, default=10, help='类别数')
parser.add_argument('--noise_dim', type=int, default=100, help='噪声维度')
parser.add_argument('--output_dir', type=str, default='./generated_images', help='生成图像保存目录')
parser.add_argument('--num_samples', type=int, default=20, help='生成图像数量')
args = parser.parse_args()

# 保证输出目录存在
os.makedirs(args.output_dir, exist_ok=True)

# 设备选择
device = torch.device("cuda")

# 加载模型
generator = DCgen_model(noise_dim=args.noise_dim, label_dim=args.n_classes).to(device)
generator.load_state_dict(torch.load(args.checkpoint, map_location=device))
generator.eval()

# 生成固定标签和噪声
fixed_labels = torch.tensor([i for i in range(args.n_classes)] * (args.num_samples // args.n_classes)).to(device)
fixed_labels = get_one_hot(fixed_labels, n_classes=args.n_classes, device=device)
fixed_noise = torch.randn(fixed_labels.shape[0], args.noise_dim).to(device)

# 生成图像
with torch.no_grad():
    fake_images = generator(fixed_noise, fixed_labels)
    fake_images = (fake_images + 1) / 2  # Tanh 输出范围 [-1, 1] 映射到 [0, 1]

    # 保存图像
    save_image(fake_images, os.path.join(args.output_dir, 'generated_samples_40_2.png'), nrow=args.n_classes)
    print(f'✅ 图像已保存到 {args.output_dir}/generated_samples_40_2.png')
