import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import os
from CVAE import CVAE  # 导入你的CVAE模型类

# 配置参数（需与训练时保持一致）
latent_dim = 128
n_classes = 10
input_dim = 28 * 28
hidden_dim = 256
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# 创建保存生成结果的目录
os.makedirs("./test_results/CVAE", exist_ok=True)

def get_one_hot(label, n_classes, device):
    """将标签转换为one-hot编码"""
    return torch.nn.functional.one_hot(label, n_classes).float().to(device)

def load_model(weight_path):
    """加载训练好的模型权重"""
    model = CVAE(
        label_dim=n_classes,
        input_dim=input_dim,
        hidden_dim=hidden_dim,
        latent_dim=latent_dim
    ).to(device)
    
    # 加载权重
    model.load_state_dict(torch.load(weight_path, map_location=device))
    model.eval()  # 设置为评估模式
    print(f"模型已从 {weight_path} 加载完成")
    return model

def generate_digits(model, num_samples=10, specific_digits=None):
    """
    生成数字图像
    
    参数:
        model: 加载好的CVAE模型
        num_samples: 生成样本数量（当specific_digits为None时有效）
        specific_digits: 指定生成的数字列表，如[0,1,2,3]，优先级高于num_samples
    """
    with torch.no_grad():  # 关闭梯度计算，节省内存
        # 确定生成的标签
        if specific_digits is not None:
            # 使用指定的数字标签
            labels = torch.tensor(specific_digits, device=device)
            num_samples = len(specific_digits)
        else:
            # 随机生成标签（0-9均匀分布）
            labels = torch.randint(0, n_classes, (num_samples,), device=device)
        
        # 转换为one-hot编码
        labels_oh = get_one_hot(labels, n_classes, device)
        
        # 生成潜在向量
        noise = torch.randn(num_samples, latent_dim, device=device)
        
        # 生成图像
        generated = model.gen(znoise=noise, label=labels_oh).detach().cpu().numpy()
        generated = generated.reshape(-1, 28, 28)  # 恢复为28x28图像
        
        return generated, labels.cpu().numpy()  # 返回生成的图像和对应的标签

def plot_generated(images, labels, save_path=None):
    """绘制生成的图像并显示对应的标签"""
    num_samples = len(images)
    # 计算绘图网格尺寸（自动适应样本数量）
    rows = (num_samples + 4) // 5  # 每行最多5个样本
    cols = min(5, num_samples)
    
    plt.figure(figsize=(cols * 2, rows * 2))
    for i in range(num_samples):
        plt.subplot(rows, cols, i + 1)
        plt.imshow(images[i], cmap='gray')
        plt.title(f"Digit: {labels[i]}")  # 显示对应的数字
        plt.axis('off')
    
    plt.tight_layout()
    
    # 保存图像（如果指定了路径）
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"生成结果已保存至 {save_path}")
    
    plt.show()

if __name__ == "__main__":
    # 模型权重路径（请替换为你的实际权重路径）
    model_weight_path = "./runs/param/cvae-100.pth"
    
    # 加载模型
    model = load_model(model_weight_path)
    
    # 示例1：生成10个随机数字（0-9各一个）
    print("生成0-9各一个数字...")
    generated_imgs, labels = generate_digits(model, specific_digits=list(range(10)))
    plot_generated(
        generated_imgs, 
        labels, 
        save_path="./test_results/CVAE/generated_0-9.png"
    )
    
    # 示例2：生成20个随机数字（随机类别）
    print("生成20个随机数字...")
    generated_imgs, labels = generate_digits(model, num_samples=20)
    plot_generated(
        generated_imgs, 
        labels, 
        save_path="./test_results/CVAE/generated_random_20.png"
    )
    
    # 示例3：生成多个相同数字（如生成10个数字"5"）
    print("生成10个数字'8'...")
    generated_imgs, labels = generate_digits(model, specific_digits=[8]*10)
    plot_generated(
        generated_imgs, 
        labels, 
        save_path="./test_results/CVAE/generated_5s.png"
    )