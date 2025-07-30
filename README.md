# DL_study

## 项目简介
欢迎来到 **DL_study** 仓库！这里是我学习深度学习（Deep Learning,）过程中的实践记录，专注于用 PyTorch 探索经典模型与算法，涵盖判别式和生成式模型！

## 核心内容
### 1. 技术栈
- **框架**：PyTorch  
- **领域**：深度学习基础实践
- **数据集**：MNIST（手写数字识别数据集）  

### 2. 关键项目结构
```
DL_study/
├── VAE
├── GAN
├── ResNet
├── VGG
├── ...
└── requirement.txt      # 我自用的虚拟环境配置，其实没有这么麻烦，正常配置好pytorch,cuda然后缺啥补啥就好了
```

```
具体某个模型的项目： 打开工程文件夹ResNet->model.py->train.py->test.py
DL_study/ResNet(举例)
├── ***(模型名称.py/model.py） #定义模型参数
├── runs/          # 训练结果存储（模型训练结果、中间结果等）
├── train.py  # 训练脚本（数据加载、训练流程、可视化）
├── img/          #模型架构 | 训练可视化保存结果（也可能直接裸奔在主项目目录）
└── test.py      # 测试训练模型效果
#每个项目可能有略微差别，注意区分
```




## 快速开始
### 1. 环境依赖
```bash
# 推荐通过 Anaconda/Miniconda 配置环境
conda create -n dl_study python=3.10
conda activate dl_study
pip install torch torchvision matplotlib tqdm

# 验证环境（需有 GPU 或正确配置 PyTorch 环境）
python -c "import torch; print(torch.__version__); print(torch.cuda.is_available())"
```

### 2. 训练 
```bash
python train.py
```

### 3. 查看训练成果
1. **损失曲线** 
2. **结果可视化**
3. **使用训练好的模型**


## 扩展与进阶
### 1. 功能扩展建议
- **支持更多数据集**：修改 `get_data()` 加载 Fashion-MNIST、CIFAR-10
- **优化生成效果**：尝试调整网络结构、学习率调度器（`torch.optim.lr_scheduler`）  
- **交互式生成**：结合 Flask/Dash 搭建网页，输入数字标签实时生成图像  


## 2.贡献与交流
- **反馈建议**：发现 Bug 或优化思路？欢迎提 Issue 或直接 PR  
- **学习交流**：想探讨深度学习实践细节？可通过 GitHub 私信或我的其他社交平台（若公开）联系  
- **项目使用**：项目遵循MIT开源协议，可以pull下来学习或修改
- **后续计划**：逐步补充生成式模型，持续完善深度学习实践库  

用代码探索 AI 边界，欢迎 Star 支持我的学习记录！✨  
（最后更新：2025-07-30，增加VAE & CVAE 实现）