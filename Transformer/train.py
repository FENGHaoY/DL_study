import torch.nn as nn
import torch
import os
import requests
import tiktoken
import torch.utils.data as dataparser
import matplotlib.pyplot as plt
from model import Transformer
from tqdm import tqdm
# 超参数
context_size = 24
batch_size = 48
num_heads = 4
num_layers = 6
lr = 0.0001
epochs = 20
d_model = 64
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
save_path = f"Top-k-best_model-{epochs}.pth"


def initialize_weights(model):
    """
    对模型中的参数进行初始化：
    - Linear层权重用 Xavier 正态分布初始化（也叫 Glorot 正态）
    - Linear层偏置初始化为 0
    - LayerNorm 层的权重初始化为 1，偏置初始化为 0
    """
    for name, param in model.named_parameters():
        if param.dim() > 1:  # 权重矩阵，比如 Linear.weight
            nn.init.xavier_normal_(param)
        else:  # 偏置向量，比如 Linear.bias 或 LayerNorm.bias
            nn.init.zeros_(param)

    # LayerNorm 层的权重一般初始化为1，偏置为0，下面单独处理
    for module in model.modules():
        if isinstance(module, nn.LayerNorm):
            nn.init.ones_(module.weight)
            nn.init.zeros_(module.bias)

def get_text():
    if not os.path.exists("sales_textbook.txt"):
        url = 'https://huggingface.co/datasets/goendalf666/sales-textbook_for_convincing_and_selling/raw/main/sales_textbook.txt'
        with open('sales_textbook.txt', 'w') as f:
            f.write(requests.get(url).text)
    with open('sales_textbook.txt') as f:
        return f.read()


def Tokenization(text, train_valid_split=0.9):
    encoding = tiktoken.get_encoding("cl100k_base")
    tokens = torch.tensor(encoding.encode(text), dtype=torch.long)
    train_size = int(len(tokens) * train_valid_split)
    train_data = tokens[:train_size]
    valid_data = tokens[train_size:]
    print(f'tokens length: {len(tokens)}')
    print(f'train_data length: {len(train_data)}')
    print(f'valid_data length: {len(valid_data)}')
    return train_data, valid_data, max(tokens).item(), encoding


def create_dataset(data, context_size):
    inputs, targets = [], []
    for i in range(0, len(data) - context_size):
        inputs.append(data[i:i + context_size])
        targets.append(data[i + 1:i + context_size + 1])
    return torch.stack(inputs), torch.stack(targets)


def get_data(data, context_size, batch):
    inputs, targets = create_dataset(data, context_size)
    dataset = dataparser.TensorDataset(inputs, targets)
    return dataparser.DataLoader(dataset, batch_size=batch, shuffle=True)


if __name__ == "__main__":
    text = get_text()
    train_data, valid_data, max_token_value, encoding = Tokenization(text)

    train_loader = get_data(train_data, context_size, batch_size)
    valid_loader = get_data(valid_data, context_size, batch_size)

    model = Transformer(d_model, num_heads, num_layers, context_size, max_token_value).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr,weight_decay=1e-4)
    initialize_weights(model)
    
    train_losses = []
    valid_losses = []
    best_val_loss = float("inf")

    for epoch in range(epochs):
        # ---------- 训练 ----------
        model.train()
        total_loss = 0
        for batch_x, batch_y in tqdm(train_loader):
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            optimizer.zero_grad()
            _, loss = model(batch_x, targets=batch_y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        avg_train_loss = total_loss / len(train_loader)

        # ---------- 验证 ----------
        model.eval()
        total_val_loss = 0
        with torch.no_grad():
            for batch_x, batch_y in valid_loader:
                batch_x, batch_y = batch_x.to(device), batch_y.to(device)
                _, loss = model(batch_x, targets=batch_y)
                total_val_loss += loss.item()
        avg_val_loss = total_val_loss / len(valid_loader)

        train_losses.append(avg_train_loss)
        valid_losses.append(avg_val_loss)

        print(f"Epoch {epoch+1}/{epochs} | Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f}")

        # ---------- 保存最优模型 ----------
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(model.state_dict(), save_path)
            print(f"✅ Best model saved at epoch {epoch+1} with val loss {avg_val_loss:.4f}")

        # ---------- 文本生成测试 ----------
        seed_tokens = train_data[:context_size].unsqueeze(0).to(device)  # 取开头做种子
        generated = model.generate(seed_tokens, max_length=50)[0].tolist()
        print("Sample Generation:")
        print(encoding.decode(generated))
        print("-" * 40)

    # ---------- 绘制 Loss 曲线 ----------
    plt.figure()
    plt.plot(train_losses, label="Train Loss")
    plt.plot(valid_losses, label="Valid Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training & Validation Loss")
    plt.legend()
    plt.savefig("loss_curve.png")
    print("📉 Loss curve saved as loss_curve.png")

    print(f"🏆 Best model saved to {save_path}")
