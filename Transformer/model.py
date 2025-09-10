import torch.nn as nn
import torch
import torch.nn.functional as F
from torchinfo import summary
def sample_next_token(logits, top_k=50):
    # logits: [batch_size, vocab_size]
    logits = logits / 1.0  # temperature = 1.0，可调
    top_logits, top_indices = torch.topk(logits, top_k)
    probs = F.softmax(top_logits, dim=-1)
    next_token = torch.multinomial(probs, num_samples=1)
    return top_indices.gather(-1, next_token)
class FFN(nn.Module):
    def __init__(self, d_model):
        super(FFN, self).__init__()
        self.fc1 = nn.Linear(d_model, d_model * 4)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(d_model * 4, d_model)
        self.dropout = nn.Dropout(p=0.1)

    def forward(self, x):
        return self.dropout(self.fc2(self.relu(self.fc1(x))))


class Attention(nn.Module):
    def __init__(self, d_model, head_dim, context_size):
        super(Attention, self).__init__()
        self.head_dim = head_dim
        self.context_size = context_size

        self.Wq = nn.Linear(d_model, head_dim, bias=False)
        self.Wk = nn.Linear(d_model, head_dim, bias=False)
        self.Wv = nn.Linear(d_model, head_dim, bias=False)

    def forward(self, x):
        batch_size, seq_length, _ = x.size()
        q = self.Wq(x)
        k = self.Wk(x)
        v = self.Wv(x)

        mask = torch.triu(torch.ones(seq_length, seq_length, device=x.device), diagonal=1).bool()

        scores = torch.matmul(q, k.transpose(-2, -1)) / (self.head_dim ** 0.5)
        scores = scores.masked_fill(mask, float('-inf'))
        attn = F.softmax(scores, dim=-1)

        return torch.matmul(attn, v)


class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads, context_size):
        super(MultiHeadAttention, self).__init__()
        self.head_dim = d_model // num_heads
        self.heads = nn.ModuleList([Attention(d_model, self.head_dim, context_size) for _ in range(num_heads)])
        self.fc = nn.Linear(d_model, d_model)

    def forward(self, x):
        out = torch.cat([head(x) for head in self.heads], dim=-1)
        return self.fc(out)


class Transformer_block(nn.Module):
    def __init__(self, d_model, num_heads, context_size):
        super(Transformer_block, self).__init__()
        self.attn = MultiHeadAttention(d_model, num_heads, context_size)
        self.ffn = FFN(d_model)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

    def forward(self, x):
        x = x + self.norm1(self.attn(x))
        x = x + self.norm2(self.ffn(x))
        return x


class Transformer(nn.Module):
    def __init__(self, d_model, num_heads, num_layers, context_size, max_token_value):
        super(Transformer, self).__init__()
        self.embedding = nn.Embedding(max_token_value + 1, d_model)
        self.d_model = d_model
        self.context_size = context_size
        self.layers = nn.ModuleList([Transformer_block(d_model, num_heads, context_size) for _ in range(num_layers)])
        self.final_linear = nn.Linear(d_model, max_token_value + 1)

    def forward(self, x, targets=None):
        batch_size, seq_length = x.shape
        x = self.embedding(x)

        pos = torch.arange(0, seq_length, dtype=torch.float32, device=x.device).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, self.d_model, 2, device=x.device) *
                            (-torch.log(torch.tensor(10000.0, device=x.device)) / self.d_model))

        pos_enc = torch.zeros(seq_length, self.d_model, device=x.device)
        pos_enc[:, 0::2] = torch.sin(pos * div_term)
        pos_enc[:, 1::2] = torch.cos(pos * div_term)

        x = x + pos_enc

        for layer in self.layers:
            x = layer(x)

        logits = self.final_linear(x)

        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
            return logits, loss
        return logits


    # def generate(self, x, max_length=50):
    #     self.eval()
    #     generated = x
    #     for _ in range(max_length):
    #         with torch.no_grad():
    #             logits = self.forward(generated)
    #             next_token = sample_next_token(logits[:, -1, :], top_k=50)
    #             generated = torch.cat([generated, next_token], dim=1)
    #     return generated
    def generate(self, x, max_length=80):
        self.eval()
        generated = x
        # 计算还能生成多少个token（总长度不超过max_length）
        remaining = max_length - generated.size(1)  # generated.size(1)是当前长度
        if remaining <= 0:
            return generated  # 初始长度已超过限制，直接返回
        
        for _ in range(remaining):  # 只生成remaining个新token
            with torch.no_grad():
                logits = self.forward(generated)
                next_token = sample_next_token(logits[:, -1, :], top_k=50)
                generated = torch.cat([generated, next_token], dim=1)
        return generated

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    d_model = 512
    num_heads = 8
    num_layers = 6
    context_size = 16
    max_token_value = 10000  # Example value, adjust as needed
    batch_size = 4
    model = Transformer(d_model, num_heads, num_layers, context_size, max_token_value).to(device)
    summary(model, input_size=(batch_size, context_size), dtypes=[torch.long])