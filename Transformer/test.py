import torch
import tiktoken
from model import Transformer  # 你之前定义的模型文件名是 model.py

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 参数必须和训练时保持一致
context_size = 32
num_heads = 4
num_layers = 6
d_model = 64

max_token_value = 100069
  # 这里需要你替换成训练时的 max_token_value

# 加载编码器
encoding = tiktoken.get_encoding("cl100k_base")

# 初始化模型并加载权重
model = Transformer(d_model, num_heads, num_layers, context_size, max_token_value).to(device)
model.load_state_dict(torch.load("Top-k-best_model-20.pth", map_location=device))
model.eval()


def generate_text(model, prompt, max_new_tokens=50):
    # 将 prompt 转成 token id
    tokens = encoding.encode(prompt)
    # 保证输入不超过 context_size，或者截断
    tokens = tokens[-context_size:]
    input_ids = torch.tensor(tokens, dtype=torch.long, device=device).unsqueeze(0)  # [1, seq_len]

    generated = input_ids

    for _ in range(max_new_tokens):
        logits = model.generate(generated, max_length=max_new_tokens)
        # model.generate 返回的是完整序列，这里直接取返回结果
        generated = logits

    generated_ids = generated[0].tolist()
    return encoding.decode(generated_ids)


if __name__ == "__main__":
    prompt = 'Respond in 2-3 sentences to a customer who says, "I’m not sure if this is right for me." Focus on empathy and asking a question to encourage them to share more.'
    print("Prompt:\n", prompt)
    output_text = generate_text(model, prompt, max_new_tokens=50)
    print("Generated text:\n", output_text)
