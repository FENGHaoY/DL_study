from my_model import MLP, predict
import torch, numpy

# 选择设备
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 加载模型并移动到设备
model = MLP(4)
model.load_state_dict(torch.load("MLP_more.pth", map_location=device))  # 确保加载到正确的设备
model.to(device)  # 移动模型到指定设备
model.eval()

# 测试样例
test_example = [5.1, 3.5, 1.4, 0.2]
yhat = predict(test_example, model)
yhat = numpy.argmax(yhat,axis=1)
print(yhat)
