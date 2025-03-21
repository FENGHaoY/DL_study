import torch
from number_shibie import CNN
from PIL import Image
from torchvision.transforms import Compose, Normalize, ToTensor
trans = Compose([ToTensor(),
                 Normalize((0.1307,),(0.3081,))])
image_path = './images/9_images.png'
img = Image.open(image_path).convert('L').resize((28,28))
img = trans(img).unsqueeze(0)
model = CNN(1)
model.load_state_dict(torch.load("mnist_cnn_cpu.pth"))
model.eval()

with torch.no_grad():
    output = model(img)
    predict_class = torch.argmax(output, dim=1).item()

print(f"预测结果是{predict_class}")