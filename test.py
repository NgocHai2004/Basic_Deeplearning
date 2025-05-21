import torch
from torchvision import transforms
from PIL import Image
from Model import SimpleNeuralNetwork
import matplotlib.pyplot as plt
model = SimpleNeuralNetwork(num_classes=10)
model.load_state_dict(torch.load('final_model.pth'))  # path tới file pth bạn vừa tải
model.eval()
image_path = 'Screenshot 2025-05-21 172844.png'
image = Image.open(image_path).convert('RGB')

# Hiển thị ảnh
plt.imshow(image)
plt.axis('off')
plt.title("Ảnh test")
plt.show()

# Transform ảnh
transform = transforms.Compose([
    transforms.Resize((32, 32)),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

image_tensor = transform(image).unsqueeze(0)

# Predict ảnh
with torch.no_grad():
    output = model(image_tensor)
    predicted_class = torch.argmax(output, dim=1).item()

# In số dự đoán
print("Số dự đoán là:", predicted_class)