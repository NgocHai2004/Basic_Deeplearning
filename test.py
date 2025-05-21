import sys
import os
import torch
from torchvision import transforms
from PIL import Image
from Bulid_model import SimpleNeuralNetwork
import matplotlib.pyplot as plt

# Thêm đường dẫn
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Load model
model = SimpleNeuralNetwork(num_classes=10)
model.load_state_dict(torch.load('Model/final_model.pth'))
model.eval()

# Load ảnh
image_path = 'Anh_test.png'
image = Image.open(image_path).convert('RGB')

# Transform ảnh
transform = transforms.Compose([
    transforms.Resize((32, 32)),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])
image_tensor = transform(image).unsqueeze(0)

# Dự đoán
with torch.no_grad():
    output = model(image_tensor)
    predicted_class = torch.argmax(output, dim=1).item()

# Danh sách tên lớp CIFAR-10
classes = ['airplane', 'automobile', 'bird', 'cat', 'deer',
           'dog', 'frog', 'horse', 'ship', 'truck']

# Lấy tên class dự đoán
predicted_name = classes[predicted_class-1]

# Hiển thị ảnh kèm tên con vật dự đoán
plt.imshow(image)
plt.axis('off')
plt.title(f"Dự đoán: {predicted_name}", fontsize=14, color='blue')
plt.show()

# In ra console
print("Số dự đoán là:", predicted_class)
print("Tên dự đoán là:", predicted_name)
