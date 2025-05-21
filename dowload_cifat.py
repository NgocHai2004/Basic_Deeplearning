import torchvision.datasets as datasets

# Thư mục bạn muốn lưu CIFAR-10
data_dir = './data'

# Tải dataset về nếu chưa có
train_dataset = datasets.CIFAR10(root=data_dir, train=True, download=True)
test_dataset = datasets.CIFAR10(root=data_dir, train=False, download=True)