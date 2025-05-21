import os
import pickle
from pathlib import Path
import numpy as np
import cv2
from torchvision import transforms

class Dataset_folder:
    def __init__(self, root, train, transform = None):
        '''
        root: đường dẫn thư mục gốc
        train: xác định muốn lấy bộ train hay test
        transform: đưa ảnh về có kích thước hay chuẩn hóa như nào
        '''
        self.transform:str = transform
        self.root: str = root
        if train:
            data_files = []
            for i in range(1, 6):
                data = os.path.join(self.root, f"data_batch_{i}")
                data_files.append(data)
        else:
            data_files = [os.path.join(self.root, "test_batch")]
        
        self.images = []
        self.labels = []

        for data_file in data_files:
            with open(data_file, "rb") as file: 
                data = pickle.load(file, encoding='bytes')  
                self.images.extend(data[b'data'])
                self.labels.extend(data[b'labels'])
        
    def __len__(self):
        return(len(self.labels))

    def __getitem__(self,item):
        image = self.images[item]
        image = np.reshape(image,(3,32,32)).astype(np.float32) / 255.0
        # image = np.transpose(image, (1, 2, 0))
        label = self.labels[item]
        return image,label

