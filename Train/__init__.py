import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from Dataset import Dataset_folder
from Bulid_model import SimpleNeuralNetwork
from torch.utils.data import DataLoader
import torch.nn as nn
import torch
from pathlib import Path
from sklearn.metrics import classification_report


class Train:
    def __init__(self,data_train,data_test,num_epoch,model):
        '''
        data_train: ma trận ảnh kèm label của dữ liệu train
        data_test: ma trận ảnh kèm label của dữ liệu test
        num_epoch: số epoch huấn luyện
        model: model vào để huấn luyện mô hình
        '''
        self.data_train:Dataset_folder = data_train
        self.data_test:DataLoader = data_test
        self.num_epoch:int = num_epoch
        self.model:SimpleNeuralNetwork = model

    def Inter_in_epoch(self):
        train_dataloader = DataLoader(
            dataset=self.data_train,
            batch_size=64,
            shuffle=True,
            num_workers=4,
            drop_last=True
    )
        test_dataloader = DataLoader(
            dataset=self.data_test,
            batch_size=64,
            shuffle=False,
            num_workers=4,
            drop_last=False
        )   
        return train_dataloader,test_dataloader
    
    def train_model(self):
        train_dataloader,test_dataloader = self.Inter_in_epoch()
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.SGD(self.model.parameters(), lr=1e-3, momentum=0.9)
        

        for epoch in range(self.num_epoch):
            self.model.train()
            for images, labels in train_dataloader:
                # forward
                outputs = self.model(images)
                loss_value = criterion(outputs, labels)
                # backward
                optimizer.zero_grad()
                loss_value.backward()
                optimizer.step()

            self.model.eval()
            all_predictions = []
            all_labels = []
            for images, labels in test_dataloader:
                all_labels.extend(labels)
                # no backward
                with torch.no_grad():
                    predictions = self.model(images) 
                    indices = torch.argmax(predictions.cpu(), dim=1)
                    all_predictions.extend(indices)
                    loss_value = criterion(predictions, labels)
            all_labels = [label.item() for label in all_labels]
            all_predictions = [prediction.item() for prediction in all_predictions]
            print("Epoch {}".format(epoch+1))
            print(classification_report(all_labels, all_predictions))



    
    



