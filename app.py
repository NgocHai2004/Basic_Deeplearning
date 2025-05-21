import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from Dataset import Dataset_folder
from Model import SimpleNeuralNetwork
from Train import Train
from torch.utils.data import DataLoader
import torch.nn as nn
import torch
from pathlib import Path
from sklearn.metrics import classification_report

def main():
    data_train = Dataset_folder(root=r"cifat/cifar-10-batches-py",train=True)
    data_test = Dataset_folder(root=r"cifat/cifar-10-batches-py",train=False)
    Train(data_train, data_test, 100, SimpleNeuralNetwork(num_classes=10)).train_model()
    
if __name__ == '__main__':
    main()