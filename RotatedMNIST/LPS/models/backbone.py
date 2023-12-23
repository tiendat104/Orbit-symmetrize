import torch 
import torch.nn as nn
import random
import numpy as np

def initialize_weights_xavier(m):
    if isinstance(m, nn.Linear) or isinstance(m, nn.Conv2d):
        nn.init.xavier_uniform_(m.weight)
        nn.init.constant_(m.bias, 0)
    
    if isinstance(m, nn.BatchNorm2d):
        nn.init.constant_(m.weight, 1)
        nn.init.constant_(m.bias, 0)
        
class CNN(nn.Module):
    def __init__(self, dropout=0.4, seed=0):
        super().__init__()
        self.n_classes = 10
        # 7 layers 
        layers = []
        l1 = self.build_layer(cin=1, cout=32, k=3, stride=1)
        layers.extend(l1)
        l2 = self.build_layer(cin=32, cout=32, k=3, stride=1)
        layers.extend(l2)
        l3 = self.build_layer(cin=32, cout=32, k=3, stride=1)
        layers.extend(l3)
        
        l4 = self.build_layer(cin=32, cout=64, k=5, stride=2, dropout=dropout)
        layers.extend(l4)
        l5 = self.build_layer(cin=64, cout=64, k=3, stride=1)
        layers.extend(l5)
        l6 = self.build_layer(cin=64, cout=64, k=3, stride=1)
        layers.extend(l6)
        
        l7 = self.build_layer(cin=64, cout=128, k=5, stride=2, dropout=dropout)
        layers.extend(l7)
        self.layers = nn.Sequential(*layers)
        
        self.classifier = nn.Linear(128, self.n_classes)
        
        # deterministic implementation
        torch.manual_seed(seed)
        random.seed(seed)
        np.random.seed(seed)
        torch.backends.cudnn.benchmark = False
        
        # weight initialization
        self.apply(initialize_weights_xavier)
    
    def build_layer(self, cin: int, cout: int, k: int, stride: int, padding: int =  0, dropout = 0):
        cl = nn.Conv2d(cin, cout, k, stride, padding)
        bn = nn.BatchNorm2d(cout)
        act = nn.ReLU(True)
        layers = [cl, bn, act]
        if dropout > 0:
            layers.append(nn.Dropout(dropout))
        return layers
        
    def forward(self, images): 
        # images: [B, 1, 28, 28]
        B = images.shape[0]
        x = self.layers(images) # [B, 128, 1, 1]
        x = x.squeeze(-1).squeeze(-1) # [B, 128]
        x = self.classifier(x) # [B, 10]
        return x