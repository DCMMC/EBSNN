import torch 
import torch.nn as nn
import torch.nn.functional as F
from overall import *


class DPCNN(nn.Module):
    __name__ = 'DPCNN'

    def __init__(self):
        super(DPCNN, self).__init__()
        
        self.cnn1 = nn.Sequential(
            nn.Conv2d(1, 200, kernel_size = (1,4), stride=(1, 3)),
            #nn.Dropout(0.05)
        )

        self.cnn2 = nn.Sequential(
            nn.Conv2d(1, 200, kernel_size = (200, 5), stride=(200, 1)),
            nn.Dropout(0.05)
        )

        self.pool = nn.MaxPool2d(kernel_size = (1, 95), stride=(1, 95))

        self.fc = nn.Sequential(
            nn.Linear(200, 100),
            nn.Linear(100, 50),
            nn.Dropout(0.05),
            nn.Linear(50, NUM_CLASS)
        )

    def forward(self, x):  #batch * 1500
        batch_size = x.size(0)
        x = x.view(batch_size, 1, 1, -1)
        x = self.cnn1(x)
        
        print(x.size())

        x = x.view(batch_size, 1, 200, -1)
        x = self.cnn2(x)
        print(x.size())
        
        x = x.view(batch_size, 1, 200, -1)
        x = self.pool(x)
        print(x.size())
        
        x = x.view(batch_size,200)
        out = self.fc(x)
        return out


