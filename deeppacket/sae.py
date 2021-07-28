import torch 
import torch.nn as nn
import torch.nn.functional as F
from overall import *

class SAE(nn.Module):
    __name__ = 'SAE'

    def __init__(self):
        super(SAE, self).__init__()
        
        self.fc = nn.Sequential(
            nn.Linear(1500, 400),
            nn.Linear(400, 300),
            nn.Linear(300, 200),
            nn.Linear(200, 100),
            nn.Linear(100, 50),
            nn.Dropout(0.05),
            nn.Linear(50, NUM_CLASS)

        )

    def forward(self, x):  #batch * 1500
        out = self.fc(x)
        return out


