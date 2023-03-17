from torch import nn
import torch
from collections import OrderedDict
from torchvision.models import resnet50

class MTL(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = resnet50(pretrained=True)
        self.n_features = self.net.fc.in_features
        self.net.fc = nn.Identity()
        self.net.gender_fc = nn.Sequential(OrderedDict([
            ('linear', nn.Linear(self.n_features,self.n_features)),
            ('relu1', nn.ReLU()),
            ('final', nn.Linear(self.n_features, 2)),
#             ('final', nn.Linear(self.n_features, 1)),
        ]))
        self.net.age_fc = nn.Sequential(OrderedDict([
            ('linear', nn.Linear(self.n_features,self.n_features)),
            ('relu1', nn.ReLU()),
            ('final', nn.Linear(self.n_features, 1))
        ]))
        
    def forward(self, x):
        age_head = self.net.age_fc(self.net(x))
        gender_head = self.net.gender_fc(self.net(x))
        return age_head, gender_head