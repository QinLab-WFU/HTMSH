import torch
import math
import torchvision
import torch.nn as nn
import torch.nn.functional as F
class LabelNet(nn.Module):

    def __init__(self, label_dim,code_len):
        super(LabelNet, self).__init__()
        self.fc1 = nn.Linear(label_dim, (label_dim+code_len)//2)
        self.fc2 = nn.Linear((label_dim+code_len)//2, code_len)


    def forward(self, x):
        feat = F.relu(self.fc1(x))
        hid = self.fc2(feat)
        code = torch.tanh(self.alpha * hid)
        return feat, hid, code


    def set_alpha(self, epoch):
        self.alpha  = math.pow((1.0 * epoch + 1.0), 0.5)

import math

# Calculate alpha values for epoch from 0 to 200
# epoch = 200
# alpha_values = [(epoch, math.pow((1.0 * epoch + 1.0), 0.5)) for epoch in range(201)]
# print(alpha_values)