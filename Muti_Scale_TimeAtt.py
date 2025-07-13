import torch
from torch import nn
from TimeAttentionM_quarter import TimeAttention_quarter
from TimeAttentionM_half import TimeAttention_half

class MutiScaleTimeAttention(nn.Module):
    def __init__(self, time_att_choice="half"):   
        super(MutiScaleTimeAttention, self).__init__()
        if time_att_choice == "half":
            print("use 0.5s chunk")
            self.timeAtt = TimeAttention_half()
        elif time_att_choice == "quarter":
            print("use 0.25s chunk")
            self.timeAtt = TimeAttention_quarter()
        else:
            raise Exception("Incorrect selection of chunk size!")
        self.linearConv = nn.Conv2d(32, 32, kernel_size=1, bias=False)

    def forward(self, x):
        branch1 = x  # residual
        branch2 = x
        branch2 = self.timeAtt(branch2)*branch2
        branch = self.linearConv(branch2)
        branch = branch+branch1  
        return branch
