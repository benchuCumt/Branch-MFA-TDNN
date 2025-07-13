import torch
from torch import nn
from TimeAttentionM_quarter import TimeAttention_quarter
from TimeAttentionM_half import TimeAttention_half

class MutiScaleTimeAttention(nn.Module):
    def __init__(self, time_att_choice="half"):   # 无重叠
        super(MutiScaleTimeAttention, self).__init__()
        if time_att_choice == "half":
            print("use 0.5s chunk")
            self.timeAtt = TimeAttention_half()
        elif time_att_choice == "quarter":
            print("use 0.25s chunk")
            self.timeAtt = TimeAttention_quarter()
        else:
            raise Exception("错误的分段大小选择")
        self.linearConv = nn.Conv2d(32, 32, kernel_size=1, bias=False)

    def forward(self, x):
        branch1 = x  # 残差
        branch2 = x
        branch2 = self.timeAtt(branch2)*branch2
        branch = self.linearConv(branch2)
        branch = branch+branch1  # 加上残差
        return branch


# # 加载模型，参数是时间维度的维数
# model = MutiScaleTimeAttention(time_att_choice="half")
# # 准备输入数据
# input_data = torch.randn(20, 32, 20, 800)  # 输入数据的大小要与模型设定一致
#
# # 前向传播
# Att = model.forward(input_data)
# # 获取输出层大小
# print("attention形状：", Att.shape)



