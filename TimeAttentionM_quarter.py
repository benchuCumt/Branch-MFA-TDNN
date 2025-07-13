from torch import nn
import torch


def overlap_chunks(tensor, chunk_size=25, step=25):
    # 计算块的重叠
    overlap = chunk_size - step
    # 计算可以创建的块的数量
    if overlap == 0:
        num_chunks = int(tensor.size(-1) / chunk_size)
    else:
        num_chunks = int((tensor.size(-1) - chunk_size) / step + 1)
    # 创建一个列表来保存重叠的块
    chunks = []
    # 生成重叠的块
    for i in range(num_chunks):
        start_idx = i * step
        end_idx = start_idx + chunk_size
        chunk = tensor[..., start_idx:end_idx]
        chunks.append(chunk)
    return chunks


class TimeAttention_quarter(nn.Module):
    def __init__(self, ratio=3):  # 这里的dims要根据分段数确定
        super(TimeAttention_quarter, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.dims = 25
        # 不同段使用不同的全连接层，来表示数据的不同分布
        self.fc1 = nn.Sequential(nn.Conv2d(self.dims, self.dims // ratio, 1, bias=False),
                                 nn.ReLU(),
                                 nn.Conv2d(self.dims // ratio, self.dims, 1, bias=False))
        self.fc2 = nn.Sequential(nn.Conv2d(self.dims, self.dims // ratio, 1, bias=False),
                                 nn.ReLU(),
                                 nn.Conv2d(self.dims // ratio, self.dims, 1, bias=False))
        self.fc3 = nn.Sequential(nn.Conv2d(self.dims, self.dims // ratio, 1, bias=False),
                                 nn.ReLU(),
                                 nn.Conv2d(self.dims // ratio, self.dims, 1, bias=False))
        self.fc4 = nn.Sequential(nn.Conv2d(self.dims, self.dims // ratio, 1, bias=False),
                                 nn.ReLU(),
                                 nn.Conv2d(self.dims // ratio, self.dims, 1, bias=False))
        self.fc5 = nn.Sequential(nn.Conv2d(self.dims, self.dims // ratio, 1, bias=False),
                                 nn.ReLU(),
                                 nn.Conv2d(self.dims // ratio, self.dims, 1, bias=False))
        self.fc6 = nn.Sequential(nn.Conv2d(self.dims, self.dims // ratio, 1, bias=False),
                                 nn.ReLU(),
                                 nn.Conv2d(self.dims // ratio, self.dims, 1, bias=False))
        self.fc7 = nn.Sequential(nn.Conv2d(self.dims, self.dims // ratio, 1, bias=False),
                                 nn.ReLU(),
                                 nn.Conv2d(self.dims // ratio, self.dims, 1, bias=False))
        self.fc8 = nn.Sequential(nn.Conv2d(self.dims, self.dims // ratio, 1, bias=False),
                                 nn.ReLU(),
                                 nn.Conv2d(self.dims // ratio, self.dims, 1, bias=False))
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        outs = []
        # 用来控制全连接层序号的索引
        i = 1
        # 对x进行分组，这里训练和验证都需要分组，因为一段改成0.5秒了
        xs = overlap_chunks(x)
        for x in xs:
            x = torch.transpose(x, 1, 3)
            ap_out = self.avg_pool(x)
            # 这里过的全连接层的序号需要计算
            # 这里过的全连接层的序号需要计算
            index = i % 8
            if index == 1:
                avg_out = self.fc1(ap_out)
            elif index == 2:
                avg_out = self.fc2(ap_out)
            elif index == 3:
                avg_out = self.fc3(ap_out)
            elif index == 4:
                avg_out = self.fc4(ap_out)
            elif index == 5:
                avg_out = self.fc5(ap_out)
            elif index == 6:
                avg_out = self.fc6(ap_out)
            else:
                avg_out = self.fc7(ap_out)
            # 仅使用平均池化
            out = avg_out
            out = torch.transpose(out, 1, 3)
            outs.append(out)
            i = i + 1
        att = torch.cat(outs, dim=3)
        # 最后过sigmoid，因为有重叠
        att = self.sigmoid(att)
        return att


# # 加载模型，参数是时间维度的维数
# model = TimeAttention_half()
#
# # 准备输入数据
# input_data = torch.randn(20, 32, 20, 900)  # 输入数据的大小要与模型设定一致
#
# # 前向传播
# output = model.forward(input_data)
#
# # 获取输出层大小
# print("attention形状：", output.shape)

