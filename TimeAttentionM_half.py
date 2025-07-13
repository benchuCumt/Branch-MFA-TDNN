from torch import nn
import torch


def overlap_chunks(tensor, chunk_size=50, step=50):
    overlap = chunk_size - step
    if overlap == 0:
        num_chunks = int(tensor.size(-1) / chunk_size)
    else:
        num_chunks = int((tensor.size(-1) - chunk_size) / step + 1)
    chunks = []
    for i in range(num_chunks):
        start_idx = i * step
        end_idx = start_idx + chunk_size
        chunk = tensor[..., start_idx:end_idx]
        chunks.append(chunk)
    return chunks


class TimeAttention_half(nn.Module):
    def __init__(self, ratio=6):  
        super(TimeAttention_half, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.dims = 50
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
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        outs = []
        i = 1
        xs = overlap_chunks(x)
        for x in xs:
            x = torch.transpose(x, 1, 3)
            ap_out = self.avg_pool(x)
            # j = i mod m
            index = i % 4
            if index == 1:
                avg_out = self.fc1(ap_out)
            elif index == 2:
                avg_out = self.fc2(ap_out)
            else:
                avg_out = self.fc3(ap_out)
            out = avg_out
            out = torch.transpose(out, 1, 3)
            outs.append(out)
            i = i + 1
        att = torch.cat(outs, dim=3)
        # sigmoid
        att = self.sigmoid(att)
        return att
