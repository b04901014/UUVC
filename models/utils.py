import torch.nn as nn
import torch
import torch.nn.functional as F

class ConvLNBlock(nn.Module):
    def __init__(self, hidden_size, dropout, dilation=1):
        super().__init__()
        ks = 3 #Fix kernel size to 3
        padding = (ks - 1) // 2 * dilation
        self.conv = nn.Conv1d(hidden_size, hidden_size, kernel_size=ks, padding=padding, dilation=dilation)
        self.dropout = nn.Dropout(p=dropout)
        self.ln = nn.LayerNorm(hidden_size)
        self.linear_conv = nn.Conv1d(hidden_size, hidden_size, kernel_size=1)
        self.activation = nn.ReLU()

    def forward(self, x):
        x = x + self.dropout(self.linear_conv(self.activation(self.conv(x))))
        x = x.transpose(1, 2).contiguous()
        x = self.ln(x)
        x = x.transpose(1, 2).contiguous()
        return x

class ResBlock(nn.Module):
    def __init__(self, c_in, c_mid=128, c_out=128):
        super(ResBlock, self).__init__()
        self.leaky_relu1 = nn.LeakyReLU(negative_slope=0.1)
        self.conv1 = nn.Conv1d(c_in, c_mid, kernel_size=3, stride=1, padding=(3 - 1) // 2 * 3, dilation=3)

        self.leaky_relu2 = nn.LeakyReLU(negative_slope=0.1)
        self.conv2 = nn.Conv1d(c_mid, c_out, kernel_size=3, stride=1, padding=(3 - 1) // 2 * 3, dilation=3)

        self.conv3 = nn.Conv1d(c_in, c_out, kernel_size=1, dilation=1)

    def forward(self, x):
        y = self.conv1(self.leaky_relu1(x))
        y = self.conv2(self.leaky_relu2(y))
        y = y + self.conv3(x)
        return y


def pad(input_ele, mel_max_length=None):
    if mel_max_length:
        max_len = mel_max_length
    else:
        max_len = max([input_ele[i].size(0) for i in range(len(input_ele))])

    out_list = list()
    for i, batch in enumerate(input_ele):
        if len(batch.shape) == 1:
            one_batch_padded = F.pad(
                batch, (0, max_len - batch.size(0)), "constant", 0.0
            )
        elif len(batch.shape) == 2:
            one_batch_padded = F.pad(
                batch, (0, 0, 0, max_len - batch.size(0)), "constant", 0.0
            )
        out_list.append(one_batch_padded)
    out_padded = torch.stack(out_list)
    return out_padded

class LengthRegulator(nn.Module):
    def __init__(self):
        super(LengthRegulator, self).__init__()

    def LR(self, x, duration, max_len):
        output = list()
        mel_len = list()
        for batch, expand_target in zip(x, duration):
            expanded = self.expand(batch, expand_target)
            output.append(expanded)
            mel_len.append(expanded.shape[0])

        if max_len is not None:
            output = pad(output, max_len)
        else:
            output = pad(output)
        mel_len = torch.LongTensor(mel_len).to(x.device)
        mask = torch.arange(output.size(1), device=x.device)[None, :] >= mel_len[:, None]

        return output, mask

    def expand(self, batch, predicted):
        out = list()
        for i, vec in enumerate(batch):
            expand_size = predicted[i].item()
            out.append(vec.expand(max(int(expand_size), 0), -1))
        out = torch.cat(out, 0)
        return out

    def forward(self, x, duration, max_len):
        output, mel_len = self.LR(x, duration, max_len)
        return output, mel_len
