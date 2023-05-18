import torch
import pdb


class Net(torch.nn.Module):
    def __init__(self,
                 depth=4,
                 mult_chan=16,
                 in_channels=1,
                 out_channels=1,
                 ):
        super().__init__()
        self.depth = depth
        self.mult_chan = mult_chan
        self.in_channels = in_channels
        self.out_channels = out_channels

        self.net_recurse = _Net_recurse(n_in_channels=self.in_channels, mult_chan=self.mult_chan, depth=self.depth)
        self.conv_out = torch.nn.Conv3d(self.mult_chan, self.out_channels, kernel_size=3, padding=1)

        self.out1 = torch.nn.Conv3d(1, 1, kernel_size=1)
        self.out2 = torch.nn.Identity()

    def forward(self, x):
        x_rec = self.net_recurse(x)
        x = self.conv_out(x_rec)
        x = self.out1(x)
        x = self.out2(x)
        return x


class _Net_recurse(torch.nn.Module):
    def __init__(self, n_in_channels, mult_chan=2, depth=0):
        """Class for recursive definition of U-network.p

        Parameters:
        in_channels - (int) number of channels for input.
        mult_chan - (int) factor to determine number of output channels
        depth - (int) if 0, this subnet will only be convolutions that double the channel count.
        """
        super().__init__()
        self.depth = depth
        n_out_channels = n_in_channels * mult_chan
        self.sub_2conv_more = SubNet2Conv(n_in_channels, n_out_channels)

        if depth > 0:
            self.sub_2conv_less = SubNet2Conv(3 * n_out_channels, n_out_channels)
            self.sub_down = SubNetdown(n_out_channels, n_out_channels * 2)

            self.convt = torch.nn.ConvTranspose3d(2 * n_out_channels, 2 * n_out_channels, kernel_size=2, stride=2)
            self.bn1 = torch.nn.BatchNorm3d(n_out_channels)
            self.relu1 = torch.nn.ReLU()
            self.sub_u = _Net_recurse(n_out_channels, mult_chan=2, depth=(depth - 1))

    def forward(self, x):
        if self.depth == 0:
            return x

        elif self.depth == 4:
            x_2conv_more = self.sub_2conv_more(x)
            x_down = self.sub_down(x_2conv_more)
            x_sub_u = self.sub_u(x_down)

            x_convt = self.convt(x_sub_u)
            x_cat = torch.cat((x_2conv_more, x_convt), 1)  # concatenate

            x_2conv_less = self.sub_2conv_less(x_cat)

        else:  # depth > 0
            # x_2conv_more = self.sub_2conv_more(x)
            x_down = self.sub_down(x)
            x_sub_u = self.sub_u(x_down)

            x_convt = self.convt(x_sub_u)
            x_cat = torch.cat((x_convt, x), 1)  # concatenate  384

            x_2conv_less = self.sub_2conv_less(x_cat)
        return x_2conv_less


# 下采样操作
class SubNetdown(torch.nn.Module):
    def __init__(self, n_in, n_out, use_1x1conv=True):
        super().__init__()
        self.input = n_in
        self.output = n_out
        self.conv_down = torch.nn.Conv3d(n_in, n_out, 2, stride=2)
        self.bn0 = torch.nn.BatchNorm3d(n_out)
        self.relu0 = torch.nn.ReLU()
        self.pool = torch.nn.AvgPool3d(3, stride=2, padding=1)

        self.conv2 = torch.nn.Conv3d(n_out, n_out, kernel_size=3, padding=1)
        self.bn2 = torch.nn.BatchNorm3d(n_out)
        self.relu2 = torch.nn.LeakyReLU()
        # dropout
        self.dropout = torch.nn.Dropout(p=0.2)
        if use_1x1conv:
            self.conv3 = torch.nn.Conv3d(n_in, n_out, kernel_size=1, stride=1)
        else:
            self.conv3 = None

    def forward(self, x):

        y = self.conv_down(x)
        y = self.relu0(y)
        y = self.bn0(y)
        # dropout
        y = self.dropout(y)

        y = self.conv2(y)
        y = self.relu2(y)
        y = self.bn2(y)
        # dropout
        y = self.dropout(y)

        if self.conv3:
            x_pool = self.pool(x)
            x_end = self.conv3(x_pool)
        y += x_end
        return y


class SubNet2Conv(torch.nn.Module):
    def __init__(self, n_in, n_out, use_1x1conv=True):
        super().__init__()
        self.conv1 = torch.nn.Conv3d(n_in, n_out, kernel_size=3, padding=1)
        self.bn1 = torch.nn.BatchNorm3d(n_out)
        self.relu1 = torch.nn.LeakyReLU()
        self.conv2 = torch.nn.Conv3d(n_out, n_out, kernel_size=3, padding=1)
        self.bn2 = torch.nn.BatchNorm3d(n_out)
        self.relu2 = torch.nn.LeakyReLU()
        # dropout
        self.dropout = torch.nn.Dropout(p=0.2)
        if use_1x1conv:
            self.conv3 = torch.nn.Conv3d(n_in, n_out, kernel_size=1, stride=1)
        else:
            self.conv3 = None

    def forward(self, x):

        y = self.conv1(x)
        y = self.relu1(y)
        y = self.bn1(y)
        # dropout
        y = self.dropout(y)

        y = self.conv2(y)
        y = self.relu2(y)
        y = self.bn2(y)
        # dropout
        y = self.dropout(y)

        if self.conv3:
            x = self.conv3(x)
        y += x
        return y


from torch.utils.data import DataLoader
from torch.utils.data.dataset import Dataset


# from torchsummary import summary
class TrainSet(Dataset):
    def __init__(self, X, Y):
        # 定义好 image 的路径
        self.X, self.Y = X, Y

    def __getitem__(self, index):
        return self.X[index], self.Y[index]

    def __len__(self):
        return len(self.X)


def main():
    X_tensor = torch.ones((4, 1, 32, 256, 256))
    Y_tensor = torch.zeros((4, 1, 32, 256, 256))
    mydataset = TrainSet(X_tensor, Y_tensor)
    train_loader = DataLoader(mydataset, batch_size=2, shuffle=True)

    net = Net()
    print(net)
    import torch.nn as nn
    loss_fn = nn.MSELoss()
    optimizer = torch.optim.SGD(net.parameters(), lr=1e-3)

    # 3) Training loop
    for epoch in range(10):
        for i, (X, y) in enumerate(train_loader):
            # predict = forward pass with our model
            pred = net(X)
            loss = loss_fn(pred, y)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            print('epoch={},i={}'.format(epoch, i))


if __name__ == '__main__':
    main()
