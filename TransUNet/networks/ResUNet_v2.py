import torch
import torch.nn as nn


class ResidualConv(nn.Module):
    def __init__(self, input_dim, output_dim, stride, padding):
        super(ResidualConv, self).__init__()

        self.conv_block = nn.Sequential(
            nn.Conv2d(
                input_dim, output_dim, kernel_size=3, stride=stride, padding=padding
            ),
            nn.ReLU(),
            nn.BatchNorm2d(output_dim),
            nn.Dropout(),
            nn.Conv2d(output_dim, output_dim, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(output_dim),
            nn.Dropout(),
        )
        self.conv_skip = nn.Sequential(
            nn.AvgPool2d(kernel_size=2),
            nn.Conv2d(input_dim, output_dim, kernel_size=3, stride=1, padding=1),
            # nn.BatchNorm2d(output_dim),
        )

    def forward(self, x):
        return self.conv_block(x) + self.conv_skip(x)

class UpResidualConv(nn.Module):
    def __init__(self, input_dim, output_dim, stride, padding):
        super(UpResidualConv, self).__init__()

        self.conv_block = nn.Sequential(
            nn.Conv2d(
                input_dim, output_dim, kernel_size=3, stride=stride, padding=padding
            ),
            nn.ReLU(),
            nn.BatchNorm2d(output_dim),
            nn.Dropout(),
            nn.Conv2d(output_dim, output_dim, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(output_dim),
            nn.Dropout(),
        )
        self.conv_skip = nn.Sequential(
            nn.Conv2d(input_dim, output_dim, kernel_size=3, stride=1, padding=1),

        )

    def forward(self, x):
        return self.conv_block(x) + self.conv_skip(x)


class Upsample(nn.Module):
    def __init__(self, input_dim, output_dim, kernel, stride):
        super(Upsample, self).__init__()

        # self.upsample = nn.ConvTranspose2d(
        #     input_dim, output_dim, kernel_size=kernel, stride=stride
        # )
        self.upsample = nn.UpsamplingBilinear2d(scale_factor=2)

    def forward(self, x):
        return self.upsample(x)


class ResUnet(nn.Module):
    def __init__(self, channel, filters=[16, 32, 64, 128, 256]):
        super(ResUnet, self).__init__()

        self.input_layer = nn.Sequential(
            nn.Conv2d(channel, filters[0], kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(filters[0]),
            nn.Dropout(),
            nn.Conv2d(filters[0], filters[0], kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(filters[0]),
            nn.Dropout(),
        )
        self.input_skip = nn.Sequential(
            nn.Conv2d(channel, filters[0], kernel_size=3, padding=1)
        )

        self.residual_conv_1 = ResidualConv(filters[0], filters[1], 2, 1)
        self.residual_conv_2 = ResidualConv(filters[1], filters[2], 2, 1)
        self.residual_conv_3 = ResidualConv(filters[2], filters[3], 2, 1)

        self.bridge = ResidualConv(filters[3], filters[4], 2, 1)

        self.upsample_1 = Upsample(filters[4], filters[4], 2, 2)
        self.up_residual_conv1 = UpResidualConv(filters[4] + filters[3], filters[3], 1, 1)

        self.upsample_2 = Upsample(filters[3], filters[3], 2, 2)
        self.up_residual_conv2 = UpResidualConv(filters[3] + filters[2], filters[2], 1, 1)

        self.upsample_3 = Upsample(filters[2], filters[2], 2, 2)
        self.up_residual_conv3 = UpResidualConv(filters[2] + filters[1], filters[1], 1, 1)

        self.upsample_4 = Upsample(filters[1], filters[1], 2, 2)
        self.up_residual_conv4 = UpResidualConv(filters[1] + filters[0], filters[0], 1, 1)

        self.output_layer = nn.Sequential(
            nn.Conv2d(filters[0], 1, 1, 1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        # Encode
        x1 = self.input_layer(x) + self.input_skip(x)
        x2 = self.residual_conv_1(x1)
        x3 = self.residual_conv_2(x2)
        x4 = self.residual_conv_3(x3)
        # Bridge
        x5 = self.bridge(x4)
        # Decode
        x5 = self.upsample_1(x5)
        x6 = torch.cat([x5, x4], dim=1)

        x7 = self.up_residual_conv1(x6)

        x7 = self.upsample_2(x7)
        x8 = torch.cat([x7, x3], dim=1)

        x9 = self.up_residual_conv2(x8)

        x9 = self.upsample_3(x9)
        x10 = torch.cat([x9, x2], dim=1)

        x11 = self.up_residual_conv3(x10)

        x11 = self.upsample_4(x11)
        x12 = torch.cat([x11, x1], dim=1)

        x13 = self.up_residual_conv4(x12)

        output = self.output_layer(x13)

        return output


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
    X_tensor = torch.ones((4, 3, 256, 256))
    Y_tensor = torch.zeros((4, 3, 256, 256))
    mydataset = TrainSet(X_tensor, Y_tensor)
    train_loader = DataLoader(mydataset, batch_size=2, shuffle=True)

    net = ResUnet(channel=3)
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
