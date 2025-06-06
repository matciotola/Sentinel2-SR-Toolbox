import torch
from torch import nn
from torch.nn import functional as func

class S2_SSC_CNN_model(nn.Module):
    def __init__(self, nbands_10, nbands_20):

        super(S2_SSC_CNN_model, self).__init__()
        nbands = nbands_10 + nbands_20

        self.conv1 = nn.Conv2d(nbands,128, 3, padding='same', padding_mode='reflect')
        self.skip1 = nn.Conv2d(128,25, 3, padding='same', padding_mode='reflect')
        self.outl1 = nn.Conv2d(128+25, 128, 3, padding='same', padding_mode='reflect')


        self.conv2 = nn.Conv2d(128,128, 3, padding='same', padding_mode='reflect')
        self.skip2 = nn.Conv2d(128,25, 3, padding='same', padding_mode='reflect')
        self.outl2 = nn.Conv2d(128+25, 128, 3, padding='same', padding_mode='reflect')

        self.conv3 = nn.Conv2d(128,128, 3, padding='same', padding_mode='reflect')
        self.outl3 = nn.Conv2d(128, 128, 3, padding='same', padding_mode='reflect')

        self.conv4 = nn.Conv2d(128, nbands_20, 3, padding='same', padding_mode='reflect')

    def forward(self, bands_10, bands_20_upsampled):

        inp = torch.cat([bands_10, bands_20_upsampled], dim=1)

        x = func.leaky_relu(self.conv1(inp), negative_slope=0.01)
        skip1 = self.skip1(x)

        x = func.leaky_relu(self.conv2(x), negative_slope=0.01)
        skip2 = self.skip2(x)

        x = func.leaky_relu(self.conv3(x), negative_slope=0.01)

        y = func.leaky_relu(self.outl3(x), negative_slope=0.01)

        y = torch.cat([y, skip2], dim=1)
        y = func.leaky_relu(self.outl2(y), negative_slope=0.01)

        y = torch.cat([y, skip1], dim=1)
        y = func.leaky_relu(self.outl1(y), negative_slope=0.01)

        out = bands_20_upsampled + self.conv4(y)

        return out

if __name__ == '__main__':
    model = S2_SSC_CNN_model(4, 6)

    fake_inp_10 = torch.rand((1, 4, 256, 256))
    fake_inp_20 = torch.rand((1, 6, 256, 256))

    out = model(fake_inp_10, fake_inp_20)