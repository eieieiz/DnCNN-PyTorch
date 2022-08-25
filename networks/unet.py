import torch
import torch.nn as nn


class ConvDownBlock(nn.Module):
    """Implementation of encoder part except for the first scale."""
    def __init__(self, in_channels=3, out_channels=3, ksize=3):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, ksize, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )
    def forward(self, x):

        return self.block(x)
    
class Conv2UpBlock(nn.Module):
    """Implementation of encoder part except for the first scale."""
    def __init__(self, in_channels=144, out_channels=96, ksize=3):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, ksize, padding=1),
            nn.ReLU(),
            nn.Conv2d(out_channels, out_channels, ksize, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(96, 96, 3, stride=2, padding=1, output_padding=1)
        )
    
    def forward(self, x):
        
        return self.block(x)
    
class UNet(nn.Module):

    def __init__(self, in_channels=3, out_channels=3, features=48, ksize=3, depth=6):
        super().__init__()

        # encoder
        self.encoder = nn.ModuleList([])

        self.block1 = nn.Sequential(
            nn.Conv2d(in_channels, features, ksize, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(features, features, ksize, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2))
        self.encoder.append(self.block1)

        for _ in range(depth-2):
            self.encoder.append(ConvDownBlock(in_channels=48, out_channels=48))

        # decoder
        self.decoder = nn.ModuleList([])

        self.block2 = nn.Sequential(
            nn.Conv2d(features, features, ksize, stride=1, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(48, 48, 3, stride=2, padding=1, output_padding=1)
        )
        self.decoder.append(self.block2)
        
        self.decoder.append(Conv2UpBlock(in_channels=2*features, out_channels=2*features))
        
        for _ in range(depth-3):
            self.decoder.append(Conv2UpBlock(in_channels=3*features, out_channels=2*features))
        
        self.block3 = nn.Sequential(
            nn.Conv2d(2*features+in_channels, 64, ksize, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 32, ksize, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, out_channels, ksize, stride=1, padding=1),
        )


    def forward(self, x):
        enc_feats = []
        for block in self.encoder:
            enc_feats.append(x)
            x = block(x)

        x = self.decoder[0](x)
        for i in range(len(self.decoder)-1):
            x = torch.cat( (enc_feats[::-1][i], x), dim=1 )
            x = self.decoder[i+1](x)
        x = torch.cat( (enc_feats[0], x), dim=1 )
        x = self.block3(x)
        
        return x



        


class UNet_(nn.Module):

    def __init__(self, in_channels=3, out_channels=3):
        """Initializes U-Net."""

        super(UNet_, self).__init__()

        # Layers: enc_conv0, enc_conv1, pool1
        self._block1 = nn.Sequential(
            nn.Conv2d(in_channels, 48, 3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(48, 48, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2))

        # Layers: enc_conv(i), pool(i); i=2..5
        self._block2 = nn.Sequential(
            nn.Conv2d(48, 48, 3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2))

        # Layers: enc_conv6, upsample5
        self._block3 = nn.Sequential(
            nn.Conv2d(48, 48, 3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(48, 48, 3, stride=2, padding=1, output_padding=1))
            #nn.Upsample(scale_factor=2, mode='nearest'))

        # Layers: dec_conv5a, dec_conv5b, upsample4
        self._block4 = nn.Sequential(
            nn.Conv2d(96, 96, 3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(96, 96, 3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(96, 96, 3, stride=2, padding=1, output_padding=1))
            #nn.Upsample(scale_factor=2, mode='nearest'))

        # Layers: dec_deconv(i)a, dec_deconv(i)b, upsample(i-1); i=4..2
        self._block5 = nn.Sequential(
            nn.Conv2d(144, 96, 3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(96, 96, 3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(96, 96, 3, stride=2, padding=1, output_padding=1))
            #nn.Upsample(scale_factor=2, mode='nearest'))

        # Layers: dec_conv1a, dec_conv1b, dec_conv1c,
        self._block6 = nn.Sequential(
            nn.Conv2d(96 + in_channels, 64, 3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 32, 3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, out_channels, 3, stride=1, padding=1),
            nn.LeakyReLU(0.1))

        # Initialize weights
        self._init_weights()


    def _init_weights(self):
        """Initializes weights using He et al. (2015)."""

        for m in self.modules():
            if isinstance(m, nn.ConvTranspose2d) or isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight.data)
                m.bias.data.zero_()


    def forward(self, x):
        """Through encoder, then decoder by adding U-skip connections. """

        # Encoder
        pool1 = self._block1(x)
        pool2 = self._block2(pool1)
        pool3 = self._block2(pool2)
        pool4 = self._block2(pool3)
        pool5 = self._block2(pool4)

        # Decoder
        upsample5 = self._block3(pool5)
        concat5 = torch.cat((upsample5, pool4), dim=1)
        upsample4 = self._block4(concat5)
        concat4 = torch.cat((upsample4, pool3), dim=1)
        upsample3 = self._block5(concat4)
        concat3 = torch.cat((upsample3, pool2), dim=1)
        upsample2 = self._block5(concat3)
        concat2 = torch.cat((upsample2, pool1), dim=1)
        upsample1 = self._block5(concat2)
        concat1 = torch.cat((upsample1, x), dim=1)

        # Final activation
        return self._block6(concat1)