import torch
import torch.nn as nn

class AttentionGate(nn.Module):
    def __init__(self, gate_channels, skip_channels, inter_channels):
        super().__init__()
        self.W_g = nn.Sequential(
                nn.Conv2d(gate_channels, inter_channels, kernel_size=1, bias=False),
                nn.BatchNorm2d(inter_channels)
        )

        self.W_x = nn.Sequential(
                nn.Conv2d(skip_channels, inter_channels, kernel_size=1, bias=False),
                nn.BatchNorm2d(inter_channels)
        )

        self.psi = nn.Sequential(
                nn.Conv2d(inter_channels, 1, kernel_size=1),
                nn.Sigmoid()
        )

        self.relu = nn.ReLU(inplace=True)

    def forward(self, gate, skip):
        g = self.W_g(gate)
        x = self.W_x(skip)

        attention = self.relu(g+x)
        attention = self.psi(attention)

        return skip * attention

class UNet(nn.Module):
    def __init__(self):
        super().__init__()

        self.pool = nn.MaxPool2d(2)

        self.att4 = AttentionGate(512,512,256)
        self.att3 = AttentionGate(256,256,128)
        self.att2 = AttentionGate(128,128,64)
        self.att1 = AttentionGate(64,64,32)


        self.enc1 = DoubleConv(3, 64)
        self.enc2 = DoubleConv(64, 128)
        self.enc3 = DoubleConv(128, 256)
        self.enc4 = DoubleConv(256, 512)

        self.bottleneck = DoubleConv(512, 1024)

        self.up4 = nn.ConvTranspose2d(
                1024,
                512,
                kernel_size=2,
                stride=2,
                )
        
        self.dec4 = DoubleConv(
                1024,
                512,
                )

        self.up3 = nn.ConvTranspose2d(
                512,
                256,
                kernel_size=2,
                stride=2,
                )
        
        self.dec3 = DoubleConv(
                512,
                256,
                )

        self.up2 = nn.ConvTranspose2d(
                256,
                128,
                kernel_size=2,
                stride=2,
                )
        
        self.dec2 = DoubleConv(
                256,
                128,
                )

        self.up1 = nn.ConvTranspose2d(
                128,
                64,
                kernel_size=2,
                stride=2,
                )
        
        self.dec1 = DoubleConv(
                128,
                64,
                )

        self.output = nn.Conv2d(
                64,
                1,
                kernel_size=1,
                )

    def forward(self, x):
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))
        e4 = self.enc4(self.pool(e3))

        b = self.bottleneck(self.pool(e4))

        d4 = self.up4(b)
        e4 = self.att4(d4, e4)
        d4 = torch.cat([d4, e4], dim=1)
        d4 = self.dec4(d4)

        d3 = self.up3(d4)
        e3 = self.att3(d3, e3)
        d3 = torch.cat([d3, e3], dim=1)
        d3 = self.dec3(d3)

        d2 = self.up2(d3)
        e2 = self.att2(d2, e2)
        d2 = torch.cat([d2, e2], dim=1)
        d2 = self.dec2(d2)

        d1 = self.up1(d2)
        e1 = self.att1(d1, e1)
        d1 = torch.cat([d1, e1], dim=1)
        d1 = self.dec1(d1)

        return self.output(d1)

        

class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()

        self.conv = nn.Sequential(
                nn.Conv2d(
                    in_channels,
                    out_channels,
                    kernel_size=3,
                    padding=1,
                    ),

                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True),

                nn.Conv2d(
                    out_channels,
                    out_channels,
                    kernel_size=3,
                    padding=1,
                    ),

                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True),
                )

    def forward(self, x):
        return self.conv(x)
