import jittor as jt
from jittor import nn

class UNet(nn.Module):
    def __init__(self, in_channels=3, out_channels=3, num_classes=10):
        super(UNet, self).__init__()
        def double_conv(in_ch, out_ch):
            return nn.Sequential(
                nn.Conv(in_ch, out_ch, 3, padding=1),
                nn.BatchNorm(out_ch),
                nn.ReLU(),
                nn.Conv(out_ch, out_ch, 3, padding=1),
                nn.BatchNorm(out_ch),
                nn.ReLU()
            )
        self.down1 = double_conv(in_channels, 64)
        self.down2 = double_conv(64, 128)
        self.down3 = double_conv(128, 256)
        self.up1 = double_conv(256 + 128, 128)
        self.up2 = double_conv(128 + 64, 64)
        self.up3 = nn.Conv(64, out_channels, 1)
        self.pool = nn.Pool(2, op='maximum')
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear')
        self.time_emb = nn.Sequential(
            nn.Linear(1, 64), nn.ReLU(), nn.Linear(64, 64)
        )
        self.class_emb = nn.Sequential(
            nn.Embedding(num_classes, 64), nn.ReLU(), nn.Linear(64, 64)
        )

    def execute(self, x, t, y=None):
        t_emb = self.time_emb(t.reshape(-1, 1)).reshape(-1, 64, 1, 1)
        emb = t_emb + self.class_emb(y).reshape(-1, 64, 1, 1) if y is not None else t_emb
        x1 = self.down1(x)
        x2 = self.down2(self.pool(x1))
        x3 = self.down3(self.pool(x2))
        x = self.up1(jt.concat([self.upsample(x3), x2], dim=1))
        x = self.up2(jt.concat([self.upsample(x), x1], dim=1))
        return self.up3(x)

class FeatureExtractor(nn.Module):
    def __init__(self, in_channels=3):
        super().__init__()
        self.layer1 = nn.Sequential(nn.Conv(in_channels, 64, 3, padding=1), nn.ReLU(), nn.Pool(2, op='maximum'))
        self.layer2 = nn.Sequential(nn.Conv(64, 128, 3, padding=1), nn.ReLU(), nn.Pool(2, op='maximum'))
        self.layer3 = nn.Sequential(nn.Conv(128, 256, 3, padding=1), nn.ReLU())

    def execute(self, x):
        return [self.layer1(x), self.layer2(self.layer1(x)), self.layer3(self.layer2(self.layer1(x)))]
