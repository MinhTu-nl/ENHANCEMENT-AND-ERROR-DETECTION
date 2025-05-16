import torch
import torch.nn as nn
from torchvision.models import vgg16

class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        # Encoder
        self.enc1 = nn.Sequential(nn.Conv2d(3, 64, 4, stride=2, padding=1), nn.ReLU())
        self.enc2 = nn.Sequential(nn.Conv2d(64, 128, 4, stride=2, padding=1), nn.ReLU())
        self.enc3 = nn.Sequential(nn.Conv2d(128, 256, 4, stride=2, padding=1), nn.ReLU())
        self.enc4 = nn.Sequential(nn.Conv2d(256, 512, 4, stride=2, padding=1), nn.ReLU())

        # Decoder with skip connections
        self.dec4 = nn.Sequential(nn.ConvTranspose2d(512, 256, 4, stride=2, padding=1), nn.ReLU())
        self.dec3 = nn.Sequential(nn.ConvTranspose2d(512, 128, 4, stride=2, padding=1), nn.ReLU())
        self.dec2 = nn.Sequential(nn.ConvTranspose2d(256, 64, 4, stride=2, padding=1), nn.ReLU())
        self.dec1 = nn.Sequential(nn.ConvTranspose2d(128, 3, 4, stride=2, padding=1), nn.Tanh())

    def forward(self, x):
        e1 = self.enc1(x)
        e2 = self.enc2(e1)
        e3 = self.enc3(e2)
        e4 = self.enc4(e3)

        d4 = self.dec4(e4)
        d4 = torch.cat([d4, e3], dim=1)
        d3 = self.dec3(d4)
        d3 = torch.cat([d3, e2], dim=1)
        d2 = self.dec2(d3)
        d2 = torch.cat([d2, e1], dim=1)
        d1 = self.dec1(d2)
        return d1

class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.model = nn.Sequential(
            nn.Conv2d(3, 64, 4, stride=2, padding=1),
            nn.LeakyReLU(0.2),
            nn.Conv2d(64, 128, 4, stride=2, padding=1),
            nn.LeakyReLU(0.2),
            nn.Conv2d(128, 256, 4, stride=2, padding=1),
            nn.LeakyReLU(0.2),
            nn.Conv2d(256, 1, 4, stride=1, padding=0),
            nn.Sigmoid(),
        )

    def forward(self, x):
        return self.model(x)

class PerceptualLoss(nn.Module):
    def __init__(self, device):
        super(PerceptualLoss, self).__init__()
        vgg = vgg16(pretrained=True).features[:16].eval().to(device)
        for param in vgg.parameters():
            param.requires_grad = False
        self.vgg = vgg
        self.loss = nn.MSELoss()

    def forward(self, fake, real):
        fake_features = self.vgg(fake)
        real_features = self.vgg(real)
        return self.loss(fake_features, real_features)

def initialize_weights(model):
    for m in model.modules():
        if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
            nn.init.normal_(m.weight.data, 0.0, 0.02)
        elif isinstance(m, nn.BatchNorm2d):
            nn.init.normal_(m.weight.data, 1.0, 0.02)
            nn.init.constant_(m.bias.data, 0)