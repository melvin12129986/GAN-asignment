import torch
import torch.nn as nn

# ==========================================
# 1. Vanilla GAN (Fully-Connected / MLP)
# ==========================================

class VanillaGenerator(nn.Module):
    def __init__(self, latent_dim=64, hidden_dim1=256, hidden_dim2=512, output_dim=784):
        super(VanillaGenerator, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(hidden_dim1, hidden_dim2),
            nn.BatchNorm1d(hidden_dim2),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(hidden_dim2, output_dim),
            nn.Sigmoid()
        )

    def forward(self, z):
        img = self.net(z)
        return img.view(-1, 1, 28, 28)


class VanillaDiscriminator(nn.Module):
    def __init__(self, input_dim=784, hidden_dim1=512, hidden_dim2=256):
        super(VanillaDiscriminator, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim1, hidden_dim2),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim2, 1)
        )

    def forward(self, img):
        x = img.view(img.size(0), -1)
        logits = self.net(x)
        return logits


# ==========================================
# 2. Deep Convolutional GAN (DCGAN)
# ==========================================

def weights_init_normal(m):
    classname = m.__class__.__name__
    if classname.find("Conv") != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find("BatchNorm2d") != -1 or classname.find("BatchNorm1d") != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0.0)


class DCGANGenerator(nn.Module):
    """
    DCGAN Generator untuk citra 28x28:
    z in R^{100} -> Linear(100, 128*7*7) -> ConvTranspose2d (14x14) -> ConvTranspose2d (28x28)
    """
    def __init__(self, latent_dim=100):
        super(DCGANGenerator, self).__init__()
        self.latent_dim = latent_dim
        self.fc = nn.Sequential(
            nn.Linear(latent_dim, 128 * 7 * 7),
            nn.BatchNorm1d(128 * 7 * 7),
            nn.ReLU(True)
        )
        self.conv_blocks = nn.Sequential(
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),  # 7x7 -> 14x14
            nn.BatchNorm2d(64),
            nn.ReLU(True),
            nn.ConvTranspose2d(64, 1, kernel_size=4, stride=2, padding=1),    # 14x14 -> 28x28
            nn.Sigmoid()
        )

    def forward(self, z):
        out = self.fc(z)
        out = out.view(-1, 128, 7, 7)
        img = self.conv_blocks(out)
        return img


class DCGANDiscriminator(nn.Module):
    """
    DCGAN Discriminator untuk citra 28x28:
    Conv2d (stride 2: 28->14) -> Conv2d (stride 2: 14->7) -> Linear -> 1
    """
    def __init__(self):
        super(DCGANDiscriminator, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=4, stride=2, padding=1),  # 28x28 -> 14x14
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1), # 14x14 -> 7x7
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True)
        )
        self.fc = nn.Linear(128 * 7 * 7, 1)

    def forward(self, img):
        out = self.conv(img)
        out = out.view(out.size(0), -1)
        logits = self.fc(out)
        return logits
