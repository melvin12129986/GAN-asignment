import torch
import torch.nn as nn
import torch.nn.functional as F

class VanillaAutoencoder(nn.Module):
    """
    Vanilla Deterministic Autoencoder
    Encoder: 784 -> 256 -> 64
    Decoder: 64 -> 256 -> 784
    Aktivasi: ReLU pada hidden layer, Sigmoid pada output piksel [0, 1]
    """
    def __init__(self, input_dim=784, hidden_dim=256, latent_dim=64):
        super(VanillaAutoencoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, latent_dim),
            nn.ReLU(inplace=True)
        )
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, input_dim),
            nn.Sigmoid()
        )

    def encode(self, x):
        x = x.view(x.size(0), -1)
        return self.encoder(x)

    def decode(self, z):
        x_recon = self.decoder(z)
        return x_recon.view(-1, 1, 28, 28)

    def forward(self, x):
        z = self.encode(x)
        x_recon = self.decode(z)
        return x_recon, z


class SparseAutoencoder(nn.Module):
    """
    Sparse Autoencoder dengan penalti regularisasi L1 pada representasi laten z:
    Loss = MSE(x, x_recon) + lambda_l1 * ||z||_1
    """
    def __init__(self, input_dim=784, hidden_dim=256, latent_dim=64):
        super(SparseAutoencoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, latent_dim),
            nn.ReLU(inplace=True)
        )
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, input_dim),
            nn.Sigmoid()
        )

    def encode(self, x):
        x = x.view(x.size(0), -1)
        return self.encoder(x)

    def decode(self, z):
        x_recon = self.decoder(z)
        return x_recon.view(-1, 1, 28, 28)

    def forward(self, x):
        z = self.encode(x)
        x_recon = self.decode(z)
        return x_recon, z


class VariationalAutoencoder(nn.Module):
    """
    Variational Autoencoder (VAE)
    Encoder: 784 -> 256 -> mu(64), logvar(64)
    Reparameterization Trick: z = mu + std * eps, eps ~ N(0, I)
    Decoder: 64 -> 256 -> 784 (Sigmoid)
    Loss: Reconstruction Loss + KL Divergence
    """
    def __init__(self, input_dim=784, hidden_dim=256, latent_dim=64):
        super(VariationalAutoencoder, self).__init__()
        self.fc_enc = nn.Linear(input_dim, hidden_dim)
        self.fc_mu = nn.Linear(hidden_dim, latent_dim)
        self.fc_logvar = nn.Linear(hidden_dim, latent_dim)
        
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, input_dim),
            nn.Sigmoid()
        )

    def encode(self, x):
        x = x.view(x.size(0), -1)
        h = F.relu(self.fc_enc(x))
        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)
        return mu, logvar

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z):
        x_recon = self.decoder(z)
        return x_recon.view(-1, 1, 28, 28)

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        x_recon = self.decode(z)
        return x_recon, mu, logvar, z


class FrozenLatentDecoder(nn.Module):
    """
    Decoder terpisah untuk mengevaluasi kandungan informasi fitur laten MLP (Tahap 4):
    Arsitektur: 64 -> 256 -> 784 (Sigmoid)
    Dilatih merekonstruksi citra asli dari z_MLP yang dibekukan (requires_grad = False).
    """
    def __init__(self, latent_dim=64, hidden_dim=256, output_dim=784):
        super(FrozenLatentDecoder, self).__init__()
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, output_dim),
            nn.Sigmoid()
        )

    def forward(self, z):
        x_recon = self.decoder(z)
        return x_recon.view(-1, 1, 28, 28)
