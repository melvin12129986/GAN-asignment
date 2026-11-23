import torch
import torch.nn as nn
import torch.nn.functional as F

class MLPClassifier(nn.Module):
    """
    Multi-Layer Perceptron (MLP) Classifier: 784 -> 256 -> 64 -> 10
    Sesuai panduan Bagian 2.1 Tahap 1:
    - Layer 1: W1 in R^{256x784}, b1 in R^{256}
    - Layer 2: W2 in R^{64x256}, b2 in R^{64}
    - Layer 3: W3 in R^{10x64}, b3 in R^{10}
    - Ekstraksi fitur laten z_MLP in R^{64} pada lapisan sebelum klasifikasi akhir.
    """
    def __init__(self, input_dim=784, hidden_dim=256, latent_dim=64, num_classes=10):
        super(MLPClassifier, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, latent_dim)
        self.fc3 = nn.Linear(latent_dim, num_classes)
        
    def extract_features(self, x):
        """Memetakan input x ke representasi laten z_MLP in R^{64}."""
        x = x.view(x.size(0), -1)
        h1 = F.relu(self.fc1(x))
        z = F.relu(self.fc2(h1))
        return z

    def forward(self, x):
        z = self.extract_features(x)
        logits = self.fc3(z)
        return logits, z
