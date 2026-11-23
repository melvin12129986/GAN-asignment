import torch
import numpy as np
from torch.utils.data import DataLoader, random_split, ConcatDataset
import torchvision.datasets as datasets
import torchvision.transforms as transforms

def set_seed(seed=42):
    """Menetapkan seed acak untuk menjamin reproduktibilitas eksperimen."""
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def get_dataloaders(root="./data", batch_size=128, seed=42, num_workers=0):
    """
    Menyiapkan DataLoader Fashion-MNIST dengan aturan standardisasi:
    - Total: 70.000 citra (60k train + 10k test digabung, lalu dibagi 80%/10%/10%)
    - Train Set: 56.000 citra (80%)
    - Validation Set: 7.000 citra (10%)
    - Test Set: 7.000 citra (10%)
    - Nilai piksel dalam rentang [0.0, 1.0]
    """
    set_seed(seed)
    
    transform = transforms.Compose([
        transforms.ToTensor()
    ])
    
    raw_train = datasets.FashionMNIST(root=root, train=True, download=True, transform=transform)
    raw_test = datasets.FashionMNIST(root=root, train=False, download=True, transform=transform)
    
    full_dataset = ConcatDataset([raw_train, raw_test])
    total_len = len(full_dataset)
    assert total_len == 70000, f"Total dataset harus 70.000 citra, didapat {total_len}"
    
    train_size = int(0.80 * total_len)  # 56,000
    val_size = int(0.10 * total_len)    # 7,000
    test_size = total_len - train_size - val_size  # 7,000
    
    generator = torch.Generator().manual_seed(seed)
    train_dataset, val_dataset, test_dataset = random_split(
        full_dataset, [train_size, val_size, test_size], generator=generator
    )
    
    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers
    )
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers
    )
    test_loader = DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers
    )
    
    return train_loader, val_loader, test_loader
