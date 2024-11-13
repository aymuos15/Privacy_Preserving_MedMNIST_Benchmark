import numpy as np

from torch.utils.data import DataLoader
from torchvision import transforms

from dataset import MedMNISTDataset

from config import CONFIG

def load_data(data_path):

    data = np.load(data_path)

    train_images = data['train_images']
    train_labels = data['train_labels']
    test_images = data['test_images']
    test_labels = data['test_labels']

    data_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[.5], std=[.5])
    ])

    train_dataset = MedMNISTDataset(train_images, train_labels, transform=data_transform)
    test_dataset = MedMNISTDataset(test_images, test_labels, transform=data_transform)

    train_loader = DataLoader(train_dataset, batch_size=CONFIG['batch_size'], shuffle=True, drop_last=False)
    test_loader = DataLoader(test_dataset, batch_size=CONFIG['batch_size'], shuffle=True, drop_last=False)

    class_counts = {
        'train': np.bincount(train_labels.flatten()),
        'test': np.bincount(test_labels.flatten())
    }

    channels = train_images.shape[-1] if len(train_images.shape) == 4 else 1

    return train_loader, test_loader, class_counts, channels
