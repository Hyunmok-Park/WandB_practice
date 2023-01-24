from torchvision import datasets
from torchvision import transforms
from torch.utils.data import DataLoader

def Load_Dataset(batch_size):
    train_data = datasets.MNIST(".",
                                train=True,
                                download=True,
                                transform=transforms.ToTensor())

    test_data = datasets.MNIST(".",
                               train=False,
                               download=True,
                               transform=transforms.ToTensor())

    train_loader = DataLoader(train_data, batch_size=batch_size)
    test_loader = DataLoader(test_data, batch_size=batch_size)

    return train_loader, test_loader