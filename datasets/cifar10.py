from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torchvision
from torch.utils.data import DataLoader


class Cifar10:
    def __init__(self, data_path, train_batch_size, eval_batch_size, num_workers, pin_memory):
        self.data_path = data_path
        self.train_batch_size = train_batch_size
        self.eval_batch_size = eval_batch_size
        self.num_workers = num_workers
        self.pin_memory = pin_memory

    def get_dataloader(self):
        # Data augmentation
        train_transform = torchvision.transforms.Compose([
            torchvision.transforms.RandomHorizontalFlip(),
            torchvision.transforms.RandomCrop(32, padding=4),
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                             std=[0.229, 0.224, 0.225])
        ])
        test_transform = torchvision.transforms.Compose([
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                             std=[0.229, 0.224, 0.225])
        ])
        dataloader = torchvision.datasets.CIFAR10
        trainset = dataloader(root=self.data_path, train=True, download=True, transform=train_transform)
        trainloader = DataLoader(trainset, batch_size=self.train_batch_size, shuffle=True,
                                 num_workers=self.num_workers, pin_memory=self.pin_memory)
        testset = dataloader(root=self.data_path, train=False, download=True, transform=test_transform)
        testloader = DataLoader(testset, batch_size=self.eval_batch_size, shuffle=False, num_workers=self.num_workers)
        dataloaders = {'train': trainloader, 'test': testloader}
        return dataloaders
