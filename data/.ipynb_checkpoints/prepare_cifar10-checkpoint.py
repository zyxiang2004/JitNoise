import jittor as jt
from jittor.dataset.cifar import CIFAR10
import os

def prepare_data(data_dir="./cifar10"):
    train_set = CIFAR10(data_dir=data_dir, train=True, download=True)
    test_set = CIFAR10(data_dir=data_dir, train=False, download=True)
    return train_set, test_set

if __name__ == "__main__":
    prepare_data()
