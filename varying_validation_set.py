
# FWIW I ran into this:
# https://stackoverflow.com/questions/45016449/pycharm-debugger-instantly-exits-with-139-code
import torch

from data_loader import get_mnist_loader
from model import to_var
from train_ny import train_lre
import numpy as np
import pandas as pd

allowed_neg = 1E-4


hp = {
    'lr': 1e-2,
    'momentum': 0.9,
    'batch_size': 128,
    'num_iterations': 1001,
}

train = get_mnist_loader(hp['batch_size'], classes=[9, 4], proportion=0.95, mode="train")
test = get_mnist_loader(hp['batch_size'], classes=[9, 4], proportion=0.5, mode="test")

val_data = to_var(train.dataset.data_val, requires_grad=False)
val_labels = to_var(train.dataset.labels_val, requires_grad=False)

data_log = dict()
accuracy_log = dict()

base_seed = 123
for i in range(10):
    np.random.seed(base_seed + i)
    train = get_mnist_loader(hp['batch_size'], classes=[9, 4], proportion=0.95, mode="train")
    val_data = to_var(train.dataset.data_val, requires_grad=False)
    val_labels = to_var(train.dataset.labels_val, requires_grad=False)
    np.random.seed(base_seed)
    torch.manual_seed(base_seed)
    data_log[i], accuracy_log[i] = train_lre(hp, train, val_data, val_labels, test)

data_log = pd.concat(data_log)
data_log.index.set_names('run', level=0, inplace=True)
accuracy_log = pd.concat(accuracy_log)
accuracy_log.index.set_names('run', level=0, inplace=True)

data_log.to_pickle('VaryingValidationSet_Data.p')
accuracy_log.to_pickle('VaryingValidationSet_Accuracy.p')
