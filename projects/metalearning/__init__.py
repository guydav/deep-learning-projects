import torch
# Does this fix the hdf5 multiprocessing bug?
torch.multiprocessing.set_start_method("spawn")

from . import base_model
from . import cnnmlp
from . import dataset

from .base_model import *
from .cnnmlp import *
from .dataset import *
