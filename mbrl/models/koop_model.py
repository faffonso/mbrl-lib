from collections import OrderedDict

import hydra
import omegaconf
import torch
from torch import nn as nn
from torch.nn import functional as F

from .model import Ensemble
from .util import EnsembleLinearLayer, truncated_normal_init


def main():

    
    nn = DeepKoopman(


    print(nn)

if __name__ == "__main__":
    main()