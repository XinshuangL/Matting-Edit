import torch
import numpy
import random

torch.manual_seed(0)
torch.cuda.manual_seed_all(0)
numpy.random.seed(0)
random.seed(0)
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True

print('All seeds have been set')
