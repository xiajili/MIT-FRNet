from trains.baselines import *
from trains.modelM import *
__all__ = ['ATIO']
class ATIO():
    def __init__(self):
        self.TRAIN_MAP = {
            'MIT-FRNet': MIT-FRNet,
        }
    def getTrain(self, args):
        return self.TRAIN_MAP[args.modelName.lower()](args)
