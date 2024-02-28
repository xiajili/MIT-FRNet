import torch.nn as nn
from models.modelM import *
__all__ = ['AMIO']
MODEL_MAP = {
    'MIT-FRNet': MIT-FRNet,
}
class AMIO(nn.Module):
    def __init__(self, args):
        super(AMIO, self).__init__()
        lastModel = MODEL_MAP[args.modelName]
        self.Model = lastModel(args)
    def forward(self, text_x, audio_x, video_x):
        return self.Model(text_x, audio_x, video_x)