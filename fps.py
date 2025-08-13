import numpy as np
import torch
import time
from tqdm import tqdm
import os

def computeTime(model1, device='cuda'):
    inputs1 = torch.randn(1, 3, 384, 384)
    if device == 'cuda':
        model1 = model1.cuda()
        inputs1 = inputs1.cuda()

    model1.eval()

    time_spent = []
    for idx in tqdm(range(1000)):
        start_time = time.time()
        with torch.no_grad():
            _ = model1(inputs1,inputs1,inputs1)

        if device == 'cuda':
            torch.cuda.synchronize()  # wait for cuda to finish (cuda is asynchronous!)
        if idx > 200:
            time_spent.append(time.time() - start_time)
    print('Average speed: {:.4f} fps'.format(1 / np.mean(time_spent)))


torch.backends.cudnn.benchmark = True

from model.TriFuse import TriFuse

model1 = TriFuse().cuda()

computeTime(model1)