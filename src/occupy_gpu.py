import os
import torch
from tqdm import tqdm
import time

# https://gist.github.com/sparkydogX/845b658e3e6cef58a7bf706a9f43d7bf

# declare which gpu devices to use
cuda_devices = '1,2,3'

# Memory needed
block_mem = 8000

# Time for reservation (in seconds)
reserved_time = 360000

def occumpy_mem(cuda_device):
    torch.cuda.set_device('cuda:'+cuda_device)
    x = torch.cuda.FloatTensor(256, 1024, block_mem)
    del x


if __name__ == '__main__':
    os.environ["CUDA_VISIBLE_DEVICES"] = cuda_devices
    devices = cuda_devices.split(',')
    for i in range(len(devices)):
        print(f'Occupying Device No:', devices[i])
        occumpy_mem(str(i))

    for _ in tqdm(range(reserved_time)):
        time.sleep(1)
    print('Done')