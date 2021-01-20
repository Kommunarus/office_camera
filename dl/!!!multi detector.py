import multiprocessing as mp
import glob
import os
from dl.detec_tracking_db import  multiDetect
import torch
from multiprocessing import Process
layer = 'test'

os.environ['CUDA_VISIBLE_DEVICES'] = '0,1'  # set environment variable


def runDetect(file, device):
    with torch.no_grad():
        multiDetect(file, str(device), layer)

if __name__ == '__main__':
    path = 'data/samples'
    files = sorted(glob.glob(os.path.join(path, '*.*')))


    #pool.close()
    procs = []
    for i, file in enumerate(files):
        #print('gpu ', i%2)
        p = Process(target=runDetect, args=(file, i%2))
        procs.append(p)
        p.start()
        if (i%2==0 and i!=0) or i == (len(files)-1):
            for proc in procs:
                proc.join()
            procs = []

        print(i,len(files))