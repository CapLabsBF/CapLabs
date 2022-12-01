import joblib as jb
import torch

def Loadjob(path):

    data = jb.load(path, mmap_mode = 'r')
    return data
