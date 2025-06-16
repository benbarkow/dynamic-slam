import torch
import torch.nn.functional as F
import torchvision
import numpy as np
import cv2
import os
import math
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from skimage.filters import threshold_multiotsu

class AttentionMaskGenerator:

    def __init__(
        self,
    ):
			pass