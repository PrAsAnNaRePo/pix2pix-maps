import torch

DIR = '/home/prasanna/dataset/pix2pix-data/maps/maps/train'
VAL_DIR = '/home/prasanna/dataset/pix2pix-data/maps/maps/val'
NUM_WORKERS = 2
N_SAMPLES = 2000
IMAGE_SHAPE = (256, 256)
EPOCHS = 25
BATCH_SIZE = 16
LR = 3e-4
TEST_FREQUENCY = 5
DEVICE = torch.device('cuda')
Z_shape = (30, 30)
L1_LAMBDA = 100
MODEL_PATH = "pix2pix-v1-256-25e.pth.tar"

