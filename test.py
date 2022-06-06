import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image
import config
from models import Generator

gen = Generator().to('cuda')
gen.load_state_dict(torch.load('pix2pix-v1-256-25e.pth.tar')['gen'])
test_img = np.array(Image.open("/home/prasanna/dataset/pix2pix-data/maps/maps/val/1.jpg")) / 255.0
x = test_img[:, 600:, :]
x = cv2.resize(x, (256, 256))
plt.imshow(test_img)
plt.show()

img = torch.tensor(x.reshape(1, 3, 256, 256), dtype=torch.float32)
pred = gen(img.to('cuda'))
plt.imshow(pred.detach().cpu().reshape(256, 256, 3))
plt.show()
