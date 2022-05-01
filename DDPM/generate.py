import os
import torch
import torchvision
import torchvision.transforms as transforms
import numpy as np
from denoising_diffusion_pytorch import Unet, GaussianDiffusion, CIFARTrainer
import matplotlib.pyplot as plt

def imshow(inp, title=None):
    """Imshow for Tensor."""
    inp = inp.numpy().transpose((1, 2, 0))
    mean = np.array([0.5, 0.5, 0.5])
    std = np.array([0.5, 0.5, 0.5])
    inp = std * inp + mean
    inp = np.clip(inp, 0, 1)
    # plt.imshow(inp)
    # if title is not None:
    #     plt.title(title)
    # plt.pause(0.001)  # pause a bit so that plots are updated
    return inp

model = Unet(
    dim = 64,
    dim_mults = (1, 2, 4, 8)
).cuda()

diffusion = GaussianDiffusion(
    model,
    image_size = 32,
    timesteps = 1000,   # number of steps
    loss_type = 'l1'    # L1 or L2
).cuda()

diffusion.load_state_dict(torch.load('/home/xc2057/denoising-diffusion-pytorch/new_results/model-13.pt')["model"])

sampled_images = diffusion.sample(batch_size = 1000)
print(sampled_images.shape)

if not os.path.exists("output"):
    os.mkdir("output")

for i in range(0, sampled_images.shape[0]):
    plt.imsave(os.path.join("output", str(i) + '.png'), imshow(sampled_images[i].cpu()))

