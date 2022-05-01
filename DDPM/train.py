import torch
import torchvision
import torchvision.transforms as transforms
from denoising_diffusion_pytorch import Unet, GaussianDiffusion, CIFARTrainer

# Load CIFAR-10
transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
batch_size = 4
trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                        download=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,
                                          shuffle=True, num_workers=1)
testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                       download=True, transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size,
                                         shuffle=False, num_workers=1)
classes = ('plane', 'car', 'bird', 'cat',
           'deer', 'dog', 'frog', 'horse', 'ship', 'truck')


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

trainer = CIFARTrainer(
    diffusion,
    trainset,
    train_batch_size = 128,
    train_lr = 2e-5,
    train_num_steps = 1000000,         # total training steps
    gradient_accumulate_every = 2,    # gradient accumulation steps
    ema_decay = 0.995,                # exponential moving average decay
    amp = True,                        # turn on mixed precision
    save_and_sample_every = 40000,
    results_folder = './new_results'
)
trainer.train()


# model = Unet(
#     dim = 64,
#     dim_mults = (1, 2, 4, 8)
# )

# diffusion = GaussianDiffusion(
#     model,
#     image_size = 128,
#     timesteps = 1000,   # number of steps
#     loss_type = 'l1'    # L1 or L2
# )

# training_images = torch.randn(8, 3, 128, 128)
# loss = diffusion(training_images)
# loss.backward()
# # after a lot of training

# sampled_images = diffusion.sample(batch_size = 4)
# print(sampled_images.shape) # (4, 3, 128, 128)