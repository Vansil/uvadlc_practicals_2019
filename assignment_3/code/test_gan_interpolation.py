from a3_gan_template import Generator
from torchvision.utils import save_image
import torch
import os
import numpy as np


model_path = 'output/gan/run1/checkpoints/mnist_generator_4000.pt'

device = 'cpu'

with torch.no_grad():
    # load generator
    generator = Generator()
    generator.load_state_dict(torch.load(model_path, map_location=device))

    # sample and show
    batch_noise = torch.Tensor(64, 100, device=device).normal_()
    imgs = generator(batch_noise)
    save_image(imgs.view(-1,1,28,28), 'interpolation_test.png', nrow=8, normalize=True)

    # choose images
    id1 = int(input('Image id 1: '))
    id2 = int(input('Image id 2: '))

    # interpolate
    image1 = imgs[id1].numpy()
    image2 = imgs[id2].numpy()
    images = torch.from_numpy(np.stack([image1 + p * (image2 - image1) for p in np.linspace(0, 1, 9)]))

    # save
    save_image(images.view(-1,1,28,28), 'interpolation_out.png', nrow=9, normalize=True)