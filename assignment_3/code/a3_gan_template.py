import argparse
import os

import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torchvision.utils import save_image
from torchvision import datasets


class Generator(nn.Module):
    # TODO: momentum 0.8 as in tutorial?
    def __init__(self):
        super(Generator, self).__init__()

        self.model = nn.Sequential(
            nn.Linear(args.latent_dim, 128),
            nn.LeakyReLU(.2),
            nn.Linear(128,256),
            nn.BatchNorm1d(256),
            nn.LeakyReLU(.2),
            nn.Linear(256,512),
            nn.BatchNorm1d(512),
            nn.LeakyReLU(.2),
            nn.Linear(512,1024),
            nn.BatchNorm1d(1024),
            nn.LeakyReLU(.2),
            nn.Linear(1024,784),
            nn.Tanh()
        )

    def forward(self, z):
        # Generate images from z
        return self.model(z)


class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()

        self.model = nn.Sequential(
            nn.Linear(784,512),
            nn.LeakyReLU(.2),
            nn.Linear(512,256),
            nn.LeakyReLU(.2),
            nn.Linear(256,1),
            nn.Sigmoid()
        )

    def forward(self, img):
        # return discriminator score for img
        return self.model(img)


def train(dataloader, discriminator, generator, optimizer_G, optimizer_D):
    for epoch in range(args.n_epochs):
        for i, (imgs, _) in enumerate(dataloader):

            # Train Generator
            # ---------------
            batch_noise = torch.Tensor(args.batch_size, args.latent_dim).normal_().cuda()
            imgs_fake = generator(batch_noise)
            predictions_fake = discriminator(imgs_fake)
            loss_gen = (- predictions_fake).log().mean()

            optimizer_G.zero_grad()
            loss_gen.backward()
            optimizer_G.step()

            # Train Discriminator
            # -------------------
            predictions_real = discriminator(imgs.view(imgs.shape[0],-1).cuda())
            predictions_fake = discriminator(imgs_fake.detach())
            loss_dis = (- predictions_real.log() - (1 - predictions_fake).log()).mean()

            optimizer_D.zero_grad()
            loss_dis.backward()
            optimizer_D.step()

            # Print metrics
            if i % 50 == 0:
                print("Epoch {}   Step {}   Loss generator: {:02.3f}   Loss discriminator: {:02.3f}".format(
                    epoch, i, loss_gen, loss_dis))

            # Save Images
            # -----------
            batches_done = epoch * len(dataloader) + i
            if batches_done % args.save_interval == 0:
                save_image(imgs_fake.view(-1,1,28,28)[:25],
                           'images/{}.png'.format(batches_done),
                           nrow=5, normalize=True)


def main():
    # Create output image directory
    os.makedirs('images', exist_ok=True)

    # load data
    dataloader = torch.utils.data.DataLoader(
        datasets.MNIST('./data/mnist', train=True, download=True,
                       transform=transforms.Compose([
                           transforms.ToTensor(),
                           transforms.Normalize((0.5,),
                                                (0.5,))])),
        batch_size=args.batch_size, shuffle=True)

    # Initialize models and optimizers
    generator = Generator()
    generator.cuda()
    discriminator = Discriminator()
    discriminator.cuda()
    optimizer_G = torch.optim.Adam(generator.parameters(), lr=args.lr)
    optimizer_D = torch.optim.Adam(discriminator.parameters(), lr=args.lr)

    # Start training
    train(dataloader, discriminator, generator, optimizer_G, optimizer_D)

    # You can save your generator here to re-use it to generate images for your
    # report, e.g.:
    torch.save(generator.state_dict(), "mnist_generator.pt")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--n_epochs', type=int, default=200,
                        help='number of epochs')
    parser.add_argument('--batch_size', type=int, default=64,
                        help='batch size')
    parser.add_argument('--lr', type=float, default=0.0002,
                        help='learning rate')
    parser.add_argument('--latent_dim', type=int, default=100,
                        help='dimensionality of the latent space')
    parser.add_argument('--save_interval', type=int, default=500,
                        help='save every SAVE_INTERVAL iterations')
    args = parser.parse_args()

    main()
