import argparse
import os

import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torchvision.utils import save_image
from torchvision import datasets

from summary import GanWriter


class Generator(nn.Module):
    # TODO: momentum 0.8 as in tutorial?
    def __init__(self):
        super(Generator, self).__init__()

        self.model = nn.Sequential(
            nn.Linear(args.latent_dim if 'args' in globals() else 100, 128),
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


def train(dataloader, discriminator, generator, optimizer_G, optimizer_D, writer):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    writer.log('Device: ' + device)

    # Binary Cross-entropy Loss
    criterion = nn.BCELoss()

    for epoch in range(args.n_epochs):
        for i, (imgs, _) in enumerate(dataloader):
            batch_size = imgs.shape[0]

            # Train Generator
            # ---------------
            generator.zero_grad()

            # inference
            batch_noise = torch.Tensor(batch_size, args.latent_dim).to(device).normal_()
            imgs_fake = generator(batch_noise)
            predictions_fake = discriminator(imgs_fake)

            # loss
            # loss_gen = (- predictions_fake).log().mean()
            label = torch.full((batch_size,), 1).to(device)
            loss_gen = criterion(predictions_fake, label)
            loss_gen.backward()

            # gradient update
            optimizer_G.step()

            # Train Discriminator
            # -------------------
            discriminator.zero_grad()

            # inference
            predictions_real = discriminator(imgs.view(imgs.shape[0],-1).to(device))
            predictions_fake = discriminator(imgs_fake.detach())

            # loss
            label.fill_(1)
            loss_dis_real = criterion(predictions_real, label)
            loss_dis_real.backward()
            label.fill_(0)
            loss_dis_fake = criterion(predictions_fake, label)
            loss_dis_fake.backward()
            loss_dis = loss_dis_real + loss_dis_fake
            # loss_dis = - predictions_real.log().mean() - (1 - predictions_fake).log().mean()

            # gradient update
            optimizer_D.step()

            # Print metrics
            if i % 50 == 0:
                writer.log("Epoch {}   Step {}   Loss generator: {:02.3f}   Loss discriminator: {:02.3f}".format(
                    epoch, i, loss_gen, loss_dis))

            # Save Images and stats
            # -----------
            writer.save_stats(loss_gen, loss_dis)
            batches_done = epoch * len(dataloader) + i
            if batches_done % args.save_interval == 0:
                writer.save_images(imgs_fake, batches_done)
                writer.make_stats_plot()
                writer.save_state_dict(generator, "mnist_generator_{}.pt".format(batches_done))


def main():
    # Create output image directory
    writer = GanWriter(args.output_dir)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

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
    generator.to(device)
    discriminator = Discriminator()
    discriminator.to(device)
    optimizer_G = torch.optim.Adam(generator.parameters(), lr=args.lr)
    optimizer_D = torch.optim.Adam(discriminator.parameters(), lr=args.lr)

    # Start training
    train(dataloader, discriminator, generator, optimizer_G, optimizer_D, writer)

    # You can save your generator here to re-use it to generate images for your
    # report
    writer.save_state_dict(generator, "mnist_generator_final.pt")


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
    parser.add_argument('--output_dir', type=str, default=os.path.join('output','gan','run'),
                        help='directory to which to output')
    args = parser.parse_args()

    main()
