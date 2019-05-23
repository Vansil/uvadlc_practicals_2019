import argparse
import os

import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from torchvision.utils import make_grid
import numpy as np

from datasets.bmnist import bmnist
from summary import VaeWriter


class Encoder(nn.Module):

    def __init__(self, hidden_dim=500, z_dim=20, input_dim=784):
        super().__init__()

        self.fc_hidden = nn.Linear(input_dim, hidden_dim)
        self.fc_mean = nn.Linear(hidden_dim, z_dim)
        self.fc_logvar = nn.Linear(hidden_dim, z_dim)

    def forward(self, input):
        """
        Perform forward pass of encoder.

        Returns mean and log variance with shape [batch_size, z_dim].
        """

        hidden = torch.tanh(self.fc_hidden(input))
        mean = self.fc_mean(hidden)
        logvar = self.fc_logvar(hidden)

        return mean, logvar


class Decoder(nn.Module):

    def __init__(self, hidden_dim=500, z_dim=20, output_dim=784):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(z_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, output_dim),
            nn.Sigmoid()
        )


    def forward(self, input):
        """
        Perform forward pass of encoder.

        Returns mean with shape [batch_size, 784].
        """
        mean = self.net(input)

        return mean


class VAE(nn.Module):

    def __init__(self, hidden_dim=500, z_dim=20, img_dim=784):
        super().__init__()
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'

        self.z_dim = z_dim
        self.encoder = Encoder(hidden_dim, z_dim, img_dim)
        self.decoder = Decoder(hidden_dim, z_dim, img_dim)

    def forward(self, input):
        """
        Given input, perform an encoding and decoding step and return the
        negative average elbo for the given batch.
        """

        # Encode
        input = input.view(input.shape[0], -1)
        mean, logvar = self.encoder(input)
        var = logvar.exp()
        std = var.sqrt()
        # Sample
        epsilon = torch.Tensor(self.z_dim).to(self.device).normal_()
        encoding = mean + torch.einsum('d,nd->nd', epsilon, std)
        # Decode
        decoded = self.decoder(encoding)

        # Reconstruction loss
        loss_recon = - (
            torch.einsum('nd,nd->n', input,       decoded.log()) + 
            torch.einsum('nd,nd->n', (1 - input), (1 - decoded).log())
        )
        # Regularization loss
        loss_regul = (.5 * (var + mean.pow(2) - 1 - logvar)).sum(dim=1)

        # Average ELBo
        average_negative_elbo = (loss_recon + loss_regul).mean()
        return average_negative_elbo

    def sample(self, n_samples):
        """
        Sample n_samples from the model. Return both the sampled images
        (from bernoulli) and the means for these bernoullis (as these are
        used to plot the data manifold).
        """

        # Sample from normal distribution
        epsilon = torch.Tensor(n_samples, self.z_dim).to(self.device).normal_()
        # Decode
        im_means = self.decoder(epsilon)
        
        # Sample one image from each means vector
        side_len = np.sqrt(im_means.shape[1]) # assuming square images
        sampled_ims = im_means.bernoulli().view(-1,1,side_len,side_len)

        return sampled_ims, im_means


def epoch_iter(model, dataloader, optimizer):
    """
    Perform a single epoch for either the training or validation.
    use model.training to determine if in 'training mode' or not.

    Returns the average elbo for the complete epoch.
    """
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    epoch_losses = [] # total loss for each batch
    for batch in dataloader:
        # Evaluate negative ELBo
        loss = model(batch.to(device))
        epoch_losses.append(loss * len(batch))

        if model.training:
            # Backpropagation
            model.zero_grad()
            loss.backward()
            optimizer.step()

    average_epoch_elbo = sum(epoch_losses) / len(dataloader.dataset)

    return average_epoch_elbo


def run_epoch(model, dataloaders, optimizer):
    """
    Run a train and validation epoch and return average elbo for each.
    """
    traindataloader, valdataloader = dataloaders

    model.train()
    train_elbo = epoch_iter(model, traindataloader, optimizer)

    model.eval()
    val_elbo = epoch_iter(model, valdataloader, optimizer)

    return train_elbo, val_elbo


def main():
    writer = VaeWriter(ARGS.output_dir)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    writer.log("Device: " + device)

    data = bmnist()[:2]  # ignore test split
    model = VAE(z_dim=ARGS.zdim).to(device)
    optimizer = torch.optim.Adam(model.parameters())
    writer.log("Model:\n" + str(model))

    for epoch in range(ARGS.epochs):
        elbos = run_epoch(model, data, optimizer)
        train_elbo, val_elbo = elbos

        writer.save_stats(train_elbo, val_elbo)
        writer.log(f"[Epoch {epoch}] train elbo: {train_elbo} val_elbo: {val_elbo}")
        writer.save_elbo_plot()
        sample_imgs, _ = model.sample(25)
        writer.save_images(sample_imgs, epoch)

    # --------------------------------------------------------------------
    #  Add functionality to plot plot the learned data manifold after
    #  if required (i.e., if zdim == 2). You can use the make_grid
    #  functionality that is already imported.
    # --------------------------------------------------------------------


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', default=100, type=int,
                        help='max number of epochs')
    parser.add_argument('--zdim', default=20, type=int,
                        help='dimensionality of latent space')
    parser.add_argument('--output_dir', type=str, default=os.path.join('output','vae','run'),
                        help='directory to which to output')

    ARGS = parser.parse_args()

    main()
