import os
import numpy as np
from Bio import SeqIO
from diffusers import UNet2DModel, DDPMScheduler
from torch.utils.data import DataLoader, Dataset
import torch
import torch.nn as nn
from torch.optim import Adam
import argparse

# Constants
LATENT_DIM = 80     # Dimensionality of latent space len(ALPHABET) * HEIGHT * WIDTH = 5 * 4 * 4 = 80
HEIGHT = 4          # Height of the latent tensor
WIDTH = 4           # Width of the latent tensor
BATCH_SIZE = 16
EPOCHS = 10
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
ALPHABET = "ATGC-"  # DNA alphabet, with "-" as gap/unknown character

class DNADataset(Dataset):
    def __init__(self, fasta_file):
        self.sequences = [str(record.seq) for record in SeqIO.parse(fasta_file, "fasta")]

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        sequence = self.sequences[idx]
        encoded = self.one_hot_encode(sequence)
        return torch.tensor(encoded, dtype=torch.float32)

    @staticmethod
    def one_hot_encode(sequence):
        mapping = {char: i for i, char in enumerate(ALPHABET)}
        encoded = np.zeros((len(sequence), len(ALPHABET)), dtype=np.float32)
        for i, nucleotide in enumerate(sequence):
            encoded[i, mapping.get(nucleotide, mapping["-"])] = 1.0
        return encoded

class VAE(nn.Module):
    def __init__(self, input_dim, latent_dim):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 2 * latent_dim)  # Mean and log variance
        )
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 512),
            nn.ReLU(),
            nn.Linear(512, input_dim),
            nn.Sigmoid()
        )

    def encode(self, x):
        mean_logvar = self.encoder(x)
        mean, logvar = mean_logvar[:, :LATENT_DIM], mean_logvar[:, LATENT_DIM:]
        return mean, logvar

    def reparameterize(self, mean, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mean + eps * std

    def decode(self, z):
        return self.decoder(z)

    def forward(self, x):
        mean, logvar = self.encode(x)
        z = self.reparameterize(mean, logvar)
        return self.decode(z), mean, logvar

def train(vae_model, diffusion_model, scheduler, dataloader, epochs, vae_optimizer, diffusion_optimizer):
    mse_loss = nn.MSELoss()
    for epoch in range(epochs):
        vae_model.train()
        diffusion_model.train()
        for batch in dataloader:
            batch = batch.view(batch.size(0), -1).to(DEVICE)  # Flatten one-hot sequences

            # Train VAE
            recon, mean, logvar = vae_model(batch)
            recon_loss = mse_loss(recon, batch)
            kl_loss = -0.5 * torch.sum(1 + logvar - mean.pow(2) - logvar.exp())
            vae_loss = recon_loss + kl_loss
            vae_optimizer.zero_grad()
            vae_loss.backward()
            vae_optimizer.step()

            # Train Diffusion Model
            latent, _ = vae_model.encode(batch)
            latent = latent.view(latent.size(0), len(ALPHABET), HEIGHT, WIDTH)  # Reshape to [batch, channels, height, width]
            # Latent shape is torch.Size([16, 5, 4, 4])
            noise = torch.randn_like(latent).to(DEVICE)
            timesteps = torch.randint(0, scheduler.config.num_train_timesteps, (latent.size(0),)).to(DEVICE)
            noisy_latent = scheduler.add_noise(latent, noise, timesteps)
            predicted_noise = diffusion_model(noisy_latent, timesteps).sample  # Access the tensor from UNet2DOutput
            diffusion_loss = mse_loss(predicted_noise, noise)

            diffusion_optimizer.zero_grad()
            diffusion_loss.backward()
            diffusion_optimizer.step()

        print(f"Epoch {epoch+1}/{epochs} - VAE Loss: {vae_loss.item():.4f} - Diffusion Loss: {diffusion_loss.item():.4f}")

def main(args):
    dataset = DNADataset(args.input)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

    input_dim = len(ALPHABET) * len(dataset.sequences[0])

    vae_model = VAE(input_dim=input_dim, latent_dim=LATENT_DIM).to(DEVICE)

    diffusion_model = UNet2DModel(
        sample_size=WIDTH,                          # Width of the latent representation
        in_channels=len(ALPHABET),                  # Input channel for latent vectors
        out_channels=len(ALPHABET),                 # Output channel
        layers_per_block=2,                         # Number of layers per block
        block_out_channels=(64, 128),               # Match the number of down_block_types
        down_block_types=("DownBlock2D", "AttnDownBlock2D"),
        up_block_types=("UpBlock2D", "AttnUpBlock2D")
    ).to(DEVICE)

    scheduler = DDPMScheduler(num_train_timesteps=1000)

    vae_optimizer = Adam(vae_model.parameters(), lr=1e-3)
    diffusion_optimizer = Adam(diffusion_model.parameters(), lr=1e-4)

    train(vae_model, diffusion_model, scheduler, dataloader, EPOCHS, vae_optimizer, diffusion_optimizer)

    # Save models
    torch.save(vae_model.state_dict(), args.output_vae_model)
    torch.save(diffusion_model.state_dict(), args.output_diffusion_model)

# === Entry Point ===
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train a Latent Diffusion Model on DNA sequences.")
    parser.add_argument("--input", type=str, default="data/alignment.fasta", help="Path to the input FASTA file.")
    parser.add_argument("--output-vae-model", type=str, default="models/vae.pth", help="Path to save the trained VAE model.")
    parser.add_argument("--output-diffusion-model", type=str, default="models/diffusion.pth", help="Path to save the trained Diffusion model.")

    args = parser.parse_args()
    main(args)
