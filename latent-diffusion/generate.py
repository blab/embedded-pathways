import argparse
import os
import torch
from models import VAE, DiffusionModel, ALPHABET, SEQ_LENGTH, HEIGHT, WIDTH, LATENT_DIM
from diffusers import DDPMScheduler

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def generate_sequences(vae_model_path, diffusion_model_path, count):

    # Load trained models
    input_dim = len(ALPHABET) * SEQ_LENGTH
    vae_model = VAE(input_dim=input_dim, latent_dim=LATENT_DIM).to(DEVICE)
    if args.input_vae_model and os.path.exists(args.input_vae_model):
        print(f"Loading VAE model from {args.input_vae_model}")
        vae_model.load_state_dict(torch.load(args.input_vae_model, map_location=DEVICE, weights_only=False))
    vae_model.eval()

    diffusion_model = DiffusionModel().to(DEVICE)
    if args.input_diffusion_model and os.path.exists(args.input_diffusion_model):
        print(f"Loading Diffusion model from {args.input_diffusion_model}")
        diffusion_model.load_state_dict(torch.load(args.input_diffusion_model, map_location=DEVICE, weights_only=False))
    diffusion_model.eval()

    scheduler = DDPMScheduler(num_train_timesteps=1000)

    print(f"Generating sequences")
    generated_sequences = []
    for i in range(count):
        print("Sequence", i+1)
        # Generate noise in latent space
        latent = torch.randn((1, len(ALPHABET), HEIGHT, WIDTH), device=DEVICE)

        # Reverse diffusion process
        for t in reversed(range(scheduler.config.num_train_timesteps)):
            latent = scheduler.step(diffusion_model(latent, torch.tensor([t], device=DEVICE)).sample, t, latent).prev_sample

        # Decode latent space to one-hot sequence
        latent_flat = latent.view(1, -1)
        reconstructed_sequence = vae_model.decode(latent_flat).view(-1, len(ALPHABET))
        decoded_sequence = "".join(ALPHABET[torch.argmax(x).item()] for x in reconstructed_sequence)
        generated_sequences.append(decoded_sequence)

    return generated_sequences

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate DNA sequences from trained models")
    parser.add_argument("--input-vae-model", type=str, default="models/vae.pth", help="Path to the trained VAE model")
    parser.add_argument("--input-diffusion-model", type=str, default="models/diffusion.pth", help="Path to the trained Diffusion model")
    parser.add_argument("--output-alignment", type=str, default="results/generated.fasta", help="FASTA file to output sequences to")
    parser.add_argument("--count", type=int, default=10, help="Number of DNA sequences to generate")
    args = parser.parse_args()

    sequences = generate_sequences(args.input_vae_model, args.input_diffusion_model, args.count)

    if not os.path.exists("results/"):
        os.makedirs("results/")
    with open(args.output_alignment, 'w') as f:
        for i, seq in enumerate(sequences):
            print(">seq_", i+1, file=f, sep = '')
            print(seq, file=f)
