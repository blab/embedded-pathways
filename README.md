# Embedded Pathways

Analyzing evolution as pathways through latent representation space. This
repository analyzes SARS-CoV-2 spike protein or SARS-CoV-2 full genome sequence.
The former uses the ESM-2 protein language model that was trained on a large
corpus of protein sequences including viral proteins. There is optional fine
tuning of ESM-2 with SARS-CoV-2 spike sequences. The latter uses a custom latent
diffusion model implemented with PyTorch / Hugging Face. This is analogous to
the Stable Diffusion strategy.

# Installation

Working from https://github.com/facebookresearch/esm.

Install PyTorch (on Mac via Homebrew, other systems will need different installation)
```
brew install pytorch
```

Install ESM, Hugging Face and various dependencies with pip:

```
pip install -r requirements.txt
```

# ESM fine tune workflow

Using parameters in `config.yaml` under `ESM_fine_tune`.

Run `snakemake --cores 1 -p --snakefile ESM_fine_tune.smk`

This outputs `models/esm.bin`. This requires decent GPU resources, where `--batch-size 8` of has been tuned for a single NVIDIA 96Gb H100 GPU.

# ESM embeddings workflow

Using parameters in `config.yaml` under `ESM_embeddings`.

Run `snakemake --cores 1 -p --snakefile ESM_embeddings.smk`

This outputs
 - `results/log_likelihoods.tsv`
 - `results/embeddings.tsv`
 - `results/ordination.tsv`  

# Latent diffusion workflow

## Provision data

Run `snakemake --cores 1 -p --snakefile latent_diffusion.smk data/alignment.fasta data/metadata.tsv`

## Train latent diffusion model

Run `snakemake --cores 1 -p --snakefile latent_diffusion.smk models/vae.pth models/diffusion.pth`

## Generate sequences from latent diffusion model

Run `snakemake --cores 1 -p --snakefile latent_diffusion.smk results/generated.fasta`

## Compute embeddings and ordination

Run `snakemake --cores 1 -p --snakefile latent_diffusion.smk results/ordination.tsv`

# Cluster

Working from the new [harmony nodes](https://sciwiki.fredhutch.org/scicompannounce/2024-11-17-new-harmony-gpu-nodes/) on the Fred Hutch cluster

Log into cluster
```
ssh tbedford@maestro.fhcrc.org
```

Set up for ESM
```
module load snakemake PyTorch/2.1.2-foss-2023a-CUDA-12.1.1
pip install --user fair-esm nextstrain-augur transformers
```

Set up for latent diffusion
```
module load snakemake PyTorch/2.1.2-foss-2023a-CUDA-12.1.1
pip install --user nextstrain-augur diffusers["torch"] transformers
```

Running
```
# Submit worklow
sbatch --partition=chorus --gpus=1 snakemake --cores 1 -p --snakefile latent_diffusion.smk

# Interactive session
srun --pty -c 6 -t "2-0" --gpus=1 -p chorus /bin/zsh -i
```

Update ESM scripts
```
scp ESM/* tbedford@maestro.fhcrc.org:~/embedded-pathways/ESM/
```

Update latent diffusion scripts
```
scp latent-diffusion/* tbedford@maestro.fhcrc.org:~/embedded-pathways/latent-diffusion/
```

Grab remote results
```
scp -r tbedford@maestro.fhcrc.org:~/embedded-pathways/results/* results/
```

# Models

## Variational autoencoder

VAE model is 154M parameters

## Diffusion model

Using U-Net. Diffusion model is 15M parameters

# License

This repository is licensed under the MIT License. See the LICENSE file for
details.

## Important Disclaimer About Copyright and AI-Generated Code

Some portions of the code in this repository were generated with the assistance
of large language models (LLMs), primarily ChatGPT. Individual scripts are
commented to state their provenance. While I have reviewed, modified, and
integrated these contributions, the copyright status of LLM-generated code is
uncertain and may vary depending on jurisdiction.

As a result:

- Human-Authored Contributions: Code written by me (the repository owner) is
  explicitly licensed under the MIT License and is subject to the terms outlined
  in the LICENSE file.
- LLM-Generated Contributions: For any portions of the code generated by LLMs, I
  do not assert copyright ownership and disclaim any responsibility for the
  originality or copyright status of such code.
- User Responsibility: Users of this repository are encouraged to independently
  verify the legal status of any LLM-generated portions of the code before reuse
  or redistribution.
