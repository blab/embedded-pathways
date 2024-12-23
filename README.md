# Embedded Pathways

Analyzing evolution through embedding space in protein language models. Here I'm
using ESM-2 as it was trained on viral proteins and should include SARS-CoV-2
spike.

# Installation

Working from https://github.com/facebookresearch/esm.

Install PyTorch (on Mac via Homebrew, other systems will need different installation)
```
brew install pytorch
```

Install ESM, UMAP and various dependencies with pip:

```
pip install -r requirements.txt
```

# Workflow

Run the entire workflow with `snakemake --cores 1 -p`

## Provision data

### Provision smaller dataset

In `config.yaml`, specify
```
tree: https://data.nextstrain.org/ncov_gisaid_reference.json
root: https://data.nextstrain.org/ncov_gisaid_reference_root-sequence.json
```

Run `snakemake --cores 1 -p data/alignment.fasta data/metadata.tsv`

### Provision larger dataset

In `config.yaml`, specify
```
tree: https://data.nextstrain.org/ncov_gisaid_global_all-time.json
root: https://data.nextstrain.org/ncov_gisaid_global_all-time_root-sequence.json
```

Run `snakemake --cores 1 -p data/alignment.fasta data/metadata.tsv`

## Compute embeddings and ordination

Run `snakemake --cores 1 -p results/embeddings.tsv results/ordination.tsv`

# Cluster

Working from the new [harmony nodes](https://sciwiki.fredhutch.org/scicompannounce/2024-11-17-new-harmony-gpu-nodes/) on the Fred Hutch cluster

Set up
```
ssh tbedford@maestro.fhcrc.org
module load snakemake PyTorch/2.1.2-foss-2023a-CUDA-12.1.1
pip install --user fair-esm nextstrain-augur umap-learn
```

Running
```
# Submit worklow
sbatch --partition=chorus --gpus=1 snakemake --cores 1 -p

# Interactive session
srun --pty -c 6 -t "2-0" --gpus=1 -p chorus /bin/zsh -i

# Grab embeddings
scp tbedford@maestro.fhcrc.org:~/embedded-pathways/results/embeddings.tsv results/embeddings.tsv
```
