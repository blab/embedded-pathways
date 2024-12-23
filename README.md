# Notes

Using ESM-2 because it was trained on viral proteins and should include SARS-CoV-2 spike.

# Installation

Working from https://github.com/facebookresearch/esm.

Install PyTorch (on Mac via Homebrew, other systems will need different installation)
```
brew install pytorch
```

Install ESM
```
pip install fair-esm
```

Install other various dependences
```
pip install umap-learn requests
```

# Workflow

Run the entire workflow with `snakemake --cores 1 -p`

# Provision data

## Provision smaller dataset

In `config.yaml`, specify
```
tree: https://data.nextstrain.org/ncov_gisaid_reference.json
root: https://data.nextstrain.org/ncov_gisaid_reference_root-sequence.json
```

Run `snakemake --cores 1 -p data/alignment.fasta data/metadata.tsv`

## Provision larger dataset

In `config.yaml`, specify
```
tree: https://data.nextstrain.org/ncov_gisaid_global_all-time.json
root: https://data.nextstrain.org/ncov_gisaid_global_all-time_root-sequence.json
```

Run `snakemake --cores 1 -p data/alignment.fasta data/metadata.tsv`

# Compute embeddings and ordination

Run `snakemake --cores 1 -p results/embeddings.tsv results/ordination.tsv`
