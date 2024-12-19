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

Install UMAP
```
pip install umap-learn
```

# Commands

## Provision smaller dataset

Run `python alignment.py --gene S --tree https://data.nextstrain.org/ncov_gisaid_reference.json --root https://data.nextstrain.org/ncov_gisaid_reference_root-sequence.json` to produce `alignment.fasta` of spike AA sequences from https://nextstrain.org/ncov/gisaid/reference.

Run `python metadata.py --tree https://data.nextstrain.org/ncov_gisaid_reference.json` to produce `metadata.tsv` from https://nextstrain.org/ncov/gisaid/reference.

## Provision larger dataset

Run `python alignment.py --gene S` to produce `alignment.fasta` of spike AA sequences from https://nextstrain.org/ncov/gisaid/global/all-time.

Run `python metadata.py` to produce `metadata.tsv` from https://nextstrain.org/ncov/gisaid/global/all-time.

## Analysis

Run `python embeddings.py` to produce `embeddings.tsv` from the file `alignment.fasta`.

Run `python ordination.py` to produce `ordination.tsv` from the file `embeddings.tsv`.
