import argparse
import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

def parse_arguments():
    parser = argparse.ArgumentParser(description="Run UMAP, t-SNE, and PCA on CLS vectors to produce 2D ordinations.")
    parser.add_argument("--input", type=str, default="embeddings.tsv", help="Input TSV file containing CLS vectors (default: embeddings.tsv).")
    parser.add_argument("--output", type=str, default="ordination.tsv", help="Output TSV file to save 2D coordinates (default: ordination_results.tsv).")
    return parser.parse_args()

def load_embeddings(input_file):
    """Load embeddings from a TSV file."""
    df = pd.read_csv(input_file, sep="\t", header=None)
    sequence_ids = df.iloc[:, 0].values  # First column contains sequence IDs
    embeddings = df.iloc[:, 1:].values  # Remaining columns contain embeddings
    return sequence_ids, embeddings

def run_tsne(embeddings):
    """Run t-SNE to reduce embeddings to 2D."""
    tsne = TSNE(n_components=2, random_state=42, init="random")
    tsne_results = tsne.fit_transform(embeddings)
    return tsne_results

def run_pca(embeddings):
    """Run PCA to reduce embeddings to 5D."""
    pca = PCA(n_components=5, random_state=42)
    pca_results = pca.fit_transform(embeddings)
    return pca_results

def save_results(sequence_ids, tsne_results, pca_results, output_file):
    """Save combined t-SNE, and PCA results to a TSV file."""
    df = pd.DataFrame({
        'id': sequence_ids,
        'tsne_1': tsne_results[:, 0],
        'tsne_2': tsne_results[:, 1],
        'pca_1': pca_results[:, 0],
        'pca_2': pca_results[:, 1],
        'pca_3': pca_results[:, 2],
        'pca_4': pca_results[:, 3],
        'pca_5': pca_results[:, 4]
    })
    df.to_csv(output_file, sep="\t", index=False)
    print(f"Saved results to {output_file}")

def main():
    args = parse_arguments()

    # Load embeddings
    print(f"Loading embeddings from {args.input}...")
    sequence_ids, embeddings = load_embeddings(args.input)

    # Run t-SNE
    print("Running t-SNE for dimensionality reduction...")
    tsne_results = run_tsne(embeddings)

    # Run PCA
    print("Running PCA for dimensionality reduction...")
    pca_results = run_pca(embeddings)

    # Save results
    print(f"Saving combined results to {args.output}...")
    save_results(sequence_ids, tsne_results, pca_results, args.output)

if __name__ == "__main__":
    main()
