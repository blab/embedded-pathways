import argparse
import pandas as pd
import numpy as np
import umap

def parse_arguments():
    parser = argparse.ArgumentParser(description="Run UMAP on CLS vectors to produce a 2D ordination.")
    parser.add_argument("--input", type=str, default="embeddings.tsv", help="Input TSV file containing CLS vectors (default: embeddings.tsv).")
    parser.add_argument("--output", type=str, default="umap.tsv", help="Output TSV file to save 2D UMAP coordinates (default: umap.tsv).")
    return parser.parse_args()

def load_embeddings(input_file):
    """Load embeddings from a TSV file."""
    df = pd.read_csv(input_file, sep="\t", header=None)
    sequence_ids = df.iloc[:, 0].values  # First column contains sequence IDs
    embeddings = df.iloc[:, 1:].values  # Remaining columns contain embeddings
    return sequence_ids, embeddings

def run_umap(embeddings):
    """Run UMAP to reduce embeddings to 2D."""
    reducer = umap.UMAP(n_components=2, random_state=42)
    umap_results = reducer.fit_transform(embeddings)
    return umap_results

def save_umap_results(sequence_ids, umap_results, output_file):
    """Save UMAP 2D results to a TSV file."""
    df = pd.DataFrame({
        'Sequence_ID': sequence_ids,
        'UMAP1': umap_results[:, 0],
        'UMAP2': umap_results[:, 1]
    })
    df.to_csv(output_file, sep="\t", index=False)
    print(f"Saved UMAP results to {output_file}")

def main():
    args = parse_arguments()

    # Load embeddings
    print(f"Loading embeddings from {args.input}...")
    sequence_ids, embeddings = load_embeddings(args.input)

    # Run UMAP
    print("Running UMAP for dimensionality reduction...")
    umap_results = run_umap(embeddings)

    # Save UMAP results
    print(f"Saving 2D UMAP coordinates to {args.output}...")
    save_umap_results(sequence_ids, umap_results, args.output)

if __name__ == "__main__":
    main()
