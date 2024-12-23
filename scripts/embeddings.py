import torch
import esm
import argparse
from Bio import SeqIO
import pandas as pd

def parse_arguments():
    parser = argparse.ArgumentParser(description="Extract CLS token embeddings from AA sequences in FASTA format using ESM-2.")
    parser.add_argument("--input", type=str, default="alignment.fasta", help="Input FASTA file containing AA sequences (default: aligned.fasta).")
    parser.add_argument("--output", type=str, default="embeddings.tsv", help="Output TSV file to save CLS vectors (default: embeddings.tsv).")
    return parser.parse_args()

def load_sequences(fasta_file):
    """Load amino acid sequences from a FASTA file."""
    sequences = []
    for record in SeqIO.parse(fasta_file, "fasta"):
        sequences.append((record.id, str(record.seq)))
    return sequences

def extract_cls_embeddings(sequences, model, batch_converter):
    """Extract CLS embeddings for a list of sequences."""
    all_cls_vectors = []
    for i, (seq_id, sequence) in enumerate(sequences):
        print(f"Processing sequence {i+1}/{len(sequences)}: {seq_id}")
        data = [(seq_id, sequence)]
        batch_labels, batch_strs, batch_tokens = batch_converter(data)

        with torch.no_grad():
            results = model(batch_tokens, repr_layers=[33], return_contacts=False)
        embeddings = results["representations"][33]
        cls_embedding = embeddings[:, 0, :].squeeze(0)  # Extract CLS vector

        all_cls_vectors.append((seq_id, cls_embedding.cpu().numpy()))
    return all_cls_vectors

def save_to_tsv(cls_vectors, output_file):
    """Save CLS vectors to a TSV file."""
    records = []
    for seq_id, vector in cls_vectors:
        record = [seq_id] + vector.tolist()
        records.append(record)

    # Convert to DataFrame and save
    df = pd.DataFrame(records)
    df.to_csv(output_file, sep="\t", index=False, header=False)
    print(f"Saved CLS vectors to {output_file}")

def main():
    args = parse_arguments()

    # Load model and tokenizer
    print("Loading ESM-2 model...")
    model, alphabet = esm.pretrained.esm2_t33_650M_UR50D()
    batch_converter = alphabet.get_batch_converter()
    model.eval()  # Disable dropout for evaluation

    # Load sequences from FASTA file
    print(f"Loading sequences from {args.input}...")
    sequences = load_sequences(args.input)

    # Extract CLS embeddings
    print("Extracting CLS embeddings...")
    cls_vectors = extract_cls_embeddings(sequences, model, batch_converter)

    # Save embeddings to TSV file
    print(f"Saving embeddings to {args.output}...")
    save_to_tsv(cls_vectors, args.output)

if __name__ == "__main__":
    main()
