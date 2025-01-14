# code from ChatGPT
import torch
import esm
import argparse
from Bio import SeqIO
import pandas as pd

def parse_arguments():
    parser = argparse.ArgumentParser(description="Extract CLS token embeddings from AA sequences in FASTA format using ESM-2.")
    parser.add_argument("--input", type=str, default="alignment.fasta", help="Input FASTA file containing AA sequences (default: aligned.fasta).")
    parser.add_argument("--output", type=str, default="embeddings.tsv", help="Output TSV file to save CLS vectors (default: embeddings.tsv).")
    parser.add_argument("--model", type=str, default=None, help="Fine-tuned model in .bin format. If not specified, the pre-trained ESM-2 model will be used.")
    return parser.parse_args()

def load_sequences(fasta_file):
    """Load amino acid sequences from a FASTA file."""
    sequences = []
    for record in SeqIO.parse(fasta_file, "fasta"):
        sequences.append((record.id, str(record.seq)))
    return sequences

def extract_cls_embeddings(sequences, model, batch_converter, device):
    """Extract CLS embeddings for a list of sequences."""
    all_cls_vectors = []
    for i, (seq_id, sequence) in enumerate(sequences):
        print(f"Processing sequence {i+1}/{len(sequences)}: {seq_id}")
        data = [(seq_id, sequence)]
        batch_labels, batch_strs, batch_tokens = batch_converter(data)

        # Move batch tokens to the specified device
        batch_tokens = batch_tokens.to(device)

        with torch.no_grad():
            results = model(batch_tokens, repr_layers=[33], return_contacts=False)
        embeddings = results["representations"][33]

        # Extract CLS vector and move to CPU
        cls_embedding = embeddings[:, 0, :].squeeze(0).to("cpu")

        all_cls_vectors.append((seq_id, cls_embedding.numpy()))
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

    # Check if CUDA is available and set the device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load model and tokenizer
    print("Loading pre-trained ESM-2 model...")
    model, alphabet = esm.pretrained.esm2_t33_650M_UR50D()
    if args.model:
        print(f"Updating with fine-tuned model from {args.model}...")
        model.load_state_dict(torch.load(f"{args.model}", map_location=device))

    batch_converter = alphabet.get_batch_converter()
    model = model.to(device)  # Move model to device
    model.eval()  # Disable dropout for evaluation

    # Load sequences from FASTA file
    print(f"Loading sequences from {args.input}...")
    sequences = load_sequences(args.input)

    # Extract CLS embeddings
    print("Extracting CLS embeddings...")
    cls_vectors = extract_cls_embeddings(sequences, model, batch_converter, device)

    # Save embeddings to TSV file
    print(f"Saving embeddings to {args.output}...")
    save_to_tsv(cls_vectors, args.output)

if __name__ == "__main__":
    main()
