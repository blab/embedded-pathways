# code from ChatGPT
import torch
import esm
import argparse
from Bio import SeqIO
import pandas as pd

def parse_arguments():
    parser = argparse.ArgumentParser(description="Extract CLS token embeddings and log likelihood from AA sequences in FASTA format using ESM-2.")
    parser.add_argument("--input", type=str, default="alignment.fasta", help="Input FASTA file containing AA sequences (default: aligned.fasta).")
    parser.add_argument("--output-log-likelihoods", type=str, default="log_likelihoods.tsv", help="Output TSV file to save log likelihoods (default: log_likelihoods.tsv).")
    parser.add_argument("--output-embeddings", type=str, default="embeddings.tsv", help="Output TSV file to save CLS vectors (default: embeddings.tsv).")
    parser.add_argument("--model", type=str, default=None, help="Fine-tuned model in .bin format. If not specified, the pre-trained ESM-2 model will be used.")
    return parser.parse_args()

def load_sequences(fasta_file):
    """Load amino acid sequences from a FASTA file."""
    sequences = []
    for record in SeqIO.parse(fasta_file, "fasta"):
        sequences.append((record.id, str(record.seq)))
    return sequences

def extract_cls_embeddings_and_log_likelihood(sequences, model, batch_converter, device):
    """Extract CLS embeddings and log likelihood for a list of sequences."""
    all_results = []
    for i, (seq_id, sequence) in enumerate(sequences):
        print(f"Processing sequence {i+1}/{len(sequences)}: {seq_id}")
        data = [(seq_id, sequence)]
        batch_labels, batch_strs, batch_tokens = batch_converter(data)

        # Move batch tokens to the specified device
        batch_tokens = batch_tokens.to(device)

        with torch.no_grad():
            results = model(batch_tokens, repr_layers=[33], return_contacts=False)
            logits = results["logits"]  # Get logits
            log_probs = torch.log_softmax(logits, dim=-1)  # Convert logits to log probabilities
            log_likelihood = log_probs.gather(2, batch_tokens.unsqueeze(-1)).sum().item()  # Compute log-likelihood

        embeddings = results["representations"][33]

        # Extract CLS vector and move to CPU
        cls_embedding = embeddings[:, 0, :].squeeze(0).to("cpu")

        all_results.append((seq_id, log_likelihood, cls_embedding.numpy()))
    return all_results

def save_to_tsv(results, log_likelihoods_file, embeddings_file):
    """Save log likelihoods and CLS vectors to separate TSV files."""
    log_likelihoods = []
    embeddings = []
    for seq_id, log_likelihood, vector in results:
        log_likelihoods.append([seq_id, log_likelihood])
        embeddings.append([seq_id] + vector.tolist())

    # Save log likelihoods
    df_log_likelihoods = pd.DataFrame(log_likelihoods)
    df_log_likelihoods.to_csv(log_likelihoods_file, sep="\t", index=False, header=False)
    print(f"Saved log likelihoods to {log_likelihoods_file}")

    # Save embeddings
    df_embeddings = pd.DataFrame(embeddings)
    df_embeddings.to_csv(embeddings_file, sep="\t", index=False, header=False)
    print(f"Saved embeddings to {embeddings_file}")

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

    # Extract CLS embeddings and log likelihood
    print("Extracting CLS embeddings and log likelihood...")
    results = extract_cls_embeddings_and_log_likelihood(sequences, model, batch_converter, device)

    # Save embeddings and log likelihoods to separate TSV files
    print(f"Saving log likelihoods to {args.output_log_likelihoods} and embeddings to {args.output_embeddings}...")
    save_to_tsv(results, args.output_log_likelihoods, args.output_embeddings)

if __name__ == "__main__":
    main()
