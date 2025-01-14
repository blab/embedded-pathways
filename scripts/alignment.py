# Code from Katie Kistler
# MIT license
"""
Given a tree.json and root-sequence.json file, finds the sequences of
each node in the tree and outputs a FASTA file with these node sequences.
If a gene is specified, the sequences will be the AA sequence of that gene
at that node. If 'nuc' is specified, the whole genome nucleotide sequence
at the node will be output. (this is default if no gene is specified).
The FASTA header is the node's name in the tree.json
"""
import argparse
import json
import requests
from augur.utils import json_to_tree
from Bio import SeqIO
from Bio.Seq import Seq
from Bio.Seq import MutableSeq
from Bio.SeqRecord import SeqRecord


def apply_muts_to_root(root_seq, list_of_muts):
    """
    Apply a list of mutations to the root sequence
    to find the sequence at a given node. The list of mutations
    is ordered from root to node, so multiple mutations at the
    same site will correctly overwrite each other
    """

    # make the root sequence mutatable
    root_plus_muts = MutableSeq(root_seq)

    # apply all mutations to root sequence
    for mut in list_of_muts:
        # subtract 1 to deal with biological numbering vs python
        mut_site = int(mut[1:-1])-1
        # get the nuc that the site was mutated TO
        mutation = mut[-1]
        # apply mutation
        root_plus_muts[mut_site] = mutation


    return root_plus_muts

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("--gene", default="nuc",
        help="Name of gene to return AA sequences for. 'nuc' will return full geneome nucleotide seq")
    parser.add_argument("--local-files", default="False",
        help="Toggle this on if you are supplying local JSON files for the tree and root sequence." +
             "Default is to fetch them from a URL")
    parser.add_argument("--tree", default="https://data.nextstrain.org/ncov_gisaid_global_all-time.json",
        help="URL for the tree.json file, or path to the local JSON file if --local-files=True")
    parser.add_argument("--root", default="https://data.nextstrain.org/ncov_gisaid_global_all-time_root-sequence.json",
        help="URL for the root-sequence.json file, or path to the local JSON file if --local-files=True")
    parser.add_argument("--output", type=str, default="alignment.fasta", help="Output FASTA file for sequences")

    args = parser.parse_args()

    # if we are fetching the JSONs from a URL
    if args.local_files == "False":
        # fetch the tree JSON from URL
        tree_json = requests.get(args.tree, headers={"accept":"application/json"}).json()
        # put tree in Bio.phylo format
        tree = json_to_tree(tree_json)
        # fetch the root JSON from URL
        root_json = requests.get(args.root, headers={"accept":"application/json"}).json()
        # get the nucleotide sequence of root
        root_seq = root_json[args.gene]

    # if we are using paths to local JSONs
    elif args.local_files == "True":
        # load tree
        with open(args.tree, 'r') as f:
            tree_json = json.load(f)
        # put tree in Bio.phylo format
        tree = json_to_tree(tree_json)
        # load root sequence file
        with open(args.root, 'r') as f:
            root_json = json.load(f)
        # get the nucleotide sequence of root
        root_seq = root_json[args.gene]

    ## Now find the node sequences

    # initialize list to store sequence records for each node
    sequence_records = []

    # find sequence at each node in the tree (includes internal nodes and terminal nodes)
    for node in tree.find_clades():

        # get path back to the root
        path = tree.get_path(node)

        # get all mutations relative to root
        muts = [branch.branch_attrs['mutations'].get(args.gene, []) for branch in path]
        # flatten the list of mutations
        muts = [item for sublist in muts for item in sublist]
        # get sequence at node
        node_seq = apply_muts_to_root(root_seq, muts)
        # strip trailing stop codons
        stripped_seq = Seq(str(node_seq).rstrip('*'))
        # strip hCoV-19/ from beginning of strain name
        strain = node.name.removeprefix('hCoV-19/')
        # only keep records without stop codons (*)
        if not '*' in stripped_seq:
            sequence_records.append(SeqRecord(stripped_seq, strain, '', ''))

    SeqIO.write(sequence_records, args.output, "fasta")
