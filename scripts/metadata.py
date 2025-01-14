# code from Trevor Bedford and Katie Kistler
# MIT license
import os, sys
import json
import requests
import Bio.Phylo
import argparse

def json_to_tree(json_dict, root=True):
    """Returns a Bio.Phylo tree corresponding to the given JSON dictionary exported
    by `tree_to_json`.
    Assigns links back to parent nodes for the root of the tree.
    """
    # Check for v2 JSON which has combined metadata and tree data.
    if root and "meta" in json_dict and "tree" in json_dict:
        json_dict = json_dict["tree"]

    node = Bio.Phylo.Newick.Clade()

    # v1 and v2 JSONs use different keys for strain names.
    if "name" in json_dict:
        node.name = json_dict["name"]
    else:
        node.name = json_dict["strain"]

    if "children" in json_dict:
        # Recursively add children to the current node.
        node.clades = [json_to_tree(child, root=False) for child in json_dict["children"]]

    # Assign all non-children attributes.
    for attr, value in json_dict.items():
        if attr != "children":
            setattr(node, attr, value)

    # Only v1 JSONs support a single `attr` attribute.
    if hasattr(node, "attr"):
        node.numdate = node.attr.get("num_date")
        node.branch_length = node.attr.get("div")

        if "translations" in node.attr:
            node.translations = node.attr["translations"]
    elif hasattr(node, "node_attrs"):
        node.branch_length = node.node_attrs.get("div")

    if root:
        node = annotate_parents_for_tree(node)

    return node

def annotate_parents_for_tree(tree):
    """Annotate each node in the given tree with its parent.
    """
    tree.root.parent = None
    for node in tree.find_clades(order="level"):
        for child in node.clades:
            child.parent = node

    # Return the tree.
    return tree

if __name__=="__main__":

    parser = argparse.ArgumentParser(description = "Download and simplify Auspice JSON as metadata TSV")
    parser.add_argument("--local-files", default="False",
        help="Toggle this on if you are supplying local JSON files for the tree and root sequence." +
             "Default is to fetch them from a URL")
    parser.add_argument("--tree", default="https://data.nextstrain.org/ncov_gisaid_global_all-time.json",
        help="URL for the tree.json file, or path to the local JSON file if --local-files=True")
    parser.add_argument('--output', type=str, default="metadata.tsv", help="output TSV file")

    args = parser.parse_args()

    # if we are fetching the JSONs from a URL
    if args.local_files == "False":
        # fetch the tree JSON from URL
        tree_json = requests.get(args.tree, headers={"accept":"application/json"}).json()
        # put tree in Bio.phylo format
        tree = json_to_tree(tree_json)

    # if we are using paths to local JSONs
    elif args.local_files == "True":
        # load tree
        with open(args.tree, 'r') as f:
            tree_json = json.load(f)
        # put tree in Bio.phylo format
        tree = json_to_tree(tree_json)

    data = []
    for n in tree.find_clades(order="postorder"):
        node_elements = {}
        node_elements["name"] = n.name.removeprefix('hCoV-19/')
        if n.parent:
            node_elements["parent"] = n.parent.name.removeprefix('hCoV-19/')
        else:
            node_elements["parent"] = None
        if hasattr(n, 'node_attrs'):
            if 'clade_membership' in n.node_attrs:
                if 'value' in n.node_attrs["clade_membership"]:
                    node_elements["clade_membership"] = n.node_attrs["clade_membership"]["value"]
            if 'S1_mutations' in n.node_attrs:
                if 'value' in n.node_attrs["S1_mutations"]:
                    node_elements["S1_mutations"] = n.node_attrs["S1_mutations"]["value"]
        data.append(node_elements)

    with open(args.output, 'w', encoding='utf-8') as handle:
        print("name", "parent", "clade", "S1_mutations", sep='\t', file=handle)
        for elem in data:
            print(elem['name'], elem['parent'], elem['clade_membership'], elem['S1_mutations'], sep='\t', file=handle)
