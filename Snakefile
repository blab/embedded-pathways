configfile: "defaults/config.yaml"

rule all:
    input:
        results = expand(
            "results/ordination.tsv"
        )

rule provision_alignment:
    output:
        alignment = "data/alignment.fasta"
    params:
        tree = config.get("tree"),
        root = config.get("root"),
        gene = "S"
    shell:
        """
        python scripts/alignment.py \
            --tree {params.tree:q} \
            --root {params.root:q} \
            --output {output.alignment:q} \
            --gene {params.gene:q}
        """

rule provision_metadata:
    output:
        metadata = "data/metadata.tsv"
    params:
        tree = config.get("tree")
    shell:
        """
        python scripts/metadata.py \
            --tree {params.tree:q} \
            --output {output.metadata:q}
        """

rule compute_embeddings:
    input:
        alignment = "data/alignment.fasta"
    output:
        embeddings = "results/embeddings.tsv"
    shell:
        """
        python scripts/embeddings.py \
            --input {input.alignment:q} \
            --output {output.embeddings:q}
        """

rule compute_ordination:
    input:
        embeddings = "results/embeddings.tsv"
    output:
        ordination = "results/ordination.tsv"
    shell:
        """
        python scripts/ordination.py \
            --input {input.embeddings:q} \
            --output {output.ordination:q}
        """
