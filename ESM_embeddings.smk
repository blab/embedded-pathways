configfile: "defaults/config.yaml"

rule all:
    input:
        ordination = config["ESM_embeddings"]["ordination"]

rule download_auspice_json:
    output:
        tree = "data/auspice_embeddings.json",
        root = "data/auspice_embeddings_root-sequence.json"
    params:
        dataset = config["ESM_embeddings"]["dataset"]
    shell:
        """
        nextstrain remote download {params.dataset:q} {output.tree:q}
        """

rule provision_alignment:
    input:
        tree = "data/auspice_embeddings.json",
        root = "data/auspice_embeddings_root-sequence.json"
    output:
        alignment = "data/alignment_embeddings.fasta"
    params:
        gene = config["ESM_embeddings"]["gene"]
    shell:
        """
        python scripts/alignment.py \
            --tree {input.tree:q} \
            --root {input.root:q} \
            --output {output.alignment:q} \
            --gene {params.gene:q}
        """

rule provision_metadata:
    input:
        tree = "data/auspice_embeddings.json"
    output:
        metadata = "data/metadata_embeddings.tsv"
    shell:
        """
        python scripts/metadata.py \
            --tree {input.tree:q} \
            --output {output.metadata:q}
        """

rule compute_embeddings:
    input:
        alignment = "data/alignment_embeddings.fasta",
    output:
        log_likelihoods = config["ESM_embeddings"]["log_likelihoods"],
        embeddings = config["ESM_embeddings"]["embeddings"]
    params:
        model = config["ESM_embeddings"]["model"],
        model_weights = config["ESM_embeddings"]["model_weights"]
    shell:
        """
        python ESM/embeddings.py \
            --input {input.alignment:q} \
            --output-log-likelihoods {output.log_likelihoods:q} \
            --output-embeddings {output.embeddings:q} \
            --model {params.model:q} \
            --model-weights {params.model_weights:q}
        """

rule compute_ordination:
    input:
        embeddings = config["ESM_embeddings"]["embeddings"]
    output:
        ordination = config["ESM_embeddings"]["ordination"]
    shell:
        """
        python scripts/ordination.py \
            --input {input.embeddings:q} \
            --output {output.ordination:q}
        """
