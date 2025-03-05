configfile: "defaults/config.yaml"

rule all:
    input:
        ordination = "results/ordination.tsv"

rule download_auspice_json:
    output:
        tree = "data/auspice.json",
        root = "data/auspice_root-sequence.json"
    params:
        dataset = config["ESM"]["dataset"]
    shell:
        """
        nextstrain remote download {params.dataset:q} {output.tree:q}
        """

rule provision_alignment:
    input:
        tree = "data/auspice.json",
        root = "data/auspice_root-sequence.json"
    output:
        alignment = "data/alignment.fasta"
    params:
        gene = config["ESM"]["gene"]
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
        tree = "data/auspice.json"
    output:
        metadata = "data/metadata.tsv"
    shell:
        """
        python scripts/metadata.py \
            --tree {input.tree:q} \
            --output {output.metadata:q}
        """

rule fine_tune:
    input:
        alignment = "data/alignment.fasta"
    output:
        model = "models/pytorch_model.bin"
    shell:
        """
        python ESM/fine-tune.py \
            --input {input.alignment:q} \
            --output-dir "models"
        """

rule compute_embeddings:
    input:
        alignment = "data/alignment.fasta",
        model = "models/pytorch_model.bin" if config["ESM"]["fine_tune"] else []
    output:
        log_likelihoods = "results/log_likelihoods.tsv",
        embeddings = "results/embeddings.tsv"
    params:
        model_param = "--model models/pytorch_model.bin" if config["ESM"]["fine_tune"] else ""
    shell:
        """
        python ESM/embeddings.py \
            --input {input.alignment:q} \
            --output-log-likelihoods {output.log_likelihoods:q} \
            --output-embeddings {output.embeddings:q} \
            {params.model_param}
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
