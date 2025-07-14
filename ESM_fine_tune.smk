configfile: "defaults/config.yaml"

rule all:
    input:
        model = config["ESM_fine_tune"]["model_weights"]

rule download_auspice_json:
    output:
        tree = "data/auspice_fine_tune.json",
        root = "data/auspice_fine_tune_root-sequence.json"
    params:
        dataset = config["ESM_fine_tune"]["dataset"]
    shell:
        """
        nextstrain remote download {params.dataset:q} {output.tree:q}
        """

rule provision_alignment:
    input:
        tree = "data/auspice_fine_tune.json",
        root = "data/auspice_fine_tune_root-sequence.json"
    output:
        alignment = "data/alignment_fine_tune.fasta"
    params:
        gene = config["ESM_fine_tune"]["gene"]
    shell:
        """
        python scripts/alignment.py \
            --tree {input.tree:q} \
            --root {input.root:q} \
            --output {output.alignment:q} \
            --gene {params.gene:q} \
            --tips-only
        """

rule fine_tune:
    input:
        alignment = "data/alignment_fine_tune.fasta"
    output:
        model_weights = config["ESM_fine_tune"]["model_weights"]
    params:
        epochs = config["ESM_fine_tune"]["epochs"],
        batch_size = config["ESM_fine_tune"]["batch_size"],
        model = config["ESM_fine_tune"]["model"]
    shell:
        """
        python ESM/fine-tune.py \
            --input {input.alignment:q} \
            --output {output.model_weights:q} \
            --epochs {params.epochs:q} \
            --batch-size {params.batch_size:q} \
            --model {params.model:q}
        """
