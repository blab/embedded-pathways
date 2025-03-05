configfile: "defaults/config.yaml"

rule all:
    input:
        model = config["ESM_fine_tune"]["model"]

rule download_auspice_json:
    output:
        tree = "data/auspice.json",
        root = "data/auspice_root-sequence.json"
    params:
        dataset = config["ESM_fine_tune"]["dataset"]
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
        alignment = "data/alignment.fasta"
    output:
        model = config["ESM_fine_tune"]["model"]
    shell:
        """
        python ESM/fine-tune.py \
            --input {input.alignment:q} \
            --output {output.model:q}
        """
