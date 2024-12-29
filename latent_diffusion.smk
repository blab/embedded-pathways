configfile: "defaults/config.yaml"

rule all:
    input:
        generated = "results/generated.fasta",
        ordination = "results/ordination.tsv"

rule provision_alignment:
    output:
        alignment = "data/alignment.fasta"
    params:
        tree = config.get("tree"),
        root = config.get("root"),
        gene = config.get("gene")
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

rule train:
    input:
        alignment = "data/alignment.fasta"
    output:
        vae_model = "models/vae.pth",
        diffusion_model = "models/diffusion.pth"
    shell:
        """
        python latent-diffusion/train.py \
            --input-alignment {input.alignment:q} \
            --output-vae-model {output.vae_model:q} \
            --output-diffusion-model {output.diffusion_model:q}
        """

rule generate:
    input:
        vae_model = "models/vae.pth",
        diffusion_model = "models/diffusion.pth"
    output:
        alignment = "results/generated.fasta"
    shell:
        """
        python latent-diffusion/generate.py \
            --input-vae-model {input.vae_model:q} \
            --input-diffusion-model {input.diffusion_model:q} \
            --output-alignment {output.alignment:q} \
        """

rule embed:
    input:
        alignment = "data/alignment.fasta",
        vae_model = "models/vae.pth"
    output:
        embeddings = "results/embeddings.tsv"
    shell:
        """
        python latent-diffusion/embed.py \
            --input-alignment {input.alignment:q} \
            --input-vae-model {input.vae_model:q} \
            --output-embeddings {output.embeddings:q} \
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
