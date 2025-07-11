import submitit

from sklearn.model_selection import ParameterGrid

from scellst.constant import PROJ_ROOT
from scellst.train import train_and_save
from sCellST_reproducibility.submit_scripts.script_constants import (
    visium_slides,
    log_dir,
    data_path, number_of_genes_exp,
)

# Configuration
config_default_path = PROJ_ROOT / "config" / "gene_default.yaml"


if __name__ == "__main__":
    # Initialize submitit executor
    executor = submitit.AutoExecutor(folder=log_dir)
    executor.update_parameters(
        slurm_array_parallelism=8,
        slurm_partition="cbio-gpu",
        mem_gb=64,
        gpus_per_node=1,
        cpus_per_task=6,
        name="train_cellst",
        timeout_min=2880,
        slurm_exclude="node005,node006",
    )

    with executor.batch():
        # # Benchmark CellST - multiple slides
        # organ= "Prostate"
        # config_kwargs = {"data_dir": data_path, "save_dir_tag": f"review-benchmark"}
        # param_grid = {
        #     "embedding_tag": [f"moco-{organ}-rn50_train", "imagenet-rn50_train"],
        #     "fold": list(range(len(visium_slides[organ]))),
        #     "genes": [f"{organ}_50_hvg_bench"],
        #     "list_training_ids": [visium_slides[organ]],
        # }
        # configurations = list(ParameterGrid(param_grid))
        # for additional_kwargs in configurations:
        #     additional_kwargs.update(config_kwargs)
        #     executor.submit(train_and_save, config_default_path, additional_kwargs)


        # Number of genes experiments
        organ = "Prostate"
        config_kwargs = {"data_dir": data_path, "save_dir_tag": "review-genes"}
        param_grid = {
            "embedding_tag": [f"moco-{organ}-rn50_train"],
            "fold": list(range(len(visium_slides[organ]))),
            "genes": number_of_genes_exp[organ],
            "list_training_ids": [visium_slides[organ]],
        }
        configurations = list(ParameterGrid(param_grid))
        for additional_kwargs in configurations:
            additional_kwargs.update(config_kwargs)
            executor.submit(train_and_save, config_default_path, additional_kwargs)

        # # Train case studies CellST
        # config_kwargs = {"data_dir": data_path, "save_dir_tag": "exp"}
        # configurations = [
            # {
            #     "embedding_tag": f"moco-TENX65-rn50_train",
            #     "genes": "marker_ovary",
            #     "list_training_ids": ["TENX65"],
            #     "shape_name": "cellvit",
            # },
            # {
            #     "embedding_tag": f"imagenet-rn50_train",
            #     "genes": "marker_ovary",
            #     "list_training_ids": ["TENX65"],
            #     "shape_name": "cellvit",
            # },
            # {
            #     "embedding_tag": f"moco-TENX65-rn50_train",
            #     "genes": "marker_ovary",
            #     "list_training_ids": ["TENX65"],
            #     "shape_name": "hoverfast"
            # },
        #     {
        #         "embedding_tag": f"moco-TENX39-rn50_train",
        #         "genes": "marker_breast",
        #         "list_training_ids": ["TENX39"],
        #         "shape_name": "cellvit",
        #     },
        # ]
        # for additional_kwargs in configurations:
        #     additional_kwargs.update(config_kwargs)
        #     executor.submit(train_and_save, config_default_path, additional_kwargs)

        # # Train experiments CellST to Xenium - Breast single slide
        # config_kwargs = {"data_dir": data_path, "save_dir_tag": "xenium"}
        # additional_kwargs = {
        #     "embedding_tag": "moco-TENX39-rn50_self",
        #     "genes": "SVG:1000",
        #     "list_training_ids": ["TENX39"],
        #     "scale": "global_scaling"
        # }
        # additional_kwargs.update(config_kwargs)
        # executor.submit(train_and_save, config_default_path, additional_kwargs)
