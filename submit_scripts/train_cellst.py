import submitit
from pathlib import Path

from sklearn.model_selection import ParameterGrid

from scellst.train import train_and_save
from scellst_reproducibility.submit_scripts.script_constants import (
    benchmark_organs,
    benchmark_genes,
    visium_slides,
    log_dir,
    data_path,
)

# Configuration
config_dir = Path("/cluster/CBIO/home/lchadoutaud1/code/CellST/config")
config_default_path = config_dir / "gene_default.yaml"
config_simulation_path = config_dir / "gene_simulation.yaml"


if __name__ == "__main__":
    # Initialize submitit executor
    executor = submitit.AutoExecutor(folder=log_dir)
    executor.update_parameters(
        slurm_array_parallelism=8,
        slurm_partition="cbio-gpu",
        mem_gb=64,
        gpus_per_node=1,
        cpus_per_task=5,
        name="train_cellst",
        timeout_min=2880,
        slurm_exclude="node005,node006",
    )

    with (executor.batch()):
        # Simulation MIL sCellST
        config_kwargs = {"data_dir": data_path, "save_dir_tag": "simulation"}
        param_grid = {
            "list_training_ids": [
                ["TENX65_sim_cell_train"],
                ["TENX65_sim_centroid_train"],
                ["TENX65_sim_random_train"],
            ],
            "lr": [1.0e-2, 1.0e-3, 1.0e-4, 1.0e-5],
            "genes": ["1000_hvg_sim", "marker_sim"],
            "embedding_tag": ["moco-TENX65-rn50", "one-hot-celltype"],
            "dataset_handler": ["mil"],
            "task_type": ["regression", "nb_mean_regression", "nb_total_regression"],
            "scale": ["no_scaling"],
        }
        configurations = list(ParameterGrid(param_grid))
        for additional_kwargs in configurations:
            additional_kwargs.update(config_kwargs)
            executor.submit(train_and_save, config_simulation_path, additional_kwargs)

        # Simulation supervised sCellST
        config_kwargs = {"data_dir": data_path, "save_dir_tag": "simulation"}
        param_grid = {
            "list_training_ids": [["TENX65_sim_cell_train"]],
            "lr": [1.0e-2, 1.0e-3, 1.0e-4, 1.0e-5],
            "genes": ["1000_hvg_sim", "marker_sim"],
            "embedding_tag": ["moco-TENX65-rn50"],
            "dataset_handler": ["supervised"],
            "task_type": ["regression"],
            "scale": ["no_scaling"],
        }
        configurations = list(ParameterGrid(param_grid))
        for additional_kwargs in configurations:
            additional_kwargs.update(config_kwargs)
            executor.submit(train_and_save, config_simulation_path, additional_kwargs)

        # Benchmark CellST
        for organ in benchmark_organs:
            config_kwargs = {"data_dir": data_path, "save_dir_tag": "benchmark"}
            param_grid = {
                "embedding_tag": [f"moco-{organ}-rn50_train"],
                "genes": benchmark_genes[organ],
                "list_training_ids": [[slide] for slide in visium_slides[organ]],
            }
            configurations = list(ParameterGrid(param_grid))
            for additional_kwargs in configurations:
                additional_kwargs.update(config_kwargs)
                executor.submit(train_and_save, config_default_path, additional_kwargs)

            # Benchmark CellST - multiple slides
            config_kwargs = {
                "data_dir": data_path,
                "save_dir_tag": "benchmark-multiple",
            }
            param_grid = {
                "embedding_tag": [f"moco-{organ}-rn50_train"],
                "fold": list(range(len(visium_slides[organ]))),
                "genes": benchmark_genes[organ],
                "list_training_ids": [visium_slides[organ]],
            }
            configurations = list(ParameterGrid(param_grid))
            for additional_kwargs in configurations:
                additional_kwargs.update(config_kwargs)
                executor.submit(train_and_save, config_default_path, additional_kwargs)

        # Train case studies CellST
        config_kwargs = {"data_dir": data_path, "save_dir_tag": "exp"}
        configurations = [
            {
                "embedding_tag": f"moco-TENX65-rn50_train",
                "genes": "HVG:1000",
                "list_training_ids": ["TENX65"],
            },
            {
                "embedding_tag": f"moco-TENX39-rn50_train",
                "genes": "HVG:1000",
                "list_training_ids": ["TENX39"],
            },
            {
                "embedding_tag": f"moco-TENX65-rn50_train",
                "genes": "marker_ovary",
                "list_training_ids": ["TENX65"],
            },
            {
                "embedding_tag": f"imagenet-rn50_train",
                "genes": "marker_ovary",
                "list_training_ids": ["TENX65"],
            },
        ]
        for additional_kwargs in configurations:
            additional_kwargs.update(config_kwargs)
            executor.submit(train_and_save, config_default_path, additional_kwargs)

        # Train experiments CellST to Xenium - Breast single slide
        config_kwargs = {"data_dir": data_path, "save_dir_tag": "xenium"}
        param_grid = {
            "embedding_tag": ["moco-TENX39-rn50_self"],
            "genes": ["SVG:1000"],
            "list_training_ids": [["TENX39"]],
        }
        configurations = list(ParameterGrid(param_grid))
        for additional_kwargs in configurations:
            additional_kwargs.update(config_kwargs)
            executor.submit(train_and_save, config_default_path, additional_kwargs)
