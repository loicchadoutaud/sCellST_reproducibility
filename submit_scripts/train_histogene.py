from sklearn.model_selection import ParameterGrid

from scellst.bench.utils_histogene import train_histogene
import submitit

from scellst_reproducibility.submit_scripts.script_constants import (
    benchmark_organs,
    visium_slides,
    benchmark_genes,
    config_dir,
    data_path,
    log_dir,
)

# Configuration
config_path = config_dir / "gene_histogene.yaml"


if __name__ == "__main__":
    # Initialize submitit executor
    executor = submitit.AutoExecutor(folder=log_dir)
    executor.update_parameters(
        slurm_array_parallelism=3,
        slurm_partition="cbio-gpu",
        mem_gb=64,
        slurm_gres="gpu:A40:1",
        cpus_per_task=4,
        name="train_histogene",
        timeout_min=2880,
    )

    with executor.batch():
        config_kwargs = {"data_dir": data_path}

        for organ in benchmark_organs:
            # HisToGene - single slide
            param_grid = {
                "genes": benchmark_genes[organ],
                "list_training_ids": [[slide] for slide in visium_slides[organ]],
            }
            configurations = list(ParameterGrid(param_grid))
            for additional_kwargs in configurations:
                additional_kwargs.update(config_kwargs)
                executor.submit(train_histogene, config_path, additional_kwargs)

            # HisToGene - multiple slides
            param_grid = {
                "genes": benchmark_genes[organ],
                "list_training_ids": [visium_slides[organ]],
                "fold": list(range(len(visium_slides[organ]))),
            }
            configurations = list(ParameterGrid(param_grid))
            for additional_kwargs in configurations:
                additional_kwargs.update(config_kwargs)
                executor.submit(train_histogene, config_path, additional_kwargs)
