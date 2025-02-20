import submitit
from sklearn.model_selection import ParameterGrid

from scellst.bench.utils_istar import train_istar
from scellst_reproducibility.submit_scripts.script_constants import (
    benchmark_organs,
    visium_slides,
    benchmark_genes,
    config_dir,
    data_path,
    log_dir,
)

# Configuration
config_path = config_dir / "gene_istar.yaml"

if __name__ == "__main__":
    # Initialize submitit executor
    executor = submitit.AutoExecutor(folder=log_dir)
    executor.update_parameters(
        slurm_array_parallelism=3,
        slurm_partition="cbio-gpu",  # Use specified partition
        mem_gb=64,  # Set memory to 64GB
        slurm_gres="gpu:A40:1",
        cpus_per_task=8,  # Request 4 cpus
        name="train_istar",
        timeout_min=2880,
    )

    with executor.batch():
        config_kwargs = {"data_dir": data_path}
        for organ in benchmark_organs:
            param_grid = {
                "genes": benchmark_genes[organ],
                "list_training_ids": [[slide] for slide in visium_slides[organ]],
            }
            configurations = list(ParameterGrid(param_grid))
            for additional_kwargs in configurations:
                additional_kwargs.update(config_kwargs)
                executor.submit(train_istar, config_path, additional_kwargs)
