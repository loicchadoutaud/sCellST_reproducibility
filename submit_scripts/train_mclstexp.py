import submitit
from sklearn.model_selection import ParameterGrid

from scellst.bench.utils_mclstexp import train_mclstexp
from scellst_reproducibility.submit_scripts.script_constants import (
    benchmark_organs,
    visium_slides,
    benchmark_genes,
    config_dir,
    data_path,
    log_dir,
)

config_path = config_dir / "gene_mclstexp.yaml"


if __name__ == "__main__":
    # Initialize submitit executor
    executor = submitit.AutoExecutor(folder=log_dir)
    executor.update_parameters(
        slurm_array_parallelism=2,
        slurm_partition="cbio-gpu",
        mem_gb=96,
        cpus_per_task=4,
        name="train_mclstexp",
        timeout_min=2880,
        slurm_gres="gpu:A40:1",
    )

    with executor.batch():
        for organ in benchmark_organs:
            config_kwargs = {"data_dir": data_path}

            # MclstExp - single slides
            param_grid = {
                "genes": benchmark_genes[organ],
                "list_training_ids": [[slide] for slide in visium_slides[organ]],
            }
            configurations = list(ParameterGrid(param_grid))
            for additional_kwargs in configurations:
                additional_kwargs.update(config_kwargs)
                executor.submit(train_mclstexp, config_path, additional_kwargs)

            # MclstExp - multiple slides
            param_grid = {
                "genes": benchmark_genes[organ],
                "list_training_ids": [visium_slides[organ]],
                "fold": list(range(len(visium_slides[organ]))),
            }
            configurations = list(ParameterGrid(param_grid))
            for additional_kwargs in configurations:
                additional_kwargs.update(config_kwargs)
                executor.submit(train_mclstexp, config_path, additional_kwargs)
