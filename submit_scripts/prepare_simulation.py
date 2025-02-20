import submitit
from pathlib import Path
from simulation.main import prepare_simulation
from scellst_reproducibility.submit_scripts.script_constants import log_dir, data_path

# Configuration
ref_adata_path = Path("data/raw_ovarian_dataset.h5ad")
embedding_key = "moco-TENX65-rn50_TENX65"
simulation_modes = ["centroid", "cell", "random"]

if __name__ == "__main__":
    # Initialize submitit executor
    executor = submitit.AutoExecutor(folder=log_dir)
    executor.update_parameters(
        slurm_array_parallelism=3,
        slurm_partition="cbio-gpu",
        mem_gb=64,
        gpus_per_node=1,
        cpus_per_task=6,
        name="prepare",
        timeout_min=2880,
    )

    # Submit jobs for each organ
    with executor.batch():  # Submit jobs in batch for efficiency
        for simulation_mode in simulation_modes:
            executor.submit(
                prepare_simulation,
                data_path,
                ref_adata_path,
                embedding_key,
                simulation_mode,
            )
