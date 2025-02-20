import submitit
from scellst.submit_function import run_ssl
from scellst_reproducibility.submit_scripts.script_constants import (
    visium_slides,
    benchmark_organs,
    log_dir,
    data_path,
)

n_gpus = 2
n_cpus_per_gpu = 6


if __name__ == "__main__":
    # Initialize submitit executor
    executor = submitit.AutoExecutor(folder=log_dir)
    executor.update_parameters(
        slurm_array_parallelism=3,
        slurm_partition="cbio-gpu",
        mem_gb=64,
        gpus_per_node=n_gpus,
        cpus_per_task=n_gpus * n_cpus_per_gpu,
        name="run_moco",
        timeout_min=2880,
        slurm_exclude="node005,node006,node009",
    )

    # Submit jobs for each organ
    with executor.batch():  # Submit jobs in batch for efficiency
        # Sample training
        for sample in ["TENX39", "TENX65"]:
            tag = f"moco-{sample}-rn50"
            executor.submit(
                run_ssl, data_path, None, [sample], tag, n_gpus, n_cpus_per_gpu
            )

        # Organ training
        for organ in benchmark_organs:
            tag = f"moco-{organ}-rn50"
            executor.submit(
                run_ssl,
                data_path,
                None,
                visium_slides[organ],
                tag,
                n_gpus,
                n_cpus_per_gpu,
            )
