from pathlib import Path

from sCellST_reproducibility.submit_scripts.script_constants import log_dir


def get_executor_cbio(
    use_gpu: bool,
    max_time=0,
    n_gpus: int = 1,
    cpus_per_task: int = 4,
    job_name="sp",
    log_path: Path = log_dir,
    slurm_array_parallelism: int | None = 4,
    exclusive: bool = False,
):
    import submitit

    # Initialize the executor.
    executor = submitit.AutoExecutor(folder=log_path)

    # Add basic parameters.
    slurm_params = {
        "ntasks-per-node": 1,
        "hint": "nomultithread",
        "exclude": "node005,node009",
    }

    # Add slurm params specific to the partition chosen
    slurm_setup = []

    if use_gpu:
        slurm_params.update(
            {
                "distribution": "block:block",
                "gres": f"gpu:{n_gpus}",
                "cpus-per-task": cpus_per_task,
                "partition": "cbio-gpu",
                "mem": "96G",
            }
        )
    else:
        slurm_params.update({"partition": "cbio-cpu", "cpus-per-task": 8, "mem": "96G"})

    executor.update_parameters(
        slurm_job_name=job_name,
        slurm_time=max_time,
        slurm_additional_parameters=slurm_params,
        slurm_setup=slurm_setup,
    )
    if slurm_array_parallelism is not None:
        executor.update_parameters(
            slurm_array_parallelism=slurm_array_parallelism,
        )
    if exclusive:
        executor.update_parameters(
            slurm_setup=["#SBATCH --exclusive"],
        )
    return executor
