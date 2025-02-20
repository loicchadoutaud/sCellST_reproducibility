import submitit
from scellst.submit_function import download_data
from scellst_reproducibility.submit_scripts.script_constants import (
    visium_slides,
    xenium_slides,
    log_dir,
    data_path,
)

if __name__ == "__main__":
    # Initialize submitit executor
    executor = submitit.AutoExecutor(folder=log_dir)
    executor.update_parameters(
        slurm_array_parallelism=8,
        slurm_partition="cbio-cpu",
        mem_gb=64,
        cpus_per_task=4,
        name="download_data",
        timeout_min=2880,
    )

    # Submit jobs for each organ
    with executor.batch():
        # Visium slides
        for list_ids in visium_slides.values():
            executor.submit(download_data, data_path, None, list_ids)

        # Xenium slides
        for list_ids in xenium_slides.values():
            executor.submit(download_data, data_path, None, list_ids)
