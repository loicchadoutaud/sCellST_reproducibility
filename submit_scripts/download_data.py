import submitit
from scellst.submit_function import download_data
from sCellST_reproducibility.submit_scripts.script_constants import (
    visium_slides,
    xenium_slides,
    log_dir,
    data_path, xenium_slides_review, not_good_visium_slides, all_visium_slides, test_slide,
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
        # executor.submit(download_data, data_path, None, all_visium_slides)
        executor.submit(download_data, data_path, None, ["TENX39"])


        # for slide_id in xenium_slides_review:
        #     executor.submit(
        #         download_data,
        #         data_path,
        #         None,
        #         [slide_id],
        #     )
