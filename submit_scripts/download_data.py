from pathlib import Path

import submitit

from scellst.cellhest_adapter.processing_utils import fetch_data
from scellst.submit_function import download_data
from sCellST_reproducibility.submit_scripts.script_constants import (
    log_dir,
    data_path, )

if __name__ == "__main__":
    fetch_data("/home/loic/Data/raw_hest", ["TENX39", "TENX65"])

    # # Initialize submitit executor
    # executor = submitit.AutoExecutor(folder=log_dir)
    # executor.update_parameters(
    #     slurm_array_parallelism=8,
    #     slurm_partition="cbio-cpu",
    #     mem_gb=64,
    #     cpus_per_task=4,
    #     name="download_data",
    #     timeout_min=2880,
    # )
    #
    # # Submit jobs for each organ
    # with executor.batch():
    #     # Visium slides
    #     # executor.submit(download_data, data_path, None, all_visium_slides)
    #     executor.submit(download_data, "/home/loic/Data/raw_hest", None, ["TENX39", "TENX65"])
    #
    #
    #     # for slide_id in xenium_slides_review:
    #     #     executor.submit(
    #     #         download_data,
    #     #         data_path,
    #     #         None,
    #     #         [slide_id],
    #     #     )
