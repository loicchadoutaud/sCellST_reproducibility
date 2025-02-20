import submitit
from scellst.submit_function import embed_cells
from scellst_reproducibility.submit_scripts.script_constants import (
    visium_slides,
    benchmark_organs,
    xenium_slides,
    data_path,
    log_dir,
)

if __name__ == "__main__":
    # Initialize submitit executor
    executor = submitit.AutoExecutor(folder=log_dir)
    executor.update_parameters(
        slurm_array_parallelism=6,
        slurm_partition="cbio-gpu",  # Use specified partition
        mem_gb=64,  # Set memory to 64GB
        gpus_per_node=1,  # Request 1 GPU
        cpus_per_task=4,
        name="embed_cell",
        timeout_min=2880,
        slurm_exclude="node005,node006",
    )

    # Submit jobs for each organ
    with executor.batch():
        for organ in benchmark_organs:
            executor.submit(
                embed_cells,
                data_path,
                organ=None,
                ids_to_query=visium_slides[organ],
                tag=f"moco-{organ}-rn50",
                model_name="resnet50",
            )

        executor.submit(
            embed_cells,
            data_path,
            organ=None,
            ids_to_query=["TENX39"],
            tag="moco-TENX39-rn50",
            model_name="resnet50",
            normalisation_type="train",
        )
        executor.submit(
            embed_cells,
            data_path,
            organ=None,
            ids_to_query=xenium_slides["Breast"],
            tag=f"moco-TENX39-rn50",
            model_name="resnet50",
            normalisation_type="self",
        )
        executor.submit(
            embed_cells,
            data_path,
            organ=None,
            ids_to_query=["TENX65"],
            tag="moco-TENX65-rn50",
            model_name="resnet50",
            normalisation_type="train",
        )
        executor.submit(
            embed_cells,
            data_path,
            organ=None,
            ids_to_query=["TENX65"],
            tag="imagenet",
            model_name="resnet50",
            normalisation_type="train",
        )
