import submitit
from pathlib import Path
from scellst.bench.utils_thitogene import eval_thitogene
from scellst_reproducibility.submit_scripts.script_constants import (
    benchmark_organs,
    visium_slides,
    data_path,
    log_dir,
)

# Configuration
config_kwargs = {
    "data_dir": data_path,
}
all_config_dir = Path("/cluster/CBIO/home/lchadoutaud1/code/CellST/models/THItoGene")

if __name__ == "__main__":
    # Initialize submitit executor
    executor = submitit.AutoExecutor(folder=log_dir)
    executor.update_parameters(
        slurm_array_parallelism=2,
        slurm_partition="cbio-gpu",
        mem_gb=64,
        slurm_gres="gpu:A40:1",
        cpus_per_task=4,
        name="eval_thitogene",
        timeout_min=2880,
    )

    # Benchmark
    with executor.batch():
        for organ in benchmark_organs:
            for config_dir in list(all_config_dir.glob(f"*{organ}*")):
                for visium_slide in visium_slides[organ]:
                    additional_kwargs = {"predict_id": visium_slide}
                    additional_kwargs.update(config_kwargs)
                    executor.submit(eval_thitogene, config_dir, additional_kwargs)
