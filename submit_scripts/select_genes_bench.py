import submitit

from scellst.bench.utils_data import prepare_list_hvg, prepare_list_svg, prepare_list_heg
from scellst_reproducibility.submit_scripts.script_constants import (
    benchmark_organs,
    visium_slides,
    data_path,
    log_dir,
)

list_n_genes = [50, 500]

if __name__ == "__main__":
    # Initialize submitit executor
    executor = submitit.AutoExecutor(folder=log_dir)
    executor.update_parameters(
        slurm_array_parallelism=8,
        slurm_partition="cbio-cpu",
        mem_gb=64,
        cpus_per_task=4,
        name="select_genes_benchmark",
        timeout_min=2880,
    )

    # Submit jobs for each organ
    with executor.batch():
        for organ in benchmark_organs:
            for n_genes in list_n_genes:
                executor.submit(
                    prepare_list_hvg, data_path, visium_slides[organ], n_genes, organ
                )
                executor.submit(
                    prepare_list_heg, data_path, visium_slides[organ], n_genes, organ
                )
                executor.submit(
                    prepare_list_svg, data_path, visium_slides[organ], n_genes, organ
                )
