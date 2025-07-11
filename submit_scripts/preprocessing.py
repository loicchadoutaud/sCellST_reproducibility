import submitit

from scellst.bench.utils_data import prepare_list_hvg, prepare_list_svg, prepare_list_heg
from sCellST_reproducibility.submit_scripts.script_constants import (
    benchmark_organs,
    visium_slides,
    data_path,
    log_dir, xenium_slides_review,
)
from scellst.constant import DATA_DIR
from scellst.utils.img_utils import compute_target_representative_stains, batch_normalize_from_h5

list_n_genes = [50, 200, 500, 1000, 2000]

if __name__ == "__main__":
    # Initialize submitit executor
    executor = submitit.AutoExecutor(folder=log_dir)
    executor.update_parameters(
        slurm_array_parallelism=8,
        slurm_partition="cbio-cpu",
        mem_gb=64,
        cpus_per_task=4,
        name="preprocessing",
        timeout_min=2880,
    )

    # # Gene selection
    # with executor.batch():
    #     for organ in benchmark_organs:
    #         for n_genes in list_n_genes:
    #             executor.submit(
    #                 prepare_list_hvg, data_path, visium_slides[organ], n_genes, organ
    #             )
    #             executor.submit(
    #                 prepare_list_heg, data_path, visium_slides[organ], n_genes, organ
    #             )
    #             executor.submit(
    #                 prepare_list_svg, data_path, visium_slides[organ], n_genes, organ
    #             )

    # Compute representative images
    for slide_id in ["TENX39"]:
        for shape_name in ["xenium_nucleus", "cellvit"]:
            executor.submit(
                compute_target_representative_stains,
                h5_path=data_path / "cell_images" / f"{slide_id}_{shape_name}.h5",
                output_path=f"rep_{slide_id}_{shape_name}.npy",
            )


    # # Apply representative images
    # target_id = "TENX39"
    # for slide_id in xenium_slides_review + [target_id]:
    # # for slide_id in [target_id]:
    #     for shape_name in ["xenium_nucleus", "cellvit"]:
    #         executor.submit(
    #             batch_normalize_from_h5,
    #             h5_path=data_path / "cell_images" / f"{slide_id}_{shape_name}.h5",
    #             target_path=DATA_DIR / "rep_stains" / f"rep_{target_id}_{shape_name}.npy",
    #             save_path=DATA_DIR / "rep_images" / f"{slide_id}_{target_id}.png",
    #             num_samples=50
    #         )
