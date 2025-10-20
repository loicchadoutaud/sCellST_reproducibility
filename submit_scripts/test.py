

from sCellST_reproducibility.submit_scripts.utils import get_executor_cbio
from scellst.constant import PROJ_ROOT
from scellst.predict import predict_and_save
from scellst.submit_function import embed_cells, download_data
from sCellST_reproducibility.submit_scripts.script_constants import (
    data_path,
    xenium_slides_review,
    )
from scellst.train import train_and_save


all_shape_names = ["xenium_nucleus", "cellvit"]
all_normalisation_types = ["train", "self"]


if __name__ == "__main__":
    # # Train experiments CellST to Xenium - Breast single slide
    # executor = get_executor_cbio(use_gpu=True, job_name="all")
    # config_default_path = PROJ_ROOT / "config" / "gene_default.yaml"
    # config_kwargs = {"data_dir": data_path, "save_dir_tag": "xenium"}
    # param_grid = {
    #     "embedding_tag": ["moco-TENX39-rn50_self"],
    #     # "genes": ["SVG:1000"],
    #     "genes": ["xenium_Breast"],
    #     "list_training_ids": [["TENX39"]],
    #     "scale": ["no_scaling", "global_scaling"]
    # }
    # configurations = list(ParameterGrid(param_grid))
    # for additional_kwargs in configurations:
    #     additional_kwargs.update(config_kwargs)
    #     executor.submit(train_and_save, config_default_path, additional_kwargs)
    #
    # # Download_cells
    # executor = get_executor_cbio(use_gpu=False, job_name="all")
    # for slide in test_slide:
    #     for shape_name in all_shape_names:
    #         executor.submit(download_data, data_path, None, [slide], shape_name)
    #
    # # Embed cells
    # executor = get_executor_cbio(use_gpu=True, job_name="all")
    # for slide in test_slide:
    #     for normalisation_type in all_normalisation_types:
    #         for shape_name in all_shape_names:
    #             executor.submit(
    #                 embed_cells,
    #                 data_path,
    #                 organ=None,
    #                 ids_to_query=[slide],
    #                 tag=f"moco-TENX39-rn50",
    #                 model_name="resnet50",
    #                 normalisation_type=normalisation_type,
    #                 shape_name=shape_name
    #             )

    # Eval cellst
    executor = get_executor_cbio(use_gpu=True, job_name="all")
    all_config_dir = PROJ_ROOT / "models" / "mil"
    config_kwargs = {
        "data_dir": data_path,
    }
    exp_dir = all_config_dir / "xenium"
    config_dir = (
        exp_dir
        / "embedding_tag=moco-TENX39-rn50_self;genes=SVG:1000;scale=global_scaling;train_slide=TENX39"
    )
    organ = "Breast"
    for slide in xenium_slides_review:
        additional_kwargs = {
            "predict_id": slide,
            "dataset_handler": "xenium",
            "shape_name": "xenium_nucleus",
            "embedding_tag": f"moco-TENX39-rn50_self",
        }
        additional_kwargs.update(config_kwargs)
        executor.submit(
            predict_and_save,
            config_dir,
            additional_kwargs,
            "inference",
            compute_metrics=True,
            save_adata=slide in ["NCBI785", "TENX95"],
        )
