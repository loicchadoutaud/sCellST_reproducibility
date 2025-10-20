import submitit

from scellst.constant import PROJ_ROOT
from scellst.predict import predict_and_save
from sCellST_reproducibility.submit_scripts.script_constants import (
    visium_slides,
    log_dir,
    data_path,
    )

# === Configuration ===
all_config_dir = PROJ_ROOT / "models" / "mil"
config_kwargs = {"data_dir": data_path}

executor = submitit.AutoExecutor(folder=log_dir)
executor.update_parameters(
    slurm_array_parallelism=8,
    slurm_partition="cbio-gpu",
    mem_gb=64,
    gpus_per_node=1,
    cpus_per_task=4,
    name="eval_scellst",
    timeout_min=2880,
    slurm_exclude="node005",
)

if __name__ == "__main__":
    with executor.batch():

        # === Benchmark + Marker Evaluation for Visium ===
        organ = "Prostate"
        # for exp_type in ["review-benchmark", "review-genes"]:
        for exp_type in ["review-genes"]:
            exp_dir = all_config_dir / exp_type
            for config_dir in exp_dir.glob(f"*{organ}*"):
                for visium_slide in visium_slides[organ]:
                    additional_kwargs = {
                        "predict_id": visium_slide,
                        **config_kwargs,
                    }
                    executor.submit(
                        predict_and_save,
                        config_dir,
                        additional_kwargs,
                        infer_mode="bag",
                        compute_metrics=True,
                    )

        # # === Case Study: Ovary Marker ===
        # executor.submit(
        #     predict_and_save,
        #     all_config_dir / "exp" / "embedding_tag=moco-TENX65-rn50_train;genes=marker_ovary;shape_name=cellvit;train_slide=TENX65",
        #     {"predict_id": "TENX65"},
        #     "inference",
        #     align=False,
        #     save_adata=True,
        # )
        #
        # # === Case Study: Moco versus Imagenet ===
        # executor.submit(
        #     predict_and_save,
        #     all_config_dir / "exp" / "embedding_tag=imagenet-rn50_train;genes=marker_ovary;shape_name=cellvit;train_slide=TENX65",
        #     {"predict_id": "TENX65"},
        #     "inference",
        #     align=False,
        #     save_adata=True,
        # )
        #
        # # === Case Study: Breast Marker ===
        # executor.submit(
        #     predict_and_save,
        #     all_config_dir / "exp" / "embedding_tag=moco-TENX39-rn50_train;genes=marker_breast;shape_name=cellvit;train_slide=TENX39",
        #     {"predict_id": "TENX39"},
        #     "inference",
        #     align=False,
        #     save_adata=True,
        # )
        #
        # # === Case Study: HoverFast vs Standard ===
        # executor.submit(
        #     predict_and_save,
        #     all_config_dir / "exp" / "embedding_tag=moco-TENX65-rn50_train;genes=marker_ovary;shape_name=hoverfast;train_slide=TENX65",
        #     {"predict_id": "TENX65"},
        #     "inference",
        #     align=False,
        #     save_adata=True,
        # )

        # # === Case Study: Xenium ===
        # config_dir = (
        #     all_config_dir / "xenium" / "embedding_tag=moco-TENX39-rn50_self;genes=SVG:1000;scale=global_scaling;train_slide=TENX39"
        # )
        # for xenium_slide in xenium_slides_review:
        #     additional_kwargs = {
        #         "predict_id": xenium_slide,
        #         "dataset_handler": "xenium",
        #         "shape_name": "xenium_nucleus",
        #         **config_kwargs,
        #     }
        #     executor.submit(
        #         predict_and_save,
        #         config_dir,
        #         additional_kwargs,
        #         "inference",
        #         compute_metrics=True,
        #         save_adata=xenium_slide in ["NCBI785", "TENX95"],
        #     )
