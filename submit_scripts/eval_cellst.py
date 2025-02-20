import submitit
from pathlib import Path
from scellst.predict import predict_and_save
from scellst_reproducibility.submit_scripts.script_constants import (
    benchmark_organs,
    visium_slides,
    xenium_slides,
    log_dir,
    data_path,
)

# Configuration
all_config_dir = Path("/cluster/CBIO/home/lchadoutaud1/code/CellST/models/mil")
config_kwargs = {
    "data_dir": data_path,
}

if __name__ == "__main__":
    # Initialize submitit executor
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

    with executor.batch():
        # Simulation
        list_predict_ids = [
            "TENX65_sim_cell_test",
            "TENX65_sim_centroid_test",
            "TENX65_sim_random_test",
        ]
        exp_dir = all_config_dir / "simulation"
        for predict_id in list_predict_ids:
            pattern = predict_id[:-4] + "train"
            additional_kwargs = {"predict_id": predict_id}
            for config_dir in exp_dir.glob(f"*train_slide={pattern}*"):
                if "mil" in config_dir.stem:
                    executor.submit(
                        predict_and_save,
                        config_dir,
                        additional_kwargs,
                        "bag",
                        compute_metrics=True,
                    )
                    executor.submit(
                        predict_and_save,
                        config_dir,
                        additional_kwargs,
                        "instance",
                        compute_metrics=True,
                    )
                else:
                    executor.submit(
                        predict_and_save,
                        config_dir,
                        additional_kwargs,
                        "instance",
                        compute_metrics=True,
                    )

        # Benchmark
        for organ in benchmark_organs:
            exp_dir = all_config_dir / "benchmark"
            for config_dir in exp_dir.iterdir():
                # Eval for all visium slides
                for visium_slide in visium_slides[organ]:
                    additional_kwargs = {
                        "predict_id": visium_slide,
                    }
                    additional_kwargs.update(config_kwargs)
                    executor.submit(
                        predict_and_save,
                        config_dir,
                        additional_kwargs,
                        infer_mode="bag",
                        compute_metrics=True,
                    )

        # Benchmark multiple slides
        for organ in benchmark_organs:
            exp_dir = all_config_dir / "benchmark-multiple"
            for config_dir in exp_dir.iterdir():
                # Eval for all visium slides
                for visium_slide in visium_slides[organ]:
                    additional_kwargs = {
                        "predict_id": visium_slide,
                    }
                    additional_kwargs.update(config_kwargs)
                    executor.submit(
                        predict_and_save,
                        config_dir,
                        additional_kwargs,
                        infer_mode="bag",
                        compute_metrics=True,
                    )

        # Case study - Ovary hvg
        exp_dir = all_config_dir / "exp"
        config_dir = (
            exp_dir
            / "embedding_tag=moco-TENX65-rn50_train;genes=HVG:1000;train_slide=TENX65"
        )
        additional_kwargs = {"predict_id": "TENX65"}
        executor.submit(
            predict_and_save,
            config_dir,
            additional_kwargs,
            "inference",
            save_adata=True,
        )

        # Case study - Breast hvg
        config_dir = (
            exp_dir
            / "embedding_tag=moco-TENX39-rn50_train;genes=HVG:1000;train_slide=TENX39"
        )
        additional_kwargs = {"predict_id": "TENX39"}
        executor.submit(
            predict_and_save,
            config_dir,
            additional_kwargs,
            "inference",
            save_adata=True,
        )

        # Case study - moco and Ovary markers
        config_dir = (
            exp_dir
            / "embedding_tag=moco-TENX65-rn50_train;genes=marker_ovary;train_slide=TENX65"
        )
        additional_kwargs = {"predict_id": "TENX65"}
        executor.submit(
            predict_and_save,
            config_dir,
            additional_kwargs,
            "inference",
            save_adata=True,
        )

        # Case study - imagenet and Ovary markers
        config_dir = (
            exp_dir
            / "embedding_tag=imagenet-rn50_train;genes=marker_ovary;train_slide=TENX65"
        )
        additional_kwargs = {"predict_id": "TENX65"}
        executor.submit(
            predict_and_save,
            config_dir,
            additional_kwargs,
            "inference",
            save_adata=True,
        )

        # Case study - Xenium dataset
        exp_dir = all_config_dir / "xenium"
        organ = "Breast"
        for config_dir in exp_dir.glob("*TENX39*"):
            for xenium_slide in xenium_slides[organ]:
                additional_kwargs = {
                    "predict_id": xenium_slide,
                    "dataset_handler": "xenium",
                }
                additional_kwargs.update(config_kwargs)
                executor.submit(
                    predict_and_save,
                    config_dir,
                    additional_kwargs,
                    "inference",
                    compute_metrics=True,
                    save_adata=True,
                )
