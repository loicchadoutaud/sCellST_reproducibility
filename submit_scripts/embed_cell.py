import submitit

from sCellST_reproducibility.submit_scripts.utils import get_executor_cbio
from scellst.submit_function import embed_cells
from sCellST_reproducibility.submit_scripts.script_constants import (
    visium_slides,
    benchmark_organs,
    xenium_slides,
    data_path,
    log_dir, xenium_slides_review,
)

if __name__ == "__main__":
    # Initialize submitit executor
    executor = get_executor_cbio(use_gpu=True, job_name="embed_cell")

    # Submit jobs for each organ
    with executor.batch():
        # # Baseline embeddings (wait for this to finish to launch the rest)
        # for organ in benchmark_organs:
        #     executor.submit(
        #         embed_cells,
        #         data_path,
        #         organ=None,
        #         ids_to_query=visium_slides[organ],
        #         tag=f"imagenet-rn50",
        #         model_name="resnet50",
        #         normalisation_type="train",
        #     )
        #
        # # Moco embeddings
        # for organ in benchmark_organs:
        #     executor.submit(
        #         embed_cells,
        #         data_path,
        #         organ=None,
        #         ids_to_query=visium_slides[organ],
        #         tag=f"moco-{organ}-rn50",
        #         model_name="resnet50",
        #         normalisation_type="train",
        #     )

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
