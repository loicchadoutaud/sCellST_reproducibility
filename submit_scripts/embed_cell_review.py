from sCellST_reproducibility.submit_scripts.utils import get_executor_cbio
from scellst.submit_function import embed_cells
from sCellST_reproducibility.submit_scripts.script_constants import (
    data_path, xenium_slides_review, test_slide,
)

if __name__ == "__main__":
    # Initialize submitit executor
    executor = get_executor_cbio(use_gpu=True, job_name="embed_cell")

    # Submit jobs for each organ
    with executor.batch():
        # # Moco embeddings
        # executor.submit(
        #     embed_cells,
        #     data_path,
        #     organ=None,
        #     ids_to_query=["TENX65"],
        #     tag="moco-TENX65-rn50",
        #     model_name="resnet50",
        #     normalisation_type="train",
        #     shape_name="hoverfast"
        # )

        # for slide in xenium_slides_review:
        for slide in test_slide:
            executor.submit(
                embed_cells,
                data_path,
                organ=None,
                ids_to_query=[slide],
                tag=f"moco-TENX39-rn50",
                model_name="resnet50",
                normalisation_type="self",
                shape_name="cellvit"
            )

