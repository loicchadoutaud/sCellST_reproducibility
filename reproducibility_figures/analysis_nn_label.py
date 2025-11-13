from pathlib import Path
import os
import numpy as np
from openslide import OpenSlide

from scellst.constant import PREDS_DIR
from sCellST_reproducibility.reproducibility_figures.utils_analyses import (
    load_visium,
    load_predictions,
    compute_deg,
    create_level_coordinates,
    crop_adata,
)
from sCellST_reproducibility.reproducibility_figures.utils_plot import (
    plot_he,
    plot_marker_genes,
    plot_list_gene_image,
    plot_top_genes_2,
)


def run_slide_analysis(
    slide_id: str,
    data_dir: Path,
    save_dir: Path,
    table_save_dir: Path,
    genes_to_plot: list[str],
    crop_coords: list[tuple[int, int]],
    crop_size: int = 4000,
    embedding_tag: str = "moco",
    num_genes: int = 1000,
    model_tag: str = "mil",
    ext: str = "svg",
):
    # Paths
    spot_adata_path = data_dir / "st" / f"{slide_id}.h5ad"
    wsi_path = data_dir / "wsis" / f"{slide_id}.tif"
    pred_filename = f"embedding_tag={embedding_tag}-{slide_id}-rn50_train;genes=HVG:{num_genes};train_slide={slide_id};test_slide={slide_id};infer_mode=inference.h5ad"
    prediction_adata_path = PREDS_DIR / "exp" / model_tag / pred_filename

    save_path = save_dir / f"label_{slide_id}"
    save_path.mkdir(exist_ok=True, parents=True)
    table_save_path = table_save_dir / f"label_{slide_id}"
    table_save_path.mkdir(exist_ok=True, parents=True)

    # Load data
    spot_adata = load_visium(spot_adata_path)
    cell_adata = load_predictions(prediction_adata_path)
    wsi = OpenSlide(wsi_path)
    um_px = float(wsi.properties.get("openslide.mpp-x", None))

    # Plot H&E slides
    plot_he(spot_adata, title="H&E", save_path=save_path / f"he.{ext}")
    spot_adata.obs["in_tissue"] = spot_adata.obs["in_tissue"].astype("category")
    plot_he(
        spot_adata,
        title="Visium slide with spots",
        save_path=save_path / f"he_spot.{ext}",
        obs_color="in_tissue",
    )

    # DEG computation and plotting
    compute_deg(cell_adata, "class")
    list_deg_genes = plot_marker_genes(cell_adata, "class", save_path)
    plot_top_genes_2(cell_adata, "class", genes_to_plot, save_path, table_save_path, add_stat_test=False)

    # Crop plots
    create_level_coordinates(spot_adata, img_level=0)
    create_level_coordinates(cell_adata, img_level=0)
    for i, crop_coord in enumerate(crop_coords):
        crop_image = wsi.read_region(location=crop_coord, level=0, size=(crop_size, crop_size))
        crop_image = np.array(crop_image.convert("RGB"))
        crop_spot_adata = crop_adata(spot_adata, crop_coord, crop_size)
        crop_cell_adata = crop_adata(cell_adata, crop_coord, crop_size)

        plot_list_gene_image(
            crop_spot_adata,
            crop_cell_adata,
            crop_image,
            genes_to_plot,
            save_path / f"local_crops_{i}.{ext}",
            0,
            um_px,
        )

    # Print cell class distribution
    print(cell_adata.obs["class"].value_counts().sort_index().to_latex())


# Example usage
if __name__ == "__main__":
    data_dir = Path("/home/loic/Data/raw_hest")
    save_dir = Path("figures")
    table_save_dir = Path("tables")

    run_slide_analysis(
        slide_id="TENX39",
        data_dir=data_dir,
        save_dir=save_dir,
        table_save_dir=table_save_dir,
        genes_to_plot=["EPCAM", "PTPRC", "INHBA"],
        crop_coords=[(10000, 10000)],
    )

    run_slide_analysis(
        slide_id="TENX65",
        data_dir=data_dir,
        save_dir=save_dir,
        table_save_dir=table_save_dir,
        genes_to_plot=["CDH1", "CD3E", "COL1A2"],
        crop_coords=[(16000, 25000)],
    )