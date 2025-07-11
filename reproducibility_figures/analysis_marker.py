from pathlib import Path
import scanpy as sc
import pandas as pd
from openslide import OpenSlide
from loguru import logger

from scellst.constant import PREDS_DIR
from sCellST_reproducibility.reproducibility_figures.utils_analyses import (
    compute_signature_scores,
    load_predictions,
)
from sCellST_reproducibility.reproducibility_figures.utils_plot import (
    plot_gene_gallery,
    plot_corr_score,
    plot_signature_score, plot_full_gallery,
)


def run_marker_visualization(
    slide_id: str,
    organ: str,
    data_dir: Path,
    save_path: Path,
    embedding_tag: str = "moco",
    model_tag: str = "mil",
    shape_name: str | None = None,
    list_score_to_plot: list[str] | None = None,
    list_gallery_to_plot: list[str] | None = None,
    include_signature_score: bool = True,
    include_gallery: bool = True,
    extra_gallery_spec: list[dict] | None = None,
) -> None:
    """
    Generate marker score plots and galleries for a given slide.

    Parameters:
    - slide_id: name of the slide (e.g., "TENX65")
    - organ: name of the organ (e.g., "ovary")
    - data_dir: path to the main data folder (should contain "st" and "wsis" subfolders)
    - save_path: output directory to save plots
    - embedding_tag: "moco", "imagenet", etc.
    - model_tag: "mil" by default
    - shape_name: Optional hovernet shape string (e.g., "hoverfast")
    - list_score_to_plot: list of score groups (e.g., cell types) to plot in the boxplots
    - list_gallery_to_plot: list of score groups (e.g., cell types) to plot in the gallery
    - include_gallery: if True, plots gallery for each score
    - extra_gallery_spec: list of dictionaries with keys: 'class_value', 'score_group', 'title'
    """
    save_path.mkdir(parents=True, exist_ok=True)

    # Construct input paths
    wsi_path = data_dir / "wsis" / f"{slide_id}.tif"
    df_marker_path = data_dir.parent / "data" / f"genes_marker_{organ}.csv"

    # Build prediction path
    components = [
        f"embedding_tag={embedding_tag}",
        f"genes=marker_{organ}",
        f"shape_name={shape_name}",
        f"train_slide={slide_id}",
        f"test_slide={slide_id}",
        "infer_mode=inference.h5ad",
    ]
    prediction_filename = ";".join(components)
    prediction_path = PREDS_DIR / "exp" / model_tag / prediction_filename

    # Load data
    logger.info(f"Loading predictions from: {prediction_path}")
    cell_adata = load_predictions(prediction_path)
    logger.info(cell_adata)
    sc.pp.normalize_total(cell_adata, target_sum=1e4)
    sc.pp.scale(cell_adata)

    logger.info(f"Loading WSI from: {wsi_path}")
    wsi = OpenSlide(wsi_path)

    logger.info(f"Loading marker genes from: {df_marker_path}")
    df_marker = pd.read_csv(df_marker_path, index_col=0)
    df_marker["group"] = df_marker["group"].replace(
        "fallopian tube secretory epithelial cell", "epithelial cell"
    )
    cell_adata.obs["class"] = cell_adata.obs["class"].replace(
        "fallopian tube secretory epithelial cell", "epithelial cell"
    )

    # Filter and compute
    df_marker = df_marker[df_marker["gene"].isin(cell_adata.var_names)]
    compute_signature_scores(cell_adata, df_marker)

    # Determine groups to plot
    if list_score_to_plot is None:
        list_score_to_plot = list(df_marker["group"].unique())
    if list_gallery_to_plot is None:
        list_gallery_to_plot = list_score_to_plot

    logger.info(f"Plotting {len(list_score_to_plot)} signature scores")
    if include_signature_score:
        plot_signature_score(
            cell_adata,
            obs_key="class",
            list_scores=list_score_to_plot,
            save_path=save_path / "gene_score.png",
            add_stat_test=False,
        )
    plot_corr_score(
        cell_adata,
        list_scores=list_score_to_plot,
        save_path=save_path / "gene_corr.png",
    )

    if include_gallery:
        list_gallery_to_plot = [g for g in list_gallery_to_plot if g in df_marker["group"].unique()]
        for grp in list_gallery_to_plot:
            logger.info(f"Plotting gallery for {grp}")
            plot_gene_gallery(
                cell_adata,
                color=grp,
                wsi=wsi,
                save_path=save_path / f"gallery_{grp}",
            )
        plot_full_gallery(img_dir=save_path, list_score_to_plot=list_gallery_to_plot, save_path=save_path / f"full_gallery")

    if extra_gallery_spec:
        for spec in extra_gallery_spec:
            class_val = spec["class_value"]
            score_group = spec["score_group"]
            title = spec.get("title", f"{score_group} classified as {class_val}")
            subset = cell_adata[cell_adata.obs["class"] == class_val]
            out_path = (
                save_path / f"gallery_{score_group}_classified_as_{class_val}"
            )
            logger.info(f"Plotting subset gallery: {title}")
            plot_gene_gallery(
                subset,
                color=score_group,
                wsi=wsi,
                save_path=out_path,
                title=title,
            )


if __name__ == '__main__':
    data_dir = Path("../../hest_data")
    save_dir = Path("figures")

    # slide_id = "TENX65"
    # run_marker_visualization(
    #     slide_id=slide_id,
    #     organ="ovary",
    #     data_dir=data_dir,
    #     save_path=save_dir / f"marker_{slide_id}",
    #     embedding_tag=f"moco-{slide_id}-rn50_train",
    #     shape_name="cellvit",
    #     list_score_to_plot=[
    #         "fibroblast",
    #         "endothelial cell",
    #         "epithelial cell",
    #         "lymphocyte",
    #         "plasma cell",
    #     ],
    #     list_gallery_to_plot=[
    #         "fibroblast",
    #         "endothelial cell",
    #         "lymphocyte",
    #         "plasma cell",
    #     ],
    #     extra_gallery_spec=[
    #         {"class_value": "Connective", "score_group": "plasma cell"},
    #         {"class_value": "Neoplastic", "score_group": "plasma cell"},
    #     ],
    # )

    # slide_id = "TENX65"
    # run_marker_visualization(
    #     slide_id=slide_id,
    #     organ="ovary",
    #     data_dir=data_dir,
    #     save_path=save_dir / f"marker_{slide_id}_imagenet",
    #     embedding_tag=f"imagenet-rn50_train",
    #     shape_name="cellvit",
    #     list_score_to_plot=[
    #         "fibroblast",
    #         "endothelial cell",
    #         "lymphocyte",
    #         "plasma cell",
    #     ],
    # )
    #
    # slide_id = "TENX65"
    # run_marker_visualization(
    #     slide_id=slide_id,
    #     organ="ovary",
    #     data_dir=data_dir,
    #     save_path=save_dir / f"marker_{slide_id}_hoverfast",
    #     embedding_tag=f"moco-{slide_id}-rn50_train",
    #     shape_name="hoverfast",
    #     list_score_to_plot=[
    #         "fibroblast",
    #         "endothelial cell",
    #         "lymphocyte",
    #         "plasma cell",
    #     ],
    #     include_signature_score=False,
    # )
    #
    slide_id = "TENX39"
    run_marker_visualization(
        slide_id=slide_id,
        organ="breast",
        data_dir=data_dir,
        save_path=save_dir / f"marker_{slide_id}",
        embedding_tag=f"moco-{slide_id}-rn50_train",
        list_gallery_to_plot=['B-cells', 'CAFs', 'Cancer Epithelial', 'Endothelial', 'Myeloid', 'Normal Epithelial', 'PVL', 'Plasmablasts', 'T-cells', 'JCHAIN'],
        shape_name="cellvit",
    )
