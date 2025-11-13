from pathlib import Path
from PIL import Image
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import ImageGrid
import pandas as pd
import scanpy as sc
from hest import load_wsi
from sCellST_reproducibility.reproducibility_figures.utils_analyses import load_visium, preprocess_adata
from scellst.constant import DATA_DIR
from scellst.plots.plot_spatial import _rasterize_points_in_axes


def plot_ffpe_vs_frozen(ffpe_path: str, frozen_path: str, output_dir: Path, ext: str = "pdf"):
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load FFPE
    wsi_ffpe, pixel_size_ffpe = load_wsi(ffpe_path)
    location_ffpe = (10_000, 10_000)
    size_ffpe = int(300 / pixel_size_ffpe)
    ffpe_img = wsi_ffpe.read_region(location_ffpe, 0, (size_ffpe, size_ffpe))

    # Load Frozen
    wsi_frozen, _ = load_wsi(frozen_path)
    location_frozen = (15_000, 10_000)
    size_frozen = int(300 / 0.3)
    frozen_img = wsi_frozen.read_region(location_frozen, 0, (size_frozen, size_frozen))

    # Plotting
    fig, axes = plt.subplots(1, 2)
    axes[0].imshow(ffpe_img)
    axes[0].axis("off")
    axes[0].set_title("Crop of FFPE slide")

    axes[1].imshow(frozen_img)
    axes[1].axis("off")
    axes[1].set_title("Crop of Frozen slide")

    fig.savefig(output_dir / f"ffpe_frozen_comp.{ext}", dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_kidney_cell_counts(figure_folder: Path, slide_names: list[str], img_tag: str, output_path: Path):
    imgs = [Image.open(figure_folder / f"{name}_{img_tag}.png") for name in slide_names]

    fig, axes = plt.subplots(3, 4, figsize=(24, 16))
    fig.subplots_adjust(wspace=0, hspace=0.1)

    for i, ax in enumerate(axes.flatten()):
        ax.imshow(imgs[i])
        ax.axis("off")
        ax.set_title(slide_names[i])

    fig.suptitle("Number of detected cells in spots for FFPE Kidney dataset", fontsize=20, y=0.92)
    fig.savefig(output_path, dpi=280, bbox_inches="tight")
    plt.close(fig)


def plot_plasma_marker(data_dir: Path, slide_id: str, organ: str, group: str, output_path: Path):
    # Paths
    spot_adata_path = data_dir / "st" / f"{slide_id}.h5ad"
    df_marker_path = DATA_DIR / f"genes_marker_{organ}.csv"

    # Load data
    spot_adata = load_visium(spot_adata_path)
    spot_adata = preprocess_adata(spot_adata)
    df_marker = pd.read_csv(df_marker_path, index_col=0)
    list_genes = df_marker.loc[df_marker["group"].eq(group), "gene"].to_list()
    list_genes = [g for g in list_genes if g in spot_adata.var_names]

    # Make plots
    fig = sc.pl.spatial(
        spot_adata,
        color=list_genes,
        img_key="downscaled_fullres",
        cmap="magma",
        ncols=5,
        show=False,
        return_fig=True,
        wspace=0.
    )

    # rasterize only the scatter artists (keeps text/axes vector in SVG/PDF)
    for ax in fig.axes:
        for coll in getattr(ax, "collections", []):
            coll.set_rasterized(True)

    fig.savefig(output_path, dpi=150, bbox_inches="tight", transparent=True)


def create_marker_gene_table(
    csv_path: str,
    selected_groups: list | None = None,
    output_latex: bool = True,
    column_format: str = "p{3cm}p{10cm}"
) -> pd.Series:
    """
    Create a LaTeX table from a gene marker CSV file based on selected cell types.

    Parameters
    ----------
    csv_path : str
        Path to the CSV file containing at least 'group' and 'gene' columns.
    selected_groups : list
        List of cell types to include in the table.
    output_latex : bool, optional
        Whether to print the LaTeX-formatted table (default is True).
    column_format : str, optional
        Column format for the LaTeX table (default is "p{3cm}p{10cm}").

    Returns
    -------
    pd.Series
        A Series where the index is the cell type and values are comma-separated gene names.
    """
    df = pd.read_csv(csv_path)

    # Filter, group, and sort genes
    if selected_groups is None:
        selected_groups = df["group"].unique()
    grouped = (
        df[df["group"].isin(selected_groups)]
        .groupby("group")["gene"]
        .apply(lambda x: ", ".join(sorted(x)))
        .sort_index()
    )

    # Rename for formatting
    grouped.index.name = "cell type"
    grouped.name = "Top 20 genes"

    if output_latex:
        print(grouped.to_latex(column_format=column_format))

    return grouped



if __name__ == "__main__":
    data_dir = Path("/home/loic/Data/raw_hest")
    save_dir = Path("figures/sup")
    table_dave_dir = Path("tables/sup")

    # # Compare FFPE vs Frozen
    # plot_ffpe_vs_frozen(
    #     ffpe_path="/home/loic/Downloads/Visium_FFPE_Human_Breast_Cancer_image.tif",
    #     frozen_path="/home/loic/Downloads/Visium_Human_Breast_Cancer_image.tif",
    #     output_dir=output_folder,
    # )
    #
    # # Kidney dataset plots
    # kidney_slide_names = [f"INT{i}" for i in range(13, 25)]
    # plot_kidney_cell_counts(
    #     figure_folder=Path("../../hest_data/cell_plots"),
    #     slide_names=kidney_slide_names,
    #     img_tag="cellvit_spot_cell_count",
    #     output_path=output_folder / "spot_cell_counts.pdf",
    # )
    # plot_kidney_cell_counts(
    #     figure_folder=Path("../../hest_data/cell_plots"),
    #     slide_names=kidney_slide_names,
    #     img_tag="cellvit_hist_cell_count",
    #     output_path=output_folder / "hist_cell_counts.pdf",
    # )

    # Plasmablast marker
    plot_plasma_marker(
        data_dir=data_dir,
        slide_id="TENX39",
        organ="breast",
        group="Plasmablasts",
        output_path=save_dir / "plasma_marker.svg",
    )

    #
    # # Tables
    # _ = create_marker_gene_table(
    #     csv_path="../../data/genes_marker_ovary.csv",
    #     selected_groups=[
    #         "fibroblast", "lymphocyte", "endothelial cell",
    #         "plasma cell", "fallopian tube secretory epithelial cell"
    #     ]
    # )
    #
    # # Tables
    # _ = create_marker_gene_table(
    #     csv_path="../../data/genes_marker_breast.csv",
    # )
