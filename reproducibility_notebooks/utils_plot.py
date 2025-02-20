import logging
import os
from itertools import combinations
from pathlib import Path

import cv2
import matplotlib as mpl
import numpy as np
import pandas as pd
import scanpy as sc
import seaborn as sns
from anndata import AnnData
from matplotlib import pyplot as plt, cm
from matplotlib.axes import Axes
from matplotlib.colors import Normalize, LogNorm
from matplotlib.patches import Patch
from numpy import ndarray
from openslide import OpenSlide
from pandas import DataFrame
from sklearn.preprocessing import MinMaxScaler
from tqdm.auto import tqdm

from scellst.constant import SUB_CLASS_LABELS, COLOR_MAP

DPI = 300

plt.rcParams.update({"font.size": 14})
sns.set_style("whitegrid")

logger = logging.getLogger(__name__)


def plot_he(adata: AnnData, title: str, save_path: str, obs_color: str = None) -> None:
    alpha = 0.5 if obs_color is not None else None
    fig, ax = plt.subplots(1, 1, figsize=(5, 4))
    sc.pl.spatial(
        adata,
        color=obs_color,
        alpha=alpha,
        img_key="downscaled_fullres",
        show=False,
        ax=ax,
    )
    ax.set_title(title)
    ax.set_axis_off()
    if ax.get_legend() is not None:
        ax.get_legend().remove()
    fig.savefig(save_path, dpi=DPI)


def create_overlayed_image(
    main_image: ndarray,
    overlay_image: ndarray,
    inset_scale: float = 0.3,
    border_color: tuple[int, int, int] = (255, 0, 0),
    border_thickness: int = 20,
):
    # Get dimensions of the main image
    h, w, _ = main_image.shape
    ho, wo, _ = overlay_image.shape

    # Calculate dimensions for the overlay image
    ratio_scale = max(h, w) / max(ho, wo)
    overlay_w = int(inset_scale * ratio_scale * wo)
    overlay_h = int(inset_scale * ratio_scale * ho)

    # Resize overlay image to fit in the corner
    overlay_resized = cv2.resize(overlay_image, (overlay_w, overlay_h))

    # Position of the overlay image (bottom-right corner of the main image)
    overlay_x = 0
    overlay_y = h - overlay_h

    # Place the overlay image on the main image
    main_image[
        overlay_y : overlay_y + overlay_h, overlay_x : overlay_x + overlay_w
    ] = overlay_resized

    # Draw a red border around the overlay image
    cv2.rectangle(
        main_image,
        (overlay_x + border_thickness // 2, overlay_y),
        (
            overlay_x + overlay_w - border_thickness // 2,
            overlay_y + overlay_h - border_thickness // 2,
        ),
        border_color,
        border_thickness,
    )

    return main_image


def create_image_with_crop(
    wsi: OpenSlide,
    crop_coords: tuple[int, int],
    crop_size: int,
    img_level: int,
    border_color: tuple[int, int, int] = (255, 0, 0),
    border_thickness: int = 20,
) -> ndarray:
    img = np.array(
        wsi.read_region(
            location=(0, 0), level=img_level, size=wsi.level_dimensions[img_level]
        )
    )[:, :, :3].copy()
    scale_factor = 1 / (2**img_level)
    hs, ws = int(scale_factor * crop_coords[0]), int(scale_factor * crop_coords[1])
    cs = int(scale_factor * crop_size)
    cv2.rectangle(img, (hs, ws), (hs + cs, ws + cs), border_color, border_thickness)
    return img


def create_gene_img(
    coords: ndarray,
    values: ndarray,
    base_image: ndarray,
    pixel_radius: int = 10,
) -> ndarray:
    # Prepare cmap
    if values.dtype == "category":
        colors = np.vectorize(COLOR_MAP.get)(values)
        colors = np.stack(colors, axis=1)
    elif values.dtype == "O":
        cmap = sns.color_palette("tab10", len(np.unique(values)))
        cmap_dict = dict(zip(np.unique(values), cmap))
        colors = np.vectorize(cmap_dict.get)(values)
        colors = np.stack(colors, axis=1)
    else:
        norm = Normalize(
            vmin=np.quantile(values, q=0.05), vmax=np.quantile(values, q=0.95)
        )
        norm_values = norm(values)
        cmap = plt.colormaps["viridis"]
        colors = cmap(norm_values)
    colors = 255 * colors

    # Add circle to image
    for i in tqdm(range(len(colors))):
        x, y = int(coords[i, 0]), int(coords[i, 1])
        r, g, b = int(colors[i, 0]), int(colors[i, 1]), int(colors[i, 2])
        base_image = cv2.circle(
            base_image, (x, y), pixel_radius, (r, g, b), thickness=-1
        )

    return base_image


def plot_gene_image(
    spot_adata: AnnData,
    cell_adata: AnnData,
    wsi_image: ndarray,
    gene_name: str,
    save_path: str,
    img_level: int,
) -> None:
    # Create image
    spot_image = create_gene_img(
        spot_adata.obsm[f"spatial_{img_level}"],
        spot_adata[:, gene_name].X.toarray().squeeze(),
        np.zeros_like(wsi_image),
        int(
            spot_adata.uns["spatial"][next(iter(spot_adata.uns["spatial"].keys()))][
                "scalefactors"
            ]["spot_diameter_fullres"]
            / 2
        )
        // (img_level + 1),
    )
    cell_image = create_gene_img(
        cell_adata.obsm[f"spatial_{img_level}"],
        cell_adata[:, gene_name].X.squeeze(),
        np.zeros_like(wsi_image),
        12 // (img_level + 1),
    )
    celltype_image = create_gene_img(
        cell_adata.obsm[f"spatial_{img_level}"],
        cell_adata.obs["class"].values,
        np.zeros_like(wsi_image),
        12 // (img_level + 1),
    )

    # Plot and save figure
    fig, axes = plt.subplots(1, 4, figsize=(24, 6))
    axes[0].imshow(wsi_image)
    axes[1].imshow(celltype_image)
    axes[2].imshow(spot_image)
    axes[3].imshow(cell_image)

    # Add plot info
    titles = ["H&E", "Cell types", f"Visium {gene_name}", f"Predicted {gene_name}"]
    for ax, title in zip(axes, titles):
        ax.axis("off")
        ax.set_title(title)
    # Create legend handles
    legend_handles = [
        Patch(color=color, label=label) for label, color in TYPE_COLOR.items()
    ]
    fig.subplots_adjust(right=0.85, wspace=0.01)
    fig.legend(
        handles=legend_handles,
        title="Cell Types",
        loc="center",
        bbox_to_anchor=(0.89, 0.5),
    )

    # Add colorbar
    norm = Normalize(vmin=0, vmax=1)
    cmap = plt.colormaps["viridis"]
    sm = cm.ScalarMappable(cmap=cmap, norm=norm)
    colorbar_position = [0.93, 0.2, 0.02, 0.5]
    colorbar_position[1] = 0.5 - colorbar_position[3] / 2
    cbar_ax = fig.add_axes(colorbar_position)
    cbar_ax.set_title(f"{gene_name}")
    fig.colorbar(sm, cax=cbar_ax)

    # Save figure
    fig.savefig(save_path, bbox_inches="tight", dpi=DPI)


def plot_list_gene_image(
    spot_adata: AnnData,
    cell_adata: AnnData,
    wsi_image: ndarray,
    list_gene_name: list[str],
    save_path: str,
    img_level: int,
) -> None:
    # Create image
    celltype_image = create_gene_img(
        cell_adata.obsm[f"spatial_{img_level}"],
        cell_adata.obs["class"].values,
        np.zeros_like(wsi_image),
        14 // (img_level + 1),
    )
    spot_images = [
        create_gene_img(
            spot_adata.obsm[f"spatial_{img_level}"],
            spot_adata[:, gene_name].X.toarray().squeeze(),
            np.zeros_like(wsi_image),
            int(
                spot_adata.uns["spatial"][next(iter(spot_adata.uns["spatial"].keys()))][
                    "scalefactors"
                ]["spot_diameter_fullres"]
                / 2
            )
            // 2**img_level,
        )
        for gene_name in list_gene_name
    ]
    cell_images = [
        create_gene_img(
            cell_adata.obsm[f"spatial_{img_level}"],
            cell_adata[:, gene_name].X.squeeze(),
            np.zeros_like(wsi_image),
            14 // (img_level + 1),
        )
        for gene_name in list_gene_name
    ]

    # Plot and save figure
    n_rows, n_cols = 2, 2 + len(list_gene_name)
    fig, axes = plt.subplots(
        n_rows,
        n_cols,
        figsize=(6 * (len(list_gene_name) + 1) + 1, 10),
        gridspec_kw={"width_ratios": [8] * (len(list_gene_name) + 1) + [2]},
    )
    axes[0, 0].imshow(wsi_image)
    axes[1, 0].imshow(celltype_image)
    for i in range(len(list_gene_name)):
        axes[0, 1 + i].imshow(spot_images[i])
        axes[1, 1 + i].imshow(cell_images[i])

    # Add plot info
    titles = ["H&E", "Cell types"] + [
        f"{m} {g}" for g in list_gene_name for m in ["Visium", "CellST"]
    ]
    for ax, title in zip(axes[:, :-1].flatten(order="F"), titles):
        ax.axis("off")
        ax.set_title(title)

    # Create legend handles
    legend_handles = [
        Patch(color=color, label=label) for label, color in COLOR_MAP.items()
    ]
    axes[0, -1].legend(
        handles=legend_handles,
        title="Cell Types",
        loc="center",
    )
    axes[0, -1].axis("off")

    # Add colorbars
    norm = Normalize(vmin=0, vmax=1)
    cmap = plt.colormaps["viridis"]
    sm = cm.ScalarMappable(cmap=cmap, norm=norm)
    cbar = fig.colorbar(sm, cax=axes[1, -1], orientation="vertical")
    cbar.set_ticks([])
    axes[1, -1].set_title("Min-max scaled\ngene expression", ha="center", fontsize=14)

    # Save figure
    fig.subplots_adjust(hspace=0.2, wspace=0.1)
    fig.savefig(save_path, bbox_inches="tight", dpi=DPI)


def plot_xenium_list_gene_image(
    adata: AnnData,
    wsi_image: ndarray,
    list_gene_name: list[str],
    save_path: str,
    img_level: int,
) -> None:
    # Create image
    cell_label_images = [
        create_gene_img(
            adata.obsm[f"spatial_{img_level}"],
            adata[:, gene_name].X.toarray().squeeze(),
            np.zeros_like(wsi_image),
            12 // (img_level + 1),
        )
        for gene_name in list_gene_name
    ]
    cell_pred_images = [
        create_gene_img(
            adata.obsm[f"spatial_{img_level}"],
            adata[:, gene_name].layers["predictions"].squeeze(),
            np.zeros_like(wsi_image),
            12 // (img_level + 1),
        )
        for gene_name in list_gene_name
    ]

    # Plot and save figure
    n_rows, n_cols = len(list_gene_name), 3
    fig, axes = plt.subplots(
        n_rows,
        n_cols,
        figsize=(20, 8),
    )
    if len(axes.shape) == 1:
        axes = axes[np.newaxis]
    for i in range(len(list_gene_name)):
        axes[i, 0].imshow(wsi_image)
        axes[i, 1].imshow(cell_label_images[i])
        axes[i, 2].imshow(cell_pred_images[i])

        axes[i, 0].set_title("H&E")
        axes[i, 1].set_title(f"Measured {list_gene_name[i]}")
        axes[i, 2].set_title(f"Predicted {list_gene_name[i]}")

    for ax in axes.flatten():
        ax.axis("off")

    # Save figure
    fig.subplots_adjust(hspace=0.2, wspace=0.1)
    fig.savefig(save_path, bbox_inches="tight", dpi=DPI)
    plt.close()


def plot_signature_score(
    adata: AnnData, obs_key: str, list_scores: list[str], save_path: str
) -> None:
    # Prepare data
    adata = adata[adata.obs[obs_key].isin(SUB_CLASS_LABELS)]
    df_plot = adata.obs[list_scores + ["class"]].copy()
    df_plot[list_scores] = MinMaxScaler().fit_transform(df_plot[list_scores])
    df_plot = df_plot.melt(
        id_vars=["class"], var_name="group", value_name="min-max scaled score"
    )

    # Plot
    fig, ax = plt.subplots(1, 1, figsize=(12, 6))
    sns.boxplot(
        data=df_plot,
        x="group",
        y="min-max scaled score",
        hue="class",
        order=list_scores,
        palette=COLOR_MAP,
        ax=ax,
    )
    ax.set_xlabel("")
    ax.set_title("Distribution of different cell type score.")
    sns.move_legend(ax, "upper left", bbox_to_anchor=(1, 1))
    sns.despine(fig)
    fig.savefig(save_path, bbox_inches="tight", dpi=DPI)


def plot_corr_score(adata: AnnData, list_scores: list[str], save_path: str) -> None:
    hm = sns.clustermap(
        data=adata.obs[list_scores].corr(),
        vmin=-1,
        vmax=1,
        annot=True,
        cmap="coolwarm",
        cbar_pos=(0.01, 0.45, 0.03, 0.4),
    )
    hm.ax_row_dendrogram.remove()
    hm.ax_heatmap.set_xticklabels(
        hm.ax_heatmap.get_xticklabels(), rotation=30, ha="right"
    )
    hm.fig.savefig(save_path, bbox_inches="tight", dpi=DPI)


def plot_top_genes_2(
    adata: AnnData, hue: str, list_scores: list[str], save_path
) -> None:
    palette = COLOR_MAP if hue == "class" else None

    adata_plot = adata[:, list_scores]
    adata_plot = adata_plot[adata_plot.obs[hue].isin(SUB_CLASS_LABELS)]
    adata_plot.X = MinMaxScaler().fit_transform(adata_plot.X)
    df = pd.DataFrame(
        data=adata_plot.X,
        index=adata_plot.obs_names,
        columns=adata_plot.var_names,
    )
    df[hue] = adata_plot.obs[hue]
    df = df.melt(id_vars=hue, var_name="gene", value_name="expression")

    fig, axes = plt.subplots(1, 1, figsize=(2 * len(list_scores), 4))
    sns.boxplot(data=df, x="gene", y="expression", hue=hue, ax=axes, palette=palette)
    axes.set_title(f"Gene score distribution\n (min-max scaled)")
    sns.move_legend(axes, "upper left", bbox_to_anchor=(1, 1))
    sns.despine()
    fig.savefig(
        os.path.join(save_path, f"gene_distribution_boxplot.png"),
        bbox_inches="tight",
        dpi=DPI,
    )


def plot_gene_gallery(
    adata: AnnData,
    color: str,
    wsi: OpenSlide,
    save_path: str,
    img_size: int = 48,
    title: str | None = None,
) -> None:
    # Get index of cells to plot
    if color in adata.obs_keys():
        values = adata.obs[color].values
    elif color in adata.var_names:
        values = adata[:, color].X.squeeze()
    else:
        raise ValueError(f"Key: {color} not found in obs columns or adata var names.")
    idxs = np.argsort(values)[-100:][::-1]

    if title is None:
        title = color

    # Plot figure
    n_rows, n_cols = 10, 10
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(3 * n_cols, 3 * n_rows))
    for i, ax in zip(idxs, axes.flatten()):
        x_center, y_center = adata.obsm["spatial"][i] - (img_size // 2)
        img = wsi.read_region(
            location=(x_center, y_center), level=0, size=(img_size, img_size)
        )
        ax.imshow(img)
        ax.set_axis_off()
    fig.subplots_adjust(wspace=0.05, hspace=0.05)
    fig.suptitle(title, y=0.92, fontsize=90)
    fig.savefig(save_path, bbox_inches="tight", dpi=DPI)
    plt.close()


def plot_marker_genes(adata: AnnData, obs_key: str, save_path: str) -> list[str]:
    print("Plotting deg...")
    adata = adata[adata.obs[obs_key].isin(SUB_CLASS_LABELS)]

    # Save dataframe
    df = sc.get.rank_genes_groups_df(adata, key="rank_" + obs_key, group=None)
    top_scores = df.groupby("group", observed=False)["scores"].nlargest(20)
    top_scores_with_string = pd.merge(
        top_scores,
        df[["group", "scores", "names"]],
        how="left",
        left_on=["group", "scores"],
        right_on=["group", "scores"],
    )
    top_scores_with_string.to_csv(os.path.join(save_path, "markers.csv"))
    marker_genes = top_scores_with_string.drop_duplicates(subset="group")

    # Plot results
    with mpl.rc_context({"font.size": 12, "patch.edgecolor": "black"}):
        fig = sc.pl.rank_genes_groups_dotplot(
            adata,
            groups=SUB_CLASS_LABELS,
            n_genes=5,
            key="rank_" + obs_key,
            var_group_rotation=0,
            standard_scale="var",
            colorbar_title="Minmax-scaled scores",
            dendrogram=False,
            show=False,
            return_fig=True,
        )
        fig.savefig(
            os.path.join(save_path, "mean_dotplots.png"), bbox_inches="tight", dpi=DPI
        )

    return marker_genes[marker_genes["group"].isin(SUB_CLASS_LABELS)]["names"].tolist()
    # return df


def plot_metrics_simulation(
    df_metrics: DataFrame,
    x: str,
    y: str,
    title: str,
    save_path: Path,
    hue: str,
    x_order: list[str] | None = None,
    hue_order: list[str] | None = None,
) -> Axes:
    if x_order is None:
        x_order = df_metrics[x].drop_duplicates().sort_values().values
    if hue_order is None:
        hue_order = df_metrics[hue].drop_duplicates().sort_values().values

    fig, axes = plt.subplots(figsize=(len(x_order) * 3, 4))

    hue_norm = None
    match hue:
        case "scenario":
            palette = "Paired"
        case "genes":
            palette = "rocket"
        case "task type":
            palette = "Set2"
        case "learning framework":
            palette = "Set2_r"
        case "lr":
            palette = "viridis"
            hue_norm = LogNorm()
        case _:
            palette = "viridis"

    print(f"Plot: {title}")
    print(df_metrics.groupby([x, hue]).size())
    print(df_metrics.groupby([x, hue])[y].mean())

    # Add plot
    sns.barplot(
        data=df_metrics,
        x=x,
        y=y,
        hue=hue,
        hue_norm=hue_norm,
        order=x_order,
        hue_order=hue_order,
        palette=palette,
        ax=axes,
    )
    axes.set_xlabel("")
    axes.set_title(title)
    sns.move_legend(axes, "upper left", bbox_to_anchor=(1, 1))

    # Save figure
    sns.despine()
    fig.savefig(save_path, bbox_inches="tight", dpi=300)
    plt.show()
    plt.close(fig)


def plot_top_enriched_cells(adata: AnnData, img_path: str, save_path) -> None:
    top_index = np.argsort(adata.X, axis=1)[:, -1]
    adata.obs["predicted_cell_type"] = adata.var_names.values[top_index]
    print(adata.obs["predicted_cell_type"].value_counts())
    sns.histplot(data=adata.obs, y="predicted_cell_type")
    plt.savefig(save_path + f"_cell_type_pred.png", bbox_inches="tight")
    plt.close()
    df_acts = pd.DataFrame(data=adata.X, index=adata.obs_names, columns=adata.var_names)
    df_acts = df_acts.melt(var_name="cell_type", value_name="score")
    sns.violinplot(df_acts, x="score", y="cell_type")
    plt.savefig(save_path + f"_cell_type_score.png", bbox_inches="tight")
    plt.close()

    for var in adata.var_names:
        print(f"Plotting {var}...")
        plot_gene_gallery(adata, var, img_path, save_path + f"_enriched_cell_{var}.png")
        plot_gene_gallery(
            adata[adata.obs["predicted_cell_type"] == var],
            var,
            img_path,
            save_path + f"_enriched_predicted_cell_{var}.png",
        )


if __name__ == "__main__":
    tips = sns.load_dataset("tips")

    ax = sns.boxplot(data=tips, x="day", y="total_bill", hue="sex")
    print_median_labels(ax)
    plt.show()
