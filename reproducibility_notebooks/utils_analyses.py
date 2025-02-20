import logging
import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scanpy as sc
from anndata import AnnData
from geopandas import GeoDataFrame
from openslide import OpenSlide
from pandas import DataFrame, Series
from tqdm.auto import tqdm


logger = logging.getLogger(__name__)
DPI = 300
GROUPS = ["connec", "inflam", "neopla"]
plt.rcParams.update({"font.size": 14})


def load_metrics(folder_metric_path: str) -> DataFrame:
    # Load output metrics
    metric_paths = [
        os.path.join(folder_metric_path, f) for f in os.listdir(folder_metric_path)
    ]
    df_metrics = [pd.read_csv(path, index_col=0) for path in tqdm(metric_paths)]
    df_metrics = pd.concat(df_metrics, axis=0, ignore_index=True)
    df_metrics, hp = add_hp_columns(df_metrics, sep_out=";", sep_in="-")
    return df_metrics


def load_all_metrics(
    folder_path: str, tag: str, marker_name: str | None = None
) -> DataFrame:
    all_metrics = []
    for object_type in ["spot", "cell"]:
        if not os.path.exists(os.path.join(folder_path, tag, object_type)):
            print(f"Skipped {object_type}")
            continue
        df_metrics = load_metrics(os.path.join(folder_path, tag, object_type))
        df_metrics = df_metrics.rename(
            columns={
                f"pcc_{object_type}": "pcc",
                f"scc_{object_type}": "scc",
            }
        )
        df_metrics["object_type"] = object_type

        all_genes = df_metrics.copy()
        all_genes["gene_category"] = "hvg"

        if marker_name is not None:
            print("Load marker genes and add rows to anndata.")
            marker_genes = pd.read_csv(
                f"../data/{marker_name}_celltype_markers.csv", index_col=0
            )
            marker_genes = marker_genes["names"].unique()
            df_marker = df_metrics[df_metrics["label_name"].isin(marker_genes)].copy()
            df_marker["gene_category"] = "marker"

            all_metrics.append(pd.concat([all_genes, df_marker]))
        else:
            all_metrics.append(all_genes)

    df_metrics = pd.concat(all_metrics).reset_index()
    df_metrics = df_metrics.rename(
        columns={col: col.replace("_", " ") for col in df_metrics.columns}
    )
    return df_metrics


def add_hp_columns(
    df: DataFrame, sep_out: str, sep_in: str
) -> tuple[DataFrame, list[str]]:
    # Step 1: Split the "tag" column into individual label-value pairs
    df["tag"] = df["tag"].str.split(sep_out)

    # Step 2: Identify all unique labels across all tags
    unique_labels = set(tag.split(sep_in)[0] for tags in df["tag"] for tag in tags)
    unique_labels = sorted(list(unique_labels))

    # expand the "tag" column into a DataFrame where each tag becomes a row
    expanded_df = df["tag"].explode()

    # split each tag into label and value
    expanded_df = expanded_df.str.split(sep_in, expand=True)
    expanded_df.columns = ["label", "value"]
    result_df = pd.get_dummies(expanded_df["label"]).mul(expanded_df["value"], axis=0)

    # group by the index (which corresponds to the original DataFrame's index) and sum up the rows
    result_df = result_df.groupby(result_df.index).sum()

    # Step 5 merge it with the original df
    df = pd.merge(df, result_df, left_index=True, right_index=True)
    return df, unique_labels


def load_cell_adata(file_path: str) -> AnnData:
    print("Loading cell adata...")
    return sc.read_h5ad(file_path)


def preprocess_adata(adata: AnnData) -> AnnData:
    print("Preprocessing cell adata...")
    # Filtering
    sc.pp.filter_genes(adata, min_counts=200)
    sc.pp.filter_genes(adata, min_cells=adata.shape[0] // 10)
    sc.pp.filter_cells(adata, min_counts=20)
    adata.var["mt"] = adata.var_names.str.startswith("MT-")
    adata = adata[:, ~(adata.var["mt"])].copy()

    # Preprocess data
    adata.layers["counts"] = adata.X.copy()
    sc.pp.normalize_total(adata, target_sum=1e4)
    sc.pp.log1p(adata)

    # Compute library size with HVG
    sc.pp.highly_variable_genes(
        adata,
        n_top_genes=5000,
        layer="counts",
        flavor="seurat_v3",
        subset=False,
    )

    return adata


def load_slide(file_path: str) -> OpenSlide:
    print("Loading slide...")
    return OpenSlide(file_path)


def load_visium(adata_path: Path) -> AnnData:
    spot_adata = sc.read_h5ad(adata_path)
    # spot_adata.var_names_make_unique()
    # spot_adata.obsm["spatial"] = spot_adata.obsm["spatial"].astype("int")
    # spot_adata.obs["in_tissue"] = spot_adata.obs["in_tissue"].astype("category")
    # spot_adata = spot_adata[spot_adata.obsm["spatial"][:, 0] >= 0].copy()
    return spot_adata


def load_predictions(cell_adata_path: str) -> AnnData:
    cell_adata = sc.read_h5ad(cell_adata_path)
    cell_adata.var_names_make_unique()
    cell_adata.obsm["spatial"] = cell_adata.obsm["spatial"] + (
        cell_adata.uns["patch_size_src"] // 2
    )  # Center cell coordinates
    return cell_adata


def create_level_coordinates(adata: AnnData, img_level: int) -> None:
    adata.obsm[f"spatial_{img_level}"] = adata.obsm["spatial"] * (1 / 2**img_level)


def crop_adata(adata: AnnData, crop_coord: tuple[int, int], crop_size: int) -> AnnData:
    x_min, y_min = crop_coord[0], crop_coord[1]
    x_max, y_max = x_min + crop_size, y_min + crop_size
    crop_adata = adata[
        (x_min < adata.obsm["spatial"][:, 0])
        & (adata.obsm["spatial"][:, 0] < x_max)
        & (y_min < adata.obsm["spatial"][:, 1])
        & (adata.obsm["spatial"][:, 1] < y_max)
    ]
    crop_adata.obsm["spatial_0"] -= np.asarray([x_min, y_min])
    return crop_adata


def crop_gdf(
    gdf: GeoDataFrame, crop_coord: tuple[int, int], crop_size: int
) -> GeoDataFrame:
    x_min, y_min = crop_coord[0], crop_coord[1]
    x_max, y_max = x_min + crop_size, y_min + crop_size
    crop_gdf = gdf.cx[x_min:x_max, y_min:y_max]
    crop_gdf["geometry"] = crop_gdf["geometry"].translate(xoff=-x_min, yoff=-y_min)
    return crop_gdf


def compute_deg(adata: AnnData, obs_key: str) -> None:
    print("Computing deg...")
    if adata.obs[obs_key].dtype != "category":
        adata.obs[obs_key] = adata.obs[obs_key].astype("category")
        adata.obs[obs_key] = adata.obs[obs_key].cat.reorder_categories(
            adata.obs[obs_key].cat.categories.sort_values()
        )
    sc.tl.rank_genes_groups(
        adata,
        groupby=obs_key,
        method="t-test",
        key_added="rank_" + obs_key,
    )


def select_best_lr_model(
    df: DataFrame, columns: list[str], metric: str, gene_cat: str
) -> DataFrame:
    grouped = (
        df[df["genes"] == gene_cat].groupby(columns)[metric].median().reset_index()
    )
    print(grouped)
    best_lr_per_model = grouped.loc[grouped.groupby(columns[0])[metric].idxmax()]
    return pd.merge(df, best_lr_per_model[columns], on=columns)


def compute_signature_scores(adata: AnnData, df_marker: DataFrame) -> None:
    # Compute scores
    for grp in df_marker["group"].unique():
        gene_list = df_marker[df_marker["group"] == grp]["gene"].tolist()
        sc.tl.score_genes(adata, gene_list=gene_list, score_name=grp)


def highlight_max(s: Series) -> list[str]:
    # Apply highlighting only for 'mean' columns
    if "mean" in s.name:  # Check if 'mean' is in the column level
        is_max = s == s.max()
        return ["font-weight: bold" if v else "" for v in is_max]
    else:
        return ["" for _ in s]  # No highlighting for other columns
