import numpy as np
import pandas as pd
import scanpy as sc
import submitit
from anndata import AnnData
from loguru import logger
from mygene import MyGeneInfo

from sCellST_reproducibility.submit_scripts.script_constants import log_dir
from scellst.constant import DATA_DIR


def load_ref_adata(
    organ: str,
) -> AnnData:
    # Filtering dict
    filter_dict= {
        "ovary": {
            "tissue": ["left ovary", "right ovary"],
        }
    }

    # Load data
    adata = sc.read_h5ad(DATA_DIR / f"raw_{organ}_dataset.h5ad", backed="r")

    # Filter values if necessary
    if organ in filter_dict.keys():
        mask = np.ones(len(adata), dtype=bool)
        for key, value in filter_dict[organ].items():
            mask = mask & adata.obs[key].isin(value)
        adata = adata[mask]

    # Prepare data
    adata = adata.to_memory()
    adata = adata.raw.to_adata()
    adata = adata[:, adata.var[adata.var["feature_type"] == "protein_coding"].index]

    # Map ensemble to HUGO
    ensembl_ids = adata.var_names.tolist()
    mg = MyGeneInfo()
    gene_info = mg.querymany(ensembl_ids, scopes="ensembl.gene", fields="symbol", species="human")
    gene_map = pd.DataFrame(gene_info)[["query", "symbol"]].dropna()
    gene_map = gene_map.drop_duplicates(subset="query")
    adata.var = adata.var.merge(gene_map, left_index=True, right_on="query", how="left")

    # Set HUGO
    adata.var_names = adata.var["symbol"].astype(str)
    adata.var_names_make_unique()
    logger.info(f"Loaded reference AnnData with shape {adata.X.shape}")

    # Filter genes
    adata.var["mt"] = adata.var_names.str.startswith("MT-")
    adata.var["ribo"] = adata.var_names.str.startswith(("RPS", "RPL"))
    mask = ~(adata.var["mt"] | adata.var["ribo"])
    adata = adata[:, mask]
    logger.info(f"After gene filtering AnnData with shape {adata.X.shape}")

    return adata

def preprocess_adata(
        adata: AnnData,
) -> None:
    # Filtering
    sc.pp.filter_cells(adata, min_genes=200)
    sc.pp.filter_genes(adata, min_counts=10)

    # Normalisation
    adata.layers["counts"] = adata.X.copy()
    sc.pp.normalize_total(adata)
    sc.pp.log1p(adata)

    logger.info("Preprocessed reference dataset.")

def extract_and_save_all_marker_genes(adata: AnnData, organ: str) -> None:
    celltype_col_dict = {
        "ovary": "cell_type",
        "breast": "celltype_major",
    }

    # Filter too small groups
    obs_names = adata.obs.groupby(celltype_col_dict[organ]).filter(lambda x: len(x) >= 100).index
    adata = adata[obs_names]

    # Extract marker genes
    sc.tl.rank_genes_groups(
        adata, groupby=celltype_col_dict[organ], method="t-test_overestim_var"
    )
    df = sc.get.rank_genes_groups_df(adata, group=None)
    if organ == "ovary":
        df["group"] = df["group"].replace(
            {"B cell": "lymphocyte", "T cell": "lymphocyte"}
        )
    top_scores = df.groupby("group")["scores"].nlargest(20)
    top_scores_with_string = pd.merge(
        top_scores,
        df[["group", "scores", "names"]],
        how="left",
        left_on=["group", "scores"],
        right_on=["group", "scores"],
    )
    top_scores_with_string.rename(columns={"names": "gene"}, inplace=True)
    top_scores_with_string.to_csv(DATA_DIR / f"genes_marker_{organ}.csv")


def main(organ: str) -> None:
    adata = load_ref_adata(organ)
    preprocess_adata(adata)
    extract_and_save_all_marker_genes(adata, organ)


if __name__ == '__main__':
    # Initialize submitit executor
    executor = submitit.AutoExecutor(folder=log_dir)
    executor.update_parameters(
        slurm_array_parallelism=3,
        slurm_partition="cbio-gpu",
        mem_gb=64,
        gpus_per_node=1,
        cpus_per_task=8,
        name="prepare",
        timeout_min=2880,
    )

    for organ in ["breast", "ovary"]:
        executor.submit(main, organ)

