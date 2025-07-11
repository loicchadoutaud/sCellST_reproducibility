from pathlib import Path


# HEST ids
visium_slides = {
    "Kidney": ["INT13", "INT14", "INT15", "INT17", "INT18", "INT19", "INT21", "INT24"],
    "Prostate": ["INT25", "INT26", "INT27", "INT28", "INT35"],
}
not_good_visium_slides = ["INT16", "INT20", "INT22", "INT23"]
all_visium_slides = [s for organ in visium_slides.keys() for s in visium_slides[organ]] + not_good_visium_slides
xenium_slides = {
    "Breast": ["NCBI785", "TENX95"],
}
xenium_slides_review = [
    "NCBI783",
    "NCBI784",
    "NCBI785",
    "TENX94",
    "TENX95",
    "TENX96",
    "TENX97",
    "TENX98",
    "TENX99",
]
test_slide =  ["NCBI784"]

# Other info
benchmark_organs = ["Kidney", "Prostate"]
benchmark_genes = {
    organ: [
        f"{organ}_50_hvg_bench",
        f"{organ}_500_hvg_bench",
        f"{organ}_50_svg_bench",
        f"{organ}_500_svg_bench",
    ]
    for organ in benchmark_organs
}
number_of_genes_exp = {
    organ: [
        f"{organ}_50_hvg_bench",
        f"{organ}_200_hvg_bench",
        f"{organ}_500_hvg_bench",
        f"{organ}_1000_hvg_bench",
        f"{organ}_2000_hvg_bench",
    ]
    for organ in benchmark_organs
}
normalisation_types = ["train", "self"]

# Data paths
data_path = Path("/cluster/CBIO/data1/lchadoutaud1/data/hest_data")
config_dir = Path("/cluster/CBIO/home/lchadoutaud1/code/CellST/config")

# Job settings
log_dir = "logs"  # Directory for logs
log_dir_path = Path(log_dir)
log_dir_path.mkdir(parents=True, exist_ok=True)
