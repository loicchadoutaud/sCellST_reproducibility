import gzip
import json
from pathlib import Path
import geopandas as gpd

from hest.io.seg_readers import write_geojson
from sCellST_reproducibility.submit_scripts.script_constants import data_path


def convert_hoverfast_to_hest(
    seg_dir: Path,
    output_seg_dir: Path,
    id: str,
) -> None:
    seg_path = seg_dir / f"{id}.json.gz"
    with gzip.open(seg_path, "rt", encoding="utf-8") as f:
        geojson_data = json.load(f)
    gdf = gpd.GeoDataFrame.from_features(geojson_data)
    gdf = gdf[["geometry", "object_type"]]
    write_geojson(gdf, output_seg_dir / f"{id}.geojson", category_key="object_type")


if __name__ == '__main__':
    seg_dir = Path("../HoverFast/output_segmentations")
    output_seg_dir = data_path / "hoverfast_seg"
    output_seg_dir.mkdir(parents=True, exist_ok=True)
    id = "TENX65"
    convert_hoverfast_to_hest(seg_dir, output_seg_dir, id)
