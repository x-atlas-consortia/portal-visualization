from vitessce import (
    Component as cm,
)
from vitessce import (
    FileType as ft,
)

from ..paths import SCATAC_SEQ_DIR, SCRNA_SEQ_DIR
from ..utils import create_coordination_values, get_conf_cells
from .base_builders import ViewConfBuilder


class AbstractScatterplotViewConfBuilder(ViewConfBuilder):
    """Base class for subclasses creating a JSON-backed scatterplot for
    "first generation" RNA-seq and ATAC-seq data like
    https://portal.hubmapconsortium.org/browse/dataset/d4493657cde29702c5ed73932da5317c
    from h5ad-to-arrow.cwl.
    """

    def get_conf_cells(self, **kwargs):
        file_paths_expected = [file["rel_path"] for file in self._files]
        # We need to check that the files we expect actually exist.
        # This is due to the volatility of the datasets.
        try:
            self._require_files(file_paths_expected, f"scatterplot files: {file_paths_expected}")
        except FileNotFoundError:
            raise
        vc, dataset = self._create_vitessce_config(dataset_name="Visualization Files")
        # The sublcass initializes _files in its __init__ method
        for file in self._files:
            dataset = dataset.add_file(**(self._replace_url_in_file(file)))
        vc = self._setup_scatterplot_view_config(vc, dataset)
        return get_conf_cells(vc)

    def _setup_scatterplot_view_config(self, vc, dataset):
        vc.add_view(cm.SCATTERPLOT, dataset=dataset, mapping="UMAP", x=0, y=0, w=9, h=12)
        vc.add_view(cm.OBS_SETS, dataset=dataset, x=9, y=0, w=3, h=12)
        return vc


class RNASeqViewConfBuilder(AbstractScatterplotViewConfBuilder):
    """Wrapper class for creating a JSON-backed scatterplot for "first generation" RNA-seq data
    like https://portal.hubmapconsortium.org/browse/dataset/c019a1cd35aab4d2b4a6ff221e92aaab
    from h5ad-to-arrow.cwl (August 2020 release).
    """

    def __init__(self, entity, groups_token, assets_endpoint, **kwargs):
        super().__init__(entity, groups_token, assets_endpoint, **kwargs)
        # All "file" Vitessce objects that do not have wrappers.
        self._files = [
            {
                "rel_path": f"{SCRNA_SEQ_DIR}.cells.json",
                "file_type": ft.OBS_SEGMENTATIONS_CELLS_JSON,
                "coordination_values": create_coordination_values(),
            },
            {
                "rel_path": f"{SCRNA_SEQ_DIR}.cells.json",
                "file_type": ft.OBS_LOCATIONS_CELLS_JSON,
                "coordination_values": create_coordination_values(),
            },
            {
                "rel_path": f"{SCRNA_SEQ_DIR}.cells.json",
                "file_type": ft.OBS_EMBEDDING_CELLS_JSON,
                "coordination_values": create_coordination_values(embeddingType="UMAP"),
            },
            {
                "rel_path": f"{SCRNA_SEQ_DIR}.cell-sets.json",
                "file_type": ft.OBS_SETS_CELL_SETS_JSON,
                "coordination_values": create_coordination_values(),
            },
        ]


class ATACSeqViewConfBuilder(AbstractScatterplotViewConfBuilder):
    """Wrapper class for creating a JSON-backed scatterplot for "first generation" ATAC-seq data
    like https://portal.hubmapconsortium.org/browse/dataset/d4493657cde29702c5ed73932da5317c
    from h5ad-to-arrow.cwl.
    """

    def __init__(self, entity, groups_token, assets_endpoint, **kwargs):
        super().__init__(entity, groups_token, assets_endpoint, **kwargs)
        # All "file" Vitessce objects that do not have wrappers.

        self._files = [
            {
                "rel_path": SCATAC_SEQ_DIR + "/umap_coords_clusters.cells.json",
                "file_type": ft.OBS_SEGMENTATIONS_CELLS_JSON,
                "coordination_values": create_coordination_values(),
            },
            {
                "rel_path": SCATAC_SEQ_DIR + "/umap_coords_clusters.cells.json",
                "file_type": ft.OBS_LOCATIONS_CELLS_JSON,
                "coordination_values": create_coordination_values(),
            },
            {
                "rel_path": SCATAC_SEQ_DIR + "/umap_coords_clusters.cells.json",
                "file_type": ft.OBS_EMBEDDING_CELLS_JSON,
                "coordination_values": create_coordination_values(embeddingType="UMAP"),
            },
            {
                "rel_path": SCATAC_SEQ_DIR + "/umap_coords_clusters.cell-sets.json",
                "file_type": ft.OBS_SETS_CELL_SETS_JSON,
                "coordination_values": create_coordination_values(),
            },
        ]
