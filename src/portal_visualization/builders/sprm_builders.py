import re
from pathlib import Path

from vitessce import (
    AnnDataWrapper,
    CoordinationType,
    ImageOmeTiffWrapper,
    ObsSegmentationsOmeTiffWrapper,
    OmeTiffWrapper,
    get_initial_coordination_scope_prefix,
)
from vitessce import (
    Component as cm,
)
from vitessce import (
    CoordinationLevel as CL,
)
from vitessce import (
    FileType as ft,
)

from ..constants import MAX_OBS_FOR_HEATMAP
from ..paths import (
    CODEX_TILE_DIR,
    IMAGE_PYRAMID_DIR,
    SPRM_JSON_DIR,
    SPRM_PYRAMID_SUBDIR,
    STITCHED_IMAGE_DIR,
    STITCHED_REGEX,
    TILE_REGEX,
)
from ..utils import (
    create_coordination_values,
    get_conf_cells,
    get_matches,
    get_ome_tiff_metadata,
    get_segmentation_alignment_scale,
)
from .base_builders import ViewConfBuilder
from .imaging_builders import ImagePyramidViewConfBuilder

# https://github.com/hubmapconsortium/portal-containers/blob/master/containers/sprm-to-anndata
# has information on how these keys are generated.
DEFAULT_SPRM_ANNDATA_FACTORS = [
    "Cell K-Means [tSNE_All_Features]",
    "Cell K-Means [Mean-All-SubRegions] Expression",
    "Cell K-Means [Mean] Expression",
    "Cell K-Means [Shape-Vectors]",
    "Cell K-Means [Texture]",
    "Cell K-Means [Total] Expression",
    "Cell K-Means [Covariance] Expression",
]

# Preferred scatterplot embedding + the matching clustering to preselect, UMAP first then t-SNE.
# Each entry is (obsm embedding path, display name, cell-set clustering name).
UMAP_EMBEDDING = ("obsm/umap", "UMAP", "Cell K-Means [UMAP_All_Features]")
TSNE_EMBEDDING = ("obsm/tsne", "t-SNE", "Cell K-Means [tSNE_All_Features]")

# Distinct colors for the first few image channels (RGB). The image controller lets users adjust.
IMAGE_CHANNEL_COLORS = [
    [255, 0, 0],
    [0, 255, 0],
    [0, 0, 255],
    [255, 255, 0],
    [255, 0, 255],
    [0, 255, 255],
]
MAX_IMAGE_CHANNELS = len(IMAGE_CHANNEL_COLORS)


class CytokitSPRMViewConfigError(Exception):
    """Raised when one of the individual SPRM view configs errors out for Cytokit"""


class SPRMViewConfBuilder(ImagePyramidViewConfBuilder):
    """Base class with shared methods for different SPRM subclasses,
    like SPRMJSONViewConfBuilder and SPRMAnnDataViewConfBuilder
    https://portal.hubmapconsortium.org/search?mapped_data_types[0]=CODEX%20%5BCytokit%20%2B%20SPRM%5D&entity_type[0]=Dataset
    """

    def _get_full_image_path(self):
        return f"{self._imaging_path_regex}/{self._image_name}" + r"\.ome\.tiff?"

    def _check_sprm_image(self, path_regex):
        """Check whether or not there is a matching SPRM image at a path.
        :param str path_regex: The path to look for the images
        :rtype: str The found image
        """
        file_paths_found = self._get_file_paths()
        found_image_files = get_matches(file_paths_found, path_regex)
        if len(found_image_files) != 1:  # pragma: no cover
            message = f'Found {len(found_image_files)} image files for SPRM uuid "{self._uuid}".'
            raise FileNotFoundError(message)
        found_image_file = found_image_files[0]
        return found_image_file

    def _get_ometiff_image_wrapper(self, found_image_file, found_image_path):
        """Create a OmeTiffWrapper object for an image, including offsets.json after calling
        _get_img_and_offset_url on the arguments to this function.
        :param str found_image_file: The path to look for the image itself
        :param str found_image_path: The folder to be replaced with the offsets path
        """
        img_url, offsets_url, _ = self._get_img_and_offset_url(
            found_image_file,
            re.escape(found_image_path),
        )
        return OmeTiffWrapper(img_url=img_url, offsets_url=offsets_url, name=self._image_name)


class SPRMJSONViewConfBuilder(SPRMViewConfBuilder):
    """Wrapper class for generating "first generation" non-stitched JSON-backed
    SPRM Vitessce configurations, like
    https://portal.hubmapconsortium.org/browse/dataset/dc31a6d06daa964299224e9c8d6cafb3
    """

    def __init__(self, entity, groups_token, assets_endpoint, **kwargs):
        # All "file" Vitessce objects that do not have wrappers.
        super().__init__(entity, groups_token, assets_endpoint, **kwargs)
        # These are both something like R001_X009_Y009 because
        # there is no mask used here or shared name with the mask data.
        self._base_name = kwargs["base_name"]
        self._image_name = kwargs["base_name"]
        self._imaging_path_regex = kwargs["imaging_path"]
        self._files = [
            {
                "rel_path": f"{SPRM_JSON_DIR}/" + f"{self._base_name}.cells.json",
                "file_type": ft.CELLS_JSON,
                "coordination_values": create_coordination_values(),
            },
            {
                "rel_path": f"{SPRM_JSON_DIR}/" + f"{self._base_name}.cell-sets.json",
                "file_type": ft.CELL_SETS_JSON,
                "coordination_values": create_coordination_values(),
            },
            {
                "rel_path": f"{SPRM_JSON_DIR}/" + f"{self._base_name}.clusters.json",
                "file_type": "clusters.json",
                "coordination_values": create_coordination_values(),
            },
        ]

    def get_conf_cells(self, **kwargs):
        found_image_file = self._check_sprm_image(self._get_full_image_path())
        vc, dataset = self._create_vitessce_config(name=self._base_name, dataset_name="SPRM")
        image_wrapper = self._get_ometiff_image_wrapper(found_image_file, self._imaging_path_regex)
        dataset = dataset.add_object(image_wrapper)
        file_paths_found = self._get_file_paths()
        if self._files[0]["rel_path"] not in file_paths_found:
            # This tile has no segmentations,
            # so only show Spatial component without cells sets, genes etc.
            vc = self._setup_view_config(vc, dataset, self.view_type, disable_3d=[self._image_name])
        else:
            # This tile has segmentations so show the analysis results.
            for file in self._files:
                path = file["rel_path"]
                try:
                    self._require_file(path, f"SPRM file {path}")
                except FileNotFoundError:
                    raise
                dataset_file = self._replace_url_in_file(file)
                dataset = dataset.add_file(**(dataset_file))
            vc = self._setup_view_config_raster_cellsets_expression_segmentation(vc, dataset)
        return get_conf_cells(vc)

    def _setup_view_config_raster_cellsets_expression_segmentation(self, vc, dataset):
        vc.add_view(cm.SPATIAL, dataset=dataset, x=3, y=0, w=7, h=8)
        vc.add_view(cm.DESCRIPTION, dataset=dataset, x=0, y=8, w=3, h=4)
        vc.add_view(cm.LAYER_CONTROLLER, dataset=dataset, x=0, y=0, w=3, h=8).set_props(disable3d=[self._image_name])
        vc.add_view(cm.OBS_SETS, dataset=dataset, x=10, y=5, w=2, h=7)
        vc.add_view(cm.FEATURE_LIST, dataset=dataset, x=10, y=0, w=2, h=5).set_props(variablesLabelOverride="antigen")
        vc.add_view(cm.HEATMAP, dataset=dataset, x=3, y=8, w=7, h=4).set_props(
            transpose=True, variablesLabelOverride="antigen"
        )
        return vc


class SPRMAnnDataViewConfBuilder(SPRMViewConfBuilder):
    """Wrapper class for generating "second generation"
    stitched AnnData-backed SPRM Vitessce configurations,
    like the dataset derived from
    https://portal.hubmapconsortium.org/browse/dataset/1c33472c68c4fb40f531b39bf6310f2d

    :param \\*\\*kwargs: { imaging_path: str, mask_path: str } for the paths
    of the image and mask relative to image_pyramid_regex
    """

    def __init__(self, entity, groups_token, assets_endpoint, **kwargs):
        super().__init__(entity, groups_token, assets_endpoint, **kwargs)
        self._base_name = kwargs["base_name"]
        self._mask_name = kwargs["mask_name"]
        self._image_name = kwargs["image_name"]
        self._imaging_path_regex = f"{self.image_pyramid_regex}/{kwargs['imaging_path']}"
        self._mask_path_regex = f"{self.image_pyramid_regex}/{kwargs['mask_path']}"

    def zarr_store(self):
        zarr_path = f"anndata-zarr/{self._image_name}-anndata.zarr"
        if self._is_zarr_zip:
            zarr_path = f"{zarr_path}.zip"
        return self._zarr_accessor.open_store(is_zip=self._is_zarr_zip, zarr_path=zarr_path)

    def _get_bitmask_image_path(self):
        return f"{self._mask_path_regex}/{self._mask_name}" + r"\.ome\.tiff?"

    @staticmethod
    def _embedding_preference(z):
        """(embedding path, display name, clustering name) preferring UMAP, falling back to t-SNE."""
        if z is not None and UMAP_EMBEDDING[0] in z:
            return UMAP_EMBEDDING
        return TSNE_EMBEDDING

    def _get_n_obs(self, z):
        """Number of cells in the SPRM AnnData store (for the heatmap size gate)."""
        if z is None:
            return 0
        if "obs" in z and "_index" in z["obs"]:
            return z["obs"]["_index"].shape[0]
        if "obsm" in z and "xy" in z["obsm"]:
            return z["obsm"]["xy"].shape[0]
        return 0  # pragma: no cover

    def get_conf_cells(self, marker=None):
        vc, dataset = self._create_vitessce_config(name=self._image_name, dataset_name="SPRM")
        file_paths_found = self._get_file_paths()
        zarr_path = f"anndata-zarr/{self._image_name}-anndata.zarr"
        # Use the group as a proxy for presence of the rest of the zarr store.
        if f"{zarr_path}.zip" in file_paths_found:  # pragma: no cover
            self._is_zarr_zip = True
            zarr_path = f"{zarr_path}.zip"
        else:  # pragma: no cover
            self._require_zarr_store(zarr_path)
        adata_url = self._build_assets_url(zarr_path, use_token=False)

        z = self.zarr_store()
        # zarr v3 iterates an array into 0-d ndarrays; read values explicitly as a list of strings.
        additional_cluster_names = z["uns/cluster_columns"][:].tolist() if "uns/cluster_columns" in z else []
        obs_set_names = sorted(set(additional_cluster_names + DEFAULT_SPRM_ANNDATA_FACTORS))
        obs_set_paths = [f"obs/{key}" for key in obs_set_names]
        n_obs = self._get_n_obs(z)
        embedding_path, embedding_name, prioritized_cell_set = self._embedding_preference(z)

        anndata_wrapper = AnnDataWrapper(
            adata_url=adata_url,
            is_zip=self._is_zarr_zip,
            obs_feature_matrix_path="X",
            obs_embedding_paths=[embedding_path],
            obs_embedding_names=[embedding_name],
            obs_set_names=obs_set_names,
            obs_set_paths=obs_set_paths,
            # Cells are shown via the (image-aligned) segmentation mask. obsm/xy centroids live in a
            # different coordinate space than the registered image and render misaligned, so omit them.
            coordination_values={"obsType": "cell"},
            request_init=self._get_request_init(),
        )
        dataset = dataset.add_object(anndata_wrapper)

        # Beta spatial/layerController views read the new image-coordination model, so the
        # expression image and the segmentation bitmask are separate layers (not MultiImageWrapper).
        found_image_file = self._check_sprm_image(self._get_full_image_path())
        img_url, offsets_url, _ = self._get_img_and_offset_url(found_image_file, self.image_pyramid_regex)
        image_metadata = get_ome_tiff_metadata(img_url)
        dataset = dataset.add_object(
            ImageOmeTiffWrapper(
                img_url=img_url,
                offsets_url=offsets_url,
                coordination_values={"fileUid": "image"},
            )
        )
        found_bitmask_file = self._check_sprm_image(self._get_bitmask_image_path())
        bitmask_url, bitmask_offsets_url, _ = self._get_img_and_offset_url(found_bitmask_file, self.image_pyramid_regex)
        # The expression image and mask are often stored at different physical pixel sizes (and units);
        # scale the segmentation into the image's coordinate space so the cells overlay it (the legacy
        # raster config aligned them implicitly). Handles unit conversion and a unitless/absent mask
        # physical size, degrading to no scaling only when neither OME-TIFF exposes physical sizes.
        segmentation_scale = get_segmentation_alignment_scale(image_metadata, get_ome_tiff_metadata(bitmask_url))
        dataset = dataset.add_object(
            ObsSegmentationsOmeTiffWrapper(
                img_url=bitmask_url,
                offsets_url=bitmask_offsets_url,
                coordinate_transformations=[{"type": "scale", "scale": segmentation_scale}],
                coordination_values={"fileUid": "segmentation-mask"},
            )
        )

        num_image_channels = min(MAX_IMAGE_CHANNELS, (image_metadata or {}).get("SizeC") or 1)
        vc = self._setup_view_config_raster_cellsets_expression_segmentation(
            vc,
            dataset,
            marker,
            n_obs=n_obs,
            obs_set_names=obs_set_names,
            num_image_channels=num_image_channels,
            embedding_name=embedding_name,
            prioritized_cell_set=prioritized_cell_set,
        )
        return get_conf_cells(vc)

    def _setup_view_config_raster_cellsets_expression_segmentation(
        self,
        vc,
        dataset,
        marker,
        n_obs=0,
        obs_set_names=(),
        num_image_channels=1,
        embedding_name="t-SNE",
        prioritized_cell_set=None,
    ):
        # Hide the heatmap for very large datasets (same gate as the AnnData builders) and let the
        # spatial/scatterplot views grow into the freed vertical space.
        include_heatmap = not self._minimal and n_obs <= MAX_OBS_FOR_HEATMAP
        views_h = 8 if include_heatmap else 12

        description = vc.add_view(cm.DESCRIPTION, dataset=dataset, x=0, y=8, w=3, h=4)
        layer_controller = vc.add_view("layerControllerBeta", dataset=dataset, x=0, y=0, w=3, h=8)
        spatial = vc.add_view("spatialBeta", dataset=dataset, x=3, y=0, w=4, h=views_h)
        scatterplot = vc.add_view(cm.SCATTERPLOT, dataset=dataset, mapping=embedding_name, x=7, y=0, w=3, h=views_h)
        cell_sets = vc.add_view(cm.OBS_SETS, dataset=dataset, x=10, y=5, w=2, h=7)
        gene_list = vc.add_view(cm.FEATURE_LIST, dataset=dataset, x=10, y=0, w=2, h=5).set_props(
            variablesLabelOverride="antigen"
        )

        views = [description, layer_controller, spatial, scatterplot, cell_sets, gene_list]
        if include_heatmap:
            heatmap = vc.add_view(cm.HEATMAP, dataset=dataset, x=3, y=8, w=7, h=4).set_props(
                variablesLabelOverride="antigen", transpose=True
            )
            views.append(heatmap)

        vc.link_views(views, [CoordinationType.OBS_TYPE], ["cell"])

        if marker:
            vc.link_views(
                views,
                [CoordinationType.FEATURE_SELECTION, CoordinationType.OBS_COLOR_ENCODING],
                [[marker], "geneSelection"],
            )
        else:
            # Color cells (scatterplot + segmentation) by the selected cell set.
            [obs_color_encoding] = vc.add_coordination(CoordinationType.OBS_COLOR_ENCODING)
            obs_color_encoding.set_value("cellSetSelection")
            for view in (spatial, scatterplot, cell_sets):
                view.use_coordination(obs_color_encoding)
            # Preselect/color by the embedding's matching clustering (UMAP if present, else t-SNE).
            if prioritized_cell_set and prioritized_cell_set in obs_set_names:
                [obs_set_selection] = vc.add_coordination(CoordinationType.OBS_SET_SELECTION)
                obs_set_selection.set_value([[prioritized_cell_set]])
                for view in (spatial, scatterplot, cell_sets):
                    view.use_coordination(obs_set_selection)

        # Wire the image and segmentation layers into the beta views. The beta spatial model does not
        # auto-discover channels, so each channel is listed explicitly (an empty/partial channel list
        # renders a null channel and crashes the view). Show the first few channels in distinct colors;
        # the layer controller lets users toggle the rest.
        image_channels = [
            {
                "spatialTargetC": channel,
                "spatialChannelColor": IMAGE_CHANNEL_COLORS[channel],
                "spatialChannelVisible": True,
                "spatialChannelOpacity": 1.0,
            }
            for channel in range(num_image_channels)
        ]
        vc.link_views_by_dict(
            [spatial, layer_controller],
            {
                "spatialTargetZ": 0,
                "spatialTargetT": 0,
                "imageLayer": CL(
                    [
                        {
                            "fileUid": "image",
                            "spatialLayerVisible": True,
                            "spatialLayerOpacity": 1.0,
                            "photometricInterpretation": "BlackIsZero",
                            "imageChannel": CL(image_channels),
                        }
                    ]
                ),
            },
            meta=True,
            scope_prefix=get_initial_coordination_scope_prefix(self._uuid, "image"),
        )
        vc.link_views_by_dict(
            [spatial, layer_controller],
            {
                "segmentationLayer": CL(
                    [
                        {
                            "fileUid": "segmentation-mask",
                            "spatialLayerVisible": True,
                            "spatialLayerOpacity": 1.0,
                            "segmentationChannel": CL(
                                [
                                    {
                                        # ponytail: channel 0 assumed to be the cell mask; obsType "cell"
                                        # joins it to the AnnData cells so it colors by the selected cell set.
                                        "spatialTargetC": 0,
                                        "obsType": "cell",
                                        "spatialChannelOpacity": 1.0,
                                        "spatialChannelVisible": True,
                                        "obsColorEncoding": "cellSetSelection",
                                        "spatialSegmentationFilled": True,
                                    }
                                ]
                            ),
                        }
                    ]
                )
            },
            meta=True,
            scope_prefix=get_initial_coordination_scope_prefix(self._uuid, "obsSegmentations"),
        )

        return vc


class MultiImageSPRMAnndataViewConfigError(Exception):
    """Raised when one of the individual SPRM view configs errors out"""


class MultiImageSPRMAnndataViewConfBuilder(ViewConfBuilder):
    """Wrapper class for generating multiple "second generation" AnnData-backed SPRM
    Vitessce configurations via SPRMAnnDataViewConfBuilder,
    used for datasets with multiple regions.
    """

    def __init__(self, entity, groups_token, assets_endpoint, **kwargs):
        super().__init__(entity, groups_token, assets_endpoint, **kwargs)
        self._expression_id = "expr"
        self._mask_id = "mask"
        self._image_pyramid_subdir = SPRM_PYRAMID_SUBDIR
        self._mask_pyramid_subdir = SPRM_PYRAMID_SUBDIR.replace(self._expression_id, self._mask_id)

    def _find_ids(self):
        """Search the image pyramid directory for all of the names of OME-TIFF files
        to use as unique identifiers.
        """
        file_paths_found = [file["rel_path"] for file in self._entity["files"]]
        full_pyramid_path = IMAGE_PYRAMID_DIR + "/" + self._image_pyramid_subdir
        pyramid_files = [file for file in file_paths_found if full_pyramid_path in file]
        found_ids = [
            Path(image_path)
            .name.replace(".ome.tiff", "")
            .replace(".ome.tif", "")
            .replace("_" + self._expression_id, "")
            for image_path in pyramid_files
        ]
        if len(found_ids) == 0:
            raise FileNotFoundError(f"Could not find images of the SPRM analysis with uuid {self._uuid}")
        return found_ids

    def get_conf_cells(self, marker=None):
        found_ids = self._find_ids()
        confs = []
        for id in sorted(found_ids):
            builder = SPRMAnnDataViewConfBuilder(
                entity=self._entity,
                groups_token=self._groups_token,
                assets_endpoint=self._assets_endpoint,
                base_name=id,
                imaging_path=self._image_pyramid_subdir,
                mask_path=self._mask_pyramid_subdir,
                image_name=f"{id}_{self._expression_id}",
                mask_name=f"{id}_{self._mask_id}",
            )
            conf = builder.get_conf_cells(marker=marker).conf
            if conf == {}:
                raise MultiImageSPRMAnndataViewConfigError(  # pragma: no cover
                    f"Cytokit SPRM assay with uuid {self._uuid} has empty view\
                        config for id '{id}'"
                )
            confs.append(conf)
        conf = confs if len(confs) > 1 else confs[0]
        return get_conf_cells(conf)


class StitchedCytokitSPRMViewConfBuilder(MultiImageSPRMAnndataViewConfBuilder):
    """Wrapper class for generating multiple "second generation" stitched AnnData-backed SPRM
    Vitessce configurations via SPRMAnnDataViewConfBuilder,
    used for datasets with multiple regions.
    These are from post-August 2020 Cytokit datasets (stitched).
    """

    # Need to override base class settings due to different directory structure
    def __init__(self, entity, groups_token, assets_endpoint, **kwargs):
        super().__init__(entity, groups_token, assets_endpoint, **kwargs)
        self._image_pyramid_subdir = STITCHED_IMAGE_DIR
        # The ids don't match exactly with the replacement because all image files have
        # stitched_expressions appended while the subdirectory only has /stitched/
        self._expression_id = "stitched_expressions"
        self._mask_pyramid_subdir = STITCHED_IMAGE_DIR.replace("expressions", "mask")
        self._mask_id = "stitched_mask"


class TiledSPRMViewConfBuilder(ViewConfBuilder):
    """Wrapper class for generating many "first generation"
    non-stitched JSON-backed SPRM Vitessce configurations,
    one per tile per region, via SPRMJSONViewConfBuilder.
    """

    def get_conf_cells(self, **kwargs):
        file_paths_found = [file["rel_path"] for file in self._entity["files"]]
        found_tiles = get_matches(file_paths_found, TILE_REGEX) or get_matches(file_paths_found, STITCHED_REGEX)
        if len(found_tiles) == 0:  # pragma: no cover
            message = f"Cytokit SPRM assay with uuid {self._uuid} has no matching tiles"
            raise FileNotFoundError(message)
        confs = []
        for tile in sorted(found_tiles):
            builder = SPRMJSONViewConfBuilder(
                entity=self._entity,
                groups_token=self._groups_token,
                assets_endpoint=self._assets_endpoint,
                base_name=tile,
                imaging_path=CODEX_TILE_DIR,
            )
            conf = builder.get_conf_cells().conf
            if conf == {}:  # pragma: no cover
                message = f"Cytokit SPRM assay with uuid {self._uuid} has empty view config"
                raise CytokitSPRMViewConfigError(message)
            confs.append(conf)
        return get_conf_cells(confs)
