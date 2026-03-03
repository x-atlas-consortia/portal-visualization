import logging
import random
import re

from requests import get
from vitessce import (
    AnnDataWrapper,
    ImageOmeTiffWrapper,
    ObsSegmentationsOmeTiffWrapper,
    get_initial_coordination_scope_prefix,
)
from vitessce import CoordinationLevel as CL

from ..paths import (
    IMAGE_METADATA_DIR,
    IMAGE_PYRAMID_DIR,
    OFFSETS_DIR,
    SEGMENTATION_SUBDIR,
    SEGMENTATION_SUPPORT_IMAGE_SUBDIR,
    SEGMENTATION_ZARR_STORES,
)
from ..utils import get_conf_cells, get_image_metadata, get_image_scale, get_matches
from .base_builders import ViewConfBuilder

logger = logging.getLogger(__name__)

zarr_path = f"{SEGMENTATION_SUBDIR}/{SEGMENTATION_ZARR_STORES}"


class SegmentationMaskBuilder(ViewConfBuilder):
    """Builder for EPIC segmentation mask datasets.

    Creates visualizations that overlay segmentation masks on base images.
    Both base images and segmentation masks are co-located in the entity's own files.
    """

    def get_conf_cells(self, **kwargs):
        """Generate Vitessce configuration for segmentation masks."""
        # Build base image configuration from entity's own files
        base_image_metadata = self._get_base_image_metadata()
        base_image_url, base_offsets_url = self._get_base_image_urls()

        # Build the Vitessce configuration
        vc, dataset = self._create_vitessce_config(dataset_name="Segmentation Masks")

        # Add base image
        dataset = dataset.add_object(
            ImageOmeTiffWrapper(
                img_url=base_image_url,
                offsets_url=base_offsets_url,
                name="Base Image",
                coordination_values={"fileUid": "base-image"},
            )
        )

        # Add segmentation mask overlays
        zarr_url = self.zarr_store_url()
        file_paths_found = [file["rel_path"] for file in self._entity["files"]]

        self._detect_zarr_format()

        found_images = list(
            get_matches(
                file_paths_found,
                IMAGE_PYRAMID_DIR + r".*\.ome\.tiff?$",
            )
        )

        # Remove the base-image pyramids from the found_images
        filtered_images = [img_path for img_path in found_images if SEGMENTATION_SUPPORT_IMAGE_SUBDIR not in img_path]

        if len(filtered_images) == 0:  # pragma: no cover
            raise FileNotFoundError(f"Dataset {self._uuid} is missing segmentation mask pyramid files")

        if len(filtered_images) >= 1:
            img_url, offsets_url, metadata_url = self.segmentations_ome_offset_url(filtered_images[0])

        segmentation_metadata = get_image_metadata(self, metadata_url)
        segmentation_scale = get_image_scale(base_image_metadata, segmentation_metadata)

        segmentations = ObsSegmentationsOmeTiffWrapper(
            img_url=img_url,
            offsets_url=offsets_url,
            coordinate_transformations=[{"type": "scale", "scale": segmentation_scale}],
            obs_types_from_channel_names=True,
            coordination_values={"fileUid": "segmentation-mask"},
        )

        dataset = dataset.add_object(segmentations)

        # Add Zarr-based segmentation objects if available
        mask_names = self.read_metadata_from_url()
        if mask_names:  # pragma: no cover
            segmentation_objects, segmentations_CL = create_segmentation_objects(self, zarr_url, mask_names)
            for obj in segmentation_objects:
                dataset.add_object(obj)

            # Configure views and coordination
            spatial_view = vc.add_view("spatialBeta", dataset=dataset)
            lc_view = vc.add_view("layerControllerBeta", dataset=dataset)

            vc.link_views_by_dict(
                [spatial_view, lc_view],
                {
                    # Neutralizing the base-image colors
                    "imageLayer": CL(
                        [
                            {
                                "photometricInterpretation": "RGB",
                            }
                        ]
                    ),
                    "segmentationLayer": CL(
                        [
                            {
                                "fileUid": "segmentation-mask",
                                "spatialLayerVisible": True,
                                "spatialLayerOpacity": 1,
                                "segmentationChannel": CL(segmentations_CL),
                            }
                        ]
                    ),
                },
                meta=True,
                scope_prefix=get_initial_coordination_scope_prefix("A", "obsSegmentations"),
            )

            vc.layout(spatial_view | lc_view)
        else:
            # No Zarr segmentations, simpler layout
            spatial_view = vc.add_view("spatialBeta", dataset=dataset)
            lc_view = vc.add_view("layerControllerBeta", dataset=dataset)
            vc.layout(spatial_view | lc_view)

        return get_conf_cells(vc)

    def _get_base_image_metadata(self):
        """Extract base image metadata from the entity's own files."""
        filtered_images = self._find_base_images()

        # Get metadata for first image (optional - may not be available in tests)
        metadata_path = re.sub(
            r"ome\.tiff?",
            "metadata.json",
            re.sub(IMAGE_PYRAMID_DIR, IMAGE_METADATA_DIR, filtered_images[0]),
        )

        # Check if metadata file exists in entity files
        file_paths = self._get_file_paths()
        if metadata_path in file_paths:
            metadata_url = self._build_assets_url(metadata_path)
            try:
                return get_image_metadata(self, metadata_url)
            except (FileNotFoundError, Exception):  # pragma: no cover
                # Metadata not available, return empty dict
                return {}
        else:
            # Metadata file not in entity files
            return {}

    def _get_base_image_urls(self):
        """Get base image and offsets URLs from the entity's own files."""
        filtered_images = self._find_base_images()
        img_path = filtered_images[0]

        img_url = self._build_assets_url(img_path)

        offsets_path = re.sub(
            r"ome\.tiff?",
            "offsets.json",
            re.sub(IMAGE_PYRAMID_DIR, OFFSETS_DIR, img_path),
        )
        offsets_url = self._build_assets_url(offsets_path)

        return img_url, offsets_url

    def _find_base_images(self):
        """Find base image pyramid files in the entity, excluding segmentation masks."""
        file_paths = self._get_file_paths()
        found_images = list(
            get_matches(
                file_paths,
                r".*" + IMAGE_PYRAMID_DIR + r".*\.ome\.tiff?$",
            )
        )
        # Filter out segmentation mask files (contain "segmentation" in filename)
        filtered_images = [img for img in found_images if "segmentation" not in img.lower()]

        if not filtered_images:  # pragma: no cover
            raise FileNotFoundError(f"Dataset {self._uuid} is missing base image pyramid files")

        return filtered_images

    def zarr_store_url(self):
        adata_url = self._build_assets_url(zarr_path, use_token=False)
        return adata_url

    def segmentations_ome_offset_url(self, img_path):
        img_url = self._build_assets_url(f"{SEGMENTATION_SUBDIR}/{img_path}")
        return (
            img_url,
            str(
                re.sub(
                    r"ome\.tiff?",
                    "offsets.json",
                    re.sub(IMAGE_PYRAMID_DIR, OFFSETS_DIR, img_url),
                )
            ),
            str(
                re.sub(
                    r"ome\.tiff?",
                    "metadata.json",
                    re.sub(IMAGE_PYRAMID_DIR, IMAGE_METADATA_DIR, img_url),
                )
            ),
        )

    def read_metadata_from_url(self):  # pragma: no cover
        mask_names = []
        url = f"{self.zarr_store_url()}/metadata.json"
        request_init = self._get_request_init() or {}
        response = get(url, **request_init)
        if response.status_code == 200:
            data = response.json()
            if isinstance(data, dict) and "mask_names" in data:
                mask_names = data["mask_names"]
            else:
                logger.warning("'mask_names' key not found in the response.")
        else:
            # in this case, the code won't execute for this
            logger.warning("Failed to retrieve metadata.json: %s - %s", response.status_code, response.reason)
        return mask_names


def create_segmentation_objects(self, base_url, mask_names):  # pragma: no cover
    segmentation_objects = []
    segmentations_CL = []
    for mask_name in mask_names:
        color_channel = generate_unique_color()
        mask_url = f"{base_url}/{mask_name}.zarr"
        if self._is_zarr_zip:
            mask_url = f"{mask_url}.zip"
        segmentations_zarr = AnnDataWrapper(
            adata_url=mask_url,
            is_zip=self._is_zarr_zip,
            obs_locations_path="obsm/X_spatial",
            obs_labels_names=mask_name,
            coordination_values={"obsType": mask_name},
        )
        seg_CL = {
            "spatialTargetC": mask_name,
            "obsType": mask_name,
            "spatialChannelOpacity": 1,
            "spatialChannelColor": color_channel,
            "obsHighlight": None,
        }
        segmentation_objects.append(segmentations_zarr)
        segmentations_CL.append(seg_CL)
    return segmentation_objects, segmentations_CL


def generate_unique_color():  # pragma: no cover
    return [random.randint(0, 255) for _ in range(3)]
