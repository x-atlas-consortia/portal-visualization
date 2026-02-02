import random
import re

from requests import get
from vitessce import (
    AnnDataWrapper,
    ImageOmeTiffWrapper,
    ObsSegmentationsOmeTiffWrapper,
    VitessceConfig,
    get_initial_coordination_scope_prefix,
)
from vitessce import CoordinationLevel as CL

from ..data_access import ImageMetadataRetriever, create_http_resource_loader
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

zarr_path = f"{SEGMENTATION_SUBDIR}/{SEGMENTATION_ZARR_STORES}"


class SegmentationMaskBuilder(ViewConfBuilder):
    """Builder for EPIC segmentation mask support datasets.

    Creates visualizations that overlay segmentation masks on base images from the parent dataset.
    Requires a parent dataset with base images.
    """

    def __init__(self, entity, groups_token, assets_endpoint, get_entity=None, parent=None, **kwargs):
        super().__init__(entity, groups_token, assets_endpoint, **kwargs)
        self._get_entity = get_entity
        self._parent_uuid = parent
        self._is_zarr_zip = False
        self._metadata_retriever = ImageMetadataRetriever(create_http_resource_loader())

    def get_conf_cells(self, **kwargs):
        """Generate Vitessce configuration for segmentation masks."""
        # Get parent entity to extract base image information
        if self._parent_uuid is None:
            raise ValueError("SegmentationMaskBuilder requires a parent dataset")

        if self._get_entity is None:
            raise ValueError("SegmentationMaskBuilder requires get_entity callback")

        parent_entity = self._get_entity(self._parent_uuid)

        # Build base image configuration from parent
        base_image_metadata = self._get_base_image_metadata(parent_entity)
        base_image_url, base_offsets_url = self._get_base_image_urls(parent_entity)

        # Build the Vitessce configuration
        vc = VitessceConfig(name="HuBMAP Data Portal", schema_version=self._schema_version)
        dataset = vc.add_dataset(name="Segmentation Masks")

        # Add base image from parent
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

        if any(".zarr.zip" in path for path in file_paths_found):
            self._is_zarr_zip = True

        found_images = list(
            get_matches(
                file_paths_found,
                IMAGE_PYRAMID_DIR + r".*\.ome\.tiff?$",
            )
        )

        # Remove the base-image pyramids from the found_images
        filtered_images = [img_path for img_path in found_images if SEGMENTATION_SUPPORT_IMAGE_SUBDIR not in img_path]

        if len(filtered_images) == 0:  # pragma: no cover
            message = f"Segmentation mask dataset with uuid {self._uuid} has no matching files"
            raise FileNotFoundError(message)

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

    def _get_base_image_metadata(self, parent_entity):
        """Extract base image metadata from parent dataset."""
        # Find base image file in parent
        parent_files = parent_entity.get("files", [])
        parent_file_paths = [file["rel_path"] for file in parent_files]

        # Look for image pyramids, excluding segmentation masks
        # Pattern allows for optional prefixes like "extras/transformations/"
        found_images = list(
            get_matches(
                parent_file_paths,
                r".*" + IMAGE_PYRAMID_DIR + r".*\.ome\.tiff?$",
            )
        )
        # Filter out segmentation mask files (contain "segmentation" in filename)
        filtered_images = [img for img in found_images if "segmentation" not in img.lower()]

        if not filtered_images:  # pragma: no cover
            raise FileNotFoundError(f"Parent dataset {self._parent_uuid} has no base images")

        # Get metadata for first image (optional - may not be available in tests)
        metadata_path = re.sub(
            r"ome\.tiff?",
            "metadata.json",
            re.sub(IMAGE_PYRAMID_DIR, IMAGE_METADATA_DIR, filtered_images[0]),
        )

        # Check if metadata file exists in parent files
        if metadata_path in parent_file_paths:
            # Build URL for parent's metadata using parent UUID
            metadata_url = f"{self._assets_endpoint}/{self._parent_uuid}/{metadata_path}"
            if self._groups_token:
                metadata_url += f"?token={self._groups_token}"

            try:
                return get_image_metadata(self, metadata_url)
            except (FileNotFoundError, Exception):  # pragma: no cover
                # Metadata not available, return empty dict
                return {}
        else:
            # Metadata file not in parent entity files
            return {}

    def _get_base_image_urls(self, parent_entity):
        """Get base image and offsets URLs from parent dataset."""
        parent_files = parent_entity.get("files", [])
        parent_file_paths = [file["rel_path"] for file in parent_files]

        # Look for image pyramids, excluding segmentation masks
        # Pattern allows for optional prefixes like "extras/transformations/"
        found_images = list(
            get_matches(
                parent_file_paths,
                r".*" + IMAGE_PYRAMID_DIR + r".*\.ome\.tiff?$",
            )
        )
        # Filter out segmentation mask files (contain "segmentation" in filename)
        filtered_images = [img for img in found_images if "segmentation" not in img.lower()]

        if not filtered_images:  # pragma: no cover
            raise FileNotFoundError(f"Parent dataset {self._parent_uuid} has no base images")

        img_path = filtered_images[0]

        # Build URLs for parent's image using parent UUID
        img_url = f"{self._assets_endpoint}/{self._parent_uuid}/{img_path}"
        if self._groups_token:
            img_url += f"?token={self._groups_token}"

        offsets_path = re.sub(
            r"ome\.tiff?",
            "offsets.json",
            re.sub(IMAGE_PYRAMID_DIR, OFFSETS_DIR, img_path),
        )
        offsets_url = f"{self._assets_endpoint}/{self._parent_uuid}/{offsets_path}"
        if self._groups_token:
            offsets_url += f"?token={self._groups_token}"

        return img_url, offsets_url

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
                print("'mask_names' key not found in the response.")
        else:
            # in this case, the code won't execute for this
            print(f"Failed to retrieve metadata.json: {response.status_code} - {response.reason}")
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
