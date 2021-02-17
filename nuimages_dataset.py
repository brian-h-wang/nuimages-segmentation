"""
nuimages_dataset.py
Brian Wang
bhw45@cornell.edu

Implements a NuImagesDataset class for use with a PyTorch data loader.

Adapted from the example at:
https://pytorch.org/tutorials/intermediate/torchvision_tutorial.html

References:
    nuImages schema - describes format of sample_data, object_ann, etc.
    https://github.com/nutonomy/nuscenes-devkit/blob/master/docs/schema_nuimages.md

    torchvision tutorial:
    https://pytorch.org/tutorials/intermediate/torchvision_tutorial.html

"""
from pathlib import Path
import numpy as np
import torch
from PIL import Image
from nuimages import NuImages
from nuimages.utils.utils import mask_decode


class NuImagesDataset(object):
    """
    Class for a NuImages dataset.

    Notes:
        - By default, the dataset removes any samples (images) which contain no annotations. These
        cause the training to crash if included. This behavior can be changed by setting the
        'remove_empty' argument when making a NuImagesDataset.
        - Currently, the dataset class skips surface annotations (drivable surfaces and ego vehicle).
        Only object annotations are included.
        - Any bounding boxes with zero height or width are removed; these cause training to crash
        if left in. The NuImages training set seems to contain one annotation with zero height.

    """

    def __init__(self, nuimages: NuImages, transforms=None, remove_empty=True):
        # If there are no annotations, this dataset holds the nuImages test set
        if len(nuimages.object_ann) == 0:
            self.has_ann = False
        # Otherwise, dataset is the train or val split
        else:
            self.has_ann = True

        assert type(nuimages) == NuImages
        self.nuimages = nuimages
        self.root_path = Path(nuimages.dataroot)
        self.transforms = transforms

        # If training, remove any samples which contain no annotations
        if remove_empty and self.has_ann:
            print("[INFO] Removing samples which contain no annotations...")
            sd_tokens_with_objects = set()
            for o in self.nuimages.object_ann:
                sd_tokens_with_objects.add(o['sample_data_token'])
            self.samples_with_objects = []
            for i, sample in enumerate(self.nuimages.sample):
                sd_token = sample['key_camera_token']
                if sd_token in sd_tokens_with_objects:
                    self.samples_with_objects.append(i)
            print("[INFO] Done. %d samples remaining out of %d." % (len(self.samples_with_objects), len(self.nuimages.sample)))
        else:
            # Keep all samples if remove_empty set to false
            self.samples_with_objects = [i for i in range(len(self.nuimages.sample))]

        # Create lookup table to convert category name to an int index
        # Speeds up creating annotations
        self._category_name_to_id = {}
        self.category_names = ['background']
        for i, category in enumerate(nuimages.category):
            # Start category IDs at 1. For torchvision compatibility, ID 0 *must* be background
            self._category_name_to_id[category['name']] = i+1
            self.category_names.append(category['name'])

        # Create lookup table that maps sample data to object annotations
        self.object_anns_dict = {}
        for o in self.nuimages.object_ann:
            object_sd_token = o['sample_data_token']
            if object_sd_token not in self.object_anns_dict.keys():
                self.object_anns_dict[object_sd_token] = []
            # Remove annotation if bounding box has zero height or width
            # NuImages training set contains one of these annotations,
            # breaks training if it isn't removed
            b = o['bbox']
            w = b[2] - b[0]
            h = b[3] - b[1]
            if w > 0 and h > 0:
                self.object_anns_dict[object_sd_token].append(o)

    def __len__(self):
        return len(self.samples_with_objects)

    def __getitem__(self, idx):
        """
        Get an item from the dataset. Returns an image tensor, and a target dict.

        See https://pytorch.org/tutorials/intermediate/torchvision_tutorial.html for the formatting
        of the target dict.

        Parameters
        ----------
        idx: int
            Index of the sample in the dataset.

        Returns
        -------
        torch.Tensor, dict
        """
        # Get a sample - i.e. an annotated camera image
        sample = self.nuimages.sample[self.samples_with_objects[idx]]
        # Get the associated sample data, representing the image associated with the sample
        sd_token = sample['key_camera_token']
        sample_data = self.nuimages.get('sample_data', sd_token)

        # Read the image file
        image = Image.open(self.root_path / sample_data['filename']).convert("RGB")

        # If this is the test split (no annotations), just return the image and None for target
        if not self.has_ann:
            return image, None

        # Get the surface and object annotations corresponding to this sample data only
        # object_anns = [o for o in self.nuimages.object_ann if o['sample_data_token'] == sd_token]
        object_anns = self.object_anns_dict[sd_token]
        # TODO surface annotations in nuscenes lack bounding boxes. Skip these for now.
        # Treat surfaces as objects, if we are learning surfaces too
        # if self.learn_surfaces:
        #     surface_anns = [o for o in self.nuimages.surface_ann if o['sample_data_token'] == sd_token]
        #     object_anns += surface_anns

        # Get bounding boxes
        # Note object_ann['bbox'] gives bbox in [xmin, ymin, xmax, ymax]
        boxes = torch.as_tensor([o['bbox'] for o in object_anns], dtype=torch.float32)

        # Get class labels for each bounding box
        category_tokens = [o['category_token'] for o in object_anns]
        categories = [self.nuimages.get('category', token) for token in category_tokens]
        labels = torch.as_tensor([self._category_name_to_id[cat['name']] for cat in categories],
                                 dtype=torch.int64)

        # Get nuimages segmentation masks
        # semantic_mask, instance_mask = self.nuimages.get_segmentation(sample['key_camera_token'])
        # The nuimages instance mask is (H by W) where each value is 0 to N (N is number of object annotations)
        # Convert this to a single (N by H by W) array
        instance_mask = get_instance_mask(self.nuimages, image, object_anns)
        masks = np.array([instance_mask == i+1 for i in range(len(object_anns))]).astype(np.uint8)
        masks = torch.as_tensor(masks)

        # Use key camera token as image identifier
        # Convert key camera token from hexadecimal to an integer, and use it as the unique identifier
        image_id = torch.as_tensor([idx]).type(torch.int64)

        # Compute area
        if boxes.shape[0] == 0:
            area = torch.as_tensor([])
        else:
            area = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])

        # Assume all instances are not crowd
        iscrowd = torch.zeros((len(object_anns),), dtype=torch.int64)

        target = {}
        target['boxes'] = boxes
        target['labels'] = labels
        target['masks'] = masks
        target['image_id'] = image_id
        target['area'] = area
        target['iscrowd'] = iscrowd

        if self.transforms is not None:
            image, target = self.transforms(image, target)

        return image, target

    def get_height_and_width(self, idx):
        sample = self.nuimages.sample[self.samples_with_objects[idx]]
        sample_data = self.nuimages.get('sample_data', sample['key_camera_token'])
        return sample_data['height'], sample_data['width']


def get_instance_mask(nuimages, image, object_anns):
    """
    Helper function to get instance masks from NuImages data.
    Avoid using the NuImages.get_segmentation() method, since this is inefficient
    (loads the image and iterates through all object annotations, which is redundant)

    Parameters
    ----------
    nuimages: NuImages
    image: Image
    object_anns: list

    Returns
    -------
    ndarray

    """
    (width, height) = image.size
    instanceseg_mask = np.zeros((height, width)).astype('int32')

    # Sort by token to ensure that objects always appear in the instance mask in the same order.
    object_anns = sorted(object_anns, key=lambda k: k['token'])

    # Draw object instances.
    # The 0 index is reserved for background; thus, the instances should start from index 1.
    for i, ann in enumerate(object_anns, start=1):
        # Get color, box, mask and name.
        category_token = ann['category_token']
        category_name = nuimages.get('category', category_token)['name']
        if ann['mask'] is None:
            continue
        mask = mask_decode(ann['mask'])

        # Draw masks for semantic segmentation and instance segmentation.
        # semseg_mask[mask == 1] = name_to_index[category_name]
        instanceseg_mask[mask == 1] = i

    return instanceseg_mask