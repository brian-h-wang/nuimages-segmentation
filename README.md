# NuImages Mask-RCNN

[Insert example image here]

A simple codebase for training Mask-RCNN to perform instance segmentation on images from the NuImages dataset.

The ``nuimages_dataset.py`` contains a NuImagesDataset class which loads NuImages samples for use with a PyTorch DataLoader. Following the example [here](https://pytorch.org/tutorials/intermediate/torchvision_tutorial.html), the dataset ``__getitem__`` method returns an RGB image, along with a dictionary containing the annotations for the image, extracted from the NuImages dataset.

The ``finetune_maskrcnn.py`` and ``maskrcnn_inference.py`` scripts are used to train a Mask-RCNN model and perform inference on new images, respectively. 

Finally, the ``utils`` directory contains several helper functions, copied from [the PyTorch GitHub repo](https://github.com/pytorch/vision/tree/master/references/detection).

# Usage

Use the `finetune_maskrcnn.py` script to fine-tune Mask-RCNN (pre-trained on the COCO dataset) for NuImages instance segmentation

Options:

[Insert options here]
