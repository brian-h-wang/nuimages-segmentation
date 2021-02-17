"""
Fine-tune a Mask-RCNN model, pretrained on the COCO dataset,
to predict instance masks on the NuImages dataset.

Reference:
https://pytorch.org/tutorials/intermediate/torchvision_tutorial.html
"""

import torch
from nuimages import NuImages
from torch.utils.data import DataLoader

from nuimages_dataset import NuImagesDataset
from utils.engine import train_one_epoch, evaluate
from utils.model_utils import get_model_instance_segmentation, get_transform, collate_fn

if __name__ == "__main__":
    nuimages = NuImages(dataroot="nuimages", version="v1.0-train", verbose=True, lazy=False)
    nuimages_val = NuImages(dataroot="nuimages", version="v1.0-val", verbose=True, lazy=False)
    transforms = get_transform(train=True)
    transforms_val = get_transform(train=False)
    dataset = NuImagesDataset(nuimages, transforms=transforms)
    dataset_val = NuImagesDataset(nuimages_val, transforms=transforms_val)

    print(f"{len(dataset)} training samples and {len(dataset_val)} val samples.")

    num_classes = len(nuimages.category) + 1  # add one for background class

    device = torch.device('cuda:1') if torch.cuda.is_available() else torch.device('cpu')

    data_loader = DataLoader(dataset, batch_size=2, shuffle=True, num_workers=4, collate_fn=collate_fn)
    data_loader_val = DataLoader(dataset_val, batch_size=1, shuffle=False, num_workers=4,
                                 collate_fn=collate_fn)

    model = get_model_instance_segmentation(num_classes)

    # Move model to GPU
    model.to(device)

    # Construct optimizer
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(params, lr=0.005,
                                    momentum=0.9, weight_decay=0.0005)
    # and a learning rate scheduler
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer,
                                                   step_size=3,
                                                   gamma=0.1)


    n_epochs = 10
    for epoch in range(n_epochs):
        train_one_epoch(model, optimizer, data_loader, device, epoch, print_freq=10)
        lr_scheduler.step()
        evaluate(model, data_loader_val, device=device)

    torch.save(model.state_dict(), "nuimages_maskrcnn.pth")




