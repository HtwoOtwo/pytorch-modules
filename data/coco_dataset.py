import torch
import numpy as np
from typing import List, Union, TypedDict
from PIL import Image
from torchvision import transforms
from torch.utils.data import Dataset

transform = transforms.Compose([transforms.ToTensor(),
                                transforms.Normalize(mean=[0.4914, 0.4832, 0.4856],
                                                     std=[0.2023, 0.2013, 0.2111]),
                                transforms.Resize((512, 512))])


def get_ground_truth(dataset):
    dataset.compute_metadata()

    gt_bboxes = []
    images = []
    gt_clss = []
    for sample in dataset.select_fields(["id", "filepath", "metadata", "ground_truth"]):
        image = transform(Image.open(sample.filepath).convert('RGB')) # C H W
        height, width = image.shape[1:]

        if sample.ground_truth is None:
            continue

        bboxes = []
        clss = []
        for det in sample.ground_truth.detections:
            tlx, tly, w, h = det.bounding_box
            bbox = [int(tlx*width), int(tly*height), int(w*width), int(h*height)]
            bboxes.append(bbox)
            clss.append(0)

        images.append(image)
        gt_bboxes.append(torch.Tensor(bboxes))
        gt_clss.append(torch.Tensor(clss))

    assert len(images) == len(gt_bboxes) and len(gt_bboxes) == len(gt_clss)
    return images, gt_bboxes, gt_clss


class COCODataset(Dataset):
    def __init__(self, dataset):
        self.dataset = dataset
        self.images, self.gt_bboxes, self.gt_clss = get_ground_truth(dataset)

    def __getitem__(self, index):
        gt_bbox = self.gt_bboxes[index]
        gt_clss = self.gt_clss[index]
        image = self.images[index]
        return image, gt_bbox, gt_clss

    def __len__(self):
        return len(self.images)

