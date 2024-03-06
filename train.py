import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from typing import List, Union, TypedDict
from dataclasses import dataclass

from data.coco_dataset import COCODataset
from det_model import DetModelDemo


import fiftyone as fo
import fiftyone.zoo as foz


@dataclass
class Annotation:
	gt_bboxes: Union[np.ndarray, torch.Tensor]
	gt_clss: Union[np.ndarray, torch.Tensor]


def train(epochs, train_loader, model, device, lr, momen):
	optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=momen)
	model.to(device)
	for e in range(epochs):
		for i, (imgs, gt_bboxes, gt_clss) in enumerate(train_loader):
			imgs = imgs.to(device)
			gt_bboxes = gt_bboxes.to(device)
			gt_clss = gt_clss.to(device)
			gt = Annotation(gt_bboxes, gt_clss)
			loss = model(imgs, gt=gt, conf=0.0001, iou_thres=0.45, mode="train")
			total_loss = sum(loss.values())
			optimizer.zero_grad()
			total_loss.backward()
			optimizer.step()
			if i % 5 == 0:
				print('epoch: {}, batch: {}, loss: {}'.format(e + 1, i + 1, total_loss.data))
	# torch.save(model, 'myAlexMnistDemo.pth') # save net model and parameters


def collate_fn(batch):
	images, gt_bboxes, gt_clss = zip(*batch)
	images = torch.stack(images, dim=0)
	gt_bboxes = torch.cat(gt_bboxes, dim=0)
	gt_clss = torch.cat(gt_clss, dim=0)

	return images, gt_bboxes, gt_clss


if __name__ == "__main__":
	# coco_dataset = COCODataset("../coco_dataset", annotation_file="../coco_dataset/annotations/instances_train2017.json")
	coco_51dataset = foz.load_zoo_dataset(
		"coco-2017",
		split="validation"
	)
	coco_dataset = COCODataset(coco_51dataset)
	train_loader = DataLoader(coco_dataset, batch_size=2, shuffle=True, num_workers=1, collate_fn=collate_fn)
	model = DetModelDemo(80)
	train(100, train_loader, model, "cpu", 0.00001, 0.9)
