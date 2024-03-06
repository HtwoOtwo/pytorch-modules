import torch
import torch.nn as nn
import torch.nn.functional as F

from modules.cnns.resnet import ResNetBlock, ConvTranspose
from modules.det_heads.anchor_free_heads import CenterNetHead


def box_iou(box1, box2, eps=1e-7):
	"""
	Args:
	    box1 (torch.Tensor): A tensor of shape (N, 4) representing N bounding boxes.
	    box2 (torch.Tensor): A tensor of shape (M, 4) representing M bounding boxes.
	    eps (float, optional): A small value to avoid division by zero. Defaults to 1e-7.

	Returns:
	    (torch.Tensor): An NxM tensor containing the pairwise IoU values for every element in box1 and box2.
	"""
	# inter(N,M) = (rb(N,M,2) - lt(N,M,2)).clamp(0).prod(2)
	(a1, a2), (b1, b2) = box1.unsqueeze(1).chunk(2, 2), box2.unsqueeze(0).chunk(2, 2)
	inter = (torch.min(a2, b2) - torch.max(a1, b1)).clamp_(0).prod(2)

	# IoU = inter / (area1 + area2 - inter)
	return inter / ((a2 - a1).prod(2) + (b2 - b1).prod(2) - inter + eps)


class DetModelDemo(nn.Module):
	def __init__(self, num_classes):
		super().__init__()
		# B 3 H W --> B 512 H/16 W/16
		self.backbone = nn.Sequential(
			ResNetBlock(3, 64, 2),
			ResNetBlock(64, 128, 2),
			ResNetBlock(128, 256, 2),
			ResNetBlock(256, 512, 2),
			ConvTranspose(512, 256, 2),
			ConvTranspose(256, 128, 2),
			ConvTranspose(128, 64, 2)
		)
		self.head = CenterNetHead(num_classes)  # need input channel 64

	def forward(self, inputs, gt=None, conf=0.0001, iou_thres=0.45, mode="predict"):
		preds = self.predict(inputs, conf)
		if mode == "predict":
			return self.decode_preds(preds)
		elif mode == "train":
			return self.loss(preds, gt, iou_thres)
		else:
			raise RuntimeError("Invalid mode.")

	def decode_preds(self, preds):
		pass

	def loss(self, preds, gt, iou_thres) -> dict:
		pred_bboxes = preds[..., :-2]
		pred_clss = preds[..., -2]
		gt_bboxes = gt.gt_bboxes
		gt_clss = gt.gt_clss
		iou = box_iou(pred_bboxes, gt_bboxes)

		match_preds_idx, match_gts_idx = torch.where(iou > iou_thres)

		new_pred_bboxes = pred_bboxes[match_preds_idx, ...]
		new_gt_bboxes = gt_bboxes[match_gts_idx, ...]
		new_pred_clss = pred_clss[match_preds_idx]
		new_gt_clss = gt_clss[match_gts_idx]

		print(gt_bboxes)

		bbox_loss = F.binary_cross_entropy_with_logits(new_pred_bboxes, new_gt_bboxes)
		clss_loss = F.binary_cross_entropy_with_logits(new_pred_clss, new_gt_clss)
		return {"bbox_loss": bbox_loss, "clss_loss": clss_loss}

	def predict(self, x, conf=0.001):
		# B C H W
		input_shape = x.shape
		hm, wh, offset = self._forward(x)
		return self._post_process(hm, wh, offset, conf, input_shape)

	def _forward(self, x):
		feat = self.backbone(x)
		return self.head(feat)

	def _post_process(self, hm, wh, offset, conf, input_shape):
		input_h, input_w = input_shape[-2:]
		b, c, h, w = hm.shape
		yv, xv = torch.meshgrid(torch.arange(0, h), torch.arange(0, w))  # h x w
		xv = xv.unsqueeze(0).expand(b, -1, -1)
		yv = yv.unsqueeze(0).expand(b, -1, -1)

		# B C H W -> B H W C
		hm = hm.permute(0, 2, 3, 1)
		wh = wh.permute(0, 2, 3, 1)
		offset = offset.permute(0, 2, 3, 1)

		class_conf, class_pred = torch.max(hm, dim=-1)  # b h w c -> b h w 1
		mask = class_conf > conf

		center_wh = wh[mask]
		center_offset = offset[mask]

		center_x = torch.unsqueeze(xv[mask].flatten().float() + center_offset[..., 0], -1)
		center_y = torch.unsqueeze(yv[mask].flatten().float() + center_offset[..., 1], -1)

		half_w, half_h = torch.unsqueeze(center_wh[..., 0] / 2, -1), torch.unsqueeze(center_wh[..., 1] / 2, -1)
		boxes = torch.cat([center_x - half_w, center_y - half_h, center_x + half_w, center_y + half_h], dim=-1)

		# normalize
		boxes[..., [0, 2]] = (boxes[..., [0, 2]] / w) * input_w
		boxes[..., [1, 3]] = (boxes[..., [1, 3]] / h) * input_h

		clamped_boxes = boxes.clone()
		clamped_boxes[:, 0::2] = torch.clamp(boxes[:, 0::2], min=0, max=input_w)  # 限制 x1 和 x2
		clamped_boxes[:, 1::2] = torch.clamp(boxes[:, 1::2], min=0, max=input_h)  # 限制 y1 和 y2

		clss = torch.unsqueeze(class_pred[mask], -1)
		conf = torch.unsqueeze(class_conf[mask], -1)

		detects = torch.cat([clamped_boxes, clss, conf], dim=-1)
		return detects


if __name__ == "__main__":
	det_net = DetModelDemo(8)  # 8 classes
	input = torch.rand((2, 3, 512, 512))
	output = det_net.predict(input, 0.6)
	print(output.shape)

	preds = torch.tensor([[0, 0, 0, 0],
	                      [50, 55, 190, 180],
	                      [10, 10, 100, 110]])

	gts = torch.tensor([[15, 15, 100, 100],
	                      [50, 50, 180, 180]])

	iou = box_iou(preds, gts)
	print(iou)
	match_preds, match_gts = torch.where(iou > 0.45)
	new_preds = preds[match_preds, ...]
	new_gts = gts[match_gts, ...]

	print(new_preds)
	print(new_gts)

