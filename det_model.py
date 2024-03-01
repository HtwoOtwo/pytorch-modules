import torch
import torch.nn as nn

from modules.cnns.resnet import ResNetBlock, ConvTranspose
from modules.heads.anchor_free_heads import CenterNetHead


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
		self.head = CenterNetHead(num_classes) # need input channel 64

	def forward(self, inputs, gt=None, conf=0.0001, mode="predict"):
		if mode == "predict":
			preds = self.predict(inputs, conf)
			return self.decode_preds(preds)
		elif mode == "train":
			return self.loss(inputs, gt)
		else:
			raise RuntimeError("Invalid mode.")

	def loss(self, inputs, gt):
		pass

	def decode_preds(self, preds):
		# get bboxes, clss, conf
		pass

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
		yv, xv = torch.meshgrid(torch.arange(0, h), torch.arange(0, w)) # h x w
		xv = xv.unsqueeze(0).expand(b, -1, -1)
		yv = yv.unsqueeze(0).expand(b, -1, -1)

		# B C H W -> B H W C
		hm = hm.permute(0, 2, 3, 1)
		wh = wh.permute(0, 2, 3, 1)
		offset = offset.permute(0, 2, 3, 1)

		class_conf, class_pred = torch.max(hm, dim=-1) # b h w c -> b h w 1
		mask = class_conf > conf

		center_wh = wh[mask]
		center_offset = offset[mask]

		center_x = torch.unsqueeze(xv[mask].flatten().float()+ center_offset[..., 0], -1)
		center_y = torch.unsqueeze(yv[mask].flatten().float() + center_offset[..., 1], -1)

		half_w, half_h = torch.unsqueeze(center_wh[..., 0] / 2, -1), torch.unsqueeze(center_wh[..., 1] / 2, -1)
		boxes = torch.cat([center_x - half_w, center_y - half_h, center_x + half_w, center_y + half_h],  dim=-1)
		# normalize
		boxes[..., [0, 2]] = (boxes[..., [0, 2]] / w) * input_w
		boxes[..., [1, 3]] = (boxes[..., [1, 3]] / h) * input_h
		clss = torch.unsqueeze(class_pred[mask], -1)
		conf = torch.unsqueeze(class_conf[mask], -1)

		detects = torch.cat([boxes, clss, conf], dim=-1)
		return detects



if __name__ == "__main__":
	det_net = DetModelDemo(8) # 8 classes
	input = torch.rand((2, 3, 512, 512))
	output = det_net.predict(input, 0.8)
	print(output)


