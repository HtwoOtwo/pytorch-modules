import torch
import json
import os
import cv2
import numpy as np
from typing import Tuple, Union, TypedDict, List
import pycocotools.mask as maskUtils
from tqdm import tqdm
from multiprocessing.pool import ThreadPool
from loguru import logger
from torch.utils.data import Dataset


EXTENSIONS = [".jpg", ".jpeg", ".png", ".bmp", ".gif", ".tiff"]
NUM_THREADS = min(8, max(1, os.cpu_count() - 1))
BOXTYPE = Union[np.ndarray, List[int], List[float], torch.Tensor] # xywh or normed xywh
MASKTYPE = Union[np.ndarray, List[np.ndarray], torch.Tensor] # bool matrix array
CLSSTYPE = Union[np.ndarray, List[int], torch.Tensor]


def build_annotations_index(coco_data) -> dict:
    annotations_index = {}

    def process_annotation(annotation):
        image_id = annotation['image_id']
        filename = next(image['file_name'] for image in coco_data['images'] if image['id'] == image_id)
        if filename not in annotations_index:
            annotations_index[filename] = []
        annotations_index[filename].append(annotation)

    with ThreadPool(NUM_THREADS) as pool:
        list(tqdm(pool.imap(func=process_annotation,
                            iterable=coco_data.get('annotations', [])),
                  total=len(coco_data.get('annotations', [])),
                  desc="retrieving data..."))

    return annotations_index


def load_json(json_file: str) -> dict:
    try:
        with open(json_file, 'r') as file:
            data = json.load(file)
        return data
    except FileNotFoundError:
        logger.warning(f"Error: File '{json_file}' not found.")
        return {}
    except json.JSONDecodeError as e:
        logger.warning(f"Error decoding JSON in file '{json_file}': {e}")
        return {}


def poly2mask(mask_ann: Union[list, dict], img_h: int,
              img_w: int) -> np.ndarray:
    if isinstance(mask_ann, list):
        # polygon
        rles = maskUtils.frPyObjects(mask_ann, img_h, img_w)
        rle = maskUtils.merge(rles)
    elif isinstance(mask_ann['counts'], list):
        # uncompressed RLE
        rle = maskUtils.frPyObjects(mask_ann, img_h, img_w)
    else:
        # rle
        rle = mask_ann
    mask = maskUtils.decode(rle)
    return mask


def process_mask(mask_ann: Union[list, dict], img_h: int,
                 img_w: int) -> np.ndarray:
    # polygon
    if isinstance(mask_ann, list):
        mask_ann = [
            np.array(polygon) for polygon in mask_ann
            if len(polygon) % 2 == 0 and len(polygon) >= 6
        ]
        if len(mask_ann) == 0:
            # ignore this ann
            mask_ann = [np.zeros(6)]
    elif isinstance(mask_ann, dict) and \
            not (mask_ann.get('counts') is not None and
                 mask_ann.get('size') is not None and
                 isinstance(mask_ann['counts'], (list, str))):
        mask_ann = [np.zeros(6)]
    gt_mask = poly2mask(mask_ann, img_h, img_w)
    return gt_mask


class Annotation(TypedDict):
    gt_boxes: BOXTYPE
    gt_clss: CLSSTYPE
    gt_masks: MASKTYPE
    img_shape: Tuple


class BaseDataset(Dataset):
    def __init__(self, dataset_path, meta_data=None):
        self.dataset_path = os.path.abspath(dataset_path)
        self.image_path = self.get_image_path()

    def get_image_path(self):
        image_paths = []
        for dirpath, _, filenames in os.walk(self.dataset_path):
            for filename in filenames:
                if filename.lower().endswith(tuple(EXTENSIONS)):
                    image_path = os.path.join(dirpath, filename)
                    image_paths.append(image_path)
        return image_paths

    def __getitem__(self, index):
        annotations = self.get_annotations(index)
        return annotations

    def __len__(self):
        return len(self.image_path)

    def get_annotations(self, index, norm=False) -> Annotation:
        raise NotImplementedError("you need return dict like {'box': ...}")


class COCODataset(BaseDataset):
    def __init__(self, dataset_path, meta_data=None, annotation_file=None):
        super().__init__(dataset_path, meta_data)
        self.annotation_file = annotation_file
        if self.annotation_file is not None:
            self.annotation_data = load_json(self.annotation_file)
        else:
            self.annotation_data = {}
        self.annotations_by_name = build_annotations_index(self.annotation_data)

    def get_annotations(self, index, norm=False) -> Annotation:
        image_path = self.image_path[index]
        h, w = cv2.imread(image_path).shape[:2]
        image_name = os.path.basename(image_path)
        anns_of_image = self.annotations_by_name.get(image_name, [])

        gt_boxes = [annotation.get("bbox", []) for annotation in anns_of_image]
        gt_clss = [annotation.get("category_id", []) for annotation in anns_of_image]
        gt_masks = [process_mask(annotation.get("segmentation", []), h, w) for annotation in anns_of_image]
        img_shape = (h, w)

        annotations: Annotation = {"gt_boxes": gt_boxes,
                                   "gt_clss": gt_clss,
                                   "gt_masks": gt_masks,
                                   "img_shape": img_shape}
        return annotations
