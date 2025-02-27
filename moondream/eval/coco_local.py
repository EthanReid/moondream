import argparse
import os
import random
import json
import torch
import numpy as np

from PIL import Image
from typing import List, Tuple
from tqdm import tqdm

from ..torch.config import MoondreamConfig
from ..torch.moondream import MoondreamModel
from ..torch.weights import load_weights_into_model


def calculate_iou(
    box1: Tuple[float, float, float, float],
    box2: Tuple[float, float, float, float],
) -> float:
    """Calculate IoU between two boxes (x1, y1, x2, y2 format)"""
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])

    intersection = max(0, x2 - x1) * max(0, y2 - y1)
    box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
    box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])
    return intersection / (box1_area + box2_area - intersection)


def calculate_map(
    ground_truth_boxes: List[List[Tuple[float, float, float, float]]],
    predicted_boxes: List[List[Tuple[float, float, float, float, float]]],
    iou_threshold: float = 0.5,
) -> float:
    """
    Calculate mAP for object detection

    Args:
        ground_truth_boxes: List (per image) of ground truth boxes [(x1, y1, x2, y2)]
        predicted_boxes: List (per image) of predicted boxes [(x1, y1, x2, y2, confidence)]
        iou_threshold: IoU threshold to consider a detection as correct

    Returns:
        Average Precision for the class
    """
    total_ap = 0.0
    num_images = len(ground_truth_boxes)

    for gt_boxes, pred_boxes in zip(ground_truth_boxes, predicted_boxes):
        # Sort predictions by confidence descending
        pred_boxes = sorted(pred_boxes, key=lambda x: x[4], reverse=True)

        num_gt = len(gt_boxes)
        if num_gt == 0:
            continue

        tp = np.zeros(len(pred_boxes))
        fp = np.zeros(len(pred_boxes))
        gt_matched = [False] * num_gt

        for pred_idx, pred_box in enumerate(pred_boxes):
            max_iou = 0.0
            max_idx = -1
            for gt_idx, gt_box in enumerate(gt_boxes):
                if gt_matched[gt_idx]:
                    continue
                iou = calculate_iou(pred_box[:4], gt_box)
                if iou > max_iou:
                    max_iou = iou
                    max_idx = gt_idx
            if max_iou >= iou_threshold:
                tp[pred_idx] = 1
                gt_matched[max_idx] = True
            else:
                fp[pred_idx] = 1

        cumsum_tp = np.cumsum(tp)
        cumsum_fp = np.cumsum(fp)
        recalls = cumsum_tp / num_gt
        precisions = cumsum_tp / (cumsum_tp + cumsum_fp + 1e-6)

        ap = 0.0
        for t in np.arange(0.0, 1.1, 0.1):
            if np.sum(recalls >= t) == 0:
                p = 0
            else:
                p = np.max(precisions[recalls >= t])
            ap += p / 11.0
        total_ap += ap

    return total_ap / num_images if num_images > 0 else 0.0


def get_total_map(results_by_label, frequency_by_label):
    total_count = 0
    total_map = 0
    for results, frequency in zip(
        results_by_label.values(), frequency_by_label.values()
    ):
        total_map += sum(results)
        total_count += frequency
    return total_map / total_count if total_count > 0 else 0.0


class CocoDataset:
    def __init__(self, annotation_file, img_dir, transform=None):
        self.annotation_file = annotation_file
        self.img_dir = img_dir
        self.transform = transform

        with open(self.annotation_file, "r") as f:
            data = json.load(f)

        self.images = data["images"]
        self.annotations = data["annotations"]
        self.categories = data.get("categories", [])
        self.class_id_to_name = {cat["id"]: cat["name"] for cat in self.categories}

        self.img_id_to_anns = {}
        for ann in self.annotations:
            img_id = ann["image_id"]
            if img_id not in self.img_id_to_anns:
                self.img_id_to_anns[img_id] = []
            self.img_id_to_anns[img_id].append(ann)

        self.ids = []
        self.id_to_img = {}
        for img_info in self.images:
            img_id = img_info["id"]
            if (
                img_id in self.img_id_to_anns
                and len(self.img_id_to_anns[img_id]) > 0
                and len(self.img_id_to_anns[img_id]) <= 150
            ):
                self.ids.append(img_id)
                self.id_to_img[img_id] = img_info
        random.shuffle(self.ids)

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, idx):
        image_id = self.ids[idx]
        img_info = self.id_to_img[image_id]
        file_name = img_info["file_name"]
        width = img_info["width"]
        height = img_info["height"]
        img_path = os.path.join(self.img_dir, file_name)
        image = Image.open(img_path).convert("RGB")

        ann_list = self.img_id_to_anns[image_id]
        gt_label_to_boxes = {}
        for ann in ann_list:
            label = ann["category_id"]
            if label not in gt_label_to_boxes:
                gt_label_to_boxes[label] = []
            bbox = ann["bbox"]

            gt_label_to_boxes[label].append(
                (bbox[0], bbox[1], bbox[0] + bbox[2], bbox[1] + bbox[3])
            )

        return {
            "image": image,
            "gt_label_to_boxes": gt_label_to_boxes,
            "width": width,
            "height": height,
            "image_id": image_id,
        }


def eval_coco_map(model, annotation_file, img_dir, iou_threshold=0.5, debug=False):
    """
    Evaluate mAP on the COCO dataset loaded via CocoDataset.
    For each image, the model is run for each unique label using the label name from the dataset.
    """
    dataset = CocoDataset(annotation_file, img_dir)
    total = 0
    results_by_label = {}
    frequency_by_label = {}

    for data in tqdm(dataset, disable=debug, desc="COCO mAP"):
        image = data["image"]
        width = data["width"]
        height = data["height"]
        total += 1
        gt_label_to_boxes = data["gt_label_to_boxes"]

        label_mapping = dataset.class_id_to_name

        for label in gt_label_to_boxes.keys():
            # Get the label name from the mapping.
            label_name = label_mapping.get(label, "unknown")
            encoded_image = model.encode_image(image)
            model_answer = model.detect(encoded_image, label_name)["objects"]

            moondream_boxes = []
            for box in model_answer:
                moondream_boxes.append(
                    (
                        box["x_min"] * width,
                        box["y_min"] * height,
                        box["x_max"] * width,
                        box["y_max"] * height,
                        1.0,  # confidence
                    )
                )
            map_result = calculate_map(
                [gt_label_to_boxes[label]], [moondream_boxes], iou_threshold
            )
            if debug and map_result == 0:
                print(
                    f"0 mAP for image {data['image_id']} and label {label} ({label_name})"
                )

            results_by_label.setdefault(label, []).append(map_result)
            frequency_by_label[label] = frequency_by_label.get(label, 0) + 1

        if debug and total % 100 == 0:
            current_map = get_total_map(results_by_label, frequency_by_label) * 100
            print(f"Processed {total} images, current overall mAP: {current_map:.2f}")

    return {"total_map": get_total_map(results_by_label, frequency_by_label)}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model", type=str, required=True, help="Path to model weights"
    )
    parser.add_argument(
        "--annotation_file",
        type=str,
        default="datasets/rf/test/_annotations.coco.json",
        help="Path to COCO annotation JSON file",
    )
    parser.add_argument(
        "--img_dir",
        type=str,
        default="/home/ethan/moondream/datasets/rf/test",
        help="Directory containing images",
    )
    parser.add_argument("--debug", action="store_true", help="Enable debug mode")
    parser.add_argument("--gpu", type=int, default=0, help="GPU device id to use")
    args = parser.parse_args()

    if torch.cuda.is_available():
        torch.cuda.set_device(args.gpu)
        device = torch.device(f"cuda:{args.gpu}")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")

    # Initialize and load the model
    config = MoondreamConfig()
    model = MoondreamModel(config)
    load_weights_into_model(args.model, model)
    model.compile()
    model.to(device)

    result = eval_coco_map(
        model, args.annotation_file, args.img_dir, iou_threshold=0.5, debug=args.debug
    )
    print(f"Overall mAP: {result['total_map'] * 100:.2f}")
