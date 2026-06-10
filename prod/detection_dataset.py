from __future__ import annotations

import json
import random
from collections import defaultdict
from pathlib import Path

import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision.transforms import functional as TF


OFFICIAL_SPLIT_FILES = {
    "train": "instances_train2017.json",
    "val": "instances_val2017.json",
    "test": "instances_test2017.json",
}


def normalize_split(split: str) -> str:
    normalized = split.strip().lower()
    if normalized not in OFFICIAL_SPLIT_FILES:
        expected = ", ".join(sorted(OFFICIAL_SPLIT_FILES))
        raise ValueError(f"Split invalido: {split}. Esperados: {expected}")
    return normalized


def normalize_image_size(image_size):
    if image_size is None:
        return None

    if isinstance(image_size, int):
        return (image_size, image_size)

    if isinstance(image_size, (tuple, list)) and len(image_size) == 2:
        height, width = int(image_size[0]), int(image_size[1])
        if height <= 0 or width <= 0:
            raise ValueError("image_size debe tener alto y ancho mayores a cero")
        return (height, width)

    raise ValueError("image_size debe ser None, un int o una tupla/lista (alto, ancho)")


def find_coco_root(data_dir) -> Path:
    base_dir = Path(data_dir)

    candidates = [
        base_dir,
        base_dir / "raw" / "CarDD",
        base_dir / "CarDD",
        base_dir / "CarDD_release",
        base_dir / "CarDD_release" / "CarDD_COCO",
        base_dir / "CarDD" / "CarDD_COCO",
    ]

    checked_paths = []
    for candidate in candidates:
        checked_paths.append(candidate)
        if (candidate / "annotations").exists():
            return candidate

        coco_candidate = candidate / "CarDD_COCO"
        checked_paths.append(coco_candidate)
        if (coco_candidate / "annotations").exists():
            return coco_candidate

    checked = "\n".join(str(path) for path in checked_paths)
    raise FileNotFoundError(
        "No se encontro CarDD_COCO. Rutas verificadas:\n" + checked
    )


def _resolve_annotation_file(coco_root: Path, split: str, annotation_file=None) -> Path:
    if annotation_file is not None:
        path = Path(annotation_file)
        if not path.exists():
            raise FileNotFoundError(f"No existe el archivo de anotaciones: {path}")
        return path

    annotation_path = coco_root / "annotations" / OFFICIAL_SPLIT_FILES[split]
    if not annotation_path.exists():
        raise FileNotFoundError(f"No existe el archivo de anotaciones: {annotation_path}")
    return annotation_path


def _load_category_mappings(annotation_paths):
    categories = {}

    for annotation_path in annotation_paths:
        if not annotation_path.exists():
            continue

        data = json.loads(annotation_path.read_text(encoding="utf-8"))
        for category in data.get("categories", []):
            categories[category["id"]] = category["name"]

    if not categories:
        raise ValueError("No se pudieron cargar categorias desde las anotaciones COCO.")

    class_to_idx = {"background": 0}
    idx_to_class = {0: "background"}

    for idx, category_id in enumerate(sorted(categories), start=1):
        class_name = categories[category_id]
        class_to_idx[class_name] = idx
        idx_to_class[idx] = class_name

    return categories, class_to_idx, idx_to_class


def _relative_to_base(image_path: Path, base_dir: Path) -> str:
    try:
        return str(image_path.relative_to(base_dir))
    except ValueError:
        return str(image_path)


def build_split_records(
    data_dir,
    splits=None,
    include_empty=False,
    annotation_files=None,
    coco_root=None,
):
    base_dir = Path(data_dir)
    coco_root = find_coco_root(base_dir) if coco_root is None else Path(coco_root)

    requested_splits = splits or tuple(OFFICIAL_SPLIT_FILES)
    requested_splits = [normalize_split(split) for split in requested_splits]

    resolved_annotation_files = {}
    annotation_files = annotation_files or {}
    for split in requested_splits:
        resolved_annotation_files[split] = _resolve_annotation_file(
            coco_root=coco_root,
            split=split,
            annotation_file=annotation_files.get(split),
        )

    all_annotation_paths = [
        coco_root / "annotations" / filename
        for filename in OFFICIAL_SPLIT_FILES.values()
    ]
    all_annotation_paths.extend(path for path in resolved_annotation_files.values())
    category_id_to_name, class_to_idx, idx_to_class = _load_category_mappings(all_annotation_paths)

    split_records = {}
    skipped_images = defaultdict(int)
    all_box_labels = []
    csv_rows = []

    for split in requested_splits:
        annotation_path = resolved_annotation_files[split]
        data = json.loads(annotation_path.read_text(encoding="utf-8"))

        images_by_id = {image["id"]: image for image in data.get("images", [])}
        annotations_by_image = defaultdict(list)
        for annotation in data.get("annotations", []):
            annotations_by_image[annotation["image_id"]].append(annotation)

        records = []
        for image_id, image_info in images_by_id.items():
            filename = image_info.get("file_name")
            image_path = coco_root / f"{split}2017" / filename

            if not image_path.exists():
                skipped_images[split] += 1
                continue

            boxes = []
            labels = []
            areas = []
            iscrowd = []
            label_names = []

            for annotation in annotations_by_image.get(image_id, []):
                x, y, width, height = annotation["bbox"]
                xmin = float(x)
                ymin = float(y)
                xmax = float(x + width)
                ymax = float(y + height)

                if xmax <= xmin or ymax <= ymin:
                    continue

                category_name = category_id_to_name[annotation["category_id"]]
                label_idx = class_to_idx[category_name]

                boxes.append([xmin, ymin, xmax, ymax])
                labels.append(label_idx)
                areas.append(float(annotation.get("area", width * height)))
                iscrowd.append(int(annotation.get("iscrowd", 0)))
                label_names.append(category_name)
                all_box_labels.append(category_name)

            if not boxes and not include_empty:
                skipped_images[split] += 1
                continue

            record = {
                "image_path": _relative_to_base(image_path, base_dir),
                "image_id": int(image_id),
                "boxes": boxes,
                "labels": labels,
                "label_names": label_names,
                "area": areas,
                "iscrowd": iscrowd,
                "split": split,
                "width": int(image_info.get("width", 0)),
                "height": int(image_info.get("height", 0)),
            }
            records.append(record)

            unique_label_names = sorted(set(label_names))
            image_label = (
                unique_label_names[0]
                if len(unique_label_names) == 1
                else "multiple_damage"
            )
            csv_rows.append(
                {
                    "image_path": record["image_path"],
                    "label": image_label,
                    "split": split,
                }
            )

        split_records[split] = records

    metadata = {
        "coco_root": coco_root,
        "class_to_idx": class_to_idx,
        "idx_to_class": idx_to_class,
        "skipped_images": dict(skipped_images),
        "csv_rows": csv_rows,
        "all_box_labels": all_box_labels,
    }
    return split_records, metadata


class ComposeDetection:
    def __init__(self, transforms_list):
        self.transforms_list = transforms_list

    def __call__(self, image, target):
        for transform in self.transforms_list:
            image, target = transform(image, target)
        return image, target


class ToTensorDetection:
    def __call__(self, image, target):
        image = TF.to_tensor(image)
        return image, target


class RandomHorizontalFlipDetection:
    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, image, target):
        if random.random() >= self.p:
            return image, target

        if not isinstance(image, torch.Tensor):
            image = TF.to_tensor(image)

        _, _, width = image.shape
        image = torch.flip(image, dims=[2])

        boxes = target["boxes"].clone()
        if boxes.numel() > 0:
            xmin = boxes[:, 0].clone()
            xmax = boxes[:, 2].clone()
            boxes[:, 0] = width - xmax
            boxes[:, 2] = width - xmin
            target["boxes"] = boxes

        return image, target


class CarDamageDetectionDataset(Dataset):
    def __init__(
        self,
        data_dir,
        split,
        transform=None,
        image_size=None,
        resize=False,
        include_empty=False,
        model_name=None,
        annotation_file=None,
    ):
        self.data_dir = Path(data_dir)
        self.split = normalize_split(split)
        self.transform = transform
        self.image_size = normalize_image_size(image_size)
        self.model_name = (model_name or "generic").strip().lower()
        self.include_empty = include_empty
        self.annotation_file = annotation_file
        self.resize = bool(resize)

        if self.model_name in {"fasterrcnn", "faster_r_cnn", "faster-rcnn"}:
            self.resize = bool(resize)

        if self.resize and self.image_size is None:
            raise ValueError("Si resize=True, image_size no puede ser None")

        split_records, metadata = build_split_records(
            data_dir=self.data_dir,
            splits=[self.split],
            include_empty=self.include_empty,
            annotation_files={self.split: self.annotation_file} if self.annotation_file else None,
        )

        self.coco_root = metadata["coco_root"]
        self.class_to_idx = metadata["class_to_idx"]
        self.idx_to_class = metadata["idx_to_class"]
        self.skipped_images = metadata["skipped_images"]
        self.records = split_records[self.split]

    def __len__(self):
        return len(self.records)

    def _build_target(self, record):
        if record["boxes"]:
            boxes = torch.tensor(record["boxes"], dtype=torch.float32)
        else:
            boxes = torch.zeros((0, 4), dtype=torch.float32)

        return {
            "boxes": boxes,
            "labels": torch.tensor(record["labels"], dtype=torch.int64),
            "image_id": torch.tensor([record["image_id"]], dtype=torch.int64),
            "area": torch.tensor(record["area"], dtype=torch.float32),
            "iscrowd": torch.tensor(record["iscrowd"], dtype=torch.int64),
        }

    def _apply_resize(self, image, target):
        if not self.resize or self.image_size is None:
            return image, target

        target_height, target_width = self.image_size
        original_width, original_height = image.size

        if original_height == target_height and original_width == target_width:
            return image, target

        scale_x = target_width / original_width
        scale_y = target_height / original_height

        image = image.resize((target_width, target_height), Image.BILINEAR)

        boxes = target["boxes"].clone()
        if boxes.numel() > 0:
            boxes[:, [0, 2]] *= scale_x
            boxes[:, [1, 3]] *= scale_y
            target["boxes"] = boxes

        if target["area"].numel() > 0:
            target["area"] = target["area"] * (scale_x * scale_y)

        return image, target

    def __getitem__(self, idx):
        record = self.records[idx]
        image_path = self.data_dir / record["image_path"]
        image = Image.open(image_path).convert("RGB")
        target = self._build_target(record)

        image, target = self._apply_resize(image, target)

        if self.transform:
            image, target = self.transform(image, target)
        else:
            image = TF.to_tensor(image)

        return image, target


def collate_fn(batch):
    images, targets = zip(*batch)
    return list(images), list(targets)
