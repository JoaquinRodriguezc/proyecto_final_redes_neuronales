from __future__ import annotations

import json
import random
import subprocess
import sys
import zipfile
from collections import defaultdict
from itertools import groupby
from pathlib import Path

import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision.transforms import functional as TF


CARDD_VIEW_URL = "https://drive.google.com/file/d/1bbyqVCKZX5Ur5Zg-uKj0jD0maWAVeOLx/view"
CARDD_FILE_ID = "1bbyqVCKZX5Ur5Zg-uKj0jD0maWAVeOLx"
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
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



def ensure_gdown():
    try:
        import gdown

        return gdown
    except ImportError:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "gdown"])
        import gdown

        return gdown



def download_cardd_zip(file_id=CARDD_FILE_ID, zip_path=None):
    zip_path = Path(zip_path) if zip_path is not None else Path("data") / "CarDD_release.zip"

    if zip_path.exists() and zipfile.is_zipfile(zip_path):
        return zip_path

    if zip_path.exists() and not zipfile.is_zipfile(zip_path):
        zip_path.unlink()

    gdown = ensure_gdown()
    zip_path.parent.mkdir(parents=True, exist_ok=True)
    gdown.download(id=file_id, output=str(zip_path), quiet=False)

    if not zipfile.is_zipfile(zip_path):
        raise zipfile.BadZipFile("La descarga no produjo un archivo ZIP válido.")

    return zip_path



def extract_cardd_zip(zip_path, extract_dir=None):
    zip_path = Path(zip_path)
    extract_dir = Path(extract_dir) if extract_dir is not None else zip_path.parent

    if not zip_path.exists():
        raise FileNotFoundError(f"No existe el ZIP en {zip_path}")

    if not zipfile.is_zipfile(zip_path):
        raise zipfile.BadZipFile(f"{zip_path} no es un ZIP válido")

    extract_dir.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path, "r") as zip_ref:
        zip_ref.extractall(extract_dir)

    return extract_dir



def find_dataset_root(data_dir) -> Path:
    base_dir = Path(data_dir)
    candidates = [
        base_dir / "raw" / "CarDD",
        base_dir / "CarDD",
        base_dir / "CarDD_release",
        base_dir / "CarDD_release" / "CarDD_COCO",
        base_dir / "CarDD" / "CarDD_COCO",
    ]

    for candidate in candidates:
        if candidate.exists():
            return candidate

    checked = "\n".join(str(path) for path in candidates)
    raise FileNotFoundError("No se encontro el dataset CarDD. Rutas verificadas:\n" + checked)



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
    raise FileNotFoundError("No se encontro CarDD_COCO. Rutas verificadas:\n" + checked)


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

        self.coco_root = self._find_coco_root()
        self.annotation_path = self._resolve_annotation_file()
        self.annotation_data = self._load_json(self.annotation_path)
        self.images = self.annotation_data.get("images", [])
        self.image_ids = [int(image["id"]) for image in self.images]
        self.images_by_id = {int(image["id"]): image for image in self.images}
        self.annotations_by_image = self._group_annotations_by_image(
            self.annotation_data.get("annotations", [])
        )
        self.class_to_idx = self._build_class_to_idx()
        self.idx_to_class = {idx: name for name, idx in self.class_to_idx.items()}
        self.category_id_to_name = self._load_category_id_to_name()
        self.skipped_images = {}

    def __len__(self):
        return len(self.image_ids)

    def __getitem__(self, idx):
        image, target = self.get_raw_sample(idx)

        image, target = self._apply_resize(image, target)

        if self.transform:
            image, target = self.transform(image, target)
        else:
            image = TF.to_tensor(image)

        return image, target

    def get_raw_sample(self, idx):
        image_info = self.images[idx]
        image_id = int(image_info["id"])
        image_path = self._image_path_from_info(image_info)

        if not image_path.exists():
            raise FileNotFoundError(f"No existe la imagen: {image_path}")

        image = Image.open(image_path).convert("RGB")
        target = self._build_target(image_id)
        return image, target

    def _find_coco_root(self) -> Path:
        return find_coco_root(self.data_dir)

    def _resolve_annotation_file(self) -> Path:
        if self.annotation_file is not None:
            path = Path(self.annotation_file)
            if not path.exists():
                raise FileNotFoundError(f"No existe el archivo de anotaciones: {path}")
            return path

        annotation_path = self.coco_root / "annotations" / OFFICIAL_SPLIT_FILES[self.split]
        if not annotation_path.exists():
            raise FileNotFoundError(f"No existe el archivo de anotaciones: {annotation_path}")
        return annotation_path

    def _annotation_paths_for_categories(self):
        paths = [
            self.coco_root / "annotations" / filename
            for filename in OFFICIAL_SPLIT_FILES.values()
            if (self.coco_root / "annotations" / filename).exists()
        ]

        resolved_path = self._resolve_annotation_file()
        if resolved_path not in paths:
            paths.append(resolved_path)

        return paths

    def _load_json(self, path: Path):
        return json.loads(path.read_text(encoding="utf-8"))

    def _build_class_to_idx(self):
        categories = {}
        category_lists = [
            self._load_json(annotation_path).get("categories", [])
            for annotation_path in self._annotation_paths_for_categories()
        ]
        categories = {
            int(category["id"]): category["name"]
            for category_list in category_lists
            for category in category_list
        }

        if not categories:
            raise ValueError("No se pudieron cargar categorias desde las anotaciones COCO.")

        class_to_idx = {"background": 0}
        for idx, category_id in enumerate(sorted(categories), start=1):
            class_to_idx[categories[category_id]] = idx
        return class_to_idx

    def _load_category_id_to_name(self):
        category_lists = [
            self._load_json(annotation_path).get("categories", [])
            for annotation_path in self._annotation_paths_for_categories()
        ]
        category_id_to_name = {
            int(category["id"]): category["name"]
            for category_list in category_lists
            for category in category_list
        }

        if not category_id_to_name:
            raise ValueError("No se pudieron cargar categorias desde las anotaciones COCO.")

        return category_id_to_name

    def _relative_to_base(self, image_path: Path) -> str:
        try:
            return str(image_path.relative_to(self.data_dir))
        except ValueError:
            return str(image_path)

    def _group_annotations_by_image(self, annotations):
        ordered_annotations = sorted(
            annotations,
            key=lambda annotation: int(annotation["image_id"]),
        )
        return {
            int(image_id): list(group)
            for image_id, group in groupby(
                ordered_annotations,
                key=lambda annotation: int(annotation["image_id"]),
            )
        }

    def _convert_bbox_xywh_to_xyxy(self, bbox):
        x, y, width, height = bbox
        xmin = float(x)
        ymin = float(y)
        xmax = float(x + width)
        ymax = float(y + height)
        return xmin, ymin, xmax, ymax, float(width), float(height)

    def _image_path_from_info(self, image_info):
        filename = image_info.get("file_name")
        return self.coco_root / f"{self.split}2017" / filename

    def _build_target(self, image_id):
        annotations = self.annotations_by_image.get(int(image_id), [])
        converted_annotations = [
            (annotation, *self._convert_bbox_xywh_to_xyxy(annotation["bbox"]))
            for annotation in annotations
        ]
        valid_annotations = [
            (annotation, xmin, ymin, xmax, ymax, width, height)
            for annotation, xmin, ymin, xmax, ymax, width, height in converted_annotations
            if xmax > xmin and ymax > ymin
        ]

        boxes = (
            torch.tensor(
                [[xmin, ymin, xmax, ymax] for _, xmin, ymin, xmax, ymax, _, _ in valid_annotations],
                dtype=torch.float32,
            )
            if valid_annotations
            else torch.zeros((0, 4), dtype=torch.float32)
        )
        labels = torch.tensor(
            [
                self.class_to_idx[self.category_id_to_name[int(annotation["category_id"])]]
                for annotation, *_ in valid_annotations
            ],
            dtype=torch.int64,
        )
        area = torch.tensor(
            [
                float(annotation.get("area", width * height))
                for annotation, _, _, _, _, width, height in valid_annotations
            ],
            dtype=torch.float32,
        )
        iscrowd = torch.tensor(
            [int(annotation.get("iscrowd", 0)) for annotation, *_ in valid_annotations],
            dtype=torch.int64,
        )

        return {
            "boxes": boxes,
            "labels": labels,
            "image_id": torch.tensor([int(image_id)], dtype=torch.int64),
            "area": area,
            "iscrowd": iscrowd,
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



def collate_fn(batch):
    images, targets = zip(*batch)
    return list(images), list(targets)
