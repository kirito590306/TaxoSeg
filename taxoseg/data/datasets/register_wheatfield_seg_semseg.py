import json
import os
from pathlib import Path

from detectron2.data import DatasetCatalog, MetadataCatalog


REPO_ROOT = Path(__file__).resolve().parents[3]
DATA_ROOT = Path(
    os.getenv("TAXOSEG_DATASETS", str(REPO_ROOT / "data"))
).expanduser()
SPLIT_ROOT = DATA_ROOT / "WheatField-Seg"
TAXONOMY_PATH = SPLIT_ROOT / "mappings" / "taxonomy.json"
SPLIT_DATASET_NAMES = {
    "train": "wheatfield_seg_train_semseg",
    "test": "wheatfield_seg_test_semseg",
    "val": "wheatfield_seg_val_semseg",
}
WHEATFIELD_SPLIT_CLASSES = [
    (0, "background"),
    (1, "wheat head scab"),
    (2, "wheat powdery mildew"),
    (3, "wheat stripe rust"),
]


def _load_annotation(root, split):
    ann_path = root / f"annotation_{split}.json"
    if not ann_path.exists():
        raise FileNotFoundError(
            f"Missing annotation file: {ann_path}. See data/README.md for dataset layout."
        )
    with ann_path.open("r", encoding="utf-8-sig") as f:
        return json.load(f)


def _build(root, split, sem_seg_folder="annotations"):
    annotation = _load_annotation(root, split)
    data = []
    for im in annotation["images"]:
        image_path = root / "images" / split / im["file_name"]
        sem_seg_path = root / sem_seg_folder / split / f"{Path(im['file_name']).stem}.png"
        data.append(
            {
                "file_name": str(image_path),
                "sem_seg_file_name": str(sem_seg_path),
                "height": im["height"],
                "width": im["width"],
                "image_id": im.get("id"),
            }
        )
    return data


def _register_split_dataset(split, dataset_name):
    DatasetCatalog.register(
        dataset_name,
        lambda split=split: _build(SPLIT_ROOT, split, sem_seg_folder="annotations"),
    )
    names = [name for _, name in WHEATFIELD_SPLIT_CLASSES]
    meta = MetadataCatalog.get(dataset_name)
    meta.stuff_classes = names
    meta.ignore_label = 255
    meta.evaluator_type = "wheatfield_sem_seg"
    meta.dataset_root = str(SPLIT_ROOT)
    meta.image_root = str(SPLIT_ROOT / "images" / split)
    meta.sem_seg_root = str(SPLIT_ROOT / "annotations" / split)
    meta.taxonomy_path = str(TAXONOMY_PATH)
    meta.original_to_contiguous_id = {idx: idx for idx in range(len(names))}
    meta.stuff_dataset_id_to_contiguous_id = {idx: idx for idx in range(len(names))}


for split_name, dataset_name in SPLIT_DATASET_NAMES.items():
    _register_split_dataset(split_name, dataset_name)
