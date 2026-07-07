#!/usr/bin/env python3
import argparse
import json
import os
from pathlib import Path

import numpy as np
from PIL import Image

DEFAULT_TAXONOMY_PATH = (
    Path(os.getenv("TAXOSEG_DATASETS", "data")).expanduser()
    / "WheatField-Seg"
    / "mappings"
    / "taxonomy.json"
)


def load_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def ensure_dir(path):
    Path(path).mkdir(parents=True, exist_ok=True)


def load_taxonomy(path):
    records = load_json(path)
    if not isinstance(records, list):
        raise ValueError(f"taxonomy must be list: {path}")
    normalized = []
    for idx, rec in enumerate(records):
        if not isinstance(rec, dict):
            raise ValueError(f"taxonomy record at index {idx} must be object")
        for key in ("class_id", "fine", "crop", "disease"):
            if key not in rec:
                raise ValueError(f"taxonomy record at index {idx} missing key: {key}")
        class_id = int(rec["class_id"])
        fine = str(rec["fine"]).strip()
        crop = str(rec["crop"]).strip()
        disease = str(rec["disease"]).strip()
        if not fine or not crop or not disease:
            raise ValueError(f"taxonomy class_id {class_id} has empty fine/crop/disease")
        normalized.append(
            {
                "class_id": class_id,
                "fine": fine,
                "crop": crop,
                "disease": disease,
            }
        )
    normalized.sort(key=lambda x: x["class_id"])
    for expected, rec in enumerate(normalized, start=1):
        if rec["class_id"] != expected:
            raise ValueError(
                f"taxonomy class_id must be contiguous from 1, got {rec['class_id']} at position {expected}"
            )
    return normalized


def build_eval_maps(taxonomy_records):
    crop_names = []
    disease_names = []
    crop_to_id = {}
    disease_to_id = {}
    fine_to_crop = np.zeros(len(taxonomy_records) + 1, dtype=np.int64)
    fine_to_disease = np.zeros(len(taxonomy_records) + 1, dtype=np.int64)

    for rec in taxonomy_records:
        cid = rec["class_id"]
        crop_name = rec["crop"]
        disease_name = rec["disease"]
        if crop_name not in crop_to_id:
            crop_to_id[crop_name] = len(crop_names) + 1
            crop_names.append(crop_name)
        if disease_name not in disease_to_id:
            disease_to_id[disease_name] = len(disease_names) + 1
            disease_names.append(disease_name)
        fine_to_crop[cid] = crop_to_id[crop_name]
        fine_to_disease[cid] = disease_to_id[disease_name]
    return fine_to_crop, fine_to_disease, crop_names, disease_names


def compute_confusion(gt, pred, num_classes):
    mask = (gt >= 0) & (gt < num_classes)
    gt = gt[mask].astype(np.int64)
    pred = pred[mask].astype(np.int64)
    idx = gt * num_classes + pred
    cm = np.bincount(idx, minlength=num_classes * num_classes)
    return cm.reshape(num_classes, num_classes)


def per_class_iou(cm, start_id=1):
    ious = []
    per = []
    for k in range(start_id, cm.shape[0]):
        inter = cm[k, k]
        union = cm[k, :].sum() + cm[:, k].sum() - inter
        if union > 0:
            iou = inter / union
            ious.append(iou)
        else:
            iou = None
        per.append((k, iou, int(inter), int(union)))
    miou = sum(ious) / len(ious) if ious else 0.0
    return miou, per


def main():
    parser = argparse.ArgumentParser(description="Crop/Disease aggregated evaluation")
    parser.add_argument("--gt-dir", required=True, help="GT mask directory")
    parser.add_argument("--pred-dir", required=True, help="Pred mask directory")
    parser.add_argument(
        "--taxonomy-path",
        default=str(DEFAULT_TAXONOMY_PATH),
        help=(
            "Path to taxonomy.json. Defaults to the WheatField-Seg taxonomy file "
            "resolved from TAXOSEG_DATASETS, or the repository data directory if "
            "TAXOSEG_DATASETS is unset."
        ),
    )
    parser.add_argument("--out-dir", required=True, help="Output directory")
    parser.add_argument("--ignore-label", type=int, default=255)
    args = parser.parse_args()

    gt_dir = Path(args.gt_dir)
    pred_dir = Path(args.pred_dir)
    taxonomy_path = Path(args.taxonomy_path)
    out_dir = Path(args.out_dir)
    ensure_dir(out_dir)

    taxonomy_records = load_taxonomy(taxonomy_path)
    crop_map, disease_map, crop_names, disease_names = build_eval_maps(taxonomy_records)

    num_crops = len(crop_names) + 1
    num_diseases = len(disease_names) + 1

    cm_crop = np.zeros((num_crops, num_crops), dtype=np.int64)
    cm_disease = np.zeros((num_diseases, num_diseases), dtype=np.int64)

    gt_files = sorted(gt_dir.glob("*.png"))
    if not gt_files:
        raise ValueError(f"no gt png in {gt_dir}")

    for gt_path in gt_files:
        pred_path = pred_dir / gt_path.name
        if not pred_path.exists():
            continue
        gt = np.array(Image.open(gt_path))
        pred = np.array(Image.open(pred_path))
        if gt.shape != pred.shape:
            raise ValueError(f"shape mismatch: {gt_path.name} {gt.shape} vs {pred.shape}")

        if args.ignore_label is not None:
            valid = gt != args.ignore_label
            gt = np.where(valid, gt, 0)
            pred = np.where(valid, pred, 0)

        gt_crop = crop_map[gt]
        pred_crop = crop_map[pred]
        gt_dis = disease_map[gt]
        pred_dis = disease_map[pred]

        cm_crop += compute_confusion(gt_crop, pred_crop, num_crops)
        cm_disease += compute_confusion(gt_dis, pred_dis, num_diseases)

    crop_miou, per_crop = per_class_iou(cm_crop, start_id=1)
    disease_miou, per_dis = per_class_iou(cm_disease, start_id=1)

    # accuracy on GT foreground
    fg_crop = cm_crop[1:, 1:].sum()
    fg_crop_correct = np.trace(cm_crop[1:, 1:])
    crop_acc = fg_crop_correct / fg_crop if fg_crop > 0 else 0.0

    fg_dis = cm_disease[1:, 1:].sum()
    fg_dis_correct = np.trace(cm_disease[1:, 1:])
    disease_acc = fg_dis_correct / fg_dis if fg_dis > 0 else 0.0

    summary = {
        "pixel": {
            "crop_miou": crop_miou,
            "disease_miou": disease_miou,
            "crop_accuracy": crop_acc,
            "disease_accuracy": disease_acc,
        }
    }
    (out_dir / "eval_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=True), encoding="utf-8"
    )

    # write per-crop iou
    with open(out_dir / "eval_per_crop_iou.csv", "w", newline="", encoding="utf-8") as f:
        f.write("crop_id,crop_name,iou,inter,union\n")
        for cid, iou, inter, union in per_crop:
            name = crop_names[cid - 1] if 0 < cid <= len(crop_names) else str(cid)
            f.write(f"{cid},{name},{'' if iou is None else f'{iou:.6f}'},{inter},{union}\n")

    with open(out_dir / "eval_per_disease_iou.csv", "w", newline="", encoding="utf-8") as f:
        f.write("disease_id,disease_name,iou,inter,union\n")
        for did, iou, inter, union in per_dis:
            name = disease_names[did - 1] if 0 < did <= len(disease_names) else str(did)
            f.write(f"{did},{name},{'' if iou is None else f'{iou:.6f}'},{inter},{union}\n")

    print("crop_miou", f"{crop_miou:.6f}")
    print("disease_miou", f"{disease_miou:.6f}")
    print("crop_accuracy", f"{crop_acc:.6f}")
    print("disease_accuracy", f"{disease_acc:.6f}")
    print(f"saved to {out_dir}")


if __name__ == "__main__":
    main()
