"""
This file may have been modified by Bytedance Ltd. and/or its affiliates (“Bytedance's Modifications”).
All Bytedance's Modifications are Copyright (year) Bytedance Ltd. and/or its affiliates.

Reference: https://github.com/facebookresearch/Mask2Former/blob/main/mask2former/maskformer_model.py
"""
import json
from pathlib import Path
from typing import Tuple

import torch
from torch import nn
from torch.nn import functional as F

from detectron2.config import configurable
from detectron2.data import MetadataCatalog
from detectron2.modeling import META_ARCH_REGISTRY, build_backbone, build_sem_seg_head
from detectron2.modeling.backbone import Backbone
from detectron2.modeling.postprocessing import sem_seg_postprocess
from detectron2.structures import Boxes, ImageList, Instances, BitMasks
from detectron2.utils.comm import get_world_size
from detectron2.utils.events import get_event_storage
from detectron2.utils.memory import retry_if_cuda_oom

from .modeling.criterion import SetCriterion
from .modeling.matcher import HungarianMatcher
from .utils.misc import is_dist_avail_and_initialized


from .modeling.transformer_decoder.fcclip_transformer_decoder import MaskPooling, get_classification_logits
VILD_PROMPT = [
    "a photo of a plant with {}.",
    "a close-up photo of a plant with {}.",
    "a photo of a crop affected by {}.",
    "a photo of {} symptoms on a plant.",
    "a close-up of {} symptoms on plant tissue.",
    "a photo of diseased plant tissue showing {}.",
    "a photo of a plant leaf with {}.",
    "a close-up photo of a leaf with {}.",
    "a photo of a plant fruit with {}.",
    "a close-up photo of fruit lesions caused by {}.",
]

TAXONOMY_FIELDS = (
    "class_id",
    "fine",
    "crop",
    "disease",
    "symptom",
    "organ",
    "form",
    "shape",
    "color",
    "texture",
)


def _normalize_label(label: str) -> str:
    return " ".join(str(label).strip().lower().split())


def _normalize_taxonomy_value(value) -> str:
    text = "" if value is None else str(value).strip()
    if text.lower() in {"", "nan", "none", "null"}:
        return ""
    return text


def _compose_structured_fine_name(record: dict) -> str:
    fine = record["fine"]
    crop = record["crop"]
    disease = record["disease"]
    symptom = record["symptom"]
    organ = record["organ"]
    form = record["form"]
    shape = record["shape"]
    color = record["color"]
    texture = record["texture"]

    clauses = []
    disease_lower = disease.lower()
    disease_phrase = (
        disease if "disease" in disease_lower else f"{disease} disease"
    )
    if crop and disease:
        clauses.append(f"{disease_phrase} on {crop}")
    elif disease:
        clauses.append(disease_phrase)
    elif crop:
        clauses.append(f"disease on {crop}")
    if organ:
        clauses.append(f"affecting {organ}")
    if symptom:
        clauses.append(f"with {symptom} symptoms")
    if form:
        clauses.append(f"lesion form {form}")
    if shape:
        clauses.append(f"{shape} shape")
    if color:
        clauses.append(f"{color} color")
    if texture:
        clauses.append(f"{texture} texture")
    if not clauses:
        return fine
    return f"{fine}, " + ", ".join(clauses)


@META_ARCH_REGISTRY.register()
class FCCLIP(nn.Module):
    """
    Main class for mask classification semantic segmentation architectures.
    """

    @configurable
    def __init__(
        self,
        *,
        backbone: Backbone,
        sem_seg_head: nn.Module,
        criterion: nn.Module,
        num_queries: int,
        object_mask_threshold: float,
        overlap_threshold: float,
        train_metadata,
        test_metadata,
        size_divisibility: int,
        sem_seg_postprocess_before_inference: bool,
        pixel_mean: Tuple[float],
        pixel_std: Tuple[float],
        # inference
        semantic_on: bool,
        panoptic_on: bool,
        instance_on: bool,
        test_topk_per_image: int,
        # Model arguments
        geometric_ensemble_alpha: float,
        geometric_ensemble_beta: float,
        ensemble_on_valid_mask: bool,
        global_crop_weight: float,
        global_crop_warmup_iters: int,
        global_crop_warmup_off_iters: int,
        hier_crop_weight: float,
        hier_disease_weight: float,
        a2f_hkd_on: bool,
        a2f_hkd_weight: float,
        a2f_hkd_warmup_iters: int,
        a2f_hkd_alpha: float,
        a2f_hkd_beta: float,
        a2f_hkd_temp: float,
        a2f_hkd_fusion: str,
        a2f_hkd_eps: float,
        a2f_hkd_crop_conf_tau: float,
        a2f_hkd_disease_conf_tau: float,
        a2f_hkd_fine_margin_tau: float,
        fine_prompt_fusion_on: bool,
        fine_short_weight: float,
        fine_long_weight: float,
        fine_long_warmup_iters: int,
        mask_adapter_on: bool,
        mask_adapter_mask_threshold: float,
        mask_adapter_reg_weight: float,
        mask_adapter_reg_warmup_iters: int,
        mask_adapter_reg_decay_iters: int,
    ):
        """
        Args:
            backbone: a backbone module, must follow detectron2's backbone interface
            sem_seg_head: a module that predicts semantic segmentation from backbone features
            criterion: a module that defines the loss
            num_queries: int, number of queries
            object_mask_threshold: float, threshold to filter query based on classification score
                for panoptic segmentation inference
            overlap_threshold: overlap threshold used in general inference for panoptic segmentation
            metadata: dataset meta, get `thing` and `stuff` category names for panoptic
                segmentation inference
            size_divisibility: Some backbones require the input height and width to be divisible by a
                specific integer. We can use this to override such requirement.
            sem_seg_postprocess_before_inference: whether to resize the prediction back
                to original input size before semantic segmentation inference or after.
                For high-resolution datasets, resizing predictions before inference
                may cause OOM error.
            pixel_mean, pixel_std: list or tuple with #channels element, representing
                the per-channel mean and std to be used to normalize the input image
            semantic_on: bool, whether to output semantic segmentation prediction
            instance_on: bool, whether to output instance segmentation prediction
            panoptic_on: bool, whether to output panoptic segmentation prediction
            test_topk_per_image: int, instance segmentation parameter, keep topk instances per image
        """
        super().__init__()
        self.backbone = backbone
        self.sem_seg_head = sem_seg_head
        self.criterion = criterion
        self.num_queries = num_queries
        self.overlap_threshold = overlap_threshold
        self.object_mask_threshold = object_mask_threshold
        self.train_metadata = train_metadata
        self.test_metadata = test_metadata
        if size_divisibility < 0:
            # use backbone size_divisibility if not set
            size_divisibility = self.backbone.size_divisibility
        self.size_divisibility = size_divisibility
        self.sem_seg_postprocess_before_inference = sem_seg_postprocess_before_inference
        self.register_buffer("pixel_mean", torch.Tensor(pixel_mean).view(-1, 1, 1), False)
        self.register_buffer("pixel_std", torch.Tensor(pixel_std).view(-1, 1, 1), False)

        # additional args
        self.semantic_on = semantic_on
        self.instance_on = instance_on
        self.panoptic_on = panoptic_on
        self.test_topk_per_image = test_topk_per_image

        if not self.semantic_on:
            assert self.sem_seg_postprocess_before_inference

        # Model-specific args
        self.mask_pooling = MaskPooling()
        self.geometric_ensemble_alpha = geometric_ensemble_alpha
        self.geometric_ensemble_beta = geometric_ensemble_beta
        self.ensemble_on_valid_mask = ensemble_on_valid_mask
        self.global_crop_weight = global_crop_weight
        self.global_crop_warmup_iters = global_crop_warmup_iters
        self.global_crop_warmup_off_iters = global_crop_warmup_off_iters

        self.fine_prompt_fusion_on = fine_prompt_fusion_on
        self.fine_short_weight = fine_short_weight
        self.fine_long_weight = fine_long_weight
        self.fine_long_warmup_iters = fine_long_warmup_iters

        self.train_text_classifier_short = None
        self.test_text_classifier_short = None
        self.train_text_classifier_long = None
        self.test_text_classifier_long = None
        self.void_embedding = nn.Embedding(1, backbone.dim_latent) # use this for void
        self.fine_long_name_by_label = {}

        self.hierarchical_enabled = False
        self.enable_crop_prior_gating = False
        self.hier_crop_weight = hier_crop_weight
        self.hier_disease_weight = hier_disease_weight
        self.a2f_hkd_on = a2f_hkd_on
        self.a2f_hkd_weight = a2f_hkd_weight
        self.a2f_hkd_warmup_iters = a2f_hkd_warmup_iters
        self.a2f_hkd_alpha = a2f_hkd_alpha
        self.a2f_hkd_beta = a2f_hkd_beta
        self.a2f_hkd_temp = a2f_hkd_temp
        self.a2f_hkd_fusion = a2f_hkd_fusion
        self.a2f_hkd_eps = a2f_hkd_eps
        self.a2f_hkd_crop_conf_tau = a2f_hkd_crop_conf_tau
        self.a2f_hkd_disease_conf_tau = a2f_hkd_disease_conf_tau
        self.a2f_hkd_fine_margin_tau = a2f_hkd_fine_margin_tau
        self.mask_adapter_on = mask_adapter_on
        self.mask_adapter_mask_threshold = mask_adapter_mask_threshold
        self.mask_adapter_reg_weight = mask_adapter_reg_weight
        self.mask_adapter_reg_warmup_iters = mask_adapter_reg_warmup_iters
        self.mask_adapter_reg_decay_iters = mask_adapter_reg_decay_iters
        self.crop_text_classifier = None
        self.disease_text_classifier = None
        self.crop_num_templates = None
        self.disease_num_templates = None
        self.crop_class_names = None
        self.disease_class_names = None
        self.hier_taxonomy_path = None
        self.register_buffer("fine_to_crop", torch.empty(0, dtype=torch.long), persistent=False)
        self.register_buffer("fine_to_disease", torch.empty(0, dtype=torch.long), persistent=False)
        self.register_buffer("fine_to_crop_mat", torch.empty(0), persistent=False)
        self.register_buffer("fine_to_disease_mat", torch.empty(0), persistent=False)
        self.register_buffer("crop_to_fine_mat", torch.empty(0), persistent=False)
        self.register_buffer("disease_to_fine_mat", torch.empty(0), persistent=False)
        taxonomy_path = self._resolve_hier_taxonomy_path(train_metadata)
        if taxonomy_path is not None:
            self.hierarchical_enabled = True
            self._load_hier_taxonomy(taxonomy_path)
            self._validate_hier_taxonomy_num_classes(sem_seg_head.num_classes, taxonomy_path)
            self.hier_taxonomy_path = str(taxonomy_path)

        _, self.train_num_templates_short, self.train_class_names_short = self.prepare_class_names_from_metadata(
            train_metadata, train_metadata, use_long=False
        )
        _, self.train_num_templates_long, self.train_class_names_long = self.prepare_class_names_from_metadata(
            train_metadata, train_metadata, use_long=True
        )
        self.category_overlapping_mask, self.test_num_templates_short, self.test_class_names_short = self.prepare_class_names_from_metadata(
            test_metadata, train_metadata, use_long=False
        )
        _, self.test_num_templates_long, self.test_class_names_long = self.prepare_class_names_from_metadata(
            test_metadata, train_metadata, use_long=True
        )

    def prepare_class_names_from_metadata(self, metadata, train_metadata, use_long: bool = False):
        def split_labels(x):
            res = []
            for x_ in x:
                x_ = x_.replace(', ', ',')
                x_ = x_.split(',') # there can be multiple synonyms for single class
                res.append(x_)
            return res
        # get text classifier
        try:
            class_names = split_labels(metadata.stuff_classes) # it includes both thing and stuff
            train_class_names = split_labels(train_metadata.stuff_classes)
        except:
            # this could be for insseg, where only thing_classes are available
            class_names = split_labels(metadata.thing_classes)
            train_class_names = split_labels(train_metadata.thing_classes)
        train_class_names = {l for label in train_class_names for l in label}
        category_overlapping_list = []
        for test_class_names in class_names:
            is_overlapping = not set(train_class_names).isdisjoint(set(test_class_names))
            category_overlapping_list.append(is_overlapping)
        category_overlapping_mask = torch.tensor(
            category_overlapping_list, dtype=torch.long)

        def map_fine_labels(labels):
            if not use_long or not self.fine_long_name_by_label:
                return labels
            mapped = []
            seen = set()
            for label in labels:
                raw_label = label.strip()
                if not raw_label:
                    continue
                long_label = self.fine_long_name_by_label.get(_normalize_label(raw_label), raw_label)
                if long_label in seen:
                    continue
                mapped.append(long_label)
                seen.add(long_label)
            return mapped if mapped else labels

        def fill_all_templates_ensemble(x_=''):
            res = []
            for x in x_:
                for template in VILD_PROMPT:
                    res.append(template.format(x))
            return res, len(res) // len(VILD_PROMPT)

        num_templates = []
        templated_class_names = []
        for x in class_names:
            templated_classes, templated_classes_num = fill_all_templates_ensemble(map_fine_labels(x))
            templated_class_names += templated_classes
            num_templates.append(templated_classes_num) # how many templates for current classes
        class_names = templated_class_names
        return category_overlapping_mask, num_templates, class_names

    def prepare_class_names_from_list(self, class_names):
        def split_labels(x):
            res = []
            for x_ in x:
                x_ = x_.replace(', ', ',')
                x_ = x_.split(',')
                res.append(x_)
            return res

        def fill_all_templates_ensemble(x_=''):
            res = []
            for x in x_:
                for template in VILD_PROMPT:
                    res.append(template.format(x))
            return res, len(res) // len(VILD_PROMPT)

        class_names = split_labels(class_names)
        num_templates = []
        templated_class_names = []
        for x in class_names:
            templated_classes, templated_classes_num = fill_all_templates_ensemble(x)
            templated_class_names += templated_classes
            num_templates.append(templated_classes_num)
        return num_templates, templated_class_names

    def _resolve_hier_taxonomy_path(self, metadata):
        taxonomy_path = getattr(metadata, "taxonomy_path", None)
        if taxonomy_path:
            return Path(taxonomy_path)

        for attr_name in ("dataset_root", "root", "image_root", "sem_seg_root"):
            root = getattr(metadata, attr_name, None)
            if not root:
                continue
            root = Path(root)
            for candidate_root in (root, *list(root.parents)[:4]):
                candidate = candidate_root / "mappings" / "taxonomy.json"
                if candidate.exists():
                    return candidate

        return None

    def _validate_hier_taxonomy_num_classes(self, num_classes, taxonomy_path: Path):
        expected_fine_count = int(num_classes) - 1
        actual_fine_count = int(self.fine_to_crop.shape[0]) - 1
        if actual_fine_count != expected_fine_count:
            raise ValueError(
                "Hierarchical taxonomy class count mismatch: "
                f"{taxonomy_path} has {actual_fine_count} foreground classes, "
                f"but MODEL.SEM_SEG_HEAD.NUM_CLASSES={int(num_classes)} expects "
                f"{expected_fine_count} foreground classes."
            )

    def _load_hier_taxonomy(self, taxonomy_path: Path):
        if not taxonomy_path.exists():
            raise FileNotFoundError(f"Hierarchical taxonomy not found: {taxonomy_path}")
        taxonomy_records = json.loads(taxonomy_path.read_text(encoding="utf-8"))
        if not isinstance(taxonomy_records, list):
            raise ValueError(f"taxonomy must be a list: {taxonomy_path}")

        normalized_records = []
        for idx, record in enumerate(taxonomy_records):
            if not isinstance(record, dict):
                raise ValueError(f"taxonomy record at index {idx} must be an object")
            required_keys = ("class_id", "fine", "crop", "disease")
            missing_keys = [key for key in required_keys if key not in record]
            if missing_keys:
                raise ValueError(
                    f"taxonomy record at index {idx} missing keys: {missing_keys}"
                )
            try:
                class_id = int(record["class_id"])
            except (TypeError, ValueError) as exc:
                raise ValueError(
                    f"taxonomy record at index {idx} has invalid class_id: {record['class_id']}"
                ) from exc
            normalized_record = {"class_id": class_id}
            for key in TAXONOMY_FIELDS:
                if key == "class_id":
                    continue
                normalized_record[key] = _normalize_taxonomy_value(record.get(key, ""))
            normalized_records.append(normalized_record)

        normalized_records.sort(key=lambda x: x["class_id"])
        for expected_id, record in enumerate(normalized_records, start=1):
            if record["class_id"] != expected_id:
                raise ValueError(
                    f"taxonomy class_id must be contiguous from 1, got {record['class_id']} at position {expected_id}"
                )
            if not record["fine"]:
                raise ValueError(f"taxonomy class_id {record['class_id']} has empty fine")
            if not record["crop"]:
                raise ValueError(f"taxonomy class_id {record['class_id']} has empty crop")
            if not record["disease"]:
                raise ValueError(f"taxonomy class_id {record['class_id']} has empty disease")

        crop_names = []
        disease_names = []
        crop_to_idx = {}
        disease_to_idx = {}
        fine_to_crop = [-1] * (len(normalized_records) + 1)
        fine_to_disease = [-1] * (len(normalized_records) + 1)
        fine_long_name_by_label = {}

        for record in normalized_records:
            class_id = record["class_id"]
            fine_name = record["fine"]
            crop_name = record["crop"]
            disease_name = record["disease"]
            if crop_name not in crop_to_idx:
                crop_to_idx[crop_name] = len(crop_names)
                crop_names.append(crop_name)
            if disease_name not in disease_to_idx:
                disease_to_idx[disease_name] = len(disease_names)
                disease_names.append(disease_name)
            fine_to_crop[class_id] = crop_to_idx[crop_name]
            fine_to_disease[class_id] = disease_to_idx[disease_name]
            fine_long_name_by_label[_normalize_label(fine_name)] = _compose_structured_fine_name(record)

        self.fine_long_name_by_label = fine_long_name_by_label
        self.crop_num_templates, self.crop_class_names = self.prepare_class_names_from_list(crop_names)
        self.disease_num_templates, self.disease_class_names = self.prepare_class_names_from_list(disease_names)

        fine_to_crop = torch.tensor(fine_to_crop, dtype=torch.long)
        fine_to_disease = torch.tensor(fine_to_disease, dtype=torch.long)
        self.fine_to_crop = fine_to_crop
        self.fine_to_disease = fine_to_disease

        fine_count = fine_to_crop.shape[0] - 1
        crop_count = len(crop_names)
        disease_count = len(disease_names)
        fine_to_crop_mat = torch.zeros((fine_count, crop_count), dtype=torch.float)
        fine_to_disease_mat = torch.zeros((fine_count, disease_count), dtype=torch.float)
        for fine_id in range(1, fine_to_crop.shape[0]):
            crop_id = fine_to_crop[fine_id].item()
            disease_id = fine_to_disease[fine_id].item()
            if crop_id >= 0:
                fine_to_crop_mat[fine_id - 1, crop_id] = 1.0
            if disease_id >= 0:
                fine_to_disease_mat[fine_id - 1, disease_id] = 1.0
        self.fine_to_crop_mat = fine_to_crop_mat
        self.fine_to_disease_mat = fine_to_disease_mat

        # Aux->Fine projection matrices for A2F-HKD.
        crop_to_fine_mat = fine_to_crop_mat.transpose(0, 1).contiguous()
        disease_to_fine_mat = fine_to_disease_mat.transpose(0, 1).contiguous()
        crop_to_fine_mat = crop_to_fine_mat / (crop_to_fine_mat.sum(dim=-1, keepdim=True) + 1e-6)
        disease_to_fine_mat = disease_to_fine_mat / (disease_to_fine_mat.sum(dim=-1, keepdim=True) + 1e-6)
        self.crop_to_fine_mat = crop_to_fine_mat
        self.disease_to_fine_mat = disease_to_fine_mat

    def set_metadata(self, metadata):
        self.test_metadata = metadata
        (
            self.category_overlapping_mask,
            self.test_num_templates_short,
            self.test_class_names_short,
        ) = self.prepare_class_names_from_metadata(metadata, self.train_metadata, use_long=False)
        _, self.test_num_templates_long, self.test_class_names_long = self.prepare_class_names_from_metadata(
            metadata, self.train_metadata, use_long=True
        )
        self.test_text_classifier_short = None
        self.test_text_classifier_long = None
        return

    def _encode_text_classifier(self, class_names):
        text_classifier = []
        # this is needed to avoid oom, which may happen when num of class is large
        bs = 128
        for idx in range(0, len(class_names), bs):
            text_classifier.append(
                self.backbone.get_text_classifier(class_names[idx:idx + bs], self.device).detach()
            )
        text_classifier = torch.cat(text_classifier, dim=0)
        text_classifier /= text_classifier.norm(dim=-1, keepdim=True)
        text_classifier = text_classifier.reshape(
            text_classifier.shape[0] // len(VILD_PROMPT),
            len(VILD_PROMPT),
            text_classifier.shape[-1],
        ).mean(1)
        text_classifier /= text_classifier.norm(dim=-1, keepdim=True)
        return text_classifier

    def get_text_classifier_short(self):
        if self.training:
            if self.train_text_classifier_short is None:
                self.train_text_classifier_short = self._encode_text_classifier(
                    self.train_class_names_short
                )
            return self.train_text_classifier_short, self.train_num_templates_short
        else:
            if self.test_text_classifier_short is None:
                self.test_text_classifier_short = self._encode_text_classifier(
                    self.test_class_names_short
                )
            return self.test_text_classifier_short, self.test_num_templates_short

    def get_text_classifier_long(self):
        if self.training:
            if self.train_text_classifier_long is None:
                self.train_text_classifier_long = self._encode_text_classifier(
                    self.train_class_names_long
                )
            return self.train_text_classifier_long, self.train_num_templates_long
        else:
            if self.test_text_classifier_long is None:
                self.test_text_classifier_long = self._encode_text_classifier(
                    self.test_class_names_long
                )
            return self.test_text_classifier_long, self.test_num_templates_long

    def _get_fine_prompt_fusion_weights(self):
        short_weight = max(float(self.fine_short_weight), 0.0)
        long_weight = max(float(self.fine_long_weight), 0.0)
        if self.training:
            warmup_iters = int(self.fine_long_warmup_iters)
            if warmup_iters > 0:
                cur_iter = get_event_storage().iter
                scale = min(max(float(cur_iter) / float(warmup_iters), 0.0), 1.0)
                long_weight = long_weight * scale
        total = short_weight + long_weight
        if total <= 0.0:
            return 1.0, 0.0
        return short_weight / total, long_weight / total

    def _fuse_fine_logits(self, logits_short, logits_long):
        if logits_long is None:
            return logits_short
        short_weight, long_weight = self._get_fine_prompt_fusion_weights()
        if long_weight <= 0.0:
            return logits_short
        return logits_short * short_weight + logits_long * long_weight

    def _fuse_semseg_outputs(self, outputs):
        if "pred_logits_long" not in outputs:
            return outputs
        fused_outputs = dict(outputs)
        fused_outputs["pred_logits"] = self._fuse_fine_logits(
            outputs["pred_logits"], outputs["pred_logits_long"]
        )
        aux_short = outputs.get("aux_outputs")
        aux_long = outputs.get("aux_outputs_long")
        if aux_short is not None and aux_long is not None:
            fused_aux = []
            for aux_out_short, aux_out_long in zip(aux_short, aux_long):
                aux_out = dict(aux_out_short)
                aux_out["pred_logits"] = self._fuse_fine_logits(
                    aux_out_short["pred_logits"], aux_out_long["pred_logits"]
                )
                fused_aux.append(aux_out)
            fused_outputs["aux_outputs"] = fused_aux
        fused_outputs.pop("pred_logits_long", None)
        fused_outputs.pop("aux_outputs_long", None)
        return fused_outputs

    def get_crop_disease_text_classifier(self):
        if self.crop_text_classifier is None or self.disease_text_classifier is None:
            if self.crop_class_names is None or self.disease_class_names is None:
                raise RuntimeError("Hierarchical class names are not initialized.")

            self.crop_text_classifier = self._encode_text_classifier(self.crop_class_names)
            self.disease_text_classifier = self._encode_text_classifier(self.disease_class_names)
        return (
            self.crop_text_classifier,
            self.crop_num_templates,
            self.disease_text_classifier,
            self.disease_num_templates,
        )

    @classmethod
    def from_config(cls, cfg):
        backbone = build_backbone(cfg)
        sem_seg_head = build_sem_seg_head(cfg, backbone.output_shape())

        # Loss parameters:
        deep_supervision = cfg.MODEL.MASK_FORMER.DEEP_SUPERVISION
        no_object_weight = cfg.MODEL.MASK_FORMER.NO_OBJECT_WEIGHT

        # loss weights
        class_weight = cfg.MODEL.MASK_FORMER.CLASS_WEIGHT
        dice_weight = cfg.MODEL.MASK_FORMER.DICE_WEIGHT
        mask_weight = cfg.MODEL.MASK_FORMER.MASK_WEIGHT

        # building criterion
        matcher = HungarianMatcher(
            cost_class=class_weight,
            cost_mask=mask_weight,
            cost_dice=dice_weight,
            num_points=cfg.MODEL.MASK_FORMER.TRAIN_NUM_POINTS,
        )

        weight_dict = {"loss_ce": class_weight, "loss_mask": mask_weight, "loss_dice": dice_weight}

        if deep_supervision:
            dec_layers = cfg.MODEL.MASK_FORMER.DEC_LAYERS
            aux_weight_dict = {}
            for i in range(dec_layers - 1):
                aux_weight_dict.update({k + f"_{i}": v for k, v in weight_dict.items()})
            weight_dict.update(aux_weight_dict)

        losses = ["labels", "masks"]

        criterion = SetCriterion(
            sem_seg_head.num_classes,
            matcher=matcher,
            weight_dict=weight_dict,
            eos_coef=no_object_weight,
            losses=losses,
            num_points=cfg.MODEL.MASK_FORMER.TRAIN_NUM_POINTS,
            oversample_ratio=cfg.MODEL.MASK_FORMER.OVERSAMPLE_RATIO,
            importance_sample_ratio=cfg.MODEL.MASK_FORMER.IMPORTANCE_SAMPLE_RATIO,
        )

        return {
            "backbone": backbone,
            "sem_seg_head": sem_seg_head,
            "criterion": criterion,
            "num_queries": cfg.MODEL.MASK_FORMER.NUM_OBJECT_QUERIES,
            "object_mask_threshold": cfg.MODEL.MASK_FORMER.TEST.OBJECT_MASK_THRESHOLD,
            "overlap_threshold": cfg.MODEL.MASK_FORMER.TEST.OVERLAP_THRESHOLD,
            "train_metadata": MetadataCatalog.get(cfg.DATASETS.TRAIN[0]),
            "test_metadata": MetadataCatalog.get(cfg.DATASETS.TEST[0]),
            "size_divisibility": cfg.MODEL.MASK_FORMER.SIZE_DIVISIBILITY,
            "sem_seg_postprocess_before_inference": (
                cfg.MODEL.MASK_FORMER.TEST.SEM_SEG_POSTPROCESSING_BEFORE_INFERENCE
                or cfg.MODEL.MASK_FORMER.TEST.PANOPTIC_ON
                or cfg.MODEL.MASK_FORMER.TEST.INSTANCE_ON
            ),
            "pixel_mean": cfg.MODEL.PIXEL_MEAN,
            "pixel_std": cfg.MODEL.PIXEL_STD,
            # inference
            "semantic_on": cfg.MODEL.MASK_FORMER.TEST.SEMANTIC_ON,
            "instance_on": cfg.MODEL.MASK_FORMER.TEST.INSTANCE_ON,
            "panoptic_on": cfg.MODEL.MASK_FORMER.TEST.PANOPTIC_ON,
            "test_topk_per_image": cfg.TEST.DETECTIONS_PER_IMAGE,
            "geometric_ensemble_alpha": cfg.MODEL.FC_CLIP.GEOMETRIC_ENSEMBLE_ALPHA,
            "geometric_ensemble_beta": cfg.MODEL.FC_CLIP.GEOMETRIC_ENSEMBLE_BETA,
            "ensemble_on_valid_mask": cfg.MODEL.FC_CLIP.ENSEMBLE_ON_VALID_MASK,
            "global_crop_weight": cfg.MODEL.FC_CLIP.GLOBAL_CROP_WEIGHT,
            "global_crop_warmup_iters": cfg.MODEL.FC_CLIP.GLOBAL_CROP_WARMUP_ITERS,
            "global_crop_warmup_off_iters": cfg.MODEL.FC_CLIP.GLOBAL_CROP_WARMUP_OFF_ITERS,
            "hier_crop_weight": cfg.MODEL.FC_CLIP.HIER_CROP_WEIGHT,
            "hier_disease_weight": cfg.MODEL.FC_CLIP.HIER_DISEASE_WEIGHT,
            "a2f_hkd_on": cfg.MODEL.FC_CLIP.A2F_HKD_ON,
            "a2f_hkd_weight": cfg.MODEL.FC_CLIP.A2F_HKD_WEIGHT,
            "a2f_hkd_warmup_iters": cfg.MODEL.FC_CLIP.A2F_HKD_WARMUP_ITERS,
            "a2f_hkd_alpha": cfg.MODEL.FC_CLIP.A2F_HKD_ALPHA,
            "a2f_hkd_beta": cfg.MODEL.FC_CLIP.A2F_HKD_BETA,
            "a2f_hkd_temp": cfg.MODEL.FC_CLIP.A2F_HKD_TEMP,
            "a2f_hkd_fusion": cfg.MODEL.FC_CLIP.A2F_HKD_FUSION,
            "a2f_hkd_eps": cfg.MODEL.FC_CLIP.A2F_HKD_EPS,
            "a2f_hkd_crop_conf_tau": cfg.MODEL.FC_CLIP.A2F_HKD_CROP_CONF_TAU,
            "a2f_hkd_disease_conf_tau": cfg.MODEL.FC_CLIP.A2F_HKD_DISEASE_CONF_TAU,
            "a2f_hkd_fine_margin_tau": cfg.MODEL.FC_CLIP.A2F_HKD_FINE_MARGIN_TAU,
            "fine_prompt_fusion_on": cfg.MODEL.FC_CLIP.FINE_PROMPT_FUSION_ON,
            "fine_short_weight": cfg.MODEL.FC_CLIP.FINE_SHORT_WEIGHT,
            "fine_long_weight": cfg.MODEL.FC_CLIP.FINE_LONG_WEIGHT,
            "fine_long_warmup_iters": cfg.MODEL.FC_CLIP.FINE_LONG_WARMUP_ITERS,
            "mask_adapter_on": cfg.MODEL.FC_CLIP.MASK_ADAPTER_ON,
            "mask_adapter_mask_threshold": cfg.MODEL.FC_CLIP.MASK_ADAPTER_MASK_THRESHOLD,
            "mask_adapter_reg_weight": cfg.MODEL.FC_CLIP.MASK_ADAPTER_REG_WEIGHT,
            "mask_adapter_reg_warmup_iters": cfg.MODEL.FC_CLIP.MASK_ADAPTER_REG_WARMUP_ITERS,
            "mask_adapter_reg_decay_iters": cfg.MODEL.FC_CLIP.MASK_ADAPTER_REG_DECAY_ITERS,
        }

    @property
    def device(self):
        return self.pixel_mean.device

    def forward(self, batched_inputs):
        """
        Args:
            batched_inputs: a list, batched outputs of :class:`DatasetMapper`.
                Each item in the list contains the inputs for one image.
                For now, each item in the list is a dict that contains:
                   * "image": Tensor, image in (C, H, W) format.
                   * "instances": per-region ground truth
                   * Other information that's included in the original dicts, such as:
                     "height", "width" (int): the output resolution of the model (may be different
                     from input resolution), used in inference.
        Returns:
            list[dict]:
                each dict has the results for one image. The dict contains the following keys:

                * "sem_seg":
                    A Tensor that represents the
                    per-pixel segmentation prediced by the head.
                    The prediction has shape KxHxW that represents the logits of
                    each class for each pixel.
                * "panoptic_seg":
                    A tuple that represent panoptic output
                    panoptic_seg (Tensor): of shape (height, width) where the values are ids for each segment.
                    segments_info (list[dict]): Describe each segment in `panoptic_seg`.
                        Each dict contains keys "id", "category_id", "isthing".
        """
        images = [x["image"].to(self.device) for x in batched_inputs]
        images = [(x - self.pixel_mean) / self.pixel_std for x in images]
        images = ImageList.from_tensors(images, self.size_divisibility)

        features = self.backbone(images.tensor)
        text_classifier_short, num_templates_short = self.get_text_classifier_short()
        text_classifier_short = torch.cat(
            [text_classifier_short, F.normalize(self.void_embedding.weight, dim=-1)], dim=0
        )

        text_classifier_long = None
        num_templates_long = None
        if self.fine_prompt_fusion_on:
            text_classifier_long, num_templates_long = self.get_text_classifier_long()
            text_classifier_long = torch.cat(
                [text_classifier_long, F.normalize(self.void_embedding.weight, dim=-1)], dim=0
            )

        features = dict(features)
        features["text_classifier"] = text_classifier_short
        features["num_templates"] = num_templates_short
        if self.fine_prompt_fusion_on:
            features["text_classifier_long"] = text_classifier_long
            features["num_templates_long"] = num_templates_long
        features["global_crop_on"] = bool(
            self.training and self.hierarchical_enabled and self.global_crop_weight > 0.0
        )
        crop_text_classifier_raw = None
        if self.hierarchical_enabled:
            (
                crop_text_classifier,
                crop_num_templates,
                disease_text_classifier,
                disease_num_templates,
            ) = self.get_crop_disease_text_classifier()
            crop_text_classifier_raw = crop_text_classifier
            crop_text_classifier = torch.cat(
                [crop_text_classifier, F.normalize(self.void_embedding.weight, dim=-1)], dim=0
            )
            disease_text_classifier = torch.cat(
                [disease_text_classifier, F.normalize(self.void_embedding.weight, dim=-1)], dim=0
            )
            features["text_classifier_crop"] = crop_text_classifier
            features["num_templates_crop"] = crop_num_templates
            features["text_classifier_disease"] = disease_text_classifier
            features["num_templates_disease"] = disease_num_templates

        outputs = self.sem_seg_head(features)
        if self.fine_prompt_fusion_on:
            outputs = self._fuse_semseg_outputs(outputs)

        if self.training:
            # mask classification target
            if "instances" in batched_inputs[0]:
                gt_instances = [x["instances"].to(self.device) for x in batched_inputs]
                targets = self.prepare_targets(gt_instances, images)
            else:
                targets = None

            # bipartite matching-based loss
            losses = self.criterion(outputs, targets)
            if self.hierarchical_enabled:
                losses.update(self._compute_hier_losses(outputs, targets))
                losses.update(self._compute_a2f_hkd_loss(outputs, targets))
                losses.update(self._compute_global_crop_loss(outputs, targets))
            losses.update(self._compute_mask_adapter_reg_loss(outputs))

            for k in list(losses.keys()):
                if k in self.criterion.weight_dict:
                    losses[k] *= self.criterion.weight_dict[k]
                elif k == "loss_mask_adapter_reg":
                    continue
                elif self.hierarchical_enabled and k in (
                    "loss_crop_ce",
                    "loss_disease_ce",
                    "loss_a2f_hkd",
                    "loss_global_crop",
                ):
                    continue
                else:
                    # remove this loss if not specified in `weight_dict`
                    losses.pop(k)
            return losses
        else:
            mask_cls_results = outputs["pred_logits"]
            mask_pred_results = outputs["pred_masks"]

            if self.hierarchical_enabled and self.enable_crop_prior_gating:
                with torch.no_grad():
                    global_embed = self.backbone.encode_image(images.tensor, normalize=True)
                    scale = (
                        self.backbone.clip_model.logit_scale.to(global_embed.dtype)
                        .exp()
                        .clamp(max=100)
                    )
                    crop_logits = scale * (global_embed @ crop_text_classifier_raw.to(global_embed.dtype).T)
                    crop_probs = crop_logits.softmax(-1)

                    fine_to_crop = self.fine_to_crop.to(crop_probs.device)
                    crop_prior_fine = crop_probs[:, fine_to_crop[1:]]
                    gate_bias = torch.zeros(
                        (crop_probs.shape[0], fine_to_crop.shape[0]),
                        device=crop_probs.device,
                        dtype=mask_cls_results.dtype,
                    )
                    gate_bias[:, 1:] = torch.log(crop_prior_fine + 1e-6)
                mask_cls_results[..., :-1] = mask_cls_results[..., :-1] + gate_bias.unsqueeze(1)

            # We ensemble the pred logits of in-vocab and out-vocab
            clip_feature = features["clip_vis_dense"]
            mask_for_pooling = F.interpolate(mask_pred_results, size=clip_feature.shape[-2:],
                                                mode='bilinear', align_corners=False)
            if "convnext" in self.backbone.model_name.lower():
                pooled_clip_feature = self.mask_pooling(clip_feature, mask_for_pooling)
                pooled_clip_feature = self.backbone.visual_prediction_forward(pooled_clip_feature)
            elif "rn" in self.backbone.model_name.lower():
                pooled_clip_feature = self.backbone.visual_prediction_forward(clip_feature, mask_for_pooling)
            else:
                raise NotImplementedError

            out_vocab_cls_results_short = get_classification_logits(
                pooled_clip_feature,
                text_classifier_short,
                self.backbone.clip_model.logit_scale,
                num_templates_short,
            )
            out_vocab_cls_results_long = None
            if self.fine_prompt_fusion_on:
                out_vocab_cls_results_long = get_classification_logits(
                    pooled_clip_feature,
                    text_classifier_long,
                    self.backbone.clip_model.logit_scale,
                    num_templates_long,
                )
            out_vocab_cls_results = self._fuse_fine_logits(
                out_vocab_cls_results_short, out_vocab_cls_results_long
            )
            in_vocab_cls_results = mask_cls_results[..., :-1] # remove void
            out_vocab_cls_results = out_vocab_cls_results[..., :-1] # remove void

            # Reference: https://github.com/NVlabs/ODISE/blob/main/odise/modeling/meta_arch/odise.py#L1506
            out_vocab_cls_probs = out_vocab_cls_results.softmax(-1)
            in_vocab_cls_results = in_vocab_cls_results.softmax(-1)
            category_overlapping_mask = self.category_overlapping_mask.to(self.device)

            if self.ensemble_on_valid_mask:
                # Only include out_vocab cls results on masks with valid pixels
                # We empirically find that this is important to obtain reasonable AP/mIOU score with ResNet CLIP models
                valid_masking = (mask_for_pooling > 0).to(mask_for_pooling).sum(-1).sum(-1) > 0
                valid_masking = valid_masking.to(in_vocab_cls_results.dtype).unsqueeze(-1)
                alpha = torch.ones_like(in_vocab_cls_results) * self.geometric_ensemble_alpha
                beta = torch.ones_like(in_vocab_cls_results) * self.geometric_ensemble_beta
                alpha = alpha * valid_masking
                beta = beta * valid_masking
            else:
                alpha = self.geometric_ensemble_alpha
                beta = self.geometric_ensemble_beta

            cls_logits_seen = (
                (in_vocab_cls_results ** (1 - alpha) * out_vocab_cls_probs**alpha).log()
                * category_overlapping_mask
            )
            cls_logits_unseen = (
                (in_vocab_cls_results ** (1 - beta) * out_vocab_cls_probs**beta).log()
                * (1 - category_overlapping_mask)
            )
            cls_results = cls_logits_seen + cls_logits_unseen

            # This is used to filtering void predictions.
            is_void_prob = F.softmax(mask_cls_results, dim=-1)[..., -1:]
            mask_cls_probs = torch.cat([
                cls_results.softmax(-1) * (1.0 - is_void_prob),
                is_void_prob], dim=-1)
            mask_cls_results = torch.log(mask_cls_probs + 1e-8)

            # upsample masks
            mask_pred_results = F.interpolate(
                mask_pred_results,
                size=(images.tensor.shape[-2], images.tensor.shape[-1]),
                mode="bilinear",
                align_corners=False,
            )

            del outputs

            processed_results = []
            for mask_cls_result, mask_pred_result, input_per_image, image_size in zip(
                mask_cls_results, mask_pred_results, batched_inputs, images.image_sizes
            ):
                height = input_per_image.get("height", image_size[0])
                width = input_per_image.get("width", image_size[1])
                processed_results.append({})

                if self.sem_seg_postprocess_before_inference:
                    mask_pred_result = retry_if_cuda_oom(sem_seg_postprocess)(
                        mask_pred_result, image_size, height, width
                    )
                    mask_cls_result = mask_cls_result.to(mask_pred_result)

                # semantic segmentation inference
                if self.semantic_on:
                    r = retry_if_cuda_oom(self.semantic_inference)(mask_cls_result, mask_pred_result)
                    if not self.sem_seg_postprocess_before_inference:
                        r = retry_if_cuda_oom(sem_seg_postprocess)(r, image_size, height, width)
                    processed_results[-1]["sem_seg"] = r

                # panoptic segmentation inference
                if self.panoptic_on:
                    panoptic_r = retry_if_cuda_oom(self.panoptic_inference)(mask_cls_result, mask_pred_result)
                    processed_results[-1]["panoptic_seg"] = panoptic_r

                # instance segmentation inference
                if self.instance_on:
                    instance_r = retry_if_cuda_oom(self.instance_inference)(mask_cls_result, mask_pred_result)
                    processed_results[-1]["instances"] = instance_r

            return processed_results

    def prepare_targets(self, targets, images):
        h_pad, w_pad = images.tensor.shape[-2:]
        new_targets = []
        for targets_per_image in targets:
            # pad gt
            gt_masks = targets_per_image.gt_masks
            padded_masks = torch.zeros((gt_masks.shape[0], h_pad, w_pad), dtype=gt_masks.dtype, device=gt_masks.device)
            padded_masks[:, : gt_masks.shape[1], : gt_masks.shape[2]] = gt_masks
            new_targets.append(
                {
                    "labels": targets_per_image.gt_classes,
                    "masks": padded_masks,
                }
            )
        return new_targets

    def _get_src_permutation_idx(self, indices):
        batch_idx = torch.cat([torch.full_like(src, i) for i, (src, _) in enumerate(indices)])
        src_idx = torch.cat([src for (src, _) in indices])
        return batch_idx, src_idx

    def _compute_hier_losses(self, outputs, targets):
        losses = {}
        if "pred_logits_crop" not in outputs or "pred_logits_disease" not in outputs:
            return losses

        outputs_without_aux = {k: v for k, v in outputs.items() if k != "aux_outputs"}
        indices = self.criterion.matcher(outputs_without_aux, targets)
        src_idx = self._get_src_permutation_idx(indices)

        num_masks = sum(len(t["labels"]) for t in targets)
        num_masks = torch.as_tensor(
            [num_masks], dtype=torch.float, device=next(iter(outputs.values())).device
        )
        if is_dist_avail_and_initialized():
            torch.distributed.all_reduce(num_masks)
        num_masks = torch.clamp(num_masks / get_world_size(), min=1).item()

        target_classes_o = torch.cat([t["labels"][J] for t, (_, J) in zip(targets, indices)])
        if target_classes_o.numel() == 0:
            zero = outputs["pred_logits"].sum() * 0.0
            losses["loss_crop_ce"] = zero
            losses["loss_disease_ce"] = zero
            return losses
        valid_mask = target_classes_o > 0
        if not torch.any(valid_mask):
            zero = outputs["pred_logits"].sum() * 0.0
            losses["loss_crop_ce"] = zero
            losses["loss_disease_ce"] = zero
            return losses

        target_classes_o = target_classes_o[valid_mask]
        target_crop = self.fine_to_crop.to(target_classes_o.device)[target_classes_o]
        target_disease = self.fine_to_disease.to(target_classes_o.device)[target_classes_o]

        pred_crop = outputs["pred_logits_crop"][src_idx][..., :-1][valid_mask]
        pred_disease = outputs["pred_logits_disease"][src_idx][..., :-1][valid_mask]

        loss_crop = F.cross_entropy(pred_crop, target_crop, reduction="none").sum() / num_masks
        loss_disease = F.cross_entropy(pred_disease, target_disease, reduction="none").sum() / num_masks
        losses["loss_crop_ce"] = loss_crop * self.hier_crop_weight
        losses["loss_disease_ce"] = loss_disease * self.hier_disease_weight

        return losses

    def _get_a2f_hkd_weight(self):
        weight = float(self.a2f_hkd_weight)
        if (not self.a2f_hkd_on) or weight <= 0.0:
            return 0.0
        warmup_iters = int(self.a2f_hkd_warmup_iters)
        if warmup_iters <= 0:
            return weight
        cur_iter = get_event_storage().iter
        if cur_iter >= warmup_iters:
            return weight
        if cur_iter <= 0:
            return 0.0
        return weight * float(cur_iter) / float(warmup_iters)

    def _compute_a2f_hkd_loss(self, outputs, targets):
        if (
            "pred_logits" not in outputs
            or "pred_logits_crop" not in outputs
            or "pred_logits_disease" not in outputs
        ):
            return {}

        a2f_weight = self._get_a2f_hkd_weight()
        if a2f_weight <= 0.0:
            return {}

        outputs_without_aux = {k: v for k, v in outputs.items() if k != "aux_outputs"}
        indices = self.criterion.matcher(outputs_without_aux, targets)
        src_idx = self._get_src_permutation_idx(indices)

        target_classes_o = torch.cat([t["labels"][J] for t, (_, J) in zip(targets, indices)])
        if target_classes_o.numel() == 0:
            zero = outputs["pred_logits"].sum() * 0.0
            return {"loss_a2f_hkd": zero}
        valid_mask = target_classes_o > 0
        if not torch.any(valid_mask):
            zero = outputs["pred_logits"].sum() * 0.0
            return {"loss_a2f_hkd": zero}

        pred_fine_fg = outputs["pred_logits"][src_idx][..., 1:-1][valid_mask]
        pred_crop = outputs["pred_logits_crop"][src_idx][..., :-1][valid_mask]
        pred_disease = outputs["pred_logits_disease"][src_idx][..., :-1][valid_mask]
        if pred_fine_fg.numel() == 0 or pred_crop.numel() == 0 or pred_disease.numel() == 0:
            zero = outputs["pred_logits"].sum() * 0.0
            return {"loss_a2f_hkd": zero}

        crop_to_fine = self.crop_to_fine_mat.to(device=pred_crop.device, dtype=pred_crop.dtype)
        disease_to_fine = self.disease_to_fine_mat.to(device=pred_disease.device, dtype=pred_disease.dtype)
        if crop_to_fine.numel() == 0 or disease_to_fine.numel() == 0:
            zero = outputs["pred_logits"].sum() * 0.0
            return {"loss_a2f_hkd": zero}

        temp = max(float(self.a2f_hkd_temp), 1e-6)
        eps = max(float(self.a2f_hkd_eps), 1e-12)
        alpha = max(float(self.a2f_hkd_alpha), 0.0)
        beta = max(float(self.a2f_hkd_beta), 0.0)

        teacher_crop = F.softmax(pred_crop / temp, dim=-1)
        teacher_disease = F.softmax(pred_disease / temp, dim=-1)

        active = torch.ones(pred_fine_fg.shape[0], dtype=torch.bool, device=pred_fine_fg.device)
        if self.a2f_hkd_crop_conf_tau > 0.0:
            active = active & (teacher_crop.max(dim=-1).values >= float(self.a2f_hkd_crop_conf_tau))
        if self.a2f_hkd_disease_conf_tau > 0.0:
            active = active & (teacher_disease.max(dim=-1).values >= float(self.a2f_hkd_disease_conf_tau))
        if self.a2f_hkd_fine_margin_tau > 0.0 and pred_fine_fg.shape[-1] >= 2:
            top2_vals = pred_fine_fg.topk(k=2, dim=-1).values
            fine_margin = top2_vals[:, 0] - top2_vals[:, 1]
            active = active & (fine_margin <= float(self.a2f_hkd_fine_margin_tau))

        if not torch.any(active):
            zero = outputs["pred_logits"].sum() * 0.0
            return {"loss_a2f_hkd": zero}

        teacher_from_crop = teacher_crop @ crop_to_fine
        teacher_from_disease = teacher_disease @ disease_to_fine

        if alpha <= 0.0 and beta <= 0.0:
            zero = outputs["pred_logits"].sum() * 0.0
            return {"loss_a2f_hkd": zero}

        fusion_mode = str(self.a2f_hkd_fusion).lower()
        if fusion_mode == "sum":
            teacher_fine = alpha * teacher_from_crop + beta * teacher_from_disease
        else:
            teacher_fine = (teacher_from_crop + eps).pow(alpha) * (teacher_from_disease + eps).pow(beta)
        teacher_fine = teacher_fine / (teacher_fine.sum(dim=-1, keepdim=True) + eps)
        teacher_fine = teacher_fine.detach()

        student_logits = pred_fine_fg[active]
        teacher_probs = teacher_fine[active]

        loss_hkd = F.kl_div(
            F.log_softmax(student_logits / temp, dim=-1),
            teacher_probs,
            reduction="batchmean",
        ) * (temp * temp)

        try:
            storage = get_event_storage()
            storage.put_scalar("a2f_hkd_weight", a2f_weight)
            storage.put_scalar("a2f_hkd_active_ratio", active.float().mean().item())
            storage.put_scalar("a2f_hkd_loss_raw", loss_hkd.item())
        except Exception:
            pass

        return {"loss_a2f_hkd": loss_hkd * a2f_weight}

    def _get_global_crop_weight(self):
        weight = float(self.global_crop_weight)
        if weight <= 0.0:
            return 0.0
        warmup_iters = int(self.global_crop_warmup_iters)
        off_iters = int(self.global_crop_warmup_off_iters)
        cur_iter = get_event_storage().iter
        if cur_iter < off_iters:
            return 0.0
        if warmup_iters <= 0:
            return weight
        if cur_iter >= off_iters + warmup_iters:
            return weight
        scale = float(cur_iter - off_iters) / float(warmup_iters)
        return weight * scale

    def _get_mask_adapter_reg_weight(self):
        weight = float(self.mask_adapter_reg_weight)
        if weight <= 0.0 or not self.mask_adapter_on:
            return 0.0
        warmup_iters = int(self.mask_adapter_reg_warmup_iters)
        decay_iters = int(self.mask_adapter_reg_decay_iters)
        cur_iter = get_event_storage().iter
        if warmup_iters > 0 and cur_iter < warmup_iters:
            return weight * float(cur_iter) / float(warmup_iters)
        if decay_iters > 0:
            if cur_iter >= warmup_iters + decay_iters:
                return 0.0
            scale = 1.0 - float(cur_iter - warmup_iters) / float(decay_iters)
            return weight * max(scale, 0.0)
        return weight

    def _compute_global_crop_loss(self, outputs, targets):
        if "global_crop_logits" not in outputs:
            return {}

        valid_idx = []
        target_crops = []
        for i, t in enumerate(targets):
            labels = t["labels"]
            if labels.numel() == 0:
                continue
            labels = labels[labels > 0]
            if labels.numel() == 0:
                continue
            crop_ids = self.fine_to_crop.to(labels.device)[labels]
            crop_ids = crop_ids[crop_ids >= 0]
            if crop_ids.numel() == 0:
                continue
            unique = torch.unique(crop_ids)
            if unique.numel() != 1:
                continue
            valid_idx.append(i)
            target_crops.append(unique[0])

        zero = outputs["pred_logits"].sum() * 0.0
        if not valid_idx:
            return {"loss_global_crop": zero}

        logits = outputs["global_crop_logits"]
        idx = torch.as_tensor(valid_idx, device=logits.device, dtype=torch.long)
        logits = logits.index_select(0, idx)[..., :-1]
        target = torch.stack(target_crops).to(logits.device)

        loss = F.cross_entropy(logits, target, reduction="mean")
        loss = loss * self._get_global_crop_weight()

        with torch.no_grad():
            acc = (logits.argmax(dim=-1) == target).float().mean()
            get_event_storage().put_scalar("global_crop_acc", acc.item())

        return {"loss_global_crop": loss}

    def _compute_mask_adapter_reg_loss(self, outputs):
        if "pred_masks_adapter_weights" not in outputs or not self.mask_adapter_on:
            return {}
        zero = outputs["pred_logits"].sum() * 0.0
        weight = self._get_mask_adapter_reg_weight()
        if weight <= 0.0:
            return {"loss_mask_adapter_reg": zero}

        base = outputs["pred_masks"].sigmoid()
        base = (base > self.mask_adapter_mask_threshold).to(base.dtype)
        adapter = outputs["pred_masks_adapter_weights"]
        loss = ((1.0 - base) * adapter).mean()
        return {"loss_mask_adapter_reg": loss * weight}

    def semantic_inference(self, mask_cls, mask_pred):
        mask_cls = F.softmax(mask_cls, dim=-1)[..., :-1]
        mask_pred = mask_pred.sigmoid()
        semseg = torch.einsum("qc,qhw->chw", mask_cls, mask_pred)
        return semseg

    def panoptic_inference(self, mask_cls, mask_pred):
        scores, labels = F.softmax(mask_cls, dim=-1).max(-1)
        mask_pred = mask_pred.sigmoid()
        num_classes = len(self.test_metadata.stuff_classes)
        keep = labels.ne(num_classes) & (scores > self.object_mask_threshold)
        cur_scores = scores[keep]
        cur_classes = labels[keep]
        cur_masks = mask_pred[keep]
        cur_mask_cls = mask_cls[keep]
        cur_mask_cls = cur_mask_cls[:, :-1]

        cur_prob_masks = cur_scores.view(-1, 1, 1) * cur_masks

        h, w = cur_masks.shape[-2:]
        panoptic_seg = torch.zeros((h, w), dtype=torch.int32, device=cur_masks.device)
        segments_info = []

        current_segment_id = 0

        if cur_masks.shape[0] == 0:
            # We didn't detect any mask :(
            return panoptic_seg, segments_info
        else:
            # take argmax
            cur_mask_ids = cur_prob_masks.argmax(0)
            stuff_memory_list = {}
            for k in range(cur_classes.shape[0]):
                pred_class = cur_classes[k].item()
                isthing = pred_class in self.test_metadata.thing_dataset_id_to_contiguous_id.values()
                mask_area = (cur_mask_ids == k).sum().item()
                original_area = (cur_masks[k] >= 0.5).sum().item()
                mask = (cur_mask_ids == k) & (cur_masks[k] >= 0.5)

                if mask_area > 0 and original_area > 0 and mask.sum().item() > 0:
                    if mask_area / original_area < self.overlap_threshold:
                        continue

                    # merge stuff regions
                    if not isthing:
                        if int(pred_class) in stuff_memory_list.keys():
                            panoptic_seg[mask] = stuff_memory_list[int(pred_class)]
                            continue
                        else:
                            stuff_memory_list[int(pred_class)] = current_segment_id + 1

                    current_segment_id += 1
                    panoptic_seg[mask] = current_segment_id

                    segments_info.append(
                        {
                            "id": current_segment_id,
                            "isthing": bool(isthing),
                            "category_id": int(pred_class),
                        }
                    )

            return panoptic_seg, segments_info

    def instance_inference(self, mask_cls, mask_pred):
        # mask_pred is already processed to have the same shape as original input
        image_size = mask_pred.shape[-2:]

        # [Q, K]
        scores = F.softmax(mask_cls, dim=-1)[:, :-1]
        # if this is panoptic segmentation
        if self.panoptic_on:
            num_classes = len(self.test_metadata.stuff_classes)
        else:
            num_classes = len(self.test_metadata.thing_classes)
        labels = torch.arange(num_classes, device=self.device).unsqueeze(0).repeat(self.num_queries, 1).flatten(0, 1)
        # scores_per_image, topk_indices = scores.flatten(0, 1).topk(self.num_queries, sorted=False)
        scores_per_image, topk_indices = scores.flatten(0, 1).topk(self.test_topk_per_image, sorted=False)
        labels_per_image = labels[topk_indices]

        topk_indices = topk_indices // num_classes
        # mask_pred = mask_pred.unsqueeze(1).repeat(1, self.sem_seg_head.num_classes, 1).flatten(0, 1)
        mask_pred = mask_pred[topk_indices]

        # if this is panoptic segmentation, we only keep the "thing" classes
        if self.panoptic_on:
            keep = torch.zeros_like(scores_per_image).bool()
            for i, lab in enumerate(labels_per_image):
                keep[i] = lab in self.test_metadata.thing_dataset_id_to_contiguous_id.values()

            scores_per_image = scores_per_image[keep]
            labels_per_image = labels_per_image[keep]
            mask_pred = mask_pred[keep]

        result = Instances(image_size)
        # mask (before sigmoid)
        result.pred_masks = (mask_pred > 0).float()
        result.pred_boxes = Boxes(torch.zeros(mask_pred.size(0), 4))
        # Uncomment the following to get boxes from masks (this is slow)
        # result.pred_boxes = BitMasks(mask_pred > 0).get_bounding_boxes()

        # calculate average mask prob
        mask_scores_per_image = (mask_pred.sigmoid().flatten(1) * result.pred_masks.flatten(1)).sum(1) / (result.pred_masks.flatten(1).sum(1) + 1e-6)
        result.scores = scores_per_image * mask_scores_per_image
        result.pred_classes = labels_per_image
        return result
