import numpy as np
from PIL import Image

from detectron2.evaluation import SemSegEvaluator
from detectron2.utils.file_io import PathManager


class WheatFieldSemSegEvaluator(SemSegEvaluator):
    """Semantic segmentation evaluator for WheatField-Seg closed-set labels."""

    def __init__(self, *args, original_to_contiguous_id=None, **kwargs):
        super().__init__(*args, **kwargs)
        if original_to_contiguous_id is None:
            original_to_contiguous_id = {0: 0, 1: 1, 2: 2, 3: 3}
        self._original_to_contiguous_id = dict(original_to_contiguous_id)

    def _remap_gt(self, gt):
        remapped = np.full(gt.shape, self._ignore_label, dtype=np.int64)
        for original_id, contiguous_id in self._original_to_contiguous_id.items():
            remapped[gt == int(original_id)] = int(contiguous_id)
        return remapped

    def process(self, inputs, outputs):
        for input, output in zip(inputs, outputs):
            output = output["sem_seg"].argmax(dim=0).to(self._cpu_device)
            pred = np.array(output, dtype=np.int64)
            if pred.size and pred.max() >= self._num_classes:
                raise ValueError(
                    f"WheatField prediction contains class id {int(pred.max())}, "
                    f"but closed-set evaluation only allows ids 0..{self._num_classes - 1}."
                )
            with PathManager.open(self.input_file_to_gt_file[input["file_name"]], "rb") as f:
                gt = np.array(Image.open(f), dtype=np.int64)

            gt = self._remap_gt(gt)
            gt[gt == self._ignore_label] = self._num_classes

            self._conf_matrix += np.bincount(
                (self._num_classes + 1) * pred.reshape(-1) + gt.reshape(-1),
                minlength=self._conf_matrix.size,
            ).reshape(self._conf_matrix.shape)

            self._predictions.extend(self.encode_json_sem_seg(pred, input["file_name"]))
