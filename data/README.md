# Dataset Layout

This repository does not include dataset files. Download `WheatField-Seg` separately and prepare it using the layout below.

By default the code looks for datasets under this repository's `data/` directory. You can also set:

```bash
export TAXOSEG_DATASETS=/path/to/datasets
```

The variable should point to the parent directory that contains `WheatField-Seg`.

## WheatField-Seg

Expected layout:

```text
WheatField-Seg/
  annotation_train.json
  annotation_val.json
  annotation_test.json
  images/
    train/
    val/
    test/
  annotations/
    train/
    val/
    test/
  mappings/
    taxonomy.json
```

WheatField-Seg contains 1,762 images in total.

The dataset uses contiguous semantic mask ids:

```text
0 background
1 wheat head scab
2 wheat powdery mildew
3 wheat stripe rust
```

The taxonomy file in the dataset mapping directory is required by the hierarchical losses and long-prompt branch.
