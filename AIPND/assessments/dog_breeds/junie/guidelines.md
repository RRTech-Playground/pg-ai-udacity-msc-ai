# Dog-Breeds Project Guidelines

Last updated: 2025-11-23 15:43 (local)

These guidelines explain what this project does, how to set it up and run it, what outputs to expect, and how to contribute or troubleshoot.

## 1. Purpose

This project classifies images of pets (dogs, cats, etc.) and evaluates whether the predicted label matches the actual pet breed within the filename. It is adapted from the Udacity AI Programming with Python Nanodegree dog breed classifier lab.

Primary goals:
- Run image classification with pretrained CNN architectures.
- Compare classifier labels to ground-truth labels parsed from filenames.
- Compute accuracy metrics, including dog vs. non-dog identification.

## 2. Project layout (key files)

Directory: `AIPPND/assessments/dog-breeds/`

- `check_images.py` — Main script that runs the full pipeline end-to-end and prints results/summary.
- `classify_images.py` — Uses a chosen CNN architecture to classify each image.
- `classifier.py` — Lower-level wrapper around the pretrained model(s).
- `get_input_args.py` — Defines and parses CLI arguments for main scripts.
- `get_pet_labels.py` — Extracts ground-truth labels from image filenames.
- `adjust_results4_isadog.py` — Adds dog/non-dog flags to results using a dog name list.
- `calculates_results_stats.py` — Computes metrics and summary stats.
- `print_results.py` — Formats and prints the final results.
- `print_functions_for_lab_checks.py` — Utility printing for lab checks.
- `test_classifier.py` — Simple test demonstrating the classifier call.

Data and assets:
- `data/pet_images/` — Example images (dogs and non-dogs).
- `data/dognames.txt` — List of valid dog breed names.
- `data/imagenet1000_clsid_to_human.txt` — Class index-to-name mapping (reference).
- `data/check_images.txt` — Sample expected output snapshot from a reference run.
- `data/run_models_batch.sh` — Helper to run multiple architectures in batch (local use).

## 3. Environment setup

Recommended: use Conda/Mamba with the root `environment.yml` found at the repository root.

1) Create the environment
```
conda env create -f environment.yml
# or: mamba env create -f environment.yml
```

2) Activate it
```
conda activate aipnd
# If your environment name differs, adjust accordingly.
```

3) Verify Python and key libs
```
python --version
python -c "import torch, torchvision; print(torch.__version__, torchvision.__version__)"
```

If not using the provided environment, you will need:
- Python 3.8+ (3.10 recommended)
- PyTorch and Torchvision (CPU is sufficient for the small sample set)
- Pillow, NumPy

## 4. How to run

Navigate to the project root of this sub-project:
```
cd AIPPND/assessments/dog-breeds
```

### 4.1 Quick start (default paths)
```
python check_images.py \
  --dir data/pet_images/ \
  --arch resnet \
  --dogfile data/dognames.txt
```

Common architectures: `resnet`, `alexnet`, `vgg`.

Notes:
- `--dir` should point to the folder with your images. Filenames are used to derive ground-truth labels.
- `--arch` selects the pretrained CNN backbone.
- `--dogfile` is the text file listing valid dog names.

### 4.2 Examples

- VGG, sample images:
```
python check_images.py --dir data/pet_images/ --arch vgg --dogfile data/dognames.txt
```

- AlexNet on a custom folder of images (replace with your path):
```
python check_images.py --dir /path/to/my_images/ --arch alexnet --dogfile data/dognames.txt
```

### 4.3 Batch runs

Use the helper script to run multiple architectures and capture outputs:
```
bash data/run_models_batch.sh
```

Review the generated logs or compare to `data/check_images.txt` for a sanity check.

## 5. Outputs and metrics

The pipeline prints:
- Count of images, dog images, and non-dog images.
- Match statistics between classifier label and true label.
- Percentages such as:
  - pct_match (top-1 label matches filename-derived label)
  - pct_correct_dogs / pct_correct_notdogs
  - pct_correct_breed (for dog images)

Depending on your environment and library versions, minor numeric differences can occur. Use the provided `data/check_images.txt` to gauge reasonable ranges.

## 6. Development guidelines

- Coding style: follow PEP 8, keep functions small and single-purpose.
- Docstrings: for public functions, add a short docstring describing inputs/outputs.
- Determinism: set random seeds if you extend training or sampling logic.
- I/O: avoid hardcoding absolute paths; prefer arguments or config variables.
- Performance: for many images, consider batching and avoiding repeated model loads.
- Reuse: prefer extending `classify_images.py` and `print_results.py` rather than duplicating logic.

## 7. Testing and verification

Run the sample classifier test:
```
python test_classifier.py
```

You can also perform targeted checks with the lab utilities:
```
python print_functions_for_lab_checks.py
```

For quick manual verification, run two different architectures (e.g., `resnet` and `vgg`) and compare their summary stats.

## 8. Troubleshooting

- ImportError: Torch/Torchvision not found
  - Ensure the Conda environment is active and includes PyTorch/Torchvision.

- RuntimeError: Model weights not available / network access
  - Ensure you have internet to download pretrained weights on first run, or pre-download/copy them into your cache.

- Slow performance on first inference
  - First-time model/weight load can be slow; subsequent runs are faster. Running on CPU is fine for small sample sets.

- Labels don’t match expectations
  - Confirm filenames and `data/dognames.txt`. Ensure `--dir` points to the correct folder and that filenames are clean (no extra spaces/suffixes).

## 9. Contributing

- Keep PRs small and focused; include a brief description of the change and before/after behavior.
- Add or update short comments when modifying core pipeline functions.
- Validate changes by running at least one full `check_images.py` pass and noting summary metrics.

## 10. References

- Udacity AIPND dog breed classifier lab (for educational structure and concepts)
- PyTorch Hub models (ResNet, AlexNet, VGG)
