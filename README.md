# rtu-bachelor-tasks
Task solution for RTU bachelor work

## Installation

```bash
python3.11 -m venv venv --upgrade-deps
source venv/bin/activate
python -m pip install -U -r requirements_dev.txt

# For running on Nvidia GPU:
python -m pip install -U torch torchvision --extra-index-url https://download.pytorch.org/whl/cu118

# For running on CPU:
python -m pip install -U torch torchvision --extra-index-url https://download.pytorch.org/whl/cpu

python -m pip install -U -r requirements.txt
```

## Running image converter
```bash
python -m dataset_image_converter --data-root /path/to/dataset/images/
```
Source code repo: https://github.com/kamikaze/dataset-image-converter

## Running image annotator
```bash
python -m dataset_image_annotator --data-root /path/to/dataset/images/
```
Source code repo: https://github.com/kamikaze/dataset-image-annotator

## Running dataset format benchmark
```bash
python -m dataset_format_benchmark --data-root /path/to/datasets/
```
Source code repo: https://github.com/kamikaze/dataset-format-benchmark