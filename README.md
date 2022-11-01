# rtu-bachelor-tasks
Task solution for RTU bachelor work

## Installation

```bash
python3.10 -m venv venv --upgrade-deps
source venv/bin/activate
python -m pip install -U -r requirements_dev.txt
python -m pip install -U --pre torch torchvision --extra-index-url https://download.pytorch.org/whl/nightly/cu117
python -m pip install -U -r requirements.txt
```

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