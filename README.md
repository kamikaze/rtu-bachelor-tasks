# rtu-bachelor-tasks
Task solution for RTU bachelor work

## Installation

```bash
python3.10 -m venv venv --upgrade-deps
source venv/bin/activate
python -m pip install -U -r requirements_dev.txt
python -m pip install -U -r requirements.txt
```

## Running image annotator
```bash
python -m dataset_image_annotator --data-root /path/to/dataset/images/
```

## Running dataset format benchmark
```bash
python -m dataset_format_benchmark --data-root /path/to/datasets/
```
