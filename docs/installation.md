# Installation

## Requirements

- Python 3.10+
- macOS or Linux (Apple Silicon supported)

## Install SST

```bash
git clone https://github.com/Imageomics/SST.git
cd SST
pip install -e .
```

## macOS-specific note

On Apple Silicon Macs, set `num_workers=0` in `inference.py` to avoid a multiprocessing pickling error:

```python
# inference.py
num_workers = 0
```
