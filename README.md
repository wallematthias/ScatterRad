# ScatterRad

ScatterRad is a texture-classification framework for nnUNet-style medical imaging datasets.

## Quickstart

```bash
conda activate scatterrad
pip install -e ".[dev]"
scatterrad validate Dataset002_LesionDetection
```

Set required environment variables before using the CLI:

- `SCATTERRAD_RAW`
- `SCATTERRAD_PREPROCESSED`
- `SCATTERRAD_RESULTS`
