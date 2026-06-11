# neural-operator-reproduction

From-scratch PyTorch reproduction of "Neural Operator: Learning Maps Between Function Spaces".

## Implemented operators

- `FNO1d` / `FNO2d`: Fourier neural operator layers with coordinate lifting, truncated Fourier modes, pointwise residual maps, and optional Darcy padding.
- `GNO2d`: graph neural operator with radius-neighborhood kernel integration and chunked evaluation for memory control.
- `MGNO2d`: two-level multipole-style graph operator using fine/coarse graph integration and interpolation.
- `LNO2d`: low-rank neural operator with coordinate-factorized kernel integration.

## Training

Run the original FNO scripts:

```bash
python scripts/train_burgers.py --config configs/burgers_fno1d.yaml
```

Run any Darcy operator config through the generic trainer:

```bash
python scripts/train_darcy_operator.py --config configs/darcy_fno2d.yaml
python scripts/train_darcy_operator.py --config configs/darcy_gno2d.yaml
python scripts/train_darcy_operator.py --config configs/darcy_lno2d.yaml
python scripts/train_darcy_operator.py --config configs/darcy_mgno2d.yaml
```

Each config includes a `seed` block and a `logging` block. Training writes:

- `runs/<experiment>_<timestamp>/config.yaml`
- `runs/<experiment>_<timestamp>/metrics.csv`
- `runs/<experiment>_<timestamp>/best_model.pt`
- `runs/<experiment>_<timestamp>/last_model.pt`
- `runs/<experiment>_<timestamp>/summary.json`

## Data

Expected local paths:

- `data/piececonst_r421_N1024_smooth1.mat` with keys `coeff` and `sol` for Darcy flow.
- `data/burgers_data_R10.mat` with keys `a` and `u` for Burgers. If the local file only contains `a`/`a_smooth` fields, it is not the complete supervised dataset required by the training script.

The implementation does not import code from existing neural-operator repositories.
