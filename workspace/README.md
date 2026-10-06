# workspace/

The agent is jailed to this directory: `read_file`, `write_file` and `run_python` can't reach
anything outside it. The dataset goes in `workspace/data/`:

```bash
python scripts/download_data.py
```

| File | Rows / size | What it is |
|---|---|---|
| `data/payments.csv` | 138,236 rows, ~24 MB | one row per card transaction |
| `data/merchant_data.json` | 30 merchants | one JSON object per merchant |
| `data/merchant_category_codes.csv` | 769 rows | MCC code → description |
| `data/acquirer_countries.csv` | 8 rows | acquirer → country code |

**Source:** [DABstep](https://huggingface.co/datasets/adyen/DABstep) by Adyen, licensed
[CC-BY-4.0](https://creativecommons.org/licenses/by/4.0/). The files are downloaded unchanged.
