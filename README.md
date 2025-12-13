# Adult Income Prediction

This project explores the Adult Income dataset to understand how demographic and work-related factors relate to earning more or less than \$50,000 per year. It includes a reproducible pipeline for data prep, modeling, and a Quarto report.

## Contributors
- Chun-Mien Liu
- Rebecca Rosette Nanfuka
- Roganci Fontelera
- Yonas Gebre Marie

## Requirements
- Docker Desktop (or Docker Engine)
- Make (optional, for shortcuts)
- Quarto inside the container for rendering (see setup below)

## Getting started
1) Clone and enter the repo
```bash
git clone <repo-url> IncomePrediction
cd IncomePrediction
```

2) Build the image
```bash
make build        # or: docker compose build
```

3) Start the container (JupyterLab on host port 8878)
```bash
docker compose up -d
```
- Check logs for the Jupyter token: `docker compose logs -f 522-milestone`
- Open the printed URL but swap the port to `8878` (e.g., `http://127.0.0.1:8878/?token=...`).
- Adjust the port in `docker-compose.yml` if you need a different host port.

4) In JupyterLab, open a Terminal (from the “+ Launcher” or File → New → Terminal) and activate the env
```bash
eval "$(/opt/conda/bin/conda shell.bash hook)"
conda activate 522-milestone

```

5) Run the pipeline from that JupyterLab terminal  
**All steps at once:**
```bash
make docs/index.html
```
This runs all the scripts necessary to update outdated artifacts (i.e. processed data, figures, tables).

**Here are individual steps:**
|Steps              |        Scripts    | Options | Examples |
|-------------------|-------------------|--------------------------|--------------------------|
|Download Data|scripts/download_data.py| --url<br>--write_to<br>--target-name|--url https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.data --write_to data/raw --target-name adult.csv|
|Clean data|scripts/clean_data.py| --input-path<br>--output-path|--input-path data/raw/adult.csv --output-path data/processed/clean_adult.csv|
|EDA|scripts/eda.py|--input_dir<br>--out_dir|--input_dir="data/processed/clean_adult.csv" --out_dir="artifacts/figures/"|
|Split Data|scripts/split_data.py|--input_dir<br>--out_dir|--input_dir="data/processed/clean_adult.csv" --out_dir="data/processed/"|
|Feature preprocessing (train/test)|scripts/preprocess_data.py|--input_dir<br>--out_dir|--input_dir="data/processed/train.csv" --out_dir="data/processed/"|
|Modeling|scripts/modeling.py|--data-dir<br>--artifacts-dir<br>--cv-folds<br>--top-n<br>--log-reg-iter<br>--svm-iter|--data-dir data --artifacts-dir artifacts --cv-folds 5 --top-n 10 --log-reg-iter 50 --svm-iter 30|

What each step produces (paths):
- Raw data: `data/raw/` (downloaded from UCI)
- Cleaned + split CSVs: `data/processed/clean_adult.csv`, `data/processed/train.csv`, `data/processed/test.csv`
- Validation report/logs: printed to terminal from `make data-validate` (non-blocking)
- EDA figures: `artifacts/figures/`
- Preprocessed train/test matrices: `data/processed/train_preprocessed.parquet`, `data/processed/test_preprocessed.parquet`
- Models, metrics, and tuning artifacts: `artifacts/` (tables, pickles, plots)

## Quarto report
Quarto must be available inside the container (install with `mamba install -n 522-milestone -c conda-forge quarto` if missing).

Use the JupyterLab terminal (with the env activated) to render:
```bash
quarto render reports/income-prediction.qmd
```
Preview (inside the container):
```bash
quarto preview reports/income-prediction.qmd --host 0.0.0.0 --port 8889
```
Then open `http://localhost:8889` in your browser (add `- "8889:8889"` to `docker-compose.yml` if the port is not already mapped).
Notes:
- If you want Quarto baked into the image, add `quarto` to `environment.yml`, regenerate locks, and rebuild.
- Preview uses an extra port; if you prefer not to expose it, render to HTML only.

## Clean up
- Stop services: `docker compose down`
- Remove stopped containers: `docker compose rm`

## Troubleshooting
- Jupyter URL/port mismatch: check `docker-compose.yml` port mapping and use that host port in the URL token.
- Conda not found in JupyterLab terminal: run `eval "$(/opt/conda/bin/conda shell.bash hook)"` before `conda activate 522-milestone`.
- Quarto not found: install in the container env (`mamba install -n 522-milestone -c conda-forge quarto`) or rebuild with it baked in.
- Stale artifacts: remove `data/processed/` and `artifacts/` if you need a clean run, then rerun the pipeline.

## License
Distributed under the MIT License (see `LICENSE`).

## References
- UCI Machine Learning Repository. (1996). Adult Dataset. https://archive.ics.uci.edu/dataset/2/adult
- Kohavi, R. (1996). Scaling Up the Accuracy of Naive-Bayes Classifiers: A Decision-Tree Hybrid. Proceedings of the Second International Conference on Knowledge Discovery and Data Mining (KDD).
- U.S. Census Bureau. Current Population Survey (CPS). https://www.census.gov/programs-surveys/cps.html
