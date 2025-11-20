# College Major Analysis

## Contributors
- Chun-Mien Liu
- Rebecca Rosette Nanfuka
- Roganci Fontelera
- Yonas Gebre Marie

## Project overview

This project explores the U.S. college majors dataset (`data/recent-grads.csv`) to understand how fields of study influence employment rates, salaries, and demographic outcomes. The analysis is designed to surface broad labor-market patterns (e.g., which majors have the lowest unemployment or the highest earnings) so that educators and students can get a 10,000-foot view of the trade-offs among different disciplines.

## How to run the analysis
1. Clone the repository and move into it: `git clone <repo-url> && cd CollegeMajorAnalysis`.
2. Build the Conda environment defined in `environment.yml`: `conda env create -f environment.yml`.
3. Activate the environment: `conda activate college-major-analysis`.
4. Launch Jupyter Lab from the project root: `jupyter lab`.
5. Open or create your analysis notebook (e.g., `reports/college_major_analysis.ipynb`), point it to `data/recent-grads.csv`, and execute the cells to reproduce the charts and summary metrics.

## Dependencies
- Python 3.12+
- pandas
- jupyterlab
- See `environment.yml` for the authoritative set of packages and versions used in the project.

## License
This repository is distributed under the MIT License (see `LICENSE` for the full text).
