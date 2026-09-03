# AttackDetectionML

This repo contains the code used in the paper
"Anomaly Detection with Machine Learning Algorithms in Large-Scale Power Grids",
arXiv preprint [2602.10888](https://arxiv.org/abs/2602.10888).

The code base has grown over several years, with different authors involved.
It is therefore not optimized in any way, nor thoroughly documented.
Nevertheless, this repo enables anybody to reproduce the results
presented in the paper, in accordance with modern publication principles.

## Installation

The first step is to install the virtual environment. This is most easily done with
[conda](https://docs.conda.io/projects/conda/en/latest/user-guide/install/index.html),
using the command

```sh
conda env create -f environment.yml
conda activate AttackDetectionML
```

"AttackDetectionML" is the name of the environment by default.

## Dataset

The dataset used in this project is publicly available on Zenodo:
1. Visit [https://doi.org/10.5281/zenodo.13378476](https://doi.org/10.5281/zenodo.13378476).
2. Click **Download** next to the file(s) you need.
3. Place the downloaded files in the [`raw_data/`](./raw_data/) directory of this repo.


## Execution

The multiples Python scripts in the [`codes`](./codes/) directory must be run sequentially.

### Data preprocessing

1. [`01Explore_raw_data.py`](./codes/01Explore_raw_data.py)
2. [`02Make_anomalies.py`](./codes/02Make_anomalies.py)

These scripts generate the list of on/off anomalies as well as several figures
under [`figures`](./figures) for manual dataset exploration and validation.


### Supervised models 

1. [`01GSCV.py`](./codes/supervised/01GSCV.py)
2. [`02Results_unique.py`](./codes/supervised/02Results_unique.py)

Training loops can take a significant time to run. To run in the background, e.g. on a compute server, use

```sh
nohup python -u codes/supervised/01GSCV.py > supervised.out &
```

### Unsupervised models

1. [`01GSCV.py`](./codes/unsupervised/01GSCV.py)
2. [`02Results_regression.py`](./codes/unsupervised/02Results_regression.py)
3. [`03Classification.py`](./codes/unsupervised/03Classification.py)
4. [`04Results_classification_unique.py`](./codes/unsupervised/04Results_classification_unique.py)

### LSTM

...

### Analysis of the results

1. [`04Get_LSTM_results_csv.py`](./codes/04Get_LSTM_results_csv.py)
2. [`05Results_unique_all.py`](./codes/05Results_unique_all.py)
3. [`06Hyperparameters.py`](./codes/06Hyperparameters.py)
4. [`07Algos_combo.py`](./codes/07Algos_combo.py)
5. [`08Anomalies_distribution.py`](./codes/08Anomalies_distribution.py)
6. [`09Multiple_attacks.py`](./codes/09Multiple_attacks.py)
7. [`10Noise_sensitivity.py`](./codes/10Noise_sensitivity.py)