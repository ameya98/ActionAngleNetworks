## Learning Physical Dynamics with Action-Angle Networks

[![linting: pylint](https://img.shields.io/badge/linting-pylint-yellowgreen)](https://github.com/PyCQA/pylint)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![pre-commit](https://img.shields.io/badge/pre--commit-enabled-brightgreen?logo=pre-commit&logoColor=white)](https://github.com/pre-commit/pre-commit)


The official JAX implementation of Action-Angle Networks.

### Instructions

Clone the repository:

```shell
git clone https://github.com/google-research/google-research.git
cd google-research/action_angle_networks
```

Create and activate a virtual environment:

```shell
python -m venv .venv && source .venv/bin/activate
```

Install dependencies with:

```shell
pip install --upgrade pip && pip install -r requirements.txt
```

Start training with a configuration defined under `configs/`:

```shell
python main.py --workdir=./tmp --config=configs/action_angle_flow.py
```

#### Changing Hyperparameters

Since the configuration is defined using
[config_flags](https://github.com/google/ml_collections/tree/master#config-flags),
you can override hyperparameters. For example, to change the number of training
steps and the batch size:

```shell
python main.py --workdir=./tmp --config=configs/action_angle_flow.py \
--config.num_train_steps=10 --config.batch_size=100
```

For more extensive changes, you can directly edit the configuration files, and
even add your own.

## Code Authors

Ameya Daigavane

## Note

This is a fork of [my original implementation here](https://github.com/google-research/google-research/tree/master/action_angle_networks).
