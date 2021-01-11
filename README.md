# Pytorch Something or Other

Python model template framework, I think? A tool to easily use a bunch of Pytorch featurs without rewriting the same code a million times. 

## Getting Started

### Environment Setup

Let's set up your project environment first. This project uses [poetry](https://python-poetry.org/docs/) as it's package manager. There are
two types of dependencies: **core dependencies** and **development dependencies**. Core dependencies are those that are required to be installed for the main, production release of your project or package. Development dependencies are auxiliary packages that are useful in aiding in providing functionality such as formatting, documentation or type-checking, but are non-essential for the production release. For each dependency you come across, make a determination on whether it is a core or development dependency, and add it to the `pyproject.toml` file from the command line using the following command, where the `-D` flag is to be used only for development dependencies.

```bash
poetry add [-D] <name-of-dependency>
```

When you are ready to run your code and have added all your dependencies, you can perform a `poetry lock` in order to reproducibly fix your dependency versions. This will use the pyproject.toml file to crease a poetry.lock file. Then, in order to run your code, you can use the following commands to set up a virtual environment and then run your code within the virtual envrionment. The optional `--no-dev` flag indicates that you only wish to install core dependencies.

```bash
poetry install [--no-dev]
poetry run <your-command>
```

Alternatively, you can also initialize the poetry shell (by running the following) instead of prefixing your commands with `poetry run`.

```bash
poetry shell
```

### Initializing Pre-Commit Hooks

This repository uses pre-commit hooks in order to assist you in maintaining a uniform and idiomatic code style.
If this is your first time using pre-commit hooks you can install the framework [here](https://pre-commit.com/#installation).
Once pre-commit is installed, all you need to do is execute the following command from the repository root:
```
pre-commit install
```

If you want to execute the pre-commit hooks at a time other than during the actual git commit, you can run:
```
pre-commit run --all-files
```
## Example Runs

```bash
python -m omni_model.src.omniscient_cli download-data --dataset-root "./data/cifar-10-batches-py" --dataset-name "CIFAR10"
```

```bash
python -m omni_model.src.omniscient_cli train --model-arch resnet18 --dataset-name CIFAR10 --data-split '(100,0,0)' --subset-fraction 0.1 --batch-size 8 --num-workers 8 --pretrained  --optimizer "SGD" --learning-rate 0.001 --num-epochs 1
```

