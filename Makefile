# Installation
setup_venv:
	poetry install --no-root

install_pre_commit:
	poetry run pre-commit install -c .github/hooks/.pre-commit-config.yml

install_dev: setup_venv install_pre_commit

# Dev tools
isort:
	poetry run isort src

black:
	poetry run black --config pyproject.toml src

# pre-commit flake8 runs only against staged files
flake8:
	poetry run flake8 src

format: isort black

mypy:
	poetry run mypy --incremental --install-types --show-error-codes --pretty src

pre_commit:
	poetry run pre-commit run -a -c .github/hooks/.pre-commit-config.yml

test:
	poetry run pytest src

test_cov:
	poetry run coverage run -m pytest src --cov-config=.coveragerc --junit-xml=junit/test-results.xml --cov-report=html --cov-report=xml
	poetry run coverage html
	poetry run coverage xml
	poetry run coverage report --show-missing

compile_env:
	poetry lock --no-update

build: isort black pre_commit flake8 mypy test

# Misc
jupyter:
	poetry run jupyter notebook

profile_data:
	poetry run python -m src.cli.profile_data --data-path=data/train.csv --output-dir=output/eda

prepare_data:
	poetry run python -m src.cli.prepare_data --data-dir=data/train_images --output-dir=data/train_64 --resolution=64 --n-jobs=-1
	poetry run python -m src.cli.prepare_data --data-dir=data/train_images --output-dir=data/train_128 --resolution=128 --n-jobs=-1
	poetry run python -m src.cli.prepare_data --data-dir=data/train_images --output-dir=data/train_256 --resolution=256 --n-jobs=-1
	poetry run python -m src.cli.prepare_data --data-dir=data/train_images --output-dir=data/train_512 --resolution=512 --n-jobs=-1
