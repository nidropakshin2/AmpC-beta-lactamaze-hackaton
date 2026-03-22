# Makefile для воспроизведения пайплайна QSAR + REINVENT

SHELL := /bin/bash
.ONESHELL:          # все команды цели выполняются в одном shell
.SHELLFLAGS = -e -o pipefail -c  # аналог set -e и set -o pipefail

# Путь к conda (автоопределение, но можно переопределить)
CONDA_BASE ?= $(shell conda info --base)

# Активация окружения conda (будет выполнена в каждой цели из-за .ONESHELL)
define activate_conda
	source "$(CONDA_BASE)/etc/profile.d/conda.sh" && conda activate sber_hack
endef

DATASET_FILE = data_rdkit

# Цель по умолчанию – выполнить всё последовательно
all: build metric clean reinvent sample

# Шаг 1: предобработка данных
preprocess:
	$(activate_conda)
	if [ -f "qsar/dataset/$(DATASET_FILE)_preprocessed.csv" ]; then
		echo "Preprocessed dataset already exists. Skipping preprocessing."
	else
		python qsar/preprocess_data.py --dataset_file $(DATASET_FILE)
	fi

# Шаг 2: построение QSAR-модели (зависит от preprocess)
build: 
	$(activate_conda)
	python qsar/optimize.py --dataset_train_file qsar/dataset/$(DATASET_FILE)_train.csv --dataset_test_file qsar/dataset/$(DATASET_FILE)_test.csv

metric:
	$(activate_conda)
	python qsar/metric.py --model_file qsar/model/latest.pkl

# Шаг 3: обучение агента REINVENT (зависит от build)
reinvent: 
	$(activate_conda)
	reinvent -l reinvent/logs/log.log reinvent/configs/config.toml

sample:
	$(activate_conda)
	reinvent reinvent/configs/sampling.toml

tensorboard:
	$(activate_conda)
	tensorboard --bind_all --logdir reinvent/logs/tb_0

# Опционально: очистка временных файлов
clean:
	rm -rf reinvent/logs/*

.PHONY: preprocess build reinvent sample tensorboard clean all