.PHONY: help install install-dev install-all clean test lint format run-jupyter run-tests docs

help:
	@echo "Signals to Intelligence - Available Commands"
	@echo "=============================================="
	@echo ""
	@echo "Setup:"
	@echo "  make install          Install dependencies"
	@echo "  make install-dev      Install dev dependencies"
	@echo "  make install-all      Install all dependencies"
	@echo ""
	@echo "Development:"
	@echo "  make lint             Run linters (flake8, mypy)"
	@echo "  make format           Format code (black, isort)"
	@echo "  make test             Run tests"
	@echo "  make test-cov         Run tests with coverage"
	@echo ""
	@echo "Running:"
	@echo "  make run-jupyter      Start Jupyter notebook"
	@echo "  make run-exp-ch01     Run Chapter 1 experiments"
	@echo "  make run-exp-ch02     Run Chapter 2 experiments"
	@echo "  make run-all-exp      Run all experiments"
	@echo ""
	@echo "Documentation:"
	@echo "  make docs             Build documentation"
	@echo ""
	@echo "Cleanup:"
	@echo "  make clean            Remove build artifacts"
	@echo "  make clean-cache      Remove cache files"

install:
	pip install -r requirements.txt

install-dev:
	pip install -r requirements.txt
	pip install -e ".[dev]"

install-all:
	pip install -r requirements.txt
	pip install -e ".[all]"

lint:
	flake8 code tests --max-line-length=100
	mypy code --ignore-missing-imports

format:
	black code tests
	isort code tests

test:
	pytest tests/ -v

test-cov:
	pytest tests/ -v --cov=code --cov-report=html --cov-report=term

run-jupyter:
	jupyter notebook --ip=0.0.0.0 --port=8888 --no-browser

run-exp-ch01:
	python code/ch01_dsp/fft_spectrum.py
	python code/ch01_dsp/positional_encoding.py

run-exp-ch02:
	python code/ch02_optimization/lms_vs_adam.py
	python code/ch02_optimization/mmse_vs_nn.py

run-exp-ch03:
	python code/ch03_deep_learning_fast/polynomial_vs_mlp.py
	python code/ch03_deep_learning_fast/mnist_cnn.py
	python code/ch03_deep_learning_fast/rnn_structure.py

run-exp-ch04:
	python code/ch04_transformer/self_attention.py

run-exp-ch05:
	python code/ch05_llm_basics/llm_api_demo.py

run-exp-ch06:
	python code/ch06_llm_applications/rag_demo.py

run-exp-ch07:
	python code/ch07_multimodal_llm/vit_patches.py
	python code/ch07_multimodal_llm/clip_similarity.py
	python code/ch07_multimodal_llm/high_resolution_processing.py
	python code/ch07_multimodal_llm/qwen_vl_analysis.py

run-exp-ch08:
	python code/ch08_llm_engineering/llm_engineering_demo.py

run-all-exp:
	@echo "Running all experiments..."
	make run-exp-ch01
	make run-exp-ch02
	make run-exp-ch03
	make run-exp-ch04
	make run-exp-ch05
	make run-exp-ch06
	make run-exp-ch07
	make run-exp-ch08

docs:
	@echo "Documentation is in docs/ directory"
	@echo "Start with: docs/00_introduction/README.md"

clean:
	find . -type d -name __pycache__ -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete
	find . -type d -name "*.egg-info" -exec rm -rf {} +
	rm -rf build/ dist/

clean-cache:
	rm -rf .pytest_cache/
	rm -rf .mypy_cache/
	rm -rf htmlcov/
	rm -rf .coverage

docker-up:
	docker-compose up -d

docker-down:
	docker-compose down

docker-logs:
	docker-compose logs -f jupyter
