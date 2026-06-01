.PHONY: help install install-ml install-llm install-rag install-all \
        test test-cov lint format \
        run-jupyter run-exp-ch01 run-exp-ch02 run-exp-ch03 run-exp-ch04 \
        run-exp-ch05 run-exp-ch05-offline run-exp-ch06 run-exp-ch07 run-all-exp \
        docker-up docker-up-gpu docker-down docker-logs \
        clean clean-cache

help:
	@echo ""
	@echo "Signals to Intelligence"
	@echo "======================="
	@echo ""
	@echo "环境安装："
	@echo "  make install          安装核心依赖（第1-5章离线实验，CPU）"
	@echo "  make install-ml       安装深度学习依赖（第3-4章，需要 torch）"
	@echo "  make install-llm      安装 LLM API 依赖（第5-6章，需要 API Key）"
	@echo "  make install-rag      安装 RAG 依赖（第6章，langchain/faiss）"
	@echo "  make install-all      安装全部依赖"
	@echo ""
	@echo "代码实验："
	@echo "  make run-exp-ch01     第1章：FFT、位置编码"
	@echo "  make run-exp-ch02     第2章：优化器对比"
	@echo "  make run-exp-ch03     第3章：MLP、CNN"
	@echo "  make run-exp-ch04     第4章：自注意力"
	@echo "  make run-exp-ch05     第5章：全部离线实验（不需要 API Key）"
	@echo "  make run-exp-ch06     第6章：RAG 演示"
	@echo "  make run-exp-ch07     第7章：多模态"
	@echo "  make run-all-exp      运行所有离线实验"
	@echo ""
	@echo "测试："
	@echo "  make test             运行所有测试"
	@echo "  make test-cov         运行测试并生成覆盖率报告"
	@echo ""
	@echo "Docker（配置文件在 deploy/ 目录）："
	@echo "  make docker-up        启动 CPU 版 Jupyter（第1-5章）"
	@echo "  make docker-up-gpu    启动 GPU 版 Jupyter（第5-6章开源模型）"
	@echo "  make docker-down      停止容器"
	@echo "  make docker-logs      查看容器日志"
	@echo ""
	@echo "清理："
	@echo "  make clean            删除构建产物"
	@echo "  make clean-cache      删除缓存文件"
	@echo ""

# ── 环境安装 ──────────────────────────────────────────────

install:
	pip install -r requirements.txt

install-ml:
	pip install -r requirements.txt
	pip install -e ".[ml]"

install-llm:
	pip install -r requirements.txt
	pip install -e ".[llm]"

install-rag:
	pip install -r requirements.txt
	pip install -e ".[rag]"

install-all:
	pip install -r requirements.txt
	pip install -e ".[all]"

# ── 代码实验 ──────────────────────────────────────────────

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

# 第5章：全部离线实验（不需要 API Key，不需要 GPU）
run-exp-ch05:
	python code/ch05_llm_basics/bpe_tokenization.py
	python code/ch05_llm_basics/scaling_laws.py
	python code/ch05_llm_basics/autoregressive_generation.py
	python code/ch05_llm_basics/training_data_composition.py
	python code/ch05_llm_basics/model_families_evolution.py
	python code/ch05_llm_basics/lora_visualization.py
	python code/ch05_llm_basics/rlhf_pipeline.py
	python code/ch05_llm_basics/benchmark_comparison.py

# 第5章：API 演示（需要设置 ANTHROPIC_API_KEY 或 OPENAI_API_KEY）
run-exp-ch05-api:
	python code/ch05_llm_basics/llm_api_demo.py

run-exp-ch06:
	python code/ch06_llm_applications/rag_demo.py

run-exp-ch07:
	python code/ch07_multimodal_llm/vit_patches.py
	python code/ch07_multimodal_llm/clip_similarity.py
	python code/ch07_multimodal_llm/high_resolution_processing.py
	python code/ch07_multimodal_llm/qwen_vl_analysis.py

# 运行所有离线实验（不需要 API Key，不需要 GPU）
run-all-exp:
	@echo "运行所有离线实验..."
	$(MAKE) run-exp-ch01
	$(MAKE) run-exp-ch02
	$(MAKE) run-exp-ch03
	$(MAKE) run-exp-ch04
	$(MAKE) run-exp-ch05
	$(MAKE) run-exp-ch06
	$(MAKE) run-exp-ch07
	@echo "完成。图表已保存到 assets/ 目录。"

run-jupyter:
	jupyter lab --ip=0.0.0.0 --port=8888 --no-browser

# ── 测试 ──────────────────────────────────────────────────

test:
	pytest tests/ -v

test-cov:
	pytest tests/ -v --cov=code --cov-report=html --cov-report=term

lint:
	flake8 code tests --max-line-length=100
	mypy code --ignore-missing-imports

format:
	black code tests
	isort code tests

# ── Docker（配置文件在 deploy/）────────────────────────────

docker-up:
	docker compose -f deploy/docker-compose.yml up -d
	@echo "Jupyter 已启动：http://localhost:8888"

docker-up-gpu:
	docker compose -f deploy/docker-compose.gpu.yml up -d
	@echo "GPU 版 Jupyter 已启动：http://localhost:8888"

docker-down:
	docker compose -f deploy/docker-compose.yml down
	docker compose -f deploy/docker-compose.gpu.yml down 2>/dev/null || true

docker-logs:
	docker compose -f deploy/docker-compose.yml logs -f jupyter

# ── 清理 ──────────────────────────────────────────────────

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

