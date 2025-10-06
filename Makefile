uv_install_deps:
	uv sync --no-install-project -U --extra aws
uv_install_deps_compile:
	uv sync --all-extras --no-install-project --compile --no-cache
uv_get_lock:
	uv lock
uv_update_deps:
	uv sync --no-install-project --frozen
uv_update_self:
	uv self update
uv_show_deps:
	uv pip list
uv_show_deps_tree:
	uv tree
uv_build_wheel:
	uv build --wheel
uv_create_venv:
	uv venv --python 3.13

pre_commit_install: .pre-commit-config.yaml
	pre-commit install
pre_commit_run: .pre-commit-config.yaml
	pre-commit run --all-files
pre_commit_rm_hooks:
	pre-commit --uninstall-hooks

run_load_subsample:
	python scripts/python/000-load-subsample.py
run_ids_prep:
	python scripts/python/001-prepare-ids-to-freeze.py
run_pretraining:
	python scripts/python/002-gemma-pretraining.py data=kobza tokenizer=tereshchenkoblue
run_test_pretraining:
	python scripts/python/002-gemma-pretraining.py data=kobza_local pretraining=test tokenizer=tereshchenkoblue data_processing=test
run_inference:
	python scripts/python/003-inference.py model=gemma_3_270mb tokenizer=auto

push_checkpoint:
	@if [ -z "$(CHECKPOINT_DIR)" ]; then \
		echo "Error: CHECKPOINT_DIR is required. Usage: make push_checkpoint CHECKPOINT_DIR=path/to/checkpoint HUB_MODEL_ID=username/model-name"; \
		exit 1; \
	fi
	@if [ -z "$(HUB_MODEL_ID)" ]; then \
		echo "Error: HUB_MODEL_ID is required. Usage: make push_checkpoint CHECKPOINT_DIR=path/to/checkpoint HUB_MODEL_ID=username/model-name"; \
		exit 1; \
	fi
	huggingface-cli upload $(HUB_MODEL_ID) $(CHECKPOINT_DIR) --repo-type model
