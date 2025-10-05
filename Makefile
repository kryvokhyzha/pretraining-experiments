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
