# Pretraining Experiments

|            |                                                                                                                       |
| ---------- | --------------------------------------------------------------------------------------------------------------------- |
| CI/Testing | -                                                                                                                     |
| Package    | [![Python](https://img.shields.io/badge/python-3.13-blue.svg)](https://www.python.org/downloads/release/python-3113/) |
| Meta       | [![Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff) [![GitHub License](https://img.shields.io/github/license/kryvokhyzha/pretraining-experiments)](https://github.com/kryvokhyzha/pretraining-experiments/blob/main/LICENSE) [![WandB](https://raw.githubusercontent.com/wandb/assets/main/wandb-github-badge-gradient.svg)](https://wandb.ai/site)      |

---

## 📖 About

> [!IMPORTANT]
> This is just research code written for fun over a few hours over a glass of beer🍻

This repository contains my personal **experiments with LLM pretraining**,
specifically focused on pretraining **Gemma 3 270M** models on Ukrainian text
data using the Kobza dataset.

---

## 🚀 Quick Start

### Prerequisites

> [!IMPORTANT]
>
> - Python 3.13
> - `uv` package manager
>   ([installation guide](https://docs.astral.sh/uv/getting-started/installation/))
> - Hugging Face account with API token for model access

### Running an Experiment

1. **Set up environment variables**:

   ```bash
   # Copy the example environment file and fill in your credentials
   cp .env.example .env
   # Edit .env file to add your Hugging Face token and other settings
   ```

2. **Prepare data and run a test experiment**:

   ```bash
   # Load and subsample data
   make run_load_subsample

   # Prepare token IDs to freeze (for selective freezing experiments)
   make run_ids_prep

   # Run a quick test training
   make run_test_pretraining
   ```

3. **Run a full pretraining experiment**:

   ```bash
   make run_pretraining
   ```

> [!WARNING]
> Full pretraining experiments are time-consuming. Always start with the test configuration first.

### Experiment Configuration

The experiments use **Hydra** for configuration management. All configs are in
the `configs/` directory.

---

## 🧪 Experiment Examples

### Running Different Configurations

You can modify experiments using Hydra's command-line syntax:

```bash
# Use different tokenizer
python scripts/python/002-gemma-pretraining.py tokenizer=tereshchenkoblue

# Change training parameters
python scripts/python/002-gemma-pretraining.py training.learning_rate=1e-4 training.per_device_train_batch_size=2

# Use local data
python scripts/python/002-gemma-pretraining.py data=kobza_local

# Disable experiment tracking
python scripts/python/002-gemma-pretraining.py experiment=none
```

### Available Scripts

- `scripts/python/000-load-subsample.py` - Load and subsample dataset for
  testing
- `scripts/python/001-prepare-ids-to-freeze.py` - Prepare token IDs for
  selective freezing
- `scripts/python/002-gemma-pretraining.py` - Main pretraining experiment script

### Useful Makefile Commands

```bash
# Environment setup
make uv_install_deps          # Install dependencies
make pre_commit_install       # Install code quality hooks

# Run experiments
make run_load_subsample       # Prepare sample data
make run_ids_prep             # Prepare token freezing data
make run_test_pretraining     # Quick test experiment
make run_pretraining          # Full experiment
```

---

## ⚙️ Development Environment Setup

> [!WARNING]
> This project is based on `Python 3.13` and uses `uv` for dependency management.

1. Clone the repository:

   ```bash
   git clone <repository-url>
   cd pretraining-experiments
   ```

1. Create a virtual environment:

   ```bash
   # Using venv
   uv venv --python 3.13
   ```

1. Activate the environment:

   ```bash
   # Using venv
   source .venv/bin/activate
   ```

1. Install `uv` following the
   [official documentation](https://docs.astral.sh/uv/getting-started/installation/).

1. Install dependencies:

   ```bash
   uv sync --no-install-project -U
   ```

1. Setup pre-commit hooks:

   ```bash
   pre-commit install
   ```

---

## 🔧 Troubleshooting

### Common Issues

**Problem**: `HF_TOKEN` not found error  
**Solution**: Make sure you've created a `.env` file with your Hugging Face
token:

```bash
cp .env.example .env
# Then edit .env to add your actual HF_TOKEN value
```

**Problem**: Out of memory during training  
**Solution**: Reduce batch size in training configuration:

```bash
python scripts/python/002-gemma-pretraining.py training.per_device_train_batch_size=1
```

> [!NOTE]
> Check the training logs in `outputs/` directory for more detailed error information.

---

## 📝 Contributing

1. Fork repository and create a feature branch.
2. Follow existing code style (enforced by pre-commit hooks).
3. Add tests for new functionality.
4. Submit a PR for review.
