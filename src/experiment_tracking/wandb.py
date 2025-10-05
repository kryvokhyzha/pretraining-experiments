import os

from src.experiment_tracking.base import BaseExperimentTracker


class WandbTracker(BaseExperimentTracker):
    """Weights & Biases experiment tracker."""

    def __init__(
        self,
        name: str | None = None,
        tags: list[str] | None = None,
        notes: str | None = None,
        reinit: str = "finish_previous",
        **kwargs,
    ):
        """Initialize WandbTracker with specific parameters."""
        super().__init__(name=name, tags=tags, notes=notes, reinit=reinit, **kwargs)

    def init(self, project: str, config: dict[str, any], **kwargs) -> None:
        """Initialize wandb tracking."""
        try:
            import wandb

            # Login with API key if provided
            api_key = os.getenv("WANDB_API_KEY") or os.getenv("WANDB_KEY")
            if api_key:
                wandb.login(key=api_key, relogin=False)

            # Initialize wandb
            wandb.init(
                project=project,
                config=config,
                name=self.config.get("name", None),
                tags=self.config.get("tags", None),
                notes=self.config.get("notes", None),
                reinit=self.config.get("reinit", "finish_previous"),
                **kwargs,
            )

        except ImportError:
            raise ImportError("wandb is required for WandbTracker. Install with: `pip install wandb`")

    def get_callbacks(self) -> list[any]:
        """Get wandb callbacks for the trainer."""
        try:
            from transformers.integrations import WandbCallback

            return [WandbCallback]
        except ImportError:
            return []

    def get_report_to_string(self) -> str:
        """Get the report_to string for TrainingArguments."""
        return "wandb"

    def finish(self) -> None:
        """Finish wandb tracking."""
        try:
            import wandb

            wandb.finish()
        except ImportError:
            pass
