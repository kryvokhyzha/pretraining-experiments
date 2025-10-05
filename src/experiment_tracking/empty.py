from src.experiment_tracking.base import BaseExperimentTracker


class EmptyTracker(BaseExperimentTracker):
    """No experiment tracking."""

    def init(self, project: str, config: dict[str, any], **kwargs) -> None:
        """Initialize empty tracker (no-op)."""
        pass

    def get_callbacks(self) -> list[any]:
        """Get callbacks for the trainer (empty list)."""
        return []

    def get_report_to_string(self) -> str:
        """Get the report_to string for TrainingArguments."""
        return "none"

    def finish(self) -> None:
        """Finish tracking (no-op)."""
        pass
