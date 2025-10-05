from src.experiment_tracking.base import BaseExperimentTracker


class TensorboardTracker(BaseExperimentTracker):
    """TensorBoard experiment tracker."""

    def init(self, project: str, config: dict[str, any], **kwargs) -> None:
        """Initialize TensorBoard tracking.

        TensorBoard initialization is handled by transformers automatically
        when report_to includes "tensorboard".
        """
        pass

    def get_callbacks(self) -> list[any]:
        """Get TensorBoard callbacks for the trainer."""
        try:
            from transformers.integrations import TensorBoardCallback

            return [TensorBoardCallback]
        except ImportError:
            return []

    def get_report_to_string(self) -> str:
        """Get the report_to string for TrainingArguments."""
        return "tensorboard"

    def finish(self) -> None:
        """Finish TensorBoard tracking.

        TensorBoard doesn't require explicit finishing.
        """
        pass
