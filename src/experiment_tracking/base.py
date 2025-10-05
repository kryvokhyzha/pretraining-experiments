from abc import ABC, abstractmethod


class BaseExperimentTracker(ABC):
    """Base class for experiment trackers."""

    def __init__(self, **kwargs):
        """Initialize with arbitrary keyword arguments from Hydra instantiate."""
        self.config = kwargs

    @abstractmethod
    def init(self, project: str, config: dict[str, any], **kwargs) -> None:
        """Initialize the experiment tracker."""
        pass

    @abstractmethod
    def get_callbacks(self) -> list[any]:
        """Get callbacks for the trainer."""
        pass

    @abstractmethod
    def get_report_to_string(self) -> str:
        """Get the report_to string for TrainingArguments."""
        pass

    @abstractmethod
    def finish(self) -> None:
        """Finish the experiment tracking."""
        pass
