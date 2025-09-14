from enum import Enum


class ExperimentTrackerType(str, Enum):
    """Enum for experiment tracker types."""

    WANDB = "wandb"
    TENSORBOARD = "tensorboard"
    NONE = "none"

    @classmethod
    def from_string(cls, value: str) -> "ExperimentTrackerType":
        """Convert string to ExperimentTrackerType with fallback mapping."""
        value = value.lower().strip()

        # Direct matches
        for tracker_type in cls:
            if value == tracker_type.value:
                return tracker_type

        # Alias mapping
        alias_mapping = {
            "weights_and_biases": cls.WANDB,
            "tb": cls.TENSORBOARD,
            "null": cls.NONE,
            "empty": cls.NONE,
            "": cls.NONE,
        }

        if value in alias_mapping:
            return alias_mapping[value]

        raise ValueError(f"Unknown experiment tracker type: {value}")

    def get_report_to_string(self) -> str:
        """Get the report_to string for TrainingArguments."""
        match self:
            case ExperimentTrackerType.WANDB:
                return "wandb"
            case ExperimentTrackerType.TENSORBOARD:
                return "tensorboard"
            case ExperimentTrackerType.NONE:
                return "none"
