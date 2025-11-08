from transformers import Trainer

from src.helper.logging import logger


class NoDecayEmbeddingsTrainer(Trainer):
    """Custom Trainer that disables weight decay for embedding layers.

    This is necessary when freezing specific token embeddings, as weight decay
    would still modify the frozen embeddings even when gradients are zero.
    """

    def get_decay_parameter_names(self, model) -> list[str]:
        """Override to exclude embedding parameters from weight decay.

        This prevents weight decay from being applied to embedding layers,
        which is crucial when freezing specific token embeddings.
        """
        # Get the default decay parameters using parent class method
        decay_parameters = super().get_decay_parameter_names(model)

        # Filter out embedding and lm_head parameters
        # These should not have weight decay when we're freezing tokens
        filtered_parameters = []
        for param_name in decay_parameters:
            # Exclude parameters with 'embed' or 'lm_head' in their name
            if "embed" not in param_name.lower() and "lm_head" not in param_name.lower():
                filtered_parameters.append(param_name)

        logger.debug(
            f"Excluding {len(decay_parameters) - len(filtered_parameters)} "
            "embedding/lm_head parameters from weight decay"
        )

        return filtered_parameters
