import os
from typing import Any, Callable, Dict, List, Optional

import sagemaker
from sagemaker.remote_function import CustomFileFilter, remote

from src.helper.aws.utils import get_boto_session
from src.helper.logging import logger


class SageMakerRunner:
    """Class for running tasks in SageMaker."""

    _DEFAULT_JOB_NAME_PREFIX = "sms-classification"
    _DEFAULT_VOLUME_SIZE = 30
    _DEFAULT_ENV_VARS = {
        "LOG_LEVEL": "INFO",
        "COLORIZE": "false",
        "HF_HUB_ENABLE_HF_TRANSFER": "1",
        "VLLM_USE_MODELSCOPE": "false",
        "VLLM_LOG_LEVEL": "DEBUG",
    }
    _DEFAULT_COMMANDS = [
        "which poetry || curl -sSL https://install.python-poetry.org | POETRY_VERSION=2.1.3 python -",
        "cd /app/sagemaker_remote_function_workspace",
        "poetry config virtualenvs.create false",
        "poetry install --no-root --extras 'classification aws'",
        "pip install huggingface_hub[cli] hf_transfer",
        "pip install flashinfer-python",
        "pip install outlines==1.0.4",
    ]
    _DEFAULT_IGNORE_PATTERNS = [
        "*.ipynb",
        "data",
        "data_old",
        "docker",
        "results",
        "tests",
        "weights",
    ]

    def __init__(
        self,
        image_uri: str,
        role: str,
        instance_type: str,
        region: Optional[str] = None,
        profile_name: Optional[str] = None,
        job_name_prefix: Optional[str] = None,
        volume_size: Optional[int] = None,
        sm_remote_kwargs: Optional[Dict[str, Any]] = None,
        sm_session_kwargs: Optional[Dict[str, Any]] = None,
    ):
        """Initialize the SageMaker runner.

        Args:
        ----
            image_uri: URI of the Docker image to use
            role: IAM role for SageMaker execution
            instance_type: EC2 instance type to use
            region: AWS region
            profile_name: AWS profile name
            job_name_prefix: Prefix for SageMaker job names
            volume_size: Size of the EBS volume in GB
            sm_remote_kwargs: Additional arguments for the remote function
            sm_session_kwargs: Additional arguments for the SageMaker session

        """
        self.image_uri = image_uri
        self.role = role
        self.instance_type = instance_type
        self.job_name_prefix = job_name_prefix or self._DEFAULT_JOB_NAME_PREFIX
        self.volume_size = volume_size or self._DEFAULT_VOLUME_SIZE

        self.boto_session = get_boto_session(profile_name=profile_name, region_name=region)
        self.sm_remote_kwargs = sm_remote_kwargs or {}
        self.sm_session = sagemaker.Session(boto_session=self.boto_session, **(sm_session_kwargs or {}))

    def get_remote_settings(
        self,
        environment_variables: Optional[Dict[str, str]] = None,
        pre_execution_commands: Optional[List[str]] = None,
        ignore_patterns: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        """Get settings for the remote function."""
        env_vars = self._DEFAULT_ENV_VARS.copy()

        hf_token = os.getenv("HF_ACCESS_TOKEN")
        if hf_token:
            env_vars["HF_TOKEN"] = hf_token

        if environment_variables:
            env_vars.update(environment_variables)

        commands = pre_execution_commands if pre_execution_commands is not None else self._DEFAULT_COMMANDS

        patterns = ignore_patterns if ignore_patterns is not None else self._DEFAULT_IGNORE_PATTERNS

        settings = {
            "dependencies": None,
            "pre_execution_commands": commands,
            "sagemaker_session": self.sm_session,
            "image_uri": self.image_uri,
            "role": self.role,
            "instance_type": self.instance_type,
            "environment_variables": env_vars,
            "include_local_workdir": True,
            "custom_file_filter": CustomFileFilter(ignore_name_patterns=patterns),
            "job_name_prefix": f"{self.job_name_prefix}-{self.instance_type.replace('.', '-')}",
            "volume_size": self.volume_size,
        }

        settings.update(self.sm_remote_kwargs)

        return settings

    def run_remote_task(
        self,
        task_fn: Callable,
        task_args: Dict[str, Any],
        environment_variables: Optional[Dict[str, str]] = None,
        pre_execution_commands: Optional[List[str]] = None,
        ignore_patterns: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        """Run a task remotely in SageMaker."""
        settings = self.get_remote_settings(
            environment_variables=environment_variables,
            pre_execution_commands=pre_execution_commands,
            ignore_patterns=ignore_patterns,
        )

        logger.info(f"Remote settings: {settings}")

        @remote(**settings)
        def run_task(kwargs):
            return task_fn(**kwargs)

        logger.info(f"Starting remote task on {self.instance_type}")
        result = run_task(kwargs=task_args)
        logger.info(f"Task completed with result: {result}")

        return result
