"""Resolve model-specific serving runner hooks without hard-coding them in runtime hooks."""

from __future__ import annotations

import importlib
import os
from typing import Any


class ModelAdapterError(RuntimeError):
    pass


def load_runner_class(model: dict[str, Any] | None = None) -> type[Any]:
    """Load the serving runner class selected by the model adapter.

    Existing Qwen runs retain a compatibility default. New models should set
    ``PYPTO_MODEL_RUNNER_MODULE`` and ``PYPTO_MODEL_RUNNER_CLASS`` explicitly.
    """
    module_name = os.environ.get("PYPTO_MODEL_RUNNER_MODULE", "").strip()
    class_name = os.environ.get("PYPTO_MODEL_RUNNER_CLASS", "").strip()
    family = "" if model is None else str(model.get("family", "")).strip().lower()
    if not module_name or not class_name:
        if family not in {"", "qwen"}:
            raise ModelAdapterError(
                "new model requires PYPTO_MODEL_RUNNER_MODULE and "
                "PYPTO_MODEL_RUNNER_CLASS"
            )
        module_name = "pypto_serving.model.qwen.npu_runner"
        class_name = "Qwen314BModelRunner"
    try:
        module = importlib.import_module(module_name)
        runner = getattr(module, class_name)
    except (ImportError, AttributeError) as error:
        raise ModelAdapterError(
            f"cannot load model runner adapter {module_name}:{class_name}"
        ) from error
    if not isinstance(runner, type):
        raise ModelAdapterError(f"model runner adapter is not a class: {module_name}:{class_name}")
    return runner
