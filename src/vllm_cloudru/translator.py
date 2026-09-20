import math
import re
from typing import Any

from vllm_cloudru.models import TranslatedRecipe


class UnsupportedRecipe(ValueError):
    pass


_NUMBER = re.compile(r"-?(?:0|[1-9][0-9]*)(?:\.[0-9]+)?(?:[eE][+-]?[0-9]+)?\Z")
_MULTI_PROCESS_FIELDS = (
    "head_argv",
    "worker_argv",
    "worker_argvs",
    "prefill",
    "decode",
    "vllm_argv",
)
_PLATFORM_MANAGED_ARGUMENTS = frozenset({"host", "port", "tensor-parallel-size"})


def _argument(key: str, value: Any, parameter_type: str) -> dict[str, Any]:
    return {"key": key, "value": value, "parameterType": parameter_type}


def _parse_value(value: str) -> tuple[Any, str]:
    if not _NUMBER.fullmatch(value):
        return value, "PARAMETER_TYPE_ARG_KV"

    try:
        number = float(value) if any(marker in value for marker in ".eE") else int(value)
    except (OverflowError, ValueError):
        return value, "PARAMETER_TYPE_ARG_KV"
    if isinstance(number, float) and not math.isfinite(number):
        return value, "PARAMETER_TYPE_ARG_KV"
    return number, "PARAMETER_TYPE_ARG_KV_NUMERIC"


def _parse_arguments(argv: list[Any]) -> list[dict[str, Any]]:
    arguments = []
    index = 3

    while index < len(argv):
        option = argv[index]
        if not isinstance(option, str) or not option.startswith("--") or option == "--":
            raise UnsupportedRecipe("Аргументы рецепта имеют неизвестный формат")

        key = option[2:].split("=", 1)[0]
        if not key or "." in key:
            raise UnsupportedRecipe("Аргументы рецепта имеют неизвестный формат")

        if "=" in option:
            _, raw_value = option[2:].split("=", 1)
            value, parameter_type = _parse_value(raw_value)
            arguments.append(_argument(key, value, parameter_type))
            index += 1
            continue

        if index + 1 == len(argv) or (
            isinstance(argv[index + 1], str) and argv[index + 1].startswith("--")
        ):
            arguments.append(_argument(key, True, "PARAMETER_TYPE_ARG_FLAG"))
            index += 1
            continue

        raw_value = argv[index + 1]
        if not isinstance(raw_value, str):
            raise UnsupportedRecipe("Аргументы рецепта имеют неизвестный формат")
        value, parameter_type = _parse_value(raw_value)
        arguments.append(_argument(key, value, parameter_type))
        index += 2

    # AICODE-NOTE: Cloud.ru derives process binding and vLLM tensor parallelism
    # from the Model Run resource and gpuCount; recipe values must not override it.
    return [
        argument
        for argument in arguments
        if argument["key"] not in _PLATFORM_MANAGED_ARGUMENTS
    ]


def translate_rendering(rendering: dict[str, Any]) -> TranslatedRecipe:
    argv = rendering.get("argv")
    if (
        rendering.get("deploy_type") != "single_node"
        or not isinstance(argv, list)
        or rendering.get("companions")
        or any(field in rendering for field in _MULTI_PROCESS_FIELDS)
    ):
        raise UnsupportedRecipe("Рецепт требует нескольких процессов или узлов")
    if (
        len(argv) < 3
        or argv[:2] != ["vllm", "serve"]
        or not isinstance(argv[2], str)
        or not argv[2]
        or argv[2].startswith("-")
    ):
        raise UnsupportedRecipe("Рецепт не содержит поддерживаемую команду vllm serve")

    env = rendering.get("env", {})
    if not isinstance(env, dict) or any(
        not isinstance(value, str) or "$" in value for value in env.values()
    ):
        raise UnsupportedRecipe("Рецепт содержит неразрешённые переменные окружения")

    profile = rendering.get("hardware_profile")
    if not isinstance(profile, dict):
        profile = {}

    return TranslatedRecipe(
        model_id=argv[2],
        arguments=_parse_arguments(argv),
        environment=[
            _argument(key, value, "PARAMETER_TYPE_ENV") for key, value in env.items()
        ],
        recommended_gpu_count=profile.get("gpu_count"),
        recommended_hardware=profile.get("display_name"),
    )
