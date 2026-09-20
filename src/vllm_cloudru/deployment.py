import re
from typing import Any, Protocol

from vllm_cloudru.cloudru import (
    GPU_OPTIONS,
    CloudruCreateUncertain,
    normalize_model_run_endpoint,
    normalize_model_run_id,
)
from vllm_cloudru.models import (
    CreateInferenceRequest,
    CreateInferenceResult,
    RecipeDetail,
    TranslatedRecipe,
)
from vllm_cloudru.recipes import CLOUD_TO_RECIPE_HARDWARE
from vllm_cloudru.translator import UnsupportedRecipe

_INVALID_NAME = re.compile(r"[^a-z0-9-]+")
_REPEATED_HYPHENS = re.compile(r"-+")
_HF_ID = re.compile(r"[^/\s]+/[^/\s]+\Z")
_PROFILE_WARNING = "Для выбранного типа оборудования нет точного профиля vLLM Recipes"


class RecipeProvider(Protocol):
    async def get_detail(
        self,
        hf_id: str,
        resource_type: str,
        selection: Any = None,
    ) -> RecipeDetail: ...


class CloudProvider(Protocol):
    async def create_model_run(
        self,
        project_id: str,
        payload: dict[str, Any],
    ) -> dict[str, Any]: ...


class InvalidDeploymentRequest(ValueError):
    pass


class ConfirmationRequired(ValueError):
    def __init__(self, warnings: list[str]) -> None:
        super().__init__("Требуется подтверждение предупреждений")
        self.warnings = warnings


def normalize_name(name: str) -> str:
    normalized = _REPEATED_HYPHENS.sub("-", _INVALID_NAME.sub("-", name.lower())).strip("-")
    if not normalized or not normalized[0].isalpha():
        normalized = f"model-{normalized}"
    return normalized[:63].rstrip("-") or "model"


def _resource(request: CreateInferenceRequest) -> dict[str, Any]:
    gpu = GPU_OPTIONS.get(request.resource_type)
    if gpu is None:
        raise InvalidDeploymentRequest("Неизвестный тип GPU-ресурса")
    if request.gpu_count not in gpu["counts"]:
        raise InvalidDeploymentRequest(
            "Недопустимое количество GPU для выбранного типа ресурса"
        )
    return gpu


def _warnings(
    request: CreateInferenceRequest,
    detail: RecipeDetail,
    translated: TranslatedRecipe,
) -> list[str]:
    warnings = list(detail.compatibility_warnings)
    expected_hardware = CLOUD_TO_RECIPE_HARDWARE.get(request.resource_type)
    if (
        (expected_hardware is None or detail.rendering.get("hardware") != expected_hardware)
        and not any("профил" in warning.lower() for warning in warnings)
    ):
        warnings.append(_PROFILE_WARNING)

    return list(dict.fromkeys(warnings))


def build_payload(request: CreateInferenceRequest, detail: RecipeDetail) -> dict[str, Any]:
    if not detail.launchable or detail.translated is None:
        raise UnsupportedRecipe(detail.launch_reason)

    gpu = _resource(request)

    translated = detail.translated
    warnings = _warnings(request, detail, translated)
    if warnings and not request.confirm_warnings:
        raise ConfirmationRequired(warnings)

    if _HF_ID.fullmatch(detail.translated.model_id) is None:
        raise UnsupportedRecipe("Рецепт содержит некорректный идентификатор модели")
    repo, model = detail.translated.model_id.split("/", 1)
    task_type = (
        "ModelTaskType_EMBEDDING"
        if any(task.lower() in {"embedding", "embeddings"} for task in detail.tasks)
        else "ModelTaskType_GENERATE"
    )
    model_source: dict[str, Any] = {
        "huggingFaceRepository": {
            "repo": repo,
            "model": model,
            "revision": "main",
        }
    }
    if request.credentials.hf_secret_id:
        model_source["secretLocator"] = {
            "secretId": request.credentials.hf_secret_id,
            "version": 1,
        }
    return {
        "name": normalize_name(request.name),
        "frameworkType": "FrameworkType_VLLM",
        "resourceType": request.resource_type,
        "gpuCount": request.gpu_count,
        "gpuGbMemory": gpu["memoryGb"],
        "runtimeTemplateId": request.runtime_template_id,
        "modelTaskType": task_type,
        "modelSource": model_source,
        "servingOptions": {
            "dynamicalOptions": {
                "args": [*translated.arguments, *translated.environment],
                "loraModules": [],
            },
            "cacheOptions": {},
        },
        "scaling": {
            "minScale": request.min_scale,
            "maxScale": request.max_scale,
            "scalingRules": {"rpsType": {"value": 200}},
        },
        "options": {
            "isEnabledAuth": True,
            "isEnabledLogging": True,
            "auth": {
                "type": request.authentication_type,
                "serviceAccountId": str(request.service_account_id),
            },
            "logGroupId": str(request.log_group_id),
            "publishApiGateway": True,
        },
    }


def _result(
    payload: dict[str, Any],
    fallback_name: str,
) -> CreateInferenceResult:
    model_run_id = normalize_model_run_id(payload.get("modelRunId"))
    if model_run_id is None:
        raise CloudruCreateUncertain(502, "Cloud.ru не вернул modelRunId")

    name = payload.get("name")
    status = payload.get("status")
    endpoint = normalize_model_run_endpoint(
        model_run_id,
        payload.get("publicUrl"),
        payload.get("endpoint"),
    )
    return CreateInferenceResult(
        model_run_id=model_run_id,
        name=name.strip() if isinstance(name, str) and name.strip() else fallback_name.strip(),
        status=status.strip() if isinstance(status, str) and status.strip() else "UNKNOWN",
        endpoint=endpoint,
    )


async def deploy_inference(
    request: CreateInferenceRequest,
    *,
    recipes: RecipeProvider,
    cloud: CloudProvider,
) -> CreateInferenceResult:
    _resource(request)
    detail = await recipes.get_detail(
        request.hf_id,
        resource_type=request.resource_type,
        selection=request.configuration,
    )
    payload = build_payload(request, detail)
    # AICODE-NOTE: Propagate uncertain creates; the BFF exposes no request identity
    # that would make automatic recovery or a repeated POST safe.
    created = await cloud.create_model_run(request.credentials.project_id, payload)
    return _result(created, payload["name"])
