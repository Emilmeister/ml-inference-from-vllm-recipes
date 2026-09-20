from typing import Annotated

import httpx
from fastapi import APIRouter, HTTPException, Query, Request
from pydantic import BaseModel, ConfigDict, ValidationError

from vllm_cloudru.cloudru import CloudruCreateUncertain, CloudruError
from vllm_cloudru.deployment import (
    ConfirmationRequired,
    InvalidDeploymentRequest,
    deploy_inference,
)
from vllm_cloudru.models import (
    CloudCredentials,
    CloudOptions,
    CreateInferenceRequest,
    CreateInferenceResult,
    ModelRunSummary,
    RecipeCatalogItem,
    RecipeConfigurationSelection,
    RecipeDetail,
)
from vllm_cloudru.recipe_configuration import RecipeConfigurationError
from vllm_cloudru.recipes import (
    RecipeDataError,
    RecipeNotFound,
    RecipeRefreshThrottled,
    RecipeResourceTypeError,
)
from vllm_cloudru.translator import UnsupportedRecipe

router = APIRouter(prefix="/api")


class CredentialsEnvelope(BaseModel):
    model_config = ConfigDict(extra="forbid")

    credentials: CloudCredentials


def _cloud_error(error: CloudruError) -> HTTPException:
    return HTTPException(
        status_code=error.status_code,
        detail={"code": "cloud_error", "message": error.public_message},
    )


def _recipe_error(error: Exception) -> HTTPException:
    if isinstance(error, RecipeNotFound):
        return HTTPException(
            status_code=404,
            detail={"code": "recipe_not_found", "message": str(error)},
        )
    return HTTPException(
        status_code=502,
        detail={
            "code": "recipes_unavailable",
            "message": "Не удалось получить корректные данные vLLM Recipes",
        },
    )


def _configuration_error(error: Exception) -> HTTPException:
    return HTTPException(
        status_code=422,
        detail={"code": "invalid_recipe_configuration", "message": str(error)},
    )


@router.get("/recipes")
async def recipes(
    request: Request,
    refresh: bool = False,
    resource_type: str = "ResourceType_GPU_H100",
) -> list[RecipeCatalogItem]:
    try:
        return await request.app.state.recipes.list_catalog(
            refresh=refresh,
            resource_type=resource_type,
        )
    except RecipeResourceTypeError as error:
        raise HTTPException(
            status_code=422,
            detail={"code": "invalid_resource_type", "message": str(error)},
        ) from error
    except RecipeRefreshThrottled as error:
        raise HTTPException(
            status_code=429,
            detail={"code": "catalog_throttled", "message": str(error)},
            headers={"Retry-After": "30"},
        ) from error
    except (httpx.HTTPError, ValidationError, RecipeDataError) as error:
        raise _recipe_error(error) from error


@router.get("/recipes/{organization}/{model}")
async def recipe_detail(
    organization: str,
    model: str,
    request: Request,
    resource_type: str = "ResourceType_GPU_H100",
    variant: str | None = None,
    strategy: str | None = None,
    kv_offload: str | None = None,
    frontend: str | None = None,
    feature: Annotated[list[str] | None, Query()] = None,
    feature_mode: Annotated[list[str] | None, Query()] = None,
    features_configured: bool = False,
) -> RecipeDetail:
    try:
        modes: dict[str, str] = {}
        for item in feature_mode or []:
            key, separator, value = item.partition(":")
            if not separator or not key or not value or key in modes:
                raise RecipeConfigurationError("Некорректный feature_mode")
            modes[key] = value
        selection = RecipeConfigurationSelection(
            variant=variant,
            strategy=strategy,
            kv_offload=kv_offload,
            frontend=frontend,
            features=(feature or []) if features_configured else None,
            feature_modes=modes,
        )
    except (RecipeConfigurationError, ValidationError) as error:
        raise _configuration_error(error) from error

    try:
        return await request.app.state.recipes.get_detail(
            f"{organization}/{model}",
            resource_type=resource_type,
            selection=selection,
        )
    except RecipeConfigurationError as error:
        raise _configuration_error(error) from error
    except RecipeResourceTypeError as error:
        raise HTTPException(
            status_code=422,
            detail={"code": "invalid_resource_type", "message": str(error)},
        ) from error
    except (
        RecipeNotFound,
        httpx.HTTPError,
        ValidationError,
        RecipeDataError,
    ) as error:
        raise _recipe_error(error) from error


@router.post("/cloud/validate")
async def validate_cloud(
    body: CredentialsEnvelope,
    request: Request,
) -> dict[str, bool]:
    client = request.app.state.cloud_factory(body.credentials)
    try:
        await client.list_model_runs(body.credentials.project_id, limit=1)
    except CloudruError as error:
        raise _cloud_error(error) from error
    return {"valid": True}


@router.post("/cloud/options")
async def cloud_options(body: CredentialsEnvelope, request: Request) -> CloudOptions:
    client = request.app.state.cloud_factory(body.credentials)
    try:
        return await client.options(body.credentials.project_id)
    except CloudruError as error:
        raise _cloud_error(error) from error


@router.post("/cloud/inferences/query")
async def inferences(
    body: CredentialsEnvelope,
    request: Request,
) -> list[ModelRunSummary]:
    client = request.app.state.cloud_factory(body.credentials)
    try:
        return await client.list_model_runs(body.credentials.project_id)
    except CloudruError as error:
        raise _cloud_error(error) from error


@router.post("/cloud/inferences", status_code=201)
async def create_inference(
    body: CreateInferenceRequest,
    request: Request,
) -> CreateInferenceResult:
    client = request.app.state.cloud_factory(body.credentials)
    try:
        return await deploy_inference(
            body,
            recipes=request.app.state.recipes,
            cloud=client,
        )
    except ConfirmationRequired as error:
        raise HTTPException(
            status_code=409,
            detail={"code": "confirmation_required", "warnings": error.warnings},
        ) from error
    except UnsupportedRecipe as error:
        raise HTTPException(
            status_code=422,
            detail={"code": "unsupported_recipe", "message": str(error)},
        ) from error
    except InvalidDeploymentRequest as error:
        raise HTTPException(
            status_code=422,
            detail={"code": "invalid_deployment", "message": str(error)},
        ) from error
    except RecipeConfigurationError as error:
        raise _configuration_error(error) from error
    except RecipeNotFound as error:
        raise _recipe_error(error) from error
    except (httpx.HTTPError, ValidationError, RecipeDataError) as error:
        raise _recipe_error(error) from error
    except CloudruCreateUncertain as error:
        raise HTTPException(
            status_code=error.status_code,
            detail={
                "code": "create_uncertain",
                "message": error.public_message,
                "manualCheckRequired": True,
                "instruction": "Обновите список Model Run вручную перед повторной попыткой.",
            },
        ) from error
    except CloudruError as error:
        raise _cloud_error(error) from error
