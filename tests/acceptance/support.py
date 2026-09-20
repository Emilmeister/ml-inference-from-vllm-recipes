from __future__ import annotations

import json
from typing import Any

from playwright.sync_api import Page, Route

STORAGE_KEY = "vllm-cloudru.credentials.v1"
PROJECT_ID = "11111111-1111-1111-1111-111111111111"
SERVICE_ACCOUNT_ID = "22222222-2222-2222-2222-222222222222"
SECOND_SERVICE_ACCOUNT_ID = "33333333-3333-3333-3333-333333333333"
LOG_GROUP_ID = "44444444-4444-4444-4444-444444444444"
SECOND_LOG_GROUP_ID = "55555555-5555-5555-5555-555555555555"
HF_SECRET_ID = "66666666-6666-6666-6666-666666666666"


def fulfill_json(route: Route, payload: Any, status: int = 200) -> None:
    route.fulfill(
        status=status,
        content_type="application/json",
        body=json.dumps(payload),
    )


def store_credentials(page: Page, base_url: str, *, expires_at: int | None = None) -> None:
    page.goto(base_url)
    stored = {
        "projectId": PROJECT_ID,
        "keyId": "test-key-id",
        "keySecret": "test-key-secret",
        "hfSecretId": HF_SECRET_ID,
        "expiresAt": expires_at if expires_at is not None else 4_102_444_800_000,
    }
    page.evaluate(
        "([key, value]) => localStorage.setItem(key, JSON.stringify(value))",
        [STORAGE_KEY, stored],
    )


def catalog_item(
    hf_id: str = "Acme/Small",
    *,
    provider: str = "Acme",
    parameter_count: str = "8B",
    launchable: bool = True,
    warning: str | None = None,
    launch_reason: str | None = None,
) -> dict[str, Any]:
    return {
        "hf_id": hf_id,
        "title": hf_id.split("/", 1)[1],
        "provider": provider,
        "url": f"https://huggingface.co/{hf_id}",
        "json": f"recipes/{hf_id}.json",
        "description": f"Recipe for {hf_id}",
        "tasks": ["generate"],
        "parameter_count": parameter_count,
        "variants": {"fp8": {"precision": "FP8"}},
        "recommended_hardware": "H100",
        "recommended_gpu_count": 2,
        "compatibility_warnings": [warning] if warning else [],
        "launchable": launchable,
        "launch_reason": launch_reason,
    }


def recipe_detail(
    hf_id: str = "Acme/Small",
    *,
    frontend: str = "rust",
    launchable: bool = True,
    warning: str | None = None,
    launch_reason: str | None = None,
) -> dict[str, Any]:
    item = catalog_item(
        hf_id,
        launchable=launchable,
        warning=warning,
        launch_reason=launch_reason,
    )
    item.update(
        {
            "active_parameters": "8B",
            "context_length": 32_768,
            "min_vllm_version": "0.12.0",
            "selected_variant": "fp8",
            "rendering_path": "renderings/single-node.json",
            "rendering": {"hardware": "H100"},
            "translated": {
                "model_id": "Acme/Small-FP8",
                "arguments": [
                    {
                        "key": "tool-call-parser",
                        "value": "hermes",
                        "parameterType": "PARAMETER_TYPE_ARG_KV",
                    }
                ],
                "environment": (
                    [
                        {
                            "key": "VLLM_USE_RUST_FRONTEND",
                            "value": "1",
                            "parameterType": "PARAMETER_TYPE_ENV",
                        }
                    ]
                    if frontend == "rust"
                    else []
                ),
                "recommended_gpu_count": 2,
                "recommended_hardware": "H100",
            }
            if launchable
            else None,
            "configuration": {
                "selection": {
                    "variant": "fp8",
                    "strategy": "tensor_parallel",
                    "kv_offload": "off",
                    "frontend": frontend,
                    "features": ["tool_calling"],
                    "feature_modes": {},
                },
                "groups": [
                    {
                        "key": "variant",
                        "label": "Variant",
                        "control": "single",
                        "options": [
                            {
                                "key": "fp8",
                                "label": "FP8",
                                "available": True,
                                "selected": True,
                            },
                            {
                                "key": "bf16",
                                "label": "BF16",
                                "available": True,
                                "selected": False,
                            },
                        ],
                    },
                    {
                        "key": "strategy",
                        "label": "Strategy",
                        "control": "single",
                        "options": [
                            {
                                "key": "tensor_parallel",
                                "label": "Tensor Parallel",
                                "available": True,
                                "selected": True,
                            },
                            {
                                "key": "multi_node",
                                "label": "Multi-node",
                                "available": False,
                                "unavailable_reason": "Cloud.ru принимает один Model Run",
                                "selected": False,
                            },
                        ],
                    },
                    {
                        "key": "frontend",
                        "label": "Frontend",
                        "control": "single",
                        "options": [
                            {
                                "key": "rust",
                                "label": "Rust",
                                "available": True,
                                "selected": frontend == "rust",
                            },
                            {
                                "key": "python",
                                "label": "Python",
                                "available": True,
                                "selected": frontend == "python",
                            },
                        ],
                    },
                    {
                        "key": "features",
                        "label": "Features",
                        "control": "multiple",
                        "options": [
                            {
                                "key": "tool_calling",
                                "label": "Tool Calling",
                                "available": True,
                                "selected": True,
                            },
                            {
                                "key": "companion",
                                "label": "Companion service",
                                "available": False,
                                "unavailable_reason": "Требуется отдельный процесс",
                                "selected": False,
                            },
                        ],
                    },
                ],
            },
        }
    )
    return item


def cloud_options() -> dict[str, Any]:
    return {
        "runtimeTemplates": [
            {
                "id": "runtime-vllm",
                "frameworkType": "FrameworkType_VLLM",
                "version": "0.12.0",
                "gpus": [{"resourceType": "ResourceType_GPU_H100", "isDefault": True}],
            }
        ],
        "gpuOptions": {
            "ResourceType_GPU_H100": {
                "label": "NVIDIA H100 80 GB",
                "memoryGb": 80,
                "counts": [1, 2, 4, 6, 8],
            }
        },
        "quotas": [{"resourceType": "ResourceType_GPU_H100", "free": 8, "limit": 8}],
        "serviceAccounts": [
            {"id": SERVICE_ACCOUNT_ID, "name": "inference", "description": "Primary"},
            {
                "id": SECOND_SERVICE_ACCOUNT_ID,
                "name": "inference-second",
                "description": "Secondary",
            },
        ],
        "logGroups": [
            {
                "id": LOG_GROUP_ID,
                "name": "default",
                "description": "Default logs",
                "isDefault": True,
            },
            {
                "id": SECOND_LOG_GROUP_ID,
                "name": "audit",
                "description": "Audit logs",
                "isDefault": False,
            },
        ],
    }


def model_run(
    name: str = "demo",
    *,
    model_run_id: str = "run-demo",
    status: str = "RUNNING",
) -> dict[str, Any]:
    return {
        "modelRunId": model_run_id,
        "name": name,
        "model": "Acme/Small-FP8",
        "status": status,
        "resourceType": "ResourceType_GPU_H100",
        "gpuCount": 2,
        "scaling": {"minScale": 1, "maxScale": 1},
        "endpoint": f"https://{model_run_id}.modelrun.inference.cloud.ru/v1",
    }

