import asyncio
import copy
import json
import logging
import math
import re
from collections.abc import Awaitable, Callable
from typing import Any
from urllib.parse import urlencode, urlsplit
from uuid import UUID, uuid4

import httpx

from vllm_cloudru.models import CloudCredentials

logger = logging.getLogger(__name__)

IAM_URL = "https://iam.api.cloud.ru/api/v1/auth/token"
BFF_URL = "https://console.cloud.ru"
BFF_PREFIX = "/u-api/inference/model-run/v1"
SERVICE_ACCOUNTS_PREFIX = "/u-api/bff-console/v1/projects"
LOG_GROUPS_PREFIX = "/u-api/logaas-bff-console/v1"
GPU_OPTIONS = {
    "ResourceType_GPU_V100": {
        "label": "NVIDIA V100 32 GB",
        "memoryGb": 32,
        "counts": [1, 2, 4, 8, 16],
    },
    "ResourceType_GPU_A100_NVLINK": {
        "label": "NVIDIA A100 80 GB",
        "memoryGb": 80,
        "counts": [1, 2, 4, 6, 8],
    },
    "ResourceType_GPU_H100": {
        "label": "NVIDIA H100 80 GB",
        "memoryGb": 80,
        "counts": [1, 2, 4, 6, 8],
    },
}

_RETRY_STATUSES = {502, 503, 504}
_SAFE_REQUEST_ID = re.compile(r"[A-Za-z0-9._:-]{1,128}\Z")
_MODEL_RUN_ID = re.compile(r"[A-Za-z0-9-]{1,128}\Z")
_LOG_BODY_LIMIT = 4_000
_REDACTED = "<redacted>"
_ERROR_MESSAGES = {
    401: "Неверный Key ID или Key Secret",
    403: "У сервисного аккаунта недостаточно прав",
    409: "Имя Model Run уже занято",
    422: "Выбранная конфигурация или GPU-ресурс недоступны",
    429: "Недостаточно квоты Cloud.ru для запуска",
}


class CloudruError(RuntimeError):
    def __init__(self, status_code: int, public_message: str) -> None:
        super().__init__(public_message)
        self.status_code = status_code
        self.public_message = public_message


class CloudruCreateUncertain(CloudruError):
    pass


def _sensitive_field(key: Any) -> bool:
    if not isinstance(key, str):
        return False
    normalized = re.sub(r"[^a-z0-9]", "", key.lower())
    return any(
        marker in normalized
        for marker in ("secret", "token", "password", "authorization", "apikey")
    )


def _collect_sensitive_values(value: Any, result: set[str]) -> None:
    if isinstance(value, dict):
        for key, item in value.items():
            if _sensitive_field(key):
                if isinstance(item, str) and len(item) >= 4:
                    result.add(item)
                continue
            _collect_sensitive_values(item, result)
    elif isinstance(value, list):
        for item in value:
            _collect_sensitive_values(item, result)


def _redact(value: Any, sensitive_values: set[str]) -> Any:
    if isinstance(value, dict):
        return {
            str(key): _REDACTED if _sensitive_field(key) else _redact(item, sensitive_values)
            for key, item in value.items()
        }
    if isinstance(value, list):
        return [_redact(item, sensitive_values) for item in value]
    if isinstance(value, str):
        result = value
        for secret in sorted(sensitive_values, key=len, reverse=True):
            result = result.replace(secret, _REDACTED)
        return result
    return value


def _log_body(content: bytes, sensitive_values: set[str]) -> str:
    if not content:
        return "<empty>"
    try:
        value = json.loads(content)
    except (UnicodeDecodeError, json.JSONDecodeError):
        value = content.decode("utf-8", errors="replace")
    _collect_sensitive_values(value, sensitive_values)
    serialized = json.dumps(
        _redact(value, sensitive_values),
        ensure_ascii=False,
        separators=(",", ":"),
    )
    if len(serialized) <= _LOG_BODY_LIMIT:
        return serialized
    return f"{serialized[:_LOG_BODY_LIMIT]}…<truncated>"


def _log_http_error(response: httpx.Response) -> None:
    # AICODE-NOTE: Never log raw headers or bodies at this trust boundary;
    # response text can echo credentials supplied in the corresponding request.
    if response.extensions.get("vllm_cloudru_error_logged") is True:
        return
    response.extensions["vllm_cloudru_error_logged"] = True
    try:
        request = response.request
        sensitive_values: set[str] = set()
        authorization = request.headers.get("Authorization", "")
        if authorization.lower().startswith("bearer ") and len(authorization) > 11:
            sensitive_values.add(authorization[7:])

        query = []
        for key, value in request.url.params.multi_items():
            if _sensitive_field(key):
                if len(value) >= 4:
                    sensitive_values.add(value)
                value = _REDACTED
            query.append((key, value))
        safe_url = request.url.copy_with(
            query=urlencode(query).encode() if query else None
        )
        request_body = _log_body(request.content, sensitive_values)
        response_body = _log_body(response.content, sensitive_values)
        request_id = response.headers.get("X-Request-ID") or request.headers.get(
            "X-Request-ID", ""
        )
        if not _SAFE_REQUEST_ID.fullmatch(request_id):
            request_id = "<missing>"
        logger.error(
            "cloudru_http_error method=%s url=%s status=%s request_id=%s "
            "request_body=%s response_body=%s",
            request.method,
            safe_url,
            response.status_code,
            request_id,
            request_body,
            response_body,
        )
    except Exception as error:
        logger.error(
            "cloudru_http_error logging_failed status=%s error_type=%s",
            response.status_code,
            type(error).__name__,
        )


def _project_segment(project_id: str) -> str:
    try:
        return str(UUID(project_id))
    except (AttributeError, TypeError, ValueError):
        raise CloudruError(400, "Project ID должен быть UUID") from None


def _raise_for_status(
    response: httpx.Response,
    *,
    not_found_message: str = "Проект или ресурс не найден",
) -> None:
    if 200 <= response.status_code < 300:
        return

    _log_http_error(response)

    message = (
        not_found_message
        if response.status_code == 404
        else _ERROR_MESSAGES.get(response.status_code)
    )
    if message is None:
        message = f"Cloud.ru вернул HTTP {response.status_code}"
        request_id = response.headers.get("X-Request-ID")
        if request_id and _SAFE_REQUEST_ID.fullmatch(request_id):
            message = f"{message} (request ID: {request_id})"
    raise CloudruError(response.status_code, message)


def _json_object(
    response: httpx.Response,
    *,
    not_found_message: str = "Проект или ресурс не найден",
) -> dict[str, Any]:
    _raise_for_status(response, not_found_message=not_found_message)
    try:
        payload = response.json()
    except ValueError:
        raise CloudruError(502, "Cloud.ru вернул некорректный JSON") from None
    if not isinstance(payload, dict):
        raise CloudruError(502, "Cloud.ru вернул некорректный JSON")
    return payload


def _string(value: Any, field: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise CloudruError(502, f"Cloud.ru не вернул поле {field}")
    return value.strip()


def _optional_string(value: Any) -> str:
    return value.strip() if isinstance(value, str) else ""


def _optional_int(value: Any) -> int | None:
    return value if isinstance(value, int) and not isinstance(value, bool) else None


def normalize_model_run_id(value: Any) -> str | None:
    if not isinstance(value, str):
        return None
    normalized = value.strip()
    return normalized if _MODEL_RUN_ID.fullmatch(normalized) is not None else None


def normalize_model_run_endpoint(model_run_id: str, *candidates: Any) -> str:
    normalized_id = normalize_model_run_id(model_run_id)
    if normalized_id is None:
        return ""
    expected_host = f"{normalized_id}.modelrun.inference.cloud.ru"
    for candidate in candidates:
        endpoint = _optional_string(candidate)
        if not endpoint:
            continue
        try:
            parsed = urlsplit(endpoint)
            valid = (
                parsed.scheme == "https"
                and parsed.hostname == expected_host
                and parsed.port is None
                and parsed.username is None
                and parsed.password is None
                and parsed.path in {"/v1", "/v1/"}
                and not parsed.query
                and not parsed.fragment
            )
        except ValueError:
            continue
        if valid:
            return endpoint
    return ""


def _model_label(source: Any) -> str:
    if not isinstance(source, dict):
        return ""
    for source_type in ("huggingFaceRepository", "ollama", "modelRegistry", "modelScope"):
        candidate = source.get(source_type)
        if not isinstance(candidate, dict):
            continue
        repo = _optional_string(candidate.get("repo"))
        model = _optional_string(candidate.get("model"))
        return "/".join(part for part in (repo, model) if part)
    return ""


def _model_run_summary(raw: dict[str, Any]) -> dict[str, Any]:
    model_run_id = normalize_model_run_id(raw.get("modelRunId"))
    if model_run_id is None:
        raise CloudruError(502, "Cloud.ru вернул некорректный modelRunId")
    scaling = raw.get("scaling") if isinstance(raw.get("scaling"), dict) else {}
    endpoint = normalize_model_run_endpoint(
        model_run_id,
        raw.get("publicUrl"),
        raw.get("endpoint"),
    )
    return {
        "modelRunId": model_run_id,
        "name": _string(raw.get("name"), "name"),
        "model": _model_label(raw.get("modelSource")),
        "status": _string(raw.get("status"), "status"),
        "resourceType": _optional_string(raw.get("resourceType")),
        "gpuCount": _optional_int(raw.get("gpuCount")),
        "scaling": {
            "minScale": _optional_int(scaling.get("minScale")),
            "maxScale": _optional_int(scaling.get("maxScale")),
        },
        "endpoint": endpoint,
    }


def _runtime_template(raw: dict[str, Any]) -> dict[str, Any] | None:
    if (
        raw.get("isActive") is not True
        or raw.get("frameworkType") != "FrameworkType_VLLM"
        or not isinstance(raw.get("id"), str)
        or not raw["id"].strip()
    ):
        return None
    raw_gpus = raw.get("gpus")
    gpus = []
    if isinstance(raw_gpus, list):
        gpus = [
            {
                "resourceType": gpu["resourceType"].strip(),
                "isDefault": gpu.get("isDefault") is True,
            }
            for gpu in raw_gpus
            if isinstance(gpu, dict)
            and gpu.get("isAllowed") is True
            and isinstance(gpu.get("resourceType"), str)
            and gpu["resourceType"].strip()
        ]
    return {
        "id": raw["id"].strip(),
        "frameworkType": "FrameworkType_VLLM",
        "version": _optional_string(raw.get("version")),
        "gpus": gpus,
    }


def _quota_items(raw: Any) -> list[dict[str, Any]]:
    if not isinstance(raw, list):
        raise CloudruError(502, "Cloud.ru вернул некорректные квоты")
    result = []
    for item in raw:
        if not isinstance(item, dict):
            raise CloudruError(502, "Cloud.ru вернул некорректные квоты")
        resource_type = item.get("resourceType")
        free = item.get("free")
        limit = item.get("limit")
        if (
            not isinstance(resource_type, str)
            or not resource_type.strip()
            or not isinstance(free, (int, float))
            or isinstance(free, bool)
            or not isinstance(limit, (int, float))
            or isinstance(limit, bool)
            or not math.isfinite(free)
            or not math.isfinite(limit)
        ):
            raise CloudruError(502, "Cloud.ru вернул некорректные квоты")
        result.append(
            {"resourceType": resource_type.strip(), "free": free, "limit": limit}
        )
    return result


def _service_account_options(raw: Any) -> list[dict[str, Any]]:
    if not isinstance(raw, list):
        raise CloudruError(502, "Cloud.ru вернул некорректный список сервисных аккаунтов")
    return [
        {
            "id": _string(item.get("id"), "service account id"),
            "name": _string(item.get("name"), "service account name"),
            "description": _optional_string(item.get("description")),
        }
        for item in raw
        if isinstance(item, dict) and item.get("enabled") is True
    ]


def _log_group_options(raw: Any) -> list[dict[str, Any]]:
    if not isinstance(raw, list):
        raise CloudruError(502, "Cloud.ru вернул некорректный список log groups")
    return [
        {
            "id": _string(item.get("id"), "log group id"),
            "name": _string(item.get("name"), "log group name"),
            "description": _optional_string(item.get("description")),
            "isDefault": item.get("name") == "default",
        }
        for item in raw
        if isinstance(item, dict)
        and item.get("status") == "STATUS_ACTIVE"
        and item.get("iamStatus") == "IAM_STATUS_READY"
    ]


class CloudruClient:
    # AICODE-NOTE: This read/create-only adapter deliberately isolates the non-public
    # Console BFF. Update/delete/suspend/resume methods are intentionally absent.
    def __init__(self, credentials: CloudCredentials, http: httpx.AsyncClient) -> None:
        self._credentials = credentials
        self._http = http

    async def _retry(
        self,
        operation: Callable[[], Awaitable[httpx.Response]],
    ) -> httpx.Response:
        for attempt in range(3):
            try:
                response = await operation()
            except httpx.RequestError:
                if attempt == 2:
                    raise CloudruError(502, "Не удалось связаться с Cloud.ru") from None
            else:
                if not 200 <= response.status_code < 300:
                    _log_http_error(response)
                if response.status_code not in _RETRY_STATUSES or attempt == 2:
                    return response
            await asyncio.sleep(0.1 * 2**attempt)

        raise AssertionError("unreachable")

    async def _token(self) -> str:
        try:
            response = await self._http.post(
                IAM_URL,
                json={
                    "keyId": self._credentials.key_id,
                    "secret": self._credentials.key_secret.get_secret_value(),
                },
                follow_redirects=False,
            )
        except httpx.RequestError:
            raise CloudruError(502, "Не удалось связаться с Cloud.ru") from None
        payload = _json_object(response)
        token = payload.get("access_token")
        if not isinstance(token, str) or not token.strip():
            raise CloudruError(502, "Cloud.ru не вернул access token")
        return token.strip()

    @staticmethod
    def _headers(token: str) -> dict[str, str]:
        return {
            "Authorization": f"Bearer {token}",
            "X-Request-ID": str(uuid4()),
        }

    async def _get(
        self,
        path: str,
        token: str,
        *,
        params: dict[str, str | int] | None = None,
    ) -> dict[str, Any]:
        response = await self._retry(
            lambda: self._http.get(
                f"{BFF_URL}{path}",
                params=params,
                headers=self._headers(token),
                follow_redirects=False,
            )
        )
        return _json_object(response, not_found_message="Проект не найден или недоступен")

    async def list_model_runs(
        self,
        project_id: str,
        limit: int = 100,
        offset: int = 0,
    ) -> list[dict[str, Any]]:
        project = _project_segment(project_id)
        token = await self._token()
        payload = await self._get(
            f"{BFF_PREFIX}/{project}/modelruns",
            token,
            params={"limit": limit, "offset": offset},
        )
        modelruns = payload.get("modelRuns", payload.get("modelruns"))
        if not isinstance(modelruns, list) or any(not isinstance(item, dict) for item in modelruns):
            raise CloudruError(502, "Cloud.ru вернул некорректный список Model Run")
        return [_model_run_summary(item) for item in modelruns]

    async def options(self, project_id: str) -> dict[str, Any]:
        project = _project_segment(project_id)
        token = await self._token()
        runtime_payload, quotas, accounts_payload, log_groups_payload = await asyncio.gather(
            self._get(
                f"{BFF_PREFIX}/{project}/runtime_templates",
                token,
                params={
                    "frameworkType": "FrameworkType_VLLM",
                    "isActive": "true",
                    "limit": 100,
                    "offset": 0,
                },
            ),
            self._get(f"{BFF_PREFIX}/{project}/quotas", token),
            self._get(f"{SERVICE_ACCOUNTS_PREFIX}/{project}/service-accounts", token),
            self._get(f"{LOG_GROUPS_PREFIX}/{project}/log-groups", token),
        )
        runtime_templates = runtime_payload.get("runtimeTemplates")
        if not isinstance(runtime_templates, list):
            raise CloudruError(502, "Cloud.ru вернул некорректный список runtime templates")
        filtered = [
            normalized
            for template in runtime_templates
            if isinstance(template, dict)
            if (normalized := _runtime_template(template)) is not None
        ]
        return {
            "runtimeTemplates": filtered,
            "gpuOptions": copy.deepcopy(GPU_OPTIONS),
            "quotas": _quota_items(quotas.get("data")),
            "serviceAccounts": _service_account_options(accounts_payload.get("accounts")),
            "logGroups": _log_group_options(log_groups_payload.get("logGroups")),
        }

    async def create_model_run(
        self,
        project_id: str,
        payload: dict[str, Any],
    ) -> dict[str, Any]:
        project = _project_segment(project_id)
        token = await self._token()
        try:
            response = await self._http.post(
                f"{BFF_URL}{BFF_PREFIX}/{project}/modelruns",
                json=payload,
                headers=self._headers(token),
                follow_redirects=False,
            )
        except httpx.RequestError:
            raise CloudruCreateUncertain(
                502,
                "Не удалось подтвердить создание Model Run; проверьте список перед повтором",
            ) from None
        if response.status_code >= 500:
            _log_http_error(response)
            raise CloudruCreateUncertain(
                response.status_code,
                "Cloud.ru не подтвердил создание Model Run; проверьте список перед повтором",
            )

        _raise_for_status(
            response,
            not_found_message="Выбранный runtime template или GPU-ресурс недоступен",
        )
        try:
            result = response.json()
        except ValueError:
            result = None
        if (
            not isinstance(result, dict)
            or normalize_model_run_id(result.get("modelRunId")) is None
        ):
            raise CloudruCreateUncertain(
                502,
                "Cloud.ru вернул неполный результат создания; проверьте список перед повтором",
            )
        return result
