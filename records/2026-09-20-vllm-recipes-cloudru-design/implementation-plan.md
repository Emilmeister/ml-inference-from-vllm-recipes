# vLLM Recipes → Cloud.ru ML Inference Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a local web application that reads the official vLLM Recipes JSON API, creates Cloud.ru Model Run deployments with user-selected GPU and scaling parameters, and shows existing deployments.

**Architecture:** A React/Vite single-page application stores Cloud.ru credentials in browser `localStorage` for seven days and sends them only in Cloud operation request bodies. A stateless FastAPI backend adapts the vLLM Recipes JSON API, Cloud.ru IAM, and the isolated Cloud.ru Console BFF; it has no database and never exposes mutation methods for editing or deleting Model Runs.

**Tech Stack:** Python 3.13, FastAPI 0.141.1, httpx 0.28.1, Pydantic 2.13.5, React 19.3.0, React Router 7.18.4, Vite 8.3.0, TypeScript 7.0.2 and pnpm 10. Existing Python test dependencies predate the current test gate and are not active readiness checks.

---

## Execution rules

- Follow `docs/workflow.md` and `docs/bdd.md`.
- RED/GREEN commands retained in earlier task notes are historical proposals;
  run them only after the test gate is separately opened for that exact scope.
- Do not contact a real Cloud.ru project from tests.
- Do not write credentials, IAM tokens, or Hugging Face token values to fixtures, logs, snapshots, or reports.
- Do not add update, delete, suspend, or resume backend methods.
- Commit steps are intentionally omitted because `AGENTS.md` requires explicit authorization before committing.
- Do not create a worktree: this repository has one initial commit and the approved design record is currently untracked in the active workspace.
- Test steps below are proposals, not authorization. After the 2026-09-20 testing-policy
  change, do not create, change, or delete tests, BDD step definitions, fixtures, or
  test infrastructure until the user separately approves the concrete test scope.
  Approval of this implementation plan or of a `.feature` file does not count as
  that separate approval.

## Current implementation status

Production work in Tasks 1–12, 14 and 15 is implemented. Product Gherkin files
are DRAFT contracts only. Task 13 and every automated-test step remain deferred
until separate approval; Task 16 uses only the approved build, lint, repository
integrity and manual HTTP smoke checks.

## File map

```text
.
├── pyproject.toml
├── uv.lock
├── src/vllm_cloudru/
│   ├── __init__.py
│   ├── main.py              # FastAPI application factory and safe errors
│   ├── models.py            # Request/response contracts
│   ├── recipes.py           # vLLM Recipes discovery and rendering selection
│   ├── translator.py        # argv/env → Cloud.ru dynamic arguments
│   ├── cloudru.py           # IAM and read/create-only Console BFF adapter
│   ├── deployment.py        # Validation, warnings, payload construction
│   └── api.py               # /api routes
├── tests/
│   ├── fixtures/recipes/
│   │   ├── index.json
│   │   ├── single-node.json
│   │   └── multi-node.json
│   ├── unit/
│   │   ├── test_health.py
│   │   └── test_translator.py
│   └── acceptance/
│       ├── features/
│       │   ├── catalog.feature
│       │   ├── inferences.feature
│       │   ├── deployment.feature
│       │   └── credentials.feature
├── frontend/
│   ├── package.json
│   ├── pnpm-lock.yaml
│   ├── tsconfig.json
│   ├── vite.config.ts
│   ├── index.html
│   ├── public/illustrations/
│   └── src/
│       ├── main.tsx
│       ├── App.tsx
│       ├── api.ts
│       ├── credentials.ts
│       ├── types.ts
│       ├── styles.css
│       ├── components/
│       │   ├── AppShell.tsx
│       │   ├── Notice.tsx
│       │   └── RecipeCard.tsx
│       └── pages/
│           ├── SetupPage.tsx
│           ├── CatalogPage.tsx
│           ├── RecipePage.tsx
│           ├── InferencesPage.tsx
│           └── DeployPage.tsx
└── docs/
    ├── architecture.md
    ├── bdd.md
    └── development.md
```

### Task 1: Establish the verified Python application shell

**Files:**
- Create: `pyproject.toml`
- Create: `src/vllm_cloudru/__init__.py`
- Create: `src/vllm_cloudru/main.py`
- Create: `tests/unit/test_health.py`
- Modify: `.gitignore`

- [ ] **Step 1: Add the pinned Python manifest**

Create `pyproject.toml`:

```toml
[project]
name = "vllm-recipes-cloudru"
version = "0.1.0"
description = "Launch vLLM Recipes as Cloud.ru ML Inference Model Runs"
requires-python = ">=3.13,<3.14"
dependencies = [
  "fastapi==0.141.1",
  "httpx==0.28.1",
  "pydantic==2.13.5",
  "uvicorn[standard]==0.53.0",
]

[dependency-groups]
dev = [
  "pytest==9.1.1",
  "pytest-bdd==8.1.0",
  "pytest-playwright==0.9.0",
  "ruff==0.16.8",
]

[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"

[tool.hatch.build.targets.wheel]
packages = ["src/vllm_cloudru"]

[tool.pytest.ini_options]
pythonpath = ["src"]
testpaths = ["tests"]

[tool.ruff]
line-length = 100
target-version = "py313"

[tool.ruff.lint]
select = ["E", "F", "I", "B", "UP"]
```

Append to `.gitignore`:

```gitignore
/frontend/node_modules/
/frontend/dist/
/frontend/playwright-report/
/frontend/test-results/
/.pytest_cache/
/.ruff_cache/
```

Run:

```sh
uv sync
```

Expected: exit 0 and a new `uv.lock`.

- [ ] **Step 2: Write the failing health test**

Create `tests/unit/test_health.py`:

```python
from fastapi.testclient import TestClient

from vllm_cloudru.main import create_app


def test_health_reports_ready():
    response = TestClient(create_app()).get("/api/health")

    assert response.status_code == 200
    assert response.json() == {"status": "ready"}
```

Run:

```sh
uv run pytest tests/unit/test_health.py -v
```

Expected: FAIL because `vllm_cloudru.main` does not exist.

- [ ] **Step 3: Implement the minimal application factory**

Create `src/vllm_cloudru/__init__.py` as an empty module and `src/vllm_cloudru/main.py`:

```python
from fastapi import FastAPI


def create_app() -> FastAPI:
    app = FastAPI(title="vLLM Recipes → Cloud.ru")

    @app.get("/api/health")
    def health() -> dict[str, str]:
        return {"status": "ready"}

    return app


app = create_app()
```

Run:

```sh
uv run pytest tests/unit/test_health.py -v
uv run ruff check src tests
```

Expected: one passing test and no Ruff errors.

### Task 2: Model and translate real vLLM Recipe renderings

**Files:**
- Create: `tests/fixtures/recipes/index.json`
- Create: `tests/fixtures/recipes/single-node.json`
- Create: `tests/fixtures/recipes/multi-node.json`
- Create: `tests/unit/test_translator.py`
- Create: `src/vllm_cloudru/models.py`
- Create: `src/vllm_cloudru/translator.py`

- [ ] **Step 1: Save two exact minimized upstream examples**

Use these fixtures, preserving upstream field names:

`tests/fixtures/recipes/index.json`:

```json
[
  {
    "hf_id": "LiquidAI/LFM2.5-8B-A1B",
    "title": "LFM2.5 8B A1B",
    "provider": "LiquidAI",
    "url": "/LiquidAI/LFM2.5-8B-A1B",
    "json": "/LiquidAI/LFM2.5-8B-A1B.json"
  },
  {
    "hf_id": "zai-org/GLM-5",
    "title": "GLM-5",
    "provider": "GLM (Z-AI)",
    "url": "/zai-org/GLM-5",
    "json": "/zai-org/GLM-5.json"
  }
]
```

`tests/fixtures/recipes/single-node.json`:

```json
{
  "variant": "default",
  "hardware": "h100",
  "strategy": "single_node_tp",
  "node_count": 1,
  "deploy_type": "single_node",
  "env": {},
  "argv": [
    "vllm",
    "serve",
    "LiquidAI/LFM2.5-8B-A1B",
    "--tensor-parallel-size",
    "1",
    "--reasoning-parser",
    "qwen3",
    "--enable-auto-tool-choice",
    "--tool-call-parser",
    "lfm2"
  ],
  "hardware_profile": {
    "display_name": "H100",
    "gpu_count": 8,
    "vram_gb": 640,
    "multi_node": false
  }
}
```

`tests/fixtures/recipes/multi-node.json`:

```json
{
  "variant": "default",
  "hardware": "h200",
  "strategy": "multi_node_tp",
  "node_count": 2,
  "deploy_type": "multi_node",
  "env": {
    "GLOO_SOCKET_IFNAME": "$IFACE_NAME"
  },
  "head_argv": ["vllm", "serve", "zai-org/GLM-5", "--tensor-parallel-size", "16"],
  "worker_argv": ["vllm", "serve", "zai-org/GLM-5", "--headless"],
  "hardware_profile": {
    "display_name": "H200",
    "gpu_count": 8,
    "vram_gb": 1128,
    "multi_node": false
  }
}
```

Add a source comment beside fixture loading in the tests:

```python
# AICODE-NOTE: Fixtures are minimized from recipes.vllm.ai on 2026-09-20.
# Keep both argv and head_argv/worker_argv shapes when refreshing them.
```

- [ ] **Step 2: Write failing translator tests**

Create `tests/unit/test_translator.py`:

```python
import json
from pathlib import Path

import pytest

from vllm_cloudru.translator import UnsupportedRecipe, translate_rendering

FIXTURES = Path(__file__).parents[1] / "fixtures" / "recipes"


def load(name: str) -> dict:
    return json.loads((FIXTURES / name).read_text())


def test_single_process_argv_becomes_cloud_arguments():
    translated = translate_rendering(load("single-node.json"))

    assert translated.model_id == "LiquidAI/LFM2.5-8B-A1B"
    assert translated.arguments == [
        {
            "key": "tensor-parallel-size",
            "value": 1,
            "parameterType": "PARAMETER_TYPE_ARG_KV_NUMERIC",
        },
        {
            "key": "reasoning-parser",
            "value": "qwen3",
            "parameterType": "PARAMETER_TYPE_ARG_KV",
        },
        {
            "key": "enable-auto-tool-choice",
            "value": True,
            "parameterType": "PARAMETER_TYPE_ARG_FLAG",
        },
        {
            "key": "tool-call-parser",
            "value": "lfm2",
            "parameterType": "PARAMETER_TYPE_ARG_KV",
        },
    ]


def test_multi_process_rendering_is_view_only():
    with pytest.raises(UnsupportedRecipe, match="нескольких процессов"):
        translate_rendering(load("multi-node.json"))
```

Run:

```sh
uv run pytest tests/unit/test_translator.py -v
```

Expected: FAIL because `vllm_cloudru.translator` does not exist.

- [ ] **Step 3: Implement strict single-process translation**

Create the relevant contracts in `src/vllm_cloudru/models.py`:

```python
from typing import Any

from pydantic import BaseModel, Field, SecretStr


class CloudCredentials(BaseModel):
    project_id: str = Field(min_length=1)
    key_id: str = Field(min_length=1)
    key_secret: SecretStr
    hf_secret_id: str | None = None


class TranslatedRecipe(BaseModel):
    model_id: str
    arguments: list[dict[str, Any]]
    environment: list[dict[str, Any]]
    recommended_gpu_count: int | None = None
    recommended_hardware: str | None = None
```

Create `src/vllm_cloudru/translator.py`:

```python
from typing import Any

from .models import TranslatedRecipe


class UnsupportedRecipe(ValueError):
    pass


def _value_type(value: str) -> tuple[Any, str]:
    try:
        number = int(value)
    except ValueError:
        try:
            number = float(value)
        except ValueError:
            return value, "PARAMETER_TYPE_ARG_KV"
    return number, "PARAMETER_TYPE_ARG_KV_NUMERIC"


def translate_rendering(rendering: dict[str, Any]) -> TranslatedRecipe:
    argv = rendering.get("argv")
    if rendering.get("deploy_type") != "single_node" or not isinstance(argv, list):
        raise UnsupportedRecipe("Рецепт требует нескольких процессов или узлов")
    if len(argv) < 3 or argv[:2] != ["vllm", "serve"]:
        raise UnsupportedRecipe("Рецепт не содержит поддерживаемую команду vllm serve")

    model_id = str(argv[2])
    arguments: list[dict[str, Any]] = []
    index = 3
    while index < len(argv):
        token = argv[index]
        if not isinstance(token, str) or not token.startswith("--"):
            raise UnsupportedRecipe("Аргументы рецепта имеют неизвестный формат")
        raw = token[2:]
        if "=" in raw:
            key, value = raw.split("=", 1)
            parsed, parameter_type = _value_type(value)
            arguments.append({"key": key, "value": parsed, "parameterType": parameter_type})
            index += 1
            continue
        if index + 1 >= len(argv) or str(argv[index + 1]).startswith("--"):
            arguments.append(
                {"key": raw, "value": True, "parameterType": "PARAMETER_TYPE_ARG_FLAG"}
            )
            index += 1
            continue
        parsed, parameter_type = _value_type(str(argv[index + 1]))
        arguments.append({"key": raw, "value": parsed, "parameterType": parameter_type})
        index += 2

    environment = []
    for key, value in rendering.get("env", {}).items():
        if "$" in str(value):
            raise UnsupportedRecipe("Рецепт содержит неразрешённые переменные окружения")
        environment.append(
            {"key": key, "value": value, "parameterType": "PARAMETER_TYPE_ENV"}
        )

    profile = rendering.get("hardware_profile") or {}
    return TranslatedRecipe(
        model_id=model_id,
        arguments=arguments,
        environment=environment,
        recommended_gpu_count=profile.get("gpu_count"),
        recommended_hardware=rendering.get("hardware"),
    )
```

Run:

```sh
uv run pytest tests/unit/test_translator.py -v
```

Expected: 2 passed.

### Task 3: Discover and cache the official recipe catalog

**Files:**
- Create: `tests/unit/test_recipes.py`
- Create: `src/vllm_cloudru/recipes.py`
- Modify: `src/vllm_cloudru/models.py`

- [ ] **Step 1: Write failing discovery tests**

Create `tests/unit/test_recipes.py` with an `httpx.MockTransport` that serves:

- `/models.json` from `index.json`;
- `/LiquidAI/LFM2.5-8B-A1B.json` with `meta`, `model`, `variants` and `recommended_command.by_hardware.h100`;
- `/LiquidAI/LFM2.5-8B-A1B/hw/h100.json` from `single-node.json`.

The required assertions are:

```python
async def test_catalog_uses_published_json_links(recipe_client):
    items = await recipe_client.list_models()

    assert [item.hf_id for item in items] == [
        "LiquidAI/LFM2.5-8B-A1B",
        "zai-org/GLM-5",
    ]


async def test_h100_follows_by_hardware_link(recipe_client):
    detail = await recipe_client.get_detail(
        "LiquidAI/LFM2.5-8B-A1B",
        resource_type="ResourceType_GPU_H100",
    )

    assert detail.rendering_path.endswith("/hw/h100.json")
    assert detail.launchable is True


async def test_last_catalog_survives_refresh_failure(recipe_client, transport_state):
    first = await recipe_client.list_models()
    transport_state.fail = True
    second = await recipe_client.list_models(refresh=True)

    assert second == first
```

Add `@pytest.mark.anyio` to the async tests and add `anyio_backend` returning `"asyncio"` in the module.

Run:

```sh
uv run pytest tests/unit/test_recipes.py -v
```

Expected: FAIL because `RecipeClient` is missing.

- [ ] **Step 2: Implement same-origin link following and memory fallback**

Add response models `RecipeSummary` and `RecipeDetail` to `models.py`. Implement `RecipeClient` in `recipes.py` with:

- fixed base URL `https://recipes.vllm.ai`;
- `list_models(refresh=False)` caching the last successful list in memory;
- `get_model(hf_id)` resolving the `json` link from the cached index;
- `get_detail(hf_id, resource_type)` selecting `h100` only for H100 when present, otherwise the recipe's published recommended command;
- relative-path validation that rejects a scheme, host, path not beginning with `/`, or `..` segment;
- `httpx.AsyncClient(follow_redirects=False, timeout=20)`;
- conversion of `UnsupportedRecipe` into `launchable=False` and a Russian reason.

Use this hardware mapping:

```python
CLOUD_TO_RECIPE_HARDWARE = {
    "ResourceType_GPU_H100": "h100",
}
```

A100 and V100 intentionally use the published default rendering and receive a compatibility warning because vLLM Recipes has no corresponding hardware profile key.

Run:

```sh
uv run pytest tests/unit/test_recipes.py tests/unit/test_translator.py -v
```

Expected: all recipe tests pass.

### Task 4: Add the read/create-only Cloud.ru adapter

**Files:**
- Create: `tests/unit/test_cloudru.py`
- Create: `src/vllm_cloudru/cloudru.py`
- Modify: `src/vllm_cloudru/models.py`

- [ ] **Step 1: Write failing IAM, list, options, and create tests**

Use `httpx.MockTransport` and assert these exact requests:

```text
POST https://iam.api.cloud.ru/api/v1/auth/token
GET  https://console.cloud.ru/u-api/inference/model-run/v1/{project_id}/modelruns
GET  https://console.cloud.ru/u-api/inference/model-run/v1/{project_id}/runtime_templates
GET  https://console.cloud.ru/u-api/inference/model-run/v1/{project_id}/quotas
POST https://console.cloud.ru/u-api/inference/model-run/v1/{project_id}/modelruns
```

The test must assert:

- IAM receives `{"keyId":"key-id","secret":"key-secret"}`;
- BFF receives `Authorization: Bearer iam-token` and a UUID `X-Request-ID`;
- `list_model_runs` returns the `modelruns` array;
- `options` returns only active `FrameworkType_VLLM` templates plus GPU definitions;
- `create_model_run` performs exactly one POST even when the transport raises `ReadTimeout`;
- the client has no `delete`, `update`, `suspend`, or `resume` attributes.

Run:

```sh
uv run pytest tests/unit/test_cloudru.py -v
```

Expected: FAIL because `CloudruClient` does not exist.

- [ ] **Step 2: Implement explicit IAM and safe Cloud errors**

Create `src/vllm_cloudru/cloudru.py` with these constants:

```python
IAM_URL = "https://iam.api.cloud.ru/api/v1/auth/token"
BFF_URL = "https://console.cloud.ru"
BFF_PREFIX = "/u-api/inference/model-run/v1"

GPU_OPTIONS = {
    "ResourceType_GPU_V100": {"label": "NVIDIA V100 32 GB", "memoryGb": 32, "counts": [1, 2, 4, 8, 16]},
    "ResourceType_GPU_A100_NVLINK": {"label": "NVIDIA A100 80 GB", "memoryGb": 80, "counts": [1, 2, 4, 6, 8]},
    "ResourceType_GPU_H100": {"label": "NVIDIA H100 80 GB", "memoryGb": 80, "counts": [1, 2, 4, 6, 8]},
}
```

Implement one `httpx.AsyncClient` per application lifespan and a lightweight `CloudruClient` holding credentials and the shared HTTP client. `_token()` performs IAM exchange; each public method obtains one token and uses it for all calls in that method. GET operations retry only 502, 503, 504 and connection errors twice with short exponential backoff. `create_model_run` never retries.

Raise:

```python
class CloudruError(RuntimeError):
    def __init__(self, status_code: int, public_message: str):
        self.status_code = status_code
        self.public_message = public_message
        super().__init__(public_message)


class CloudruCreateUncertain(CloudruError):
    pass
```

Map 401 to «Неверный Key ID или Key Secret», 403 to «У сервисного аккаунта недостаточно прав», 404 to «Проект или ресурс не найден», 409 to «Имя Model Run уже занято», and all other responses to a message containing only the HTTP status and upstream request ID. Do not include response bodies in public messages or logs.

Run:

```sh
uv run pytest tests/unit/test_cloudru.py -v
```

Expected: all Cloud adapter tests pass.

### Task 5: Build and validate one Model Run payload

**Files:**
- Create: `tests/unit/test_deployment.py`
- Create: `src/vllm_cloudru/deployment.py`
- Modify: `src/vllm_cloudru/models.py`

- [ ] **Step 1: Write failing payload and warning tests**

Define a request with:

```python
request = CreateInferenceRequest(
    credentials=credentials,
    name="LFM Demo",
    hf_id="LiquidAI/LFM2.5-8B-A1B",
    resource_type="ResourceType_GPU_H100",
    gpu_count=2,
    runtime_template_id="runtime-vllm",
    min_scale=1,
    max_scale=1,
    confirm_warnings=False,
)
```

Assert:

- normalized name is `lfm-demo`;
- recipe argument `tensor-parallel-size=1` is preserved;
- selecting two GPUs produces a warning because recipe TP is one;
- without `confirm_warnings` no create call occurs;
- with confirmation, payload has user GPU count 2, memory 80, scaling 1–1, endpoint auth enabled, logging disabled, and configured HF secret UUID;
- `min_scale > max_scale` fails before Cloud calls;
- a multi-node rendering fails even with confirmation;
- `CloudruCreateUncertain` propagates without a list lookup or a second create POST;
  the UI asks the user to refresh the Model Run list and verify the outcome manually.

Run:

```sh
uv run pytest tests/unit/test_deployment.py -v
```

Expected: FAIL because deployment contracts and service are absent.

- [ ] **Step 2: Add deployment request contracts**

Add to `models.py`:

```python
class CreateInferenceRequest(BaseModel):
    credentials: CloudCredentials
    name: str = Field(min_length=1, max_length=63)
    hf_id: str = Field(pattern=r"^[^/]+/[^/]+$")
    resource_type: str
    gpu_count: int = Field(gt=0)
    runtime_template_id: str = Field(min_length=1)
    min_scale: int = Field(ge=0)
    max_scale: int = Field(ge=1)
    confirm_warnings: bool = False


class CreateInferenceResult(BaseModel):
    model_run_id: str
    name: str
    status: str
    endpoint: str
```

Use a Pydantic model validator to reject `min_scale > max_scale`.

- [ ] **Step 3: Implement minimal payload construction**

Create `deployment.py` with:

```python
NAME_RE = re.compile(r"[^a-z0-9-]+")


def normalize_name(value: str) -> str:
    normalized = NAME_RE.sub("-", value.lower())
    normalized = re.sub(r"-+", "-", normalized).strip("-")
    if not normalized or not normalized[0].isalpha():
        normalized = "model-" + normalized
    return normalized[:63].rstrip("-")
```

`build_payload` must:

1. verify resource type and count against `GPU_OPTIONS`;
2. preserve recipe arguments;
3. warn when selected resource has no exact recipe hardware profile;
4. inspect `tensor-parallel-size` and warn when it differs from selected GPU count;
5. require `confirm_warnings=True` before returning a warning-bearing payload;
6. add environment arguments after CLI arguments;
7. map recipe task `embedding` to `ModelTaskType_EMBEDDING` and all text/multimodal tasks to `ModelTaskType_GENERATE`;
8. set `modelSource.secret` to the configured secret UUID or an empty string;
9. set `scalingRules.rpsType.value` to 200.

`deploy_inference` calls Cloud create once. On `CloudruCreateUncertain` it does not
search by name and does not retry; the error propagates to the API as an uncertain
outcome that requires a manual list refresh.

Define the confirmation contract used by the API:

```python
class ConfirmationRequired(ValueError):
    def __init__(self, warnings: list[str]):
        self.warnings = warnings
        super().__init__("Требуется подтверждение предупреждений")
```

Name the public service coroutine `deploy_inference`. It accepts
`CreateInferenceRequest` plus keyword-only `recipes` and `cloud` dependencies and
returns `CreateInferenceResult`.

Run:

```sh
uv run pytest tests/unit/test_deployment.py -v
uv run pytest tests/unit -q
```

Expected: all unit tests pass.

### Task 6: Expose the backend behavior as executable Gherkin specifications

**Files:**
- Create: `tests/acceptance/features/catalog.feature`
- Create: `tests/acceptance/features/inferences.feature`
- Create: `tests/acceptance/features/deployment.feature`
- Create: `src/vllm_cloudru/api.py`
- Modify: `src/vllm_cloudru/main.py`

- [x] **Step 1: Write the catalog scenarios as a DRAFT product contract**

`catalog.feature`:

```gherkin
# language: ru
Функциональность: Каталог vLLM Recipes

  Сценарий: CAT-001. Каталог показывает все полученные рецепты
    Дано upstream вернул single-node и multi-node рецепты
    Когда пользователь запрашивает каталог
    Тогда в каталоге видны обе модели
    И у каждой модели указан статус автоматического запуска

  Сценарий: CAT-002. Multi-node рецепт доступен только для просмотра
    Дано upstream вернул multi-node рецепт
    Когда пользователь открывает рецепт
    Тогда запуск недоступен с причиной о нескольких процессах
```

Pytest-bdd step definitions are intentionally deferred by the global test gate.

- [x] **Step 2: Write inference list scenario as DRAFT**

`inferences.feature`:

```gherkin
# language: ru
Функциональность: Существующие инференсы Cloud.ru

  Сценарий: INF-001. Пользователь видит Model Run своего проекта
    Дано Cloud.ru вернул запущенный Model Run "demo"
    Когда пользователь запрашивает свои инференсы
    Тогда он видит "demo" со статусом "RUNNING"
    И видит модель, тип и количество GPU
    И видит границы масштабирования и endpoint
```

- [x] **Step 3: Write deployment scenarios as DRAFT**

`deployment.feature`:

```gherkin
# language: ru
Функциональность: Запуск рецепта в Cloud.ru

  Сценарий: DEP-001. Поддерживаемый рецепт создаёт один Model Run
    Дано настроен single-node рецепт
    И выбраны две H100 и масштабирование от 1 до 1
    Когда пользователь подтверждает предупреждение и запускает модель
    Тогда Cloud.ru получает один запрос создания
    И payload содержит две H100 и масштабирование от 1 до 1

  Сценарий: DEP-002. Несовместимый GPU требует подтверждения
    Дано настройки GPU расходятся с рецептом
    Когда пользователь запускает модель без подтверждения
    Тогда Model Run не создаётся
    И ответ содержит предупреждение о несовместимости

  Сценарий: DEP-003. Передаётся только UUID секрета Hugging Face
    Дано настроен UUID секрета Hugging Face "11111111-1111-1111-1111-111111111111"
    Когда пользователь запускает модель
    Тогда payload содержит этот UUID
    И payload не содержит значение токена Hugging Face

  Сценарий: DEP-004. Неопределённый ответ не повторяет создание
    Дано Cloud.ru принял создание, но соединение оборвалось
    Когда приложение получает неопределённый результат
    Тогда создание было отправлено ровно один раз
    И автоматический поиск Model Run не выполняется
    И пользователю предложено обновить список для ручной проверки

  Сценарий: SEC-001. Секреты отсутствуют в ошибке и журнале
    Дано Cloud.ru отклонил запрос с ошибкой, содержащей секрет
    Когда приложение возвращает ошибку пользователю
    Тогда ответ не содержит Key Secret, IAM-токен и значение HF-токена
    И журнал не содержит Key Secret, IAM-токен и значение HF-токена
```

No test runner is invoked without separate approval.

- [x] **Step 4: Implement the API routes**

Add this complete route layer to `api.py`:

```python
from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel, ConfigDict

from .cloudru import CloudruCreateUncertain, CloudruError
from .deployment import ConfirmationRequired, deploy_inference
from .models import CloudCredentials, CreateInferenceRequest
from .translator import UnsupportedRecipe

router = APIRouter(prefix="/api")


class CredentialsEnvelope(BaseModel):
    model_config = ConfigDict(extra="forbid")

    credentials: CloudCredentials


def _cloud_error(error: CloudruError) -> HTTPException:
    return HTTPException(
        status_code=error.status_code,
        detail={"code": "cloud_error", "message": error.public_message},
    )


@router.get("/recipes")
async def recipes(request: Request, refresh: bool = False):
    return await request.app.state.recipes.list_catalog(refresh=refresh)


@router.get("/recipes/{organization}/{model}")
async def recipe_detail(
    organization: str,
    model: str,
    request: Request,
    resource_type: str = "ResourceType_GPU_H100",
):
    return await request.app.state.recipes.get_detail(
        organization + "/" + model,
        resource_type=resource_type,
    )


@router.post("/cloud/validate")
async def validate_cloud(body: CredentialsEnvelope, request: Request):
    client = request.app.state.cloud_factory(body.credentials)
    try:
        await client.list_model_runs(body.credentials.project_id, limit=1)
    except CloudruError as error:
        raise _cloud_error(error) from error
    return {"valid": True}


@router.post("/cloud/options")
async def cloud_options(body: CredentialsEnvelope, request: Request):
    client = request.app.state.cloud_factory(body.credentials)
    try:
        return await client.options(body.credentials.project_id)
    except CloudruError as error:
        raise _cloud_error(error) from error


@router.post("/cloud/inferences/query")
async def inferences(body: CredentialsEnvelope, request: Request):
    client = request.app.state.cloud_factory(body.credentials)
    try:
        return await client.list_model_runs(body.credentials.project_id)
    except CloudruError as error:
        raise _cloud_error(error) from error


@router.post("/cloud/inferences", status_code=201)
async def create_inference(body: CreateInferenceRequest, request: Request):
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
    except CloudruCreateUncertain as error:
        raise HTTPException(
            status_code=error.status_code,
            detail={
                "code": "create_uncertain",
                "message": error.public_message,
                "manualCheckRequired": True,
                "instruction": "Обновите список Model Run вручную перед повтором.",
            },
        ) from error
    except CloudruError as error:
        raise _cloud_error(error) from error
```

A missing confirmation returns 409 with a concrete body such as
`{"code":"confirmation_required","warnings":["tensor-parallel-size=1, выбрано GPU: 2"]}`.

Modify `create_app` to accept optional `recipes` and `cloud_factory` arguments, store them in `app.state`, include the router, and close the production `httpx.AsyncClient` in the FastAPI lifespan.

Task 6 verification is limited to production static checks. The scenarios stay
`DRAFT` until their exact executable automation scope is separately approved.

### Task 7: Establish the React application and seven-day credential storage

**Files:**
- Create: `frontend/package.json`
- Create: `frontend/tsconfig.json`
- Create: `frontend/vite.config.ts`
- Create: `frontend/index.html`
- Create: `frontend/src/types.ts`
- Create: `frontend/src/credentials.ts`
- Create: `frontend/src/main.tsx`
- Create: `frontend/src/App.tsx`

- [x] **Step 1: Add the frontend manifest**

Create `frontend/package.json`:

```json
{
  "name": "vllm-recipes-cloudru-ui",
  "private": true,
  "version": "0.1.0",
  "type": "module",
  "scripts": {
    "dev": "vite",
    "build": "tsc -b && vite build"
  },
  "dependencies": {
    "lucide-react": "1.47.0",
    "react": "19.3.0",
    "react-dom": "19.3.0",
    "react-router-dom": "7.18.4"
  },
  "devDependencies": {
    "@types/react": "19.3.0",
    "@types/react-dom": "19.3.0",
    "@vitejs/plugin-react": "6.1.1",
    "typescript": "7.0.2",
    "vite": "8.3.0"
  },
  "packageManager": "pnpm@10.26.0"
}
```

Configure Vite to proxy `/api` to `http://127.0.0.1:8000`. Run:

```sh
pnpm --dir frontend install
```

Expected: exit 0 and a new `frontend/pnpm-lock.yaml`.

- [ ] **Step 2: Automate credential expiry scenarios after separate approval**

`credentials.test.ts` must prove:

```typescript
it("keeps credentials for less than seven days", () => {
  saveCredentials(input, 1_000);
  expect(loadCredentials(1_000 + SEVEN_DAYS_MS - 1)).toMatchObject(input);
});

it("removes credentials at seven days", () => {
  saveCredentials(input, 1_000);
  expect(loadCredentials(1_000 + SEVEN_DAYS_MS)).toBeNull();
  expect(localStorage.getItem(STORAGE_KEY)).toBeNull();
});

it("does not persist a Hugging Face token value", () => {
  saveCredentials(input, 1_000);
  expect(localStorage.getItem(STORAGE_KEY)).not.toContain("hf_");
});
```

This proposed test code is not authorized by approval of the plan.

- [x] **Step 3: Implement the exact storage contract**

`credentials.ts` exports:

```typescript
export const STORAGE_KEY = "vllm-cloudru.credentials.v1";
export const SEVEN_DAYS_MS = 7 * 24 * 60 * 60 * 1000;

export type CredentialInput = {
  projectId: string;
  keyId: string;
  keySecret: string;
  hfSecretId?: string;
};

export type StoredCredentials = CredentialInput & { expiresAt: number };

export function saveCredentials(
  input: CredentialInput,
  now = Date.now(),
): StoredCredentials {
  const stored = {
    projectId: input.projectId,
    keyId: input.keyId,
    keySecret: input.keySecret,
    hfSecretId: input.hfSecretId,
    expiresAt: now + SEVEN_DAYS_MS,
  };
  localStorage.setItem(STORAGE_KEY, JSON.stringify(stored));
  return stored;
}

export function loadCredentials(now = Date.now()): StoredCredentials | null {
  const raw = localStorage.getItem(STORAGE_KEY);
  if (!raw) return null;
  try {
    const value = JSON.parse(raw) as StoredCredentials;
    if (!value.projectId || !value.keyId || !value.keySecret || value.expiresAt <= now) {
      localStorage.removeItem(STORAGE_KEY);
      return null;
    }
    return value;
  } catch {
    localStorage.removeItem(STORAGE_KEY);
    return null;
  }
}

export function clearCredentials(): void {
  localStorage.removeItem(STORAGE_KEY);
}
```

The UI must not accept or define an HF token value field; only `hfSecretId` exists.

Verified with the production TypeScript build; no test runner was invoked.

### Task 8: Implement onboarding and the application shell

**Files:**
- Create: `frontend/src/api.ts`
- Create: `frontend/src/components/AppShell.tsx`
- Create: `frontend/src/components/Notice.tsx`
- Create: `frontend/src/pages/SetupPage.tsx`
- Modify: `frontend/src/App.tsx`
- Create: `frontend/src/styles.css`

- [ ] **Step 1: Automate setup scenarios after separate approval**

The test renders `SetupPage`, fills all four agreed fields, submits, and asserts:

- `validateCredentials` receives snake_case backend credentials;
- saved expiration is seven days;
- success navigates to `/inferences`;
- a 401 message is shown and nothing is saved.

This proposed test code is not authorized by approval of the plan.

- [x] **Step 2: Implement safe API serialization and setup**

`api.ts` defines `ApiError` and a single `post` helper. The helper must:

- call `loadCredentials()` immediately before Cloud requests;
- throw «Настройки Cloud.ru отсутствуют или истекли» before fetch when missing;
- set only `Content-Type: application/json`;
- never include credentials in a URL;
- parse the backend `message` and `code` without returning the submitted body.

`SetupPage` uses labelled inputs:

```text
PROJECT_ID
Key ID
Key Secret
Hugging Face Secret UUID (необязательно)
```

Use `type="password"` for Key Secret, `autocomplete="off"`, a visible seven-day warning, «Проверить подключение» submit button, and no analytics.

`AppShell` supplies navigation to «Мои инференсы», «Каталог» and «Настройки». `App` sends users without valid credentials to setup.

Verification uses `pnpm --dir frontend build`; no test runner is invoked.

### Task 9: Implement catalog and recipe details

**Files:**
- Create: `frontend/src/components/RecipeCard.tsx`
- Create: `frontend/src/pages/CatalogPage.tsx`
- Create: `frontend/src/pages/RecipePage.tsx`
- Modify: `frontend/src/api.ts`
- Modify: `frontend/src/App.tsx`

- [ ] **Step 1: Automate catalog interactions after separate approval**

Cover:

- all returned recipes render;
- search filters title and provider;
- task, precision and launchability filters compose;
- «Только просмотр» remains visible with its reason;
- opening a recipe shows exact vLLM arguments;
- only a launchable recipe exposes «Настроить запуск».

This proposed test code is not authorized by approval of the plan.

- [x] **Step 2: Implement the catalog with URL-backed filters**

Keep filters in `URLSearchParams` so reload preserves them. Use semantic form controls, not custom select widgets. `RecipeCard` shows provider, parameter count, task chips, precision, hardware badges, and one status label.

`RecipePage` fetches detail using the currently selected Cloud GPU resource and displays:

- model description;
- minimum vLLM version;
- variant and recommended hardware;
- arguments in a code panel;
- warnings;
- view-only reason or launch action.

Verification uses `pnpm --dir frontend build`; no test runner is invoked.

### Task 10: Show existing Model Runs without exposing mutations

**Files:**
- Create: `frontend/src/pages/InferencesPage.tsx`
- Modify: `frontend/src/api.ts`
- Modify: `frontend/src/App.tsx`

- [ ] **Step 1: Automate the safety contract after separate approval**

Render one running Model Run. Click «Редактировать» and «Удалить». Assert:

- no PUT, PATCH, or DELETE fetch occurs;
- a dialog or live-region notice says the action is available in Cloud.ru Console;
- the notice contains a link to `https://console.cloud.ru`;
- endpoint, resource type, GPU count, scaling and status remain visible.

No test file or runner is added until its exact scope is separately approved.

- [x] **Step 2: Implement the read-only dashboard**

Use a responsive list/table hybrid. Status has both text and icon. Copying the inference endpoint is allowed; changing lifecycle is not. The empty state uses `model-catalog.png` and links to the catalog.

Verified with the production TypeScript/Vite build; no test runner was invoked.

### Task 11: Implement the deployment wizard and explicit warning confirmation

**Files:**
- Create: `frontend/src/pages/DeployPage.tsx`
- Modify: `frontend/src/api.ts`
- Modify: `frontend/src/App.tsx`

- [ ] **Step 1: Automate the wizard contract after separate approval**

Test this sequence:

1. choose H100 and two GPUs;
2. choose `minScale=0` and `maxScale=2`;
3. submit and receive 409 `confirmation_required`;
4. display the warning without losing user input;
5. require an explicit checkbox;
6. resubmit with `confirm_warnings=true`;
7. show the returned Model Run ID and endpoint.

Also prove `minScale > maxScale` is blocked client-side.

No test file or runner is added until its exact scope is separately approved.

- [x] **Step 2: Implement the smallest complete wizard**

Use one page with four visually separated sections, not a state-machine framework:

- Model and normalized name;
- GPU type and allowed count;
- minimum and maximum scale;
- recipe arguments, warnings, and confirmation.

GPU choices:

```typescript
export const GPU_OPTIONS = [
  { value: "ResourceType_GPU_V100", label: "NVIDIA V100 · 32 GB", counts: [1, 2, 4, 8, 16] },
  { value: "ResourceType_GPU_A100_NVLINK", label: "NVIDIA A100 · 80 GB", counts: [1, 2, 4, 6, 8] },
  { value: "ResourceType_GPU_H100", label: "NVIDIA H100 · 80 GB", counts: [1, 2, 4, 6, 8] }
];
```

Use native number inputs for scaling. Do not default back to 1–1 after the user changes values. The final payload preview redacts all credentials and shows only recipe, GPU, count, scaling, runtime, and secret UUID presence.

The implemented fail-closed UI also persists an attempt ID in `sessionStorage`,
uses compare-and-set for late responses, and blocks a retry after an uncertain
result until the user confirms a manual list check.

Verified with `pnpm --dir frontend build`; no test runner was invoked.

### Task 12: Apply the approved visual system and project illustrations

**Files:**
- Copy: `records/2026-09-20-vllm-recipes-cloudru-design/assets/cloud-credentials.png` → `frontend/public/illustrations/cloud-credentials.png`
- Copy: `records/2026-09-20-vllm-recipes-cloudru-design/assets/model-catalog.png` → `frontend/public/illustrations/model-catalog.png`
- Copy: `records/2026-09-20-vllm-recipes-cloudru-design/assets/gpu-deployment.png` → `frontend/public/illustrations/gpu-deployment.png`
- Modify: `frontend/src/styles.css`
- Modify: all page files

- [x] **Step 1: Copy approved assets without changing the historical originals**

Use explicit source and target paths. Keep the originals in the design record. Verify every target is RGBA and referenced by a page.

- [x] **Step 2: Implement the approved visual tokens**

Define CSS custom properties:

```css
:root {
  --ink: #171612;
  --paper: #fffdf8;
  --butter: #f5dc91;
  --apricot: #f3a36f;
  --mint: #bcecc8;
  --sky: #83cef2;
  --muted: #746f66;
  --line: #ded8cc;
  --danger: #b53a35;
  --radius-lg: 28px;
  --radius-md: 18px;
  color: var(--ink);
  background: #f8e8ae;
  font-family: Manrope, "Avenir Next", system-ui, sans-serif;
}
```

Use no remote font or stylesheet request. Use system fallback if Manrope is unavailable. Add a strict CSP meta tag permitting only self assets and the local API connection. Decorative images have `alt=""`. Respect `prefers-reduced-motion`.

Use illustrations as follows:

- setup side panel: `cloud-credentials.png`;
- catalog hero or empty state: `model-catalog.png`;
- successful create state: `gpu-deployment.png`.

- [ ] **Step 3: Automate accessibility-focused component checks after separate approval**

The production UI includes labelled fields, focusable links, live error messages,
non-color status text and reduced-motion handling. Automated assertions remain
deferred by the test gate. The production build exits 0.

### Task 13: Record critical browser behavior as DRAFT Gherkin

**Files:**
- Create: `tests/acceptance/features/credentials.feature`

- [x] **Step 1: Add product scenarios**

`credentials.feature`:

```gherkin
# language: ru
Функциональность: Локальные настройки доступа

  Сценарий: CFG-001. Настройки действуют меньше семи суток
    Дано пользователь сохранил настройки Cloud.ru
    Когда прошло меньше семи суток
    Тогда приложение открывает список инференсов

  Сценарий: CFG-002. Просроченные настройки удаляются
    Дано настройки Cloud.ru были сохранены семь суток назад
    Когда пользователь открывает приложение
    Тогда приложение показывает первоначальную настройку
    И просроченной записи нет в localStorage

  Сценарий: SAFE-001. Изменение и удаление выполняются только в консоли
    Дано открыт список существующих инференсов
    Когда пользователь выбирает редактирование или удаление
    Тогда приложение предлагает перейти в Cloud.ru Console
    И mutation-запрос к backend не отправлен
```

- [ ] **Step 2: Implement real browser steps after separate approval**

If later approved, the session fixture starts the local frontend, waits for the
port with a bounded loop, and always terminates the process. Playwright routes
mock only `/api`; they record HTTP methods so SAFE-001 can assert no mutation.

Use `page.add_init_script` to control `Date.now` and seed localStorage before navigation. Do not access implementation functions from test code.

No browser dependency, step definition or runner is added or invoked without that
approval.

### Task 14: Serve the production frontend from FastAPI

**Files:**
- Modify: `src/vllm_cloudru/main.py`

- [ ] **Step 1: Automate static-serving behavior after separate approval**

Build the frontend, create the app with `frontend/dist`, and assert:

- `GET /` returns `index.html`;
- `GET /catalog` returns the SPA shell;
- `GET /assets/{built_asset}` returns the built file;
- `GET /api/health` still returns JSON;
- a missing asset returns 404 rather than `index.html`.

No static-serving test is created under the current test gate.

- [x] **Step 2: Mount the built assets and add SPA fallback**

Mount `frontend/dist/assets` at `/assets`. Register a final GET fallback that returns `index.html` only for routes without a filename suffix and never for `/api`. If `frontend/dist/index.html` is absent, keep API startup working and return 404 for frontend routes.

Verified by production build and local HTTP smoke checks. `/docs` keeps its FastAPI
meaning; unknown `/api` routes and missing assets return 404.

### Task 15: Replace template documentation with verified project documentation

**Files:**
- Modify: `AGENTS.md`
- Modify: `README.md`
- Modify: `docs/development.md`
- Modify: `docs/bdd.md`
- Create: `docs/architecture.md`
- Modify: `docs/how-to-document.md` only if an implementation-discovered documentation rule requires it

- [x] **Step 1: Update current project identity**

Set:

```text
PROJECT_NAME = vLLM Recipes → Cloud.ru ML Inference
PROJECT_PURPOSE = Локальное веб-приложение для запуска конфигураций vLLM Recipes в Cloud.ru ML Inference и просмотра существующих Model Run.
```

Remove completed `remove_after_instruction_complete` blocks from README and development docs. Preserve the general repository rules.

- [x] **Step 2: Document only verified commands**

`docs/development.md` records only commands actually verified without crossing the
test gate. pytest, Vitest and Playwright commands are intentionally not declared
as current readiness checks.

`docs/architecture.md` must contain the implemented adapters, credential lifetime and threat boundary, read/create-only Cloud scope, recipe discovery path, unsupported multi-process rule, and links to the primary upstream sources. Do not copy the implementation plan into `docs/`.

- [x] **Step 3: Refresh relevant anchors**

Keep only non-obvious invariants:

- recipe links must be followed, not synthesized;
- create POST must not automatically retry;
- update/delete/suspend/resume are intentionally absent;
- secret redaction is a trust-boundary requirement.

Use `AICODE-NOTE` comments near the corresponding code. Remove any anchor made obsolete by the implementation.

### Task 16: Final verification

**Files:**
- Review all changed files

- [x] **Step 1: Run approved backend checks**

```sh
uv run ruff check src/vllm_cloudru
```

Expected: Ruff exits 0. No test runner is invoked without separate approval.

- [x] **Step 2: Run approved frontend checks**

```sh
pnpm --dir frontend build
```

Expected: the production build exits 0. No test runner is invoked.

- [x] **Step 3: Run repository integrity**

```sh
python3 scripts/check-repo.py --strict
```

Expected: exit 0 with no warnings.

- [x] **Step 4: Inspect the exact final scope**

```sh
git status --short
git diff --check
git diff --stat
rg -n --hidden -g '!records/**' -g '!.git/**' 'AICODE-(NOTE|TO[D]O|QUESTION):' .
```

Confirm:

- no credential or token value exists in the diff;
- no Cloud update/delete/suspend/resume implementation exists;
- generated images are present in the frontend and original record;
- every product scenario is present as a DRAFT contract; automated collection
  and execution remain deferred by the test gate;
- no unrelated user file changed.

Do not commit, push, deploy, or create real Cloud.ru resources without a separate explicit instruction.
