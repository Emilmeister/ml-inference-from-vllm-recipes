from __future__ import annotations

import copy
import json
import logging
import re
import time
from typing import Any
from urllib.parse import parse_qs, urlparse

import httpx
import pytest
from fastapi.testclient import TestClient
from playwright.sync_api import Page, expect
from pytest_bdd import given, parsers, scenarios, then, when
from support import (
    HF_SECRET_ID,
    LOG_GROUP_ID,
    PROJECT_ID,
    SECOND_SERVICE_ACCOUNT_ID,
    SERVICE_ACCOUNT_ID,
    catalog_item,
    cloud_options,
    fulfill_json,
    recipe_detail,
    store_credentials,
)

from vllm_cloudru.cloudru import CloudruClient, CloudruCreateUncertain
from vllm_cloudru.main import create_app
from vllm_cloudru.models import RecipeDetail
from vllm_cloudru.recipe_configuration import RecipeConfigurationError
from vllm_cloudru.translator import translate_rendering

scenarios("features/deployment.feature")


class FakeRecipes:
    def __init__(
        self,
        detail: RecipeDetail,
        *,
        error: Exception | None = None,
        resolver: Any | None = None,
    ) -> None:
        self.detail = detail
        self.error = error
        self.resolver = resolver
        self.selections: list[Any] = []

    async def get_detail(
        self,
        _hf_id: str,
        resource_type: str,
        selection: Any = None,
        **_kwargs: Any,
    ) -> RecipeDetail:
        assert resource_type
        self.selections.append(selection)
        if self.error is not None:
            raise self.error
        if self.resolver is not None:
            return self.resolver(selection)
        return self.detail


class FakeCloud:
    def __init__(self, *, create_error: Exception | None = None) -> None:
        self.create_error = create_error
        self.create_calls: list[dict[str, Any]] = []
        self.list_calls = 0

    async def create_model_run(
        self,
        _project_id: str,
        payload: dict[str, Any],
    ) -> dict[str, Any]:
        self.create_calls.append(payload)
        if self.create_error is not None:
            raise self.create_error
        return {
            "modelRunId": "run-created",
            "name": payload["name"],
            "status": "CREATING",
            "publicUrl": "https://run-created.modelrun.inference.cloud.ru/v1",
        }

    async def list_model_runs(self, *_args: Any, **_kwargs: Any) -> list[Any]:
        self.list_calls += 1
        return []


@pytest.fixture
def deployment_state() -> dict[str, Any]:
    return {}


def backend_detail(
    *,
    hardware: str = "h100",
    warnings: list[str] | None = None,
    model_id: str = "Acme/Small-FP8",
    arguments: list[dict[str, Any]] | None = None,
) -> RecipeDetail:
    return RecipeDetail.model_validate(
        {
            "hf_id": "Acme/Small",
            "title": "Small",
            "provider": "Acme",
            "tasks": ["generate"],
            "parameter_count": "8B",
            "variants": {"fp8": {"precision": "FP8"}},
            "recommended_hardware": hardware,
            "recommended_gpu_count": 2,
            "selected_variant": "fp8",
            "rendering_path": "renderings/single-node.json",
            "rendering": {"hardware": hardware},
            "translated": {
                "model_id": model_id,
                "arguments": arguments or [],
                "environment": [],
                "recommended_gpu_count": 2,
                "recommended_hardware": hardware,
            },
            "configuration": {
                "selection": {
                    "variant": "fp8",
                    "strategy": "tensor_parallel",
                    "frontend": "rust",
                    "features": [],
                    "feature_modes": {},
                },
                "groups": [],
            },
            "launchable": True,
            "launch_reason": None,
            "compatibility_warnings": warnings or [],
        }
    )


def deployment_request() -> dict[str, Any]:
    return {
        "credentials": {
            "project_id": PROJECT_ID,
            "key_id": "test-key-id",
            "key_secret": "test-key-secret",
        },
        "name": "Small",
        "hf_id": "Acme/Small",
        "resource_type": "ResourceType_GPU_H100",
        "gpu_count": 1,
        "runtime_template_id": "runtime-vllm",
        "min_scale": 1,
        "max_scale": 1,
        "service_account_id": SERVICE_ACCOUNT_ID,
        "authentication_type": "AUTHENTICATION_TYPE_KEY_BASED",
        "log_group_id": LOG_GROUP_ID,
        "configuration": {"feature_modes": {}},
        "confirm_warnings": False,
    }


def prepare_backend(
    state: dict[str, Any],
    *,
    detail: RecipeDetail | None = None,
    recipe_error: Exception | None = None,
    resolver: Any | None = None,
    cloud_error: Exception | None = None,
) -> None:
    state["request"] = deployment_request()
    state["recipes"] = FakeRecipes(
        detail or backend_detail(),
        error=recipe_error,
        resolver=resolver,
    )
    state["cloud"] = FakeCloud(create_error=cloud_error)


def post_backend(state: dict[str, Any]) -> None:
    app = create_app(
        recipes=state["recipes"],
        cloud_factory=lambda _credentials: state["cloud"],
    )
    with TestClient(app) as client:
        state["response"] = client.post("/api/cloud/inferences", json=state["request"])


def dynamic_recipe(state: dict[str, Any], query: str) -> dict[str, Any]:
    parameters = parse_qs(query)
    frontend = parameters.get("frontend", ["rust"])[-1]
    variant = parameters.get("variant", ["fp8"])[-1]
    detail = copy.deepcopy(state.get("detail", recipe_detail(frontend=frontend)))
    features = (
        parameters.get("feature", [])
        if "features_configured" in parameters
        else detail["configuration"]["selection"]["features"]
    )
    detail["configuration"]["selection"]["frontend"] = frontend
    detail["configuration"]["selection"]["variant"] = variant
    detail["configuration"]["selection"]["features"] = features
    for group in detail["configuration"]["groups"]:
        if group["key"] == "frontend":
            for option in group["options"]:
                option["selected"] = option["key"] == frontend
        if group["key"] == "variant":
            for option in group["options"]:
                option["selected"] = option["key"] == variant
        if group["key"] == "features":
            for option in group["options"]:
                option["selected"] = option["key"] in features
    detail["translated"]["model_id"] = f"Acme/Small-{variant.upper()}"
    detail["translated"]["environment"] = (
        [
            {
                "key": "VLLM_USE_RUST_FRONTEND",
                "value": "1",
                "parameterType": "PARAMETER_TYPE_ENV",
            }
        ]
        if frontend == "rust"
        else []
    )
    detail["translated"]["arguments"] = [
        {
            "key": "frontend-engine",
            "value": frontend,
            "parameterType": "PARAMETER_TYPE_ARG_KV",
        },
    ] + ([{
            "key": "tool-call-parser",
            "value": "hermes",
            "parameterType": "PARAMETER_TYPE_ARG_KV",
        }] if "tool_calling" in features else [])
    return detail


def install_deployment_routes(page: Page, state: dict[str, Any]) -> None:
    state.setdefault("create_requests", [])
    state.setdefault("create_responses", [])
    state.setdefault("detail_requests", [])

    def api(route: Any) -> None:
        request = route.request
        parsed = urlparse(request.url)
        if parsed.path == "/api/cloud/options":
            fulfill_json(route, state.get("options", cloud_options()))
            return
        if parsed.path == "/api/recipes":
            fulfill_json(route, state.get("catalog", [catalog_item()]))
            return
        if parsed.path.startswith("/api/recipes/"):
            state["detail_requests"].append(parsed.query)
            detail = dynamic_recipe(state, parsed.query)
            if (
                state.get("defer_python_recipe")
                and parse_qs(parsed.query).get("frontend") == ["python"]
            ):
                state["pending_recipe_route"] = route
                state["pending_recipe_payload"] = detail
                return
            fulfill_json(route, detail)
            return
        if parsed.path == "/api/cloud/inferences/query":
            fulfill_json(route, [])
            return
        if parsed.path == "/api/cloud/inferences" and request.method == "POST":
            state["create_requests"].append(request.post_data_json)
            response = (
                state["create_responses"].pop(0)
                if state["create_responses"]
                else {
                    "status": 201,
                    "body": {
                        "model_run_id": "run-created",
                        "name": "small",
                        "status": "CREATING",
                        "endpoint": "https://run-created.modelrun.inference.cloud.ru/v1",
                    },
                }
            )
            fulfill_json(route, response["body"], response["status"])
            return
        fulfill_json(route, {"detail": {"message": "Unexpected API request"}}, 500)

    page.route("**/api/**", api)


def open_deployment(page: Page, frontend_url: str, state: dict[str, Any]) -> None:
    store_credentials(page, frontend_url)
    install_deployment_routes(page, state)
    page.goto(
        f"{frontend_url}/deploy?model=Acme%2FSmall&resource=ResourceType_GPU_H100"
    )
    expect(page.locator(".recipe-configurator")).to_be_visible()


def select_access(page: Page, account_id: str = SERVICE_ACCOUNT_ID) -> None:
    page.get_by_label("Сервисный аккаунт").select_option(account_id)
    expect(page.get_by_role("button", name="Проверить и создать")).to_be_enabled()


def submit_deployment(page: Page) -> None:
    page.get_by_role("button", name="Проверить и создать").click()


@given("настроен single-node рецепт")
def configured_single_node(deployment_state: dict[str, Any]) -> None:
    prepare_backend(
        deployment_state,
        detail=backend_detail(warnings=["Проверьте достаточность памяти"]),
    )


@given("выбраны две H100 и масштабирование от 1 до 1")
def two_h100(deployment_state: dict[str, Any]) -> None:
    deployment_state["request"].update(
        {"gpu_count": 2, "min_scale": 1, "max_scale": 1}
    )


@when("пользователь подтверждает предупреждение и запускает модель")
def confirm_and_launch(deployment_state: dict[str, Any]) -> None:
    deployment_state["request"]["confirm_warnings"] = True
    post_backend(deployment_state)


@then("Cloud.ru получает один запрос создания")
def one_create_request(deployment_state: dict[str, Any]) -> None:
    assert deployment_state["response"].status_code == 201
    assert len(deployment_state["cloud"].create_calls) == 1


@then("payload содержит две H100 и масштабирование от 1 до 1")
def payload_has_hardware_and_scaling(deployment_state: dict[str, Any]) -> None:
    payload = deployment_state["cloud"].create_calls[0]
    assert payload["resourceType"] == "ResourceType_GPU_H100"
    assert payload["gpuCount"] == 2
    assert payload["scaling"]["minScale"] == 1
    assert payload["scaling"]["maxScale"] == 1


@given("настройки GPU расходятся с рецептом")
def mismatched_gpu(deployment_state: dict[str, Any]) -> None:
    prepare_backend(deployment_state, detail=backend_detail(hardware="a100"))


@when("пользователь запускает модель без подтверждения")
def launch_without_confirmation(deployment_state: dict[str, Any]) -> None:
    post_backend(deployment_state)


@then("Model Run не создаётся")
def model_run_not_created(deployment_state: dict[str, Any]) -> None:
    assert deployment_state["response"].status_code == 409
    assert deployment_state["cloud"].create_calls == []


@then("ответ содержит предупреждение о несовместимости")
def mismatch_warning_returned(deployment_state: dict[str, Any]) -> None:
    detail = deployment_state["response"].json()["detail"]
    assert detail["code"] == "confirmation_required"
    assert any("профил" in warning.lower() for warning in detail["warnings"])


@given(parsers.parse('настроен UUID секрета Hugging Face "{secret_id}"'))
def configured_hf_secret(deployment_state: dict[str, Any], secret_id: str) -> None:
    prepare_backend(deployment_state)
    deployment_state["request"]["credentials"]["hf_secret_id"] = secret_id


@when("пользователь запускает модель")
def launch_model(deployment_state: dict[str, Any]) -> None:
    post_backend(deployment_state)


@then("payload содержит этот UUID")
def payload_has_hf_secret_uuid(deployment_state: dict[str, Any]) -> None:
    payload = deployment_state["cloud"].create_calls[0]
    assert payload["modelSource"]["secretLocator"] == {
        "secretId": "11111111-1111-1111-1111-111111111111",
        "version": 1,
    }


@then("payload не содержит значение токена Hugging Face")
def payload_omits_hf_token(deployment_state: dict[str, Any]) -> None:
    serialized = json.dumps(deployment_state["cloud"].create_calls[0])
    assert "test-key-secret" not in serialized
    assert "hf_token" not in serialized.lower()


@given("Cloud.ru принял создание, но соединение оборвалось")
def cloud_create_uncertain(deployment_state: dict[str, Any]) -> None:
    prepare_backend(
        deployment_state,
        cloud_error=CloudruCreateUncertain(502, "Создание не подтверждено"),
    )


@when("приложение получает неопределённый результат")
def receive_uncertain_result(deployment_state: dict[str, Any]) -> None:
    post_backend(deployment_state)


@then("создание было отправлено ровно один раз")
def create_sent_once(deployment_state: dict[str, Any]) -> None:
    assert len(deployment_state["cloud"].create_calls) == 1


@then("автоматический поиск Model Run не выполняется")
def no_automatic_lookup(deployment_state: dict[str, Any]) -> None:
    assert deployment_state["cloud"].list_calls == 0


@then("пользователю предложено обновить список для ручной проверки")
def manual_check_required(deployment_state: dict[str, Any]) -> None:
    response = deployment_state["response"]
    assert response.status_code == 502
    assert response.json()["detail"]["manualCheckRequired"] is True
    assert "Обновите список" in response.json()["detail"]["instruction"]


@given("рецепт публикует доступные variant и single-node strategy")
def configurable_recipe(
    page: Page,
    frontend_url: str,
    deployment_state: dict[str, Any],
) -> None:
    detail = recipe_detail()
    detail["configuration"]["selection"]["features"] = []
    for group in detail["configuration"]["groups"]:
        if group["key"] == "features":
            for option in group["options"]:
                option["selected"] = False
    deployment_state["detail"] = detail
    open_deployment(page, frontend_url, deployment_state)


@when("пользователь выбирает variant, strategy, frontend и features")
def choose_recipe_configuration(page: Page, deployment_state: dict[str, Any]) -> None:
    page.get_by_label("BF16", exact=True).click()
    expect(page).to_have_url(re.compile(r"[?&]variant=bf16(?:&|$)"))
    page.get_by_label("Python", exact=True).click()
    expect(page).to_have_url(re.compile(r"[?&]frontend=python(?:&|$)"))
    expect(page.get_by_label("Python", exact=True)).to_be_checked()
    page.get_by_label("Tool Calling", exact=True).click()
    page.wait_for_url(re.compile(r"[?&]feature=tool_calling(?:&|$)"))
    expect(page.get_by_label("Tool Calling", exact=True)).to_be_checked()

    deployment_state["create_responses"] = [
        {
            "status": 409,
            "body": {
                "detail": {
                    "code": "confirmation_required",
                    "warnings": ["Проверка выбранной конфигурации"],
                }
            },
        }
    ]
    select_access(page)
    submit_deployment(page)
    expect(
        page.get_by_text("Проверка выбранной конфигурации", exact=True)
    ).to_be_visible()
    browser_request = deployment_state["create_requests"][0]

    def resolver(selection: Any) -> RecipeDetail:
        arguments = [
            {
                "key": "frontend-engine",
                "value": selection.frontend,
                "parameterType": "PARAMETER_TYPE_ARG_KV",
            }
        ]
        if "tool_calling" in selection.features:
            arguments.append(
                {
                    "key": "tool-call-parser",
                    "value": "hermes",
                    "parameterType": "PARAMETER_TYPE_ARG_KV",
                }
            )
        return backend_detail(
            model_id="Acme/Small-BF16",
            arguments=arguments,
        )

    prepare_backend(deployment_state, resolver=resolver)
    deployment_state["request"] = browser_request
    post_backend(deployment_state)


@then("предпросмотр содержит checkpoint и аргументы выбранной конфигурации")
def preview_has_selected_configuration(page: Page) -> None:
    preview = page.locator(".payload-preview pre")
    expect(preview).to_contain_text('"modelId": "Acme/Small-BF16"')
    expect(preview).to_contain_text('"value": "python"')
    expect(preview).to_contain_text('"key": "tool-call-parser"')


@then("backend заново разрешает ключи конфигурации перед созданием")
def backend_resolves_selection(deployment_state: dict[str, Any]) -> None:
    assert deployment_state["response"].status_code == 201
    selection = deployment_state["recipes"].selections[0]
    assert selection.variant == "bf16"
    assert selection.strategy == "tensor_parallel"
    assert selection.frontend == "python"
    assert selection.features == ["tool_calling"]
    payload = deployment_state["cloud"].create_calls[0]
    assert payload["modelSource"]["huggingFaceRepository"]["model"] == "Small-BF16"


@given("рецепт публикует multi-node и companion-конфигурации")
def recipe_with_unsupported_options(
    page: Page,
    frontend_url: str,
    deployment_state: dict[str, Any],
) -> None:
    open_deployment(page, frontend_url, deployment_state)


@when("пользователь открывает выбор конфигурации")
def configuration_is_open(page: Page) -> None:
    expect(page.locator(".recipe-configurator")).to_be_visible()


@then("неподдерживаемые опции показаны неактивным цветом с причиной")
def unsupported_options_are_muted(page: Page) -> None:
    expect(page.locator(".config-option--disabled")).to_have_count(2)
    expect(page.get_by_text("Cloud.ru принимает один Model Run", exact=True)).to_be_visible()
    expect(page.get_by_text("Требуется отдельный процесс", exact=True)).to_be_visible()


@then("неподдерживаемые опции нельзя выбрать")
def unsupported_options_disabled(page: Page) -> None:
    expect(page.get_by_label("Multi-node", exact=True)).to_be_disabled()
    expect(page.get_by_label("Companion service", exact=True)).to_be_disabled()


@given("пользователь выбрал допустимое количество GPU")
def valid_gpu_count(deployment_state: dict[str, Any]) -> None:
    translated = translate_rendering(
        {
            "deploy_type": "single_node",
            "argv": [
                "vllm",
                "serve",
                "Acme/Small-FP8",
                "--tensor-parallel-size",
                "2",
                "--max-model-len",
                "8192",
            ],
        }
    )
    prepare_backend(
        deployment_state,
        detail=backend_detail(arguments=translated.arguments),
    )
    deployment_state["request"]["gpu_count"] = 2


@when("backend формирует payload Model Run")
def backend_builds_payload(deployment_state: dict[str, Any]) -> None:
    post_backend(deployment_state)


@then("payload содержит выбранный gpuCount")
def payload_has_selected_gpu_count(deployment_state: dict[str, Any]) -> None:
    assert deployment_state["cloud"].create_calls[0]["gpuCount"] == 2


@then("динамические аргументы не содержат tensor-parallel-size")
def tensor_parallel_argument_omitted(deployment_state: dict[str, Any]) -> None:
    arguments = deployment_state["cloud"].create_calls[0]["servingOptions"][
        "dynamicalOptions"
    ]["args"]
    assert all(argument["key"].lstrip("-") != "tensor-parallel-size" for argument in arguments)


@given("браузер передал ключ, отсутствующий в опубликованном рецепте")
def unknown_configuration_key(deployment_state: dict[str, Any]) -> None:
    prepare_backend(
        deployment_state,
        recipe_error=RecipeConfigurationError("Неизвестный variant"),
    )
    deployment_state["request"]["configuration"] = {
        "variant": "unknown",
        "feature_modes": {},
    }


@then("backend отвечает ошибкой конфигурации")
def backend_rejects_configuration(deployment_state: dict[str, Any]) -> None:
    response = deployment_state["response"]
    assert response.status_code == 422
    assert response.json()["detail"]["code"] == "invalid_recipe_configuration"


@then("запрос создания Model Run не отправляется")
def create_request_not_sent(deployment_state: dict[str, Any]) -> None:
    assert deployment_state["cloud"].create_calls == []


@given("пользователь подтвердил предупреждения текущей конфигурации")
def confirmed_current_warnings(
    page: Page,
    frontend_url: str,
    deployment_state: dict[str, Any],
) -> None:
    deployment_state["create_responses"] = [
        {
            "status": 409,
            "body": {
                "detail": {
                    "code": "confirmation_required",
                    "warnings": ["Проверьте выбранную конфигурацию"],
                }
            },
        },
        {
            "status": 409,
            "body": {
                "detail": {
                    "code": "confirmation_required",
                    "warnings": ["Проверьте новую конфигурацию"],
                }
            },
        },
    ]
    open_deployment(page, frontend_url, deployment_state)
    select_access(page)
    submit_deployment(page)
    confirmation = page.get_by_label(
        "Я понимаю предупреждения и всё равно хочу создать Model Run"
    )
    expect(confirmation).to_be_visible()
    confirmation.check()


@when("пользователь меняет любую опцию конфигурации")
def change_configuration_option(page: Page) -> None:
    page.get_by_label("Python", exact=True).click()
    page.wait_for_url(re.compile(r"[?&]frontend=python(?:&|$)"))
    expect(page.get_by_label("Python", exact=True)).to_be_checked()


@then("прежнее подтверждение больше не действует")
def old_confirmation_is_reset(page: Page) -> None:
    expect(
        page.get_by_label("Я понимаю предупреждения и всё равно хочу создать Model Run")
    ).to_have_count(0)


@then("перед созданием выполняется новая проверка предупреждений")
def warnings_checked_again(page: Page, deployment_state: dict[str, Any]) -> None:
    submit_deployment(page)
    expect(page.get_by_text("Проверьте новую конфигурацию", exact=True)).to_be_visible()
    assert len(deployment_state["create_requests"]) == 2
    before, after = deployment_state["create_requests"]
    assert before["configuration"]["frontend"] == "rust"
    assert after["configuration"]["frontend"] == "python"
    assert after["configuration"] != before["configuration"]
    assert after["confirm_warnings"] is False


@given("Cloud.ru вернул несколько активных сервисных аккаунтов")
def several_service_accounts(
    page: Page,
    frontend_url: str,
    deployment_state: dict[str, Any],
) -> None:
    open_deployment(page, frontend_url, deployment_state)
    expect(page.get_by_label("Сервисный аккаунт").locator("option")).to_have_count(3)


@when('пользователь явно выбирает аккаунт и способ «API-ключ»')
def choose_account_and_api_key(page: Page) -> None:
    page.get_by_label("Сервисный аккаунт").select_option(SECOND_SERVICE_ACCOUNT_ID)
    page.get_by_label("API-ключ", exact=True).check()
    submit_deployment(page)


@then("payload содержит идентификатор выбранного аккаунта")
def payload_has_selected_account(deployment_state: dict[str, Any]) -> None:
    assert deployment_state["create_requests"][0]["service_account_id"] == (
        SECOND_SERVICE_ACCOUNT_ID
    )


@then(parsers.parse('тип аутентификации равен "{authentication_type}"'))
def payload_has_authentication_type(
    deployment_state: dict[str, Any],
    authentication_type: str,
) -> None:
    assert deployment_state["create_requests"][0]["authentication_type"] == (
        authentication_type
    )


@given("пользователь явно выбрал сервисный аккаунт")
def explicit_service_account(
    page: Page,
    frontend_url: str,
    deployment_state: dict[str, Any],
) -> None:
    open_deployment(page, frontend_url, deployment_state)
    page.get_by_label("Сервисный аккаунт").select_option(SERVICE_ACCOUNT_ID)


@when('он выбирает способ «Ключи доступа» и создаёт Model Run')
def choose_access_keys_and_create(page: Page) -> None:
    page.get_by_label("Ключи доступа", exact=True).check()
    submit_deployment(page)


@given('Cloud.ru вернул активную готовую группу "default" и другие группы')
def available_log_groups(deployment_state: dict[str, Any]) -> None:
    deployment_state["options"] = cloud_options()


@when("пользователь открывает форму создания")
def open_create_form(
    page: Page,
    frontend_url: str,
    deployment_state: dict[str, Any],
) -> None:
    open_deployment(page, frontend_url, deployment_state)


@then('группа "default" выбрана предварительно')
def default_log_group_selected(page: Page) -> None:
    expect(page.get_by_label("Log group")).to_have_value(LOG_GROUP_ID)


@then("payload включает выбранную log group и включает логирование")
def payload_has_logging(
    page: Page,
    deployment_state: dict[str, Any],
) -> None:
    select_access(page)
    submit_deployment(page)
    assert deployment_state["create_requests"][0]["log_group_id"] == LOG_GROUP_ID

    prepare_backend(deployment_state)
    post_backend(deployment_state)
    cloud_payload = deployment_state["cloud"].create_calls[0]
    assert cloud_payload["options"]["logGroupId"] == LOG_GROUP_ID
    assert cloud_payload["options"]["isEnabledLogging"] is True


@given("пользователь отправил запрос создания Model Run")
def submitted_create_request(
    page: Page,
    frontend_url: str,
    deployment_state: dict[str, Any],
) -> None:
    deployment_state["create_responses"] = [
        {
            "status": 422,
            "body": {
                "detail": {
                    "code": "cloud_error",
                    "message": "Cloud.ru отклонил Model Run",
                }
            },
        }
    ]
    open_deployment(page, frontend_url, deployment_state)
    select_access(page)
    submit_deployment(page)


@when("Cloud.ru отклоняет создание")
def cloud_rejects_creation(deployment_state: dict[str, Any]) -> None:
    assert len(deployment_state["create_requests"]) == 1


@then("приложение показывает ошибку в модальном окне")
def creation_error_modal(page: Page) -> None:
    dialog = page.get_by_role("dialog")
    expect(dialog).to_be_visible()
    expect(dialog).to_contain_text("Cloud.ru отклонил Model Run")


@then("пользователь может закрыть модальное окно")
def close_creation_error(page: Page) -> None:
    page.get_by_role("button", name="Закрыть сообщение об ошибке").click()
    expect(page.get_by_role("dialog")).not_to_be_visible()


@given("пользователь успешно создал Model Run для модели")
def created_model_run(
    page: Page,
    frontend_url: str,
    deployment_state: dict[str, Any],
) -> None:
    open_deployment(page, frontend_url, deployment_state)
    select_access(page)
    submit_deployment(page)
    expect(page.get_by_text("Model Run создан", exact=True)).to_be_visible()


@given("вернулся в каталог")
def returned_to_catalog(
    page: Page,
    frontend_url: str,
) -> None:
    page.goto(f"{frontend_url}/catalog")
    expect(page.get_by_role("heading", name="Small")).to_be_visible()


@when('он снова выбирает «Настроить запуск» для той же модели')
def configure_same_model_again(page: Page) -> None:
    page.get_by_role("link", name="Открыть рецепт").click()
    page.get_by_role("link", name="Настроить запуск").click()


@then("приложение показывает новую форму запуска")
def fresh_deployment_form(page: Page) -> None:
    expect(page.get_by_role("heading", name="Соберите запуск.")).to_be_visible()
    expect(page.locator(".recipe-configurator")).to_be_visible()


@then("результат предыдущего создания не показывается")
def previous_result_not_shown(page: Page) -> None:
    expect(page.get_by_text("Model Run создан", exact=True)).to_have_count(0)


@then("новый запрос создания не отправляется до подтверждения формы")
def no_new_create_before_submit(deployment_state: dict[str, Any]) -> None:
    assert len(deployment_state["create_requests"]) == 1


@given("конфигуратор рецепта загружен")
def loaded_configurator(
    page: Page,
    frontend_url: str,
    deployment_state: dict[str, Any],
) -> None:
    deployment_state["defer_python_recipe"] = True
    open_deployment(page, frontend_url, deployment_state)
    page.locator(".recipe-configurator").evaluate(
        "element => element.dataset.acceptanceIdentity = 'original'"
    )


@when("пользователь переключает frontend между Rust и Python")
def switch_frontend(page: Page, deployment_state: dict[str, Any]) -> None:
    with page.expect_request(
        lambda request: (
            urlparse(request.url).path.startswith("/api/recipes/")
            and parse_qs(urlparse(request.url).query).get("frontend") == ["python"]
        )
    ):
        page.get_by_label("Python", exact=True).click()
    page.wait_for_url(re.compile(r"[?&]frontend=python(?:&|$)"))
    deadline = time.monotonic() + 5
    while "pending_recipe_route" not in deployment_state:
        if time.monotonic() >= deadline:
            pytest.fail("Python recipe request did not reach the route handler")
        page.wait_for_timeout(10)


@then("конфигуратор остаётся видимым во время пересчёта")
def configurator_stays_mounted(
    page: Page,
    deployment_state: dict[str, Any],
) -> None:
    assert "pending_recipe_route" in deployment_state
    configurator = page.locator(".recipe-configurator")
    expect(configurator).to_be_visible()
    expect(configurator).to_have_attribute("data-acceptance-identity", "original")


@then("после ответа backend выбранный frontend и зависимые параметры обновляются")
def frontend_dependencies_updated(
    page: Page,
    deployment_state: dict[str, Any],
) -> None:
    fulfill_json(
        deployment_state.pop("pending_recipe_route"),
        deployment_state.pop("pending_recipe_payload"),
    )
    expect(page.get_by_label("Python", exact=True)).to_be_checked()
    preview = page.locator(".payload-preview pre")
    expect(preview).to_contain_text('"frontend": "python"')
    expect(preview).to_contain_text('"value": "python"')
    expect(preview).not_to_contain_text("VLLM_USE_RUST_FRONTEND")


@given("Cloud.ru отклонил запрос с ошибкой, содержащей секрет")
def cloud_error_contains_secrets(deployment_state: dict[str, Any]) -> None:
    deployment_state["secrets"] = [
        "test-key-secret",
        "iam-secret-token",
        "hf-secret-token",
    ]


@when("приложение возвращает ошибку пользователю")
def application_returns_safe_error(
    deployment_state: dict[str, Any],
    caplog: pytest.LogCaptureFixture,
) -> None:
    caplog.set_level(logging.ERROR, logger="vllm_cloudru.cloudru")

    def cloud_response(request: httpx.Request) -> httpx.Response:
        if str(request.url).startswith("https://iam.api.cloud.ru/"):
            return httpx.Response(200, json={"access_token": "iam-secret-token"})
        return httpx.Response(
            400,
            headers={"X-Request-ID": "request-safe"},
            json={
                "keySecret": "test-key-secret",
                "hfToken": "hf-secret-token",
                "message": (
                    "test-key-secret iam-secret-token hf-secret-token must stay private"
                ),
            },
        )

    http = httpx.AsyncClient(transport=httpx.MockTransport(cloud_response))

    def cloud_factory(credentials: Any) -> CloudruClient:
        return CloudruClient(credentials, http)

    app = create_app(recipes=object(), cloud_factory=cloud_factory)
    with TestClient(app) as client:
        deployment_state["safe_response"] = client.post(
            "/api/cloud/inferences/query",
            json={
                "credentials": {
                    "project_id": PROJECT_ID,
                    "key_id": "test-key-id",
                    "key_secret": "test-key-secret",
                    "hf_secret_id": HF_SECRET_ID,
                }
            },
        )
        assert client.portal is not None
        client.portal.call(http.aclose)
    deployment_state["log_text"] = caplog.text


@then("ответ не содержит Key Secret, IAM-токен и значение HF-токена")
def response_has_no_secrets(deployment_state: dict[str, Any]) -> None:
    response = deployment_state["safe_response"]
    assert response.status_code == 400
    serialized = response.text
    assert all(secret not in serialized for secret in deployment_state["secrets"])


@then("журнал не содержит Key Secret, IAM-токен и значение HF-токена")
def log_has_no_secrets(deployment_state: dict[str, Any]) -> None:
    log_text = deployment_state["log_text"]
    assert all(secret not in log_text for secret in deployment_state["secrets"])
    assert "<redacted>" in log_text
