from __future__ import annotations

from typing import Any
from urllib.parse import urlparse

import pytest
from playwright.sync_api import Page, expect
from pytest_bdd import given, scenarios, then, when
from support import fulfill_json, model_run, store_credentials

scenarios("features/inferences.feature")


@pytest.fixture
def inference_state() -> dict[str, Any]:
    return {"runs": [], "requests": []}


def open_inferences(
    page: Page,
    frontend_url: str,
    state: dict[str, Any],
) -> None:
    store_credentials(page, frontend_url)

    def api(route: Any) -> None:
        request = route.request
        path = urlparse(request.url).path
        state["requests"].append((request.method, path))
        if request.method == "POST" and path == "/api/cloud/inferences/query":
            fulfill_json(route, state["runs"])
            return
        fulfill_json(route, {"detail": "Unexpected API request"}, 500)

    page.route("**/api/**", api)
    page.goto(f"{frontend_url}/inferences")
    expect(page.locator(".inference-toolbar")).to_be_visible()


def run_row(page: Page, name: str):
    return page.locator(".inference-row").filter(
        has=page.get_by_role("heading", name=name, exact=True)
    )


def assert_no_mutation_requests(state: dict[str, Any]) -> None:
    assert state["requests"]
    assert set(state["requests"]) == {("POST", "/api/cloud/inferences/query")}


@given('Cloud.ru вернул запущенный Model Run "demo"')
def running_model_run(inference_state: dict[str, Any]) -> None:
    inference_state["runs"] = [model_run()]


@when("пользователь запрашивает свои инференсы")
def request_inferences(
    page: Page,
    frontend_url: str,
    inference_state: dict[str, Any],
) -> None:
    open_inferences(page, frontend_url, inference_state)


@then('он видит "demo" со статусом "RUNNING"')
def running_status_is_visible(page: Page) -> None:
    row = run_row(page, "demo")
    expect(row).to_be_visible()
    expect(row.get_by_text("RUNNING", exact=True)).to_be_visible()


@then("видит модель, тип и количество GPU")
def model_and_gpu_are_visible(page: Page) -> None:
    row = run_row(page, "demo")
    expect(row.get_by_text("Acme/Small-FP8", exact=True)).to_be_visible()
    expect(row.get_by_text("H100 × 2", exact=True)).to_be_visible()


@then("видит границы масштабирования и endpoint")
def scaling_and_endpoint_are_visible(page: Page) -> None:
    row = run_row(page, "demo")
    expect(row.get_by_text("1 → 1", exact=True)).to_be_visible()
    expect(
        row.get_by_text(
            "https://run-demo.modelrun.inference.cloud.ru/v1",
            exact=True,
        )
    ).to_be_visible()


@given("пользователь видит существующий Model Run")
def existing_model_run(inference_state: dict[str, Any]) -> None:
    inference_state["runs"] = [model_run()]


@when("он нажимает «Редактировать» или «Удалить»")
def click_console_actions(
    page: Page,
    frontend_url: str,
    inference_state: dict[str, Any],
) -> None:
    open_inferences(page, frontend_url, inference_state)
    page.get_by_role(
        "button", name="Редактировать demo в Cloud.ru Console"
    ).click()
    page.get_by_role("button", name="Удалить demo в Cloud.ru Console").click()


@then("приложение не отправляет mutation-запрос")
def no_mutation_request(page: Page, inference_state: dict[str, Any]) -> None:
    page.wait_for_timeout(250)
    assert_no_mutation_requests(inference_state)


@then("показывает ссылку на Cloud.ru Console")
def cloud_console_link_is_visible(page: Page) -> None:
    link = page.get_by_role("link", name="Открыть Cloud.ru Console")
    expect(link).to_be_visible()
    expect(link).to_have_attribute("href", "https://console.cloud.ru")


@given("Cloud.ru вернул активный и удалённый Model Run")
def active_and_deleted_runs(inference_state: dict[str, Any]) -> None:
    inference_state["runs"] = [
        model_run("active", model_run_id="run-active"),
        model_run("deleted", model_run_id="run-deleted", status="DELETED"),
    ]


@when("пользователь открывает список инференсов")
def open_inference_list(
    page: Page,
    frontend_url: str,
    inference_state: dict[str, Any],
) -> None:
    open_inferences(page, frontend_url, inference_state)


@then("активный Model Run виден")
def active_run_is_visible(page: Page) -> None:
    expect(run_row(page, "active")).to_be_visible()


@then("удалённый Model Run скрыт")
def deleted_run_is_hidden(page: Page) -> None:
    expect(run_row(page, "deleted")).to_have_count(0)


@when("пользователь включает «Показывать удалённые»")
def show_deleted(page: Page, inference_state: dict[str, Any]) -> None:
    inference_state["requests_before_toggle"] = len(inference_state["requests"])
    page.get_by_label("Показывать удалённые").check()


@then("удалённый Model Run становится виден без нового запроса")
def deleted_run_appears_without_request(
    page: Page,
    inference_state: dict[str, Any],
) -> None:
    expect(run_row(page, "deleted")).to_be_visible()
    assert len(inference_state["requests"]) == inference_state["requests_before_toggle"]


@when("отображаются действия endpoint, редактирования и удаления")
def display_compact_actions(
    page: Page,
    frontend_url: str,
    inference_state: dict[str, Any],
) -> None:
    open_inferences(page, frontend_url, inference_state)


@then("действия показаны только иконками с доступными названиями")
def actions_are_icon_only_and_accessible(page: Page) -> None:
    labels = [
        "Скопировать endpoint demo",
        "Редактировать demo в Cloud.ru Console",
        "Удалить demo в Cloud.ru Console",
    ]
    for label in labels:
        button = page.get_by_role("button", name=label)
        expect(button).to_be_visible()
        assert button.inner_text().strip() == ""


@then("редактирование и удаление не отправляют mutation-запрос")
def compact_actions_do_not_mutate(
    page: Page,
    inference_state: dict[str, Any],
) -> None:
    page.get_by_role(
        "button", name="Редактировать demo в Cloud.ru Console"
    ).click()
    page.get_by_role("button", name="Удалить demo в Cloud.ru Console").click()
    page.wait_for_timeout(250)
    assert_no_mutation_requests(inference_state)
