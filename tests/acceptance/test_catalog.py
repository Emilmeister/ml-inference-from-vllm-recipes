from __future__ import annotations

import re
from typing import Any
from urllib.parse import urlparse

import pytest
from playwright.sync_api import Page, expect
from pytest_bdd import given, scenarios, then, when
from support import catalog_item, fulfill_json, recipe_detail, store_credentials

scenarios("features/catalog.feature")


@pytest.fixture
def catalog_state() -> dict[str, Any]:
    return {}


def open_catalog(page: Page, frontend_url: str, state: dict[str, Any]) -> None:
    store_credentials(page, frontend_url)

    def api(route: Any) -> None:
        path = urlparse(route.request.url).path
        if path == "/api/recipes":
            fulfill_json(route, state.get("catalog", []))
            return
        if path.startswith("/api/recipes/"):
            fulfill_json(route, state["detail"])
            return
        fulfill_json(route, {"detail": "Unexpected API request"}, 500)

    page.route("**/api/**", api)
    page.goto(f"{frontend_url}/catalog")


@given("upstream вернул single-node и multi-node рецепты")
def single_and_multi(catalog_state: dict[str, Any]) -> None:
    catalog_state["catalog"] = [
        catalog_item(),
        catalog_item(
            "Other/Multi",
            provider="Other",
            parameter_count="70B",
            launchable=False,
            launch_reason="Рецепт требует нескольких процессов",
        ),
    ]


@when("пользователь запрашивает каталог")
def request_catalog(page: Page, frontend_url: str, catalog_state: dict[str, Any]) -> None:
    open_catalog(page, frontend_url, catalog_state)


@then("в каталоге видны обе модели")
def both_models_are_visible(page: Page) -> None:
    expect(page.get_by_role("heading", name="Small")).to_be_visible()
    expect(page.get_by_role("heading", name="Multi")).to_be_visible()


@then("у каждой модели указан статус автоматического запуска")
def every_model_has_launch_status(page: Page) -> None:
    cards = page.locator(".recipe-card")
    expect(cards).to_have_count(2)
    expect(cards.nth(0).locator(".recipe-status")).to_contain_text("Готов к запуску")
    expect(cards.nth(1).locator(".recipe-status")).to_contain_text("Только просмотр")


@given("upstream вернул multi-node рецепт")
def multi_node_detail(catalog_state: dict[str, Any]) -> None:
    catalog_state["detail"] = recipe_detail(
        "Other/Multi",
        launchable=False,
        launch_reason="Рецепт требует нескольких процессов",
    )


@when("пользователь открывает рецепт")
def open_recipe(page: Page, frontend_url: str, catalog_state: dict[str, Any]) -> None:
    open_catalog(page, frontend_url, catalog_state)
    page.goto(f"{frontend_url}/catalog/Other/Multi")


@then("запуск недоступен с причиной о нескольких процессах")
def launch_is_disabled(page: Page) -> None:
    expect(page.get_by_text("Запуск отключён", exact=True)).to_be_visible()
    reason = page.get_by_text("Рецепт требует нескольких процессов", exact=True)
    expect(reason.first).to_be_visible()
    expect(page.get_by_role("link", name="Настроить запуск")).to_have_count(0)


@given("рецепт можно запустить, но для выбранного GPU нет точного профиля")
def warning_recipe(catalog_state: dict[str, Any]) -> None:
    catalog_state["catalog"] = [
        catalog_item(warning="Для выбранного GPU нет точного профиля")
    ]


@when("пользователь открывает каталог")
def open_catalog_page(page: Page, frontend_url: str, catalog_state: dict[str, Any]) -> None:
    open_catalog(page, frontend_url, catalog_state)


@then('рецепт имеет статус «С предупреждением»')
def recipe_has_warning_status(page: Page) -> None:
    expect(page.locator(".recipe-status")).to_contain_text("С предупреждением")


@then("запуск рецепта остаётся доступен")
def recipe_remains_launchable(page: Page) -> None:
    expect(page.get_by_role("link", name="Открыть рецепт")).to_be_visible()


@given("каталог содержит модели разных провайдеров и размеров")
def different_providers_and_sizes(catalog_state: dict[str, Any]) -> None:
    catalog_state["catalog"] = [
        catalog_item(),
        catalog_item("Acme/Large", provider="Acme", parameter_count="70B"),
        catalog_item("Other/Small", provider="Other", parameter_count="8B"),
    ]


@when("пользователь выбирает провайдера и размер модели")
def choose_provider_and_size(
    page: Page,
    frontend_url: str,
    catalog_state: dict[str, Any],
) -> None:
    open_catalog(page, frontend_url, catalog_state)
    page.get_by_label("Провайдер").select_option("Acme")
    page.get_by_label("Размер").select_option("70B")


@then("видны только модели выбранного провайдера и размера")
def only_matching_model_is_visible(page: Page) -> None:
    expect(page.locator(".recipe-card")).to_have_count(1)
    expect(page.get_by_role("heading", name="Large")).to_be_visible()


@then("выбранные фильтры сохранены в адресе страницы")
def filters_are_in_url(page: Page) -> None:
    expect(page).to_have_url(re.compile(r"[?&]provider=Acme(?:&.*)?&size=70B(?:&|$)"))
