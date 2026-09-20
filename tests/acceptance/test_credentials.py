from __future__ import annotations

import re

from playwright.sync_api import Page, expect
from pytest_bdd import given, scenarios, then, when
from support import PROJECT_ID, STORAGE_KEY, fulfill_json, store_credentials

scenarios("features/credentials.feature")

SEVEN_DAYS_MS = 7 * 24 * 60 * 60 * 1000


def route_empty_inferences(page: Page) -> None:
    page.route(
        "**/api/cloud/inferences/query",
        lambda route: fulfill_json(route, []),
    )


def save_credentials_through_setup(page: Page, frontend_url: str) -> dict[str, object]:
    page.route(
        "**/api/cloud/validate",
        lambda route: fulfill_json(route, {"valid": True}),
    )
    route_empty_inferences(page)
    page.goto(f"{frontend_url}/setup")
    page.get_by_label("PROJECT_ID").fill(PROJECT_ID)
    page.get_by_label("Key ID").fill("test-key-id")
    page.get_by_label("Key Secret").fill("test-key-secret")
    before_save = page.evaluate("Date.now()")
    page.get_by_role("button", name="Проверить подключение").click()
    expect(page).to_have_url(re.compile(r"/inferences$"))
    after_save = page.evaluate("Date.now()")
    stored = page.evaluate(
        "key => JSON.parse(localStorage.getItem(key))",
        STORAGE_KEY,
    )
    assert isinstance(stored, dict)
    expires_at = stored["expiresAt"]
    assert isinstance(expires_at, int | float)
    assert before_save + SEVEN_DAYS_MS <= expires_at <= after_save + SEVEN_DAYS_MS
    return stored


@given("пользователь сохранил настройки Cloud.ru")
def saved_credentials(page: Page, frontend_url: str) -> None:
    save_credentials_through_setup(page, frontend_url)


@when("прошло меньше семи суток")
def open_before_expiry(page: Page, frontend_url: str) -> None:
    page.goto(frontend_url)


@then("приложение открывает список инференсов")
def opens_inference_list(page: Page) -> None:
    expect(page).to_have_url(re.compile(r"/inferences$"))
    expect(page.get_by_role("heading", name="Инференсы под рукой.")).to_be_visible()


@given("настройки Cloud.ru были сохранены семь суток назад")
def expired_credentials(page: Page, frontend_url: str) -> None:
    stored = save_credentials_through_setup(page, frontend_url)
    stored["expiresAt"] = page.evaluate("Date.now()")
    page.evaluate(
        "([key, value]) => localStorage.setItem(key, JSON.stringify(value))",
        [STORAGE_KEY, stored],
    )


@when("пользователь открывает приложение")
def open_application(page: Page, frontend_url: str) -> None:
    page.goto(frontend_url)


@then("приложение показывает первоначальную настройку")
def shows_setup(page: Page) -> None:
    expect(page).to_have_url(re.compile(r"/setup$"))
    expect(page.get_by_role("heading", name="Доступ к Cloud.ru")).to_be_visible()


@then("просроченной записи нет в localStorage")
def expired_record_removed(page: Page) -> None:
    assert page.evaluate("key => localStorage.getItem(key)", STORAGE_KEY) is None


@given("в браузере сохранены действующие настройки Cloud.ru")
def valid_credentials(page: Page, frontend_url: str) -> None:
    store_credentials(page, frontend_url)
    route_empty_inferences(page)
    page.goto(f"{frontend_url}/inferences")


@when('пользователь нажимает «Очистить настройки» и подтверждает действие')
def clear_settings(page: Page) -> None:
    page.once("dialog", lambda dialog: dialog.accept())
    page.get_by_role("button", name="Очистить настройки").click()


@then("записи настроек нет в localStorage")
def credentials_record_removed(page: Page) -> None:
    assert page.evaluate("key => localStorage.getItem(key)", STORAGE_KEY) is None
