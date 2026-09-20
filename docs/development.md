# Среда, запуск и проверки

Проверенная 20 сентября 2026 года локальная среда: Darwin, Python 3.13.5,
uv 0.12.1, Node.js 24.8.0 и pnpm 10.26.0. Версии библиотек зафиксированы в
`uv.lock` и `frontend/pnpm-lock.yaml`.

## Установка зависимостей

Из корня проекта:

```sh
uv sync
pnpm --dir frontend install --frozen-lockfile
```

Обе команды изменяют только локальные окружения и файлы фиксации зависимостей.
Системные пакеты не требуются. Если в системе нет Google Chrome или Chromium,
один раз установи Playwright Chromium в локальный кэш:

```sh
uv run playwright install chromium
```

## Локальный запуск

Backend слушает только loopback-интерфейс:

```sh
uv run uvicorn vllm_cloudru.main:app --host 127.0.0.1 --port 8000
```

В другом терминале запусти Vite; `/api` проксируется в FastAPI:

```sh
pnpm --dir frontend dev
```

Открой адрес, который напечатает Vite. Для загрузки публичного каталога нужен
доступ к `recipes.vllm.ai`. Cloud.ru-запросы выполняются только после ввода
пользователем параметров сервисного аккаунта. Реальные ключи нельзя помещать в
команды, `.env`, исходники, документацию или отчёты.

## Проверки без тестовой автоматизации

Проверенные команды:

```sh
uv run ruff check src/vllm_cloudru
pnpm --dir frontend build
python3 scripts/check-repo.py
```

Ruff проверяет production Python-код. Frontend build выполняет TypeScript
компиляцию и production-сборку в `frontend/dist/`. Проверка репозитория не
запускает приложение и не обращается к сети.

## Спецификации поведения и test gate

Продуктовые сценарии в `tests/acceptance/features/` исполняются pytest-bdd.
Полноту привязки и выполнение проверяй из корня:

```sh
uv run pytest tests/acceptance --collect-only -q
uv run pytest tests/acceptance -q
```

Браузерные сценарии запускают Vite на случайном loopback-порту и используют
системный Google Chrome/Chromium, а при его отсутствии — Chromium, заранее установленный
командой Playwright выше.
Внешние HTTP-границы подменяются: тесты не обращаются к Cloud.ru или
recipes.vllm.ai и не требуют реальных секретов. Изменение сценариев, fixtures,
step definitions или иной тестовой инфраструктуры по-прежнему требует
отдельного явного согласования точного объёма.
