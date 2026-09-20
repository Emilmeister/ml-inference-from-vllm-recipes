# vLLM Recipes → Cloud.ru ML Inference

Локальное веб-приложение для выбора конфигураций из официального каталога
vLLM Recipes, создания Model Run в Cloud.ru ML Inference и просмотра уже
запущенных Model Run проекта.

## Возможности

- загружает все опубликованные рецепты с `recipes.vllm.ai` и показывает их
  пригодность для автоматического запуска;
- переводит поддерживаемый single-process `vllm serve` в параметры Model Run;
- позволяет выбрать опубликованные variant, strategy, frontend, features и
  Simple KV offload; multi-node и companion-опции показывает disabled с причиной;
- оставляет пользователю выбор типа и количества GPU, runtime template и
  границ масштабирования;
- разрешает несовпадение с рекомендациями рецепта только после явного
  подтверждения предупреждений;
- отправляет единственный create-запрос и при неопределённом результате требует
  ручной проверки списка перед новой попыткой;
- показывает существующие Model Run без API для изменения или удаления.

## Локальный запуск

Нужны Python 3.13+, uv, Node.js и pnpm. Из корня проекта установите зависимости:

```sh
uv sync
pnpm --dir frontend install --frozen-lockfile
```

Для разработки запустите backend:

```sh
uv run uvicorn vllm_cloudru.main:app --host 127.0.0.1 --port 8000
```

В другом терминале запустите frontend:

```sh
pnpm --dir frontend dev
```

Vite напечатает локальный адрес и проксирует `/api` в FastAPI. Для собранного
варианта сначала выполните `pnpm --dir frontend build`, затем откройте
`http://127.0.0.1:8000` после запуска backend.

## Учётные данные

При первом открытии пользователь вводит `PROJECT_ID`, Key ID, Key Secret и
необязательный UUID секрета Hugging Face в Cloud.ru. Они хранятся в
`localStorage` браузера в открытом виде и считаются действительными семь суток.
При следующем открытии или чтении просроченная запись удаляется и больше не
используется. Не используйте приложение на общем компьютере. Сам Hugging Face
token приложению не передаётся.

## Проверки

Без создания или запуска тестовой автоматизации:

```sh
uv run ruff check src/vllm_cloudru
pnpm --dir frontend build
python3 scripts/check-repo.py --strict
```

Продуктовые контракты находятся в `tests/acceptance/features/` в статусе
`DRAFT`. Тесты, fixtures, BDD step definitions и тестовая инфраструктура могут
меняться только после отдельного согласования конкретного объёма.

Подробности: [архитектура](docs/architecture.md),
[среда разработки](docs/development.md), [BDD-правила](docs/bdd.md).

## Источники интеграций

- [vLLM Recipes](https://github.com/vllm-project/recipes)
- [Cloud.ru ML Inference](https://cloud.ru/docs/ml-inference/ug/index)
- [официальный Cloud.ru CRUD-клиент](https://github.com/cloud-ru/evo-aifactory-skills/tree/main/cloudru-ml-inference)

Приложение не выполняет update/delete/suspend/resume. Для этих операций кнопки
интерфейса направляют пользователя в Cloud.ru Console.
