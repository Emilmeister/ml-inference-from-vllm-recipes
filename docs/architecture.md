# Архитектура приложения

Приложение однопользовательское и локальное. React/Vite отвечает за интерфейс,
FastAPI — за получение рецептов, обмен сервисных ключей на IAM-токен и изоляцию
Cloud.ru Console BFF. Собственной базы данных нет.

```text
Browser (localStorage, React)
        │ POST /api/cloud/*, GET /api/recipes/*
        ▼
FastAPI
  ├── RecipesClient ──► recipes.vllm.ai
  ├── CloudruClient ──► iam.api.cloud.ru
  └── CloudruClient ──► console.cloud.ru/u-api/inference/model-run/v1
```

## Границы компонентов

- `recipes.py` следует только опубликованным JSON-ссылкам `/models.json`,
  модельного рецепта, variant, hardware-профиля и strategy alternative. Ссылки
  не синтезируются из Hugging Face ID. Backend принимает от браузера только
  ключи выбора и заново разрешает их перед созданием Model Run.
- `recipe_configuration.py` безопасно компонует single-process frontend,
  features, feature modes и Simple KV offload. Multi-node, disaggregation и
  companion-процессы возвращаются интерфейсу как видимые disabled-опции. Detail
  preview может по явному `reconcile_configuration=true` сохранить совместимые
  feature-настройки после смены GPU и вернуть точечные корректировки; create-path
  этот режим не использует.
- `translator.py` принимает один `vllm serve` процесс. Multi-node,
  disaggregated и другие многопроцессные схемы остаются доступны для просмотра,
  но не переводятся в один Model Run. `host`, `port` и
  `tensor-parallel-size` не попадают в dynamic arguments: эти параметры
  определяет ML Inference, а tensor parallelism следует выбранному `gpuCount`.
- `deployment.py` объединяет рецепт с выбранными пользователем GPU, количеством,
  runtime template и scaling. Несовпадение с рекомендацией требует явного
  подтверждения, но не заменяет пользовательский выбор. Рецептное значение
  `gpu-memory-utilization` заменяется одним числовым аргументом из формы в
  диапазоне `0.85–0.99`. После подтверждённого отказа Cloud.ru пользователь
  может явно повторить создание без точно совпавших с `invalid_keys` аргументов;
  похожие и остальные опубликованные параметры backend разрешает и сохраняет
  заново.
- `cloudru.py` предоставляет только validate/list/options/create. Методов
  update/delete/suspend/resume в приложении нет. Список runtime templates
  запрашивается с обязательными фильтрами `FrameworkType_VLLM` и `isActive=true`,
  как это делает интерфейс Cloud.ru Console. Ответ options также объединяет
  квоты, включённые сервисные аккаунты и активные IAM-ready log groups; наружу
  возвращаются только поля, необходимые форме запуска.

## Учётные данные и доверенная граница

`PROJECT_ID`, Key ID, Key Secret и необязательный Hugging Face Secret UUID
хранятся только в `localStorage` браузера вместе с `expiresAt` на семь суток.
После срока запись не используется и удаляется при следующем открытии/чтении;
раньше её можно очистить вручную. Это осознанный
компромисс локального MVP: данные не шифруются и доступны JavaScript текущего
origin, поэтому общий компьютер использовать нельзя.

Cloud-параметры передаются FastAPI только в JSON-теле запроса. Backend не
сохраняет их, не возвращает IAM-токен и не включает внешние response body в
публичные ошибки. Hugging Face token остаётся в Cloud.ru Secret Management;
приложение знает только UUID секрета.

Каждый non-2xx ответ Cloud.ru диагностируется одной серверной log-записью:
методом, URL, статусом, request ID и ограниченными по длине JSON-телами запроса
и ответа. Заголовки не логируются, а поля `secret`, `token`, `password`,
`authorization`, `api_key` и их значения заменяются на `<redacted>`.

## Создание и неопределённый результат

Create POST не повторяется автоматически. Backend не пытается сопоставить
ресурс по имени, потому что BFF не предоставляет надёжный idempotency key.
Frontend сохраняет pending-маркер в `sessionStorage`: при потере ответа или
перезагрузке следующий запуск блокируется до ручного обновления списка Model Run
и явного разрешения новой попытки пользователем.

Ошибка `framework_registry.arguments_validation_error` со статусом `400`
означает, что ресурс не создан. Приложение показывает отклонённые ключи и риск
несовместимого запуска; только кнопка пользователя отправляет новый Create POST
без этих ключей. Если и этот повтор отклонён, новая кнопка повтора не появляется.
Такой явный повтор не применяется к сетевым ошибкам и `5xx`.

Production HTML-оболочка отдаётся с `Cache-Control: no-cache`: браузер
перепроверяет `index.html`, а неизменяемые assets продолжают адресоваться по
хэшированным именам после каждой frontend-сборки.

## Выбор конфигурации рецепта

Страница запуска хранит в URL только ключи `variant`, `strategy`, `kv_offload`,
`frontend`, `feature` и `feature_mode`. Ответ detail API содержит канонический
selection, группы доступных и заблокированных опций, выбранный rendering и
переведённый preview. При смене GPU frontend очищает аппаратно-зависимые
`variant`, `strategy` и `kv_offload`, но сохраняет feature-настройки. Detail API
согласует только недоступные элементы, возвращает причины в
`configuration_adjustments`, после чего frontend синхронизирует URL. При смене
другой родительской опции зависимые ключи очищаются, а ранее выданное
подтверждение предупреждений сбрасывается.

Create-запрос передаёт канонические ключи, но не произвольные `args`, `env` или
`model_id`. Backend повторно получает опубликованные данные, проверяет
доступность комбинации и только после этого строит Cloud.ru payload.
Пользователь явно выбирает сервисный аккаунт и способ доступа к endpoint, а
активная log group `default` только предлагается как начальное значение. В
payload включаются выбранные `auth` и `logGroupId`, а UUID Hugging Face-секрета
передаётся через `secretLocator` без значения токена.

Если Rust frontend получает опубликованный аргумент `hf-overrides`, форма перед
созданием рекомендует Python. Переключение выполняется только по решению
пользователя; отказ сохраняет Rust и не блокирует запуск. JSON-значение аргумента
не переписывается.

## Источники контрактов

- [vLLM Recipes](https://github.com/vllm-project/recipes)
- [Cloud.ru ML Inference](https://cloud.ru/docs/ml-inference/ug/index)
- [официальный Cloud.ru BFF-клиент](https://github.com/cloud-ru/evo-aifactory-skills/blob/main/cloudru-ml-inference/scripts/cloudru_client.py)
- [официальные CRUD-команды Cloud.ru](https://github.com/cloud-ru/evo-aifactory-skills/blob/main/cloudru-ml-inference/scripts/commands/crud.py)
