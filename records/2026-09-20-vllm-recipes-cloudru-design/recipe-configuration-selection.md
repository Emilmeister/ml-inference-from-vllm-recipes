# Выбор конфигурации vLLM Recipes при запуске — дизайн

**Дата:** 2026-09-20  
**Статус:** согласовано пользователем и реализовано 2026-09-20  
**Основание:** пользователь хочет выбирать конфигурацию рецепта при создании
Cloud.ru Model Run так же, как в интерактивном builder на `recipes.vllm.ai`.

## Цель и границы

На странице запуска пользователь выбирает опубликованную конфигурацию рецепта,
а не получает единственный заранее выбранный rendering. Интерфейс показывает оси
`Variant`, `Strategy`, `KV Offload`, `Nodes`, `Frontend`, `Features` и, когда он
существует, `Spec Method`.

Приложение по-прежнему создаёт ровно один Cloud.ru Model Run. Поэтому доступны
только конфигурации, выражаемые одним `vllm serve` rendering без companion-
процессов. Multi-node, Prefill/Decode Disaggregation, Mooncake, LMCache с
отдельным сервером и другие многопроцессные варианты остаются видимыми, но имеют
неактивный цвет, недоступны для выбора и объясняют причину ограничения.

Не входят в изменение:

- перенос полного upstream command synthesizer;
- произвольный редактор vLLM-аргументов;
- создание нескольких Model Run для одной конфигурации;
- автоматический запуск вспомогательных процессов;
- изменение выбранного пользователем типа или количества Cloud.ru GPU.

## Источник конфигураций

Backend использует только опубликованные данные vLLM Recipes:

1. модельный JSON из `/models.json`;
2. `variants[*].json` для отдельного checkpoint, если ссылка опубликована;
3. hardware-specific ссылку из `recommended_command.by_hardware`, если она есть;
4. рекомендованный rendering и ссылки из `recommended_command.alternatives`;
5. декларации `features`, `opt_in_features` и feature modes модельного JSON;
6. опубликованный `taxonomy.json` только для однопроцессных KV Offload options.

URL для variant, hardware и strategy не синтезируются из имени модели. Если
upstream не опубликовал нужную ссылку или вернул неизвестную структуру, вариант
виден как недоступный либо вся конфигурация получает безопасную ошибку данных.

## Разрешение конфигурации

Frontend отправляет только ключи выбора, например `fp8`, `single_node_tp`,
`simple`, `rust`, `tool_calling` и `spec_decoding:mtp`. Он никогда не отправляет
готовый `argv`, environment или Hugging Face model ID как доверенные значения.

Backend заново разрешает ключи по текущему рецепту и возвращает:

- группы и элементы выбора с `key`, `label`, `description`, `available` и
  необязательной причиной недоступности;
- фактически выбранные ключи с безопасными defaults;
- `rendering_path` и итоговый model ID выбранного variant;
- итоговые vLLM arguments и environment;
- рекомендации по оборудованию и предупреждения совместимости;
- общий флаг `launchable` и причину запрета запуска.

Порядок сборки итогового rendering:

1. выбрать опубликованный variant и его checkpoint;
2. выбрать рекомендованный либо опубликованный alternative strategy;
3. отклонить rendering, если он требует нескольких узлов, нескольких команд или
   companion-процесса;
4. удалить из базового rendering аргументы всех объявленных recipe features и
   добавить аргументы только выбранных features/modes;
5. применить однопроцессный KV Offload option, если он опубликован и совместим;
6. для Rust frontend установить `VLLM_USE_RUST_FRONTEND=1`, для Python удалить
   эту переменную;
7. удалить platform-managed bind/port и `tensor-parallel-size` перед Cloud payload.

Порядок deterministic: повторный запрос с одинаковыми ключами даёт одинаковый
model ID, arguments, environment и предупреждения при неизменном upstream JSON.

## Правило GPU и tensor parallel

Тип и количество GPU остаются отдельными полями Cloud.ru и всегда задаются
пользователем. `gpuCount` передаётся в корне Model Run payload.

По согласованному правилу проекта `tensor-parallel-size` не передаётся в
`servingOptions.dynamicalOptions.args`: этим значением управляет Cloud.ru и оно
должно соответствовать `gpuCount`. Официальный CRUD-пример Cloud.ru подтверждает,
что `gpuCount` является отдельным полем от dynamic vLLM arguments, но открытая
документация прямо не подтверждает автоматическую установку TP. Поэтому правило
считается продуктовым предположением до первого разрешённого реального запуска.

Несовпадение выбранных GPU с VRAM, hardware profile или рекомендацией рецепта
остаётся предупреждением с явным подтверждением. Оно не делает однопроцессную
конфигурацию недоступной. Структурно многопроцессная конфигурация блокируется и
не может быть подтверждена в обход ограничения.

## HTTP-контракт

Существующий detail endpoint получает необязательные query-параметры выбора:

```text
GET /api/recipes/{organization}/{model}
  ?resource_type=ResourceType_GPU_H100
  &variant=fp8
  &strategy=single_node_tp
  &kv_offload=off
  &frontend=rust
  &feature=tool_calling
  &feature=reasoning
  &feature_mode=spec_decoding:mtp
```

Ответ расширяется объектом `configuration`, который содержит группы выбора и
разрешённый rendering. Отсутствующие параметры получают upstream defaults.
Неизвестный или недоступный ключ возвращает `422 invalid_recipe_configuration`,
а не молча заменяется другим вариантом.

Create request передаёт те же ключи в объекте `recipe_configuration`. Backend
не доверяет ранее показанному detail response и повторно разрешает конфигурацию
перед созданием. Это исключает подстановку браузером произвольных arguments,
environment или checkpoint.

## Интерфейс

Новый блок «Конфигурация рецепта» располагается в мастере запуска между моделью
и GPU. Каждая ось оформлена как строка компактных pill-кнопок в существующей
тёплой редакционной визуальной системе.

- выбранная кнопка имеет синюю рамку и светлый фон;
- доступная невыбранная кнопка имеет нейтральную рамку;
- недоступная кнопка серая, не нажимается и показывает краткую причину;
- зависимые строки появляются только когда имеют смысл, например `Spec Method`
  после включения speculative decoding;
- при смене родительского выбора зависимый недоступный выбор сбрасывается на
  upstream default;
- выбор синхронизируется с URL, поэтому refresh/back/forward сохраняют его;
- preview показывает variant model ID, rendering path, arguments, environment и
  platform-managed `gpuCount`, но не credentials и не значение HF token;
- любое изменение конфигурации сбрасывает подтверждение старых предупреждений.

Клавиатурная навигация использует обычные radio/checkbox semantics. Недоступный
вариант нельзя выбрать клавиатурой или программной отправкой формы; причина
доступна не только через цвет.

## Ошибки и fail-closed поведение

- Ошибка загрузки дополнительного published JSON не оставляет старый preview и
  не разрешает create.
- Если выбранный ключ исчез после обновления upstream, detail/create возвращают
  безопасную ошибку вместо fallback на другую конфигурацию.
- Недоступная конфигурация не попадает в create payload даже при ручной подмене
  URL или JSON-запроса.
- Существующая защита одного create POST и ручной проверки неопределённого
  результата распространяется на все ключи конфигурации без ослабления.
- Кэш содержит только публичные recipe/taxonomy данные и не содержит credentials.

## Продуктовые сценарии

После просмотра дизайна DRAFT Gherkin дополняется следующими обязательствами;
definitions шагов и тестовый код остаются запрещены без отдельного согласования:

- `DEP-005`: выбранный доступный variant/strategy определяет checkpoint и
  аргументы создаваемого Model Run;
- `DEP-006`: multi-node и companion-конфигурации видимы, неактивны и не могут
  вызвать create;
- `DEP-007`: Cloud payload содержит пользовательский `gpuCount`, но не содержит
  `tensor-parallel-size` в dynamic arguments;
- `DEP-008`: недоступный или исчезнувший selection key отклоняется без создания;
- `DEP-009`: изменение конфигурации сбрасывает подтверждение предупреждений.

## Критерии готовности

Изменение готово, когда пользователь может выбрать каждую доступную для одного
Model Run ось рецепта, видит недоступные multi-process варианты серыми, получает
обновлённый preview и создаёт один Model Run с повторно разрешённой backend
конфигурацией. Production build, Ruff и repository integrity должны завершаться
с кодом 0; автоматические тесты не создаются и не запускаются без разрешения.

## Первичные источники

- <https://github.com/vllm-project/recipes/blob/main/CONTRIBUTING.md>
- <https://github.com/vllm-project/recipes/blob/main/CLAUDE.md>
- <https://github.com/vllm-project/recipes/blob/main/taxonomy.yaml>
- <https://github.com/vllm-project/vllm/blob/main/tools/recipes/README.md>
- <https://github.com/cloud-ru/evo-aifactory-skills/blob/main/cloudru-ml-inference/scripts/commands/crud.py>
