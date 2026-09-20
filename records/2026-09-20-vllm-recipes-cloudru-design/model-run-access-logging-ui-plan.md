# Model Run Access, Logging, and List UI Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add explicit service-account/auth/log-group selection to Model Run creation, a creation-error modal, deleted-run filtering, and compact list actions.

**Architecture:** Extend the existing `/api/cloud/options` aggregation and the existing create request instead of adding parallel frontend loaders. Normalize Console BFF responses at the backend trust boundary, keep create-only behavior, and implement the UI inside the current deployment and inference pages with one focused modal component.

**Tech Stack:** Python 3.13, FastAPI, Pydantic, httpx, React 19, TypeScript, Vite, lucide-react, CSS.

---

## Constraints

- Do not create, modify, delete, or run tests, fixtures, step definitions, or test infrastructure without separate approval.
- Gherkin `.feature` files remain DRAFT product specifications and may be updated before implementation.
- Do not create commits or worktrees; the repository contains preserved user changes and commits require a separate instruction.
- Do not issue real Cloud.ru create requests or place credentials and resource UUIDs in source, documentation, commands, or reports.

### Task 1: Record the observable product behavior

**Files:**
- Modify: `tests/acceptance/features/deployment.feature`
- Modify: `tests/acceptance/features/inferences.feature`

- [x] **Step 1: Add DRAFT deployment scenarios**

Append scenarios that fix the public behavior without defining implementation details:

```gherkin
  Сценарий: DEP-010. Доступ к Model Run привязан к явно выбранному аккаунту
    Дано Cloud.ru вернул несколько активных сервисных аккаунтов
    Когда пользователь явно выбирает аккаунт и способ «API-ключ»
    Тогда payload содержит идентификатор выбранного аккаунта
    И тип аутентификации равен "AUTHENTICATION_TYPE_KEY_BASED"

  Сценарий: DEP-011. Ключи доступа используют token-based аутентификацию
    Дано пользователь явно выбрал сервисный аккаунт
    Когда он выбирает способ «Ключи доступа» и создаёт Model Run
    Тогда тип аутентификации равен "AUTHENTICATION_TYPE_TOKEN_BASED"

  Сценарий: DEP-012. Для логирования выбирается доступная log group
    Дано Cloud.ru вернул активную готовую группу "default" и другие группы
    Когда пользователь открывает форму создания
    Тогда группа "default" выбрана предварительно
    И payload включает выбранную log group и включает логирование

  Сценарий: DEP-013. Ошибка создания показана в закрываемом modal
    Дано пользователь отправил запрос создания Model Run
    Когда Cloud.ru отклоняет создание
    Тогда приложение показывает ошибку в модальном окне
    И пользователь может закрыть модальное окно
```

- [x] **Step 2: Add DRAFT inference-list scenarios**

```gherkin
  Сценарий: INF-002. Удалённые Model Run скрыты по умолчанию
    Дано Cloud.ru вернул активный и удалённый Model Run
    Когда пользователь открывает список инференсов
    Тогда активный Model Run виден
    А удалённый Model Run скрыт
    Когда пользователь включает «Показывать удалённые»
    Тогда удалённый Model Run становится виден без нового запроса

  Сценарий: SAFE-002. Действия списка компактны и доступны по имени
    Дано пользователь видит существующий Model Run
    Когда отображаются действия endpoint, редактирования и удаления
    Тогда действия показаны только иконками с доступными названиями
    И редактирование и удаление не отправляют mutation-запрос
```

### Task 2: Normalize service accounts and log groups in Cloud options

**Files:**
- Modify: `src/vllm_cloudru/models.py`
- Modify: `src/vllm_cloudru/cloudru.py`

- [x] **Step 1: Extend the response models**

Add focused browser-facing models and extend `CloudOptions`:

```python
class ServiceAccountOption(BaseModel):
    model_config = ConfigDict(extra="forbid")

    id: str
    name: str
    description: str


class LogGroupOption(BaseModel):
    model_config = ConfigDict(populate_by_name=True, serialize_by_alias=True, extra="forbid")

    id: str
    name: str
    description: str
    is_default: bool = Field(alias="isDefault")


class CloudOptions(BaseModel):
    model_config = ConfigDict(populate_by_name=True, serialize_by_alias=True, extra="forbid")

    runtime_templates: list[RuntimeTemplate] = Field(alias="runtimeTemplates")
    gpu_options: dict[str, GpuOption] = Field(alias="gpuOptions")
    quotas: list[QuotaItem]
    service_accounts: list[ServiceAccountOption] = Field(alias="serviceAccounts")
    log_groups: list[LogGroupOption] = Field(alias="logGroups")
```

- [x] **Step 2: Add BFF paths and strict normalizers**

In `cloudru.py`, add path constants and small functions that reject malformed top-level lists, skip disabled accounts, and skip log groups that are not active and IAM-ready:

```python
SERVICE_ACCOUNTS_PREFIX = "/u-api/bff-console/v1/projects"
LOG_GROUPS_PREFIX = "/u-api/logaas-bff-console/v1"


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
```

- [x] **Step 3: Fetch all option sources with the same IAM token**

Use `asyncio.gather` in `CloudruClient.options` so the extra lookups do not create a serial loading waterfall:

```python
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
```

Return normalized `serviceAccounts` and `logGroups` beside the existing options. Preserve the existing safe non-2xx logging for all four requests.

### Task 3: Build the confirmed Cloud.ru create payload

**Files:**
- Modify: `src/vllm_cloudru/models.py`
- Modify: `src/vllm_cloudru/deployment.py`

- [x] **Step 1: Extend and validate the local create contract**

Add required UUID fields and the two confirmed enum values:

```python
AuthenticationType = Literal[
    "AUTHENTICATION_TYPE_KEY_BASED",
    "AUTHENTICATION_TYPE_TOKEN_BASED",
]


class CreateInferenceRequest(BaseModel):
    # existing fields stay unchanged
    service_account_id: UUID
    authentication_type: AuthenticationType
    log_group_id: UUID
```

Import `UUID` from `uuid` and `Literal` from `typing`. Pydantic performs boundary validation; use `str(...)` when writing UUIDs into the outgoing JSON.

- [x] **Step 2: Correct model source and serving options**

Construct `model_source` before the return statement:

```python
model_source: dict[str, Any] = {
    "huggingFaceRepository": {
        "repo": repo,
        "model": model,
        "revision": "main",
    }
}
if request.credentials.hf_secret_id:
    model_source["secretLocator"] = {
        "secretId": str(request.credentials.hf_secret_id),
        "version": 1,
    }
```

Replace the current `modelSource`, `servingOptions`, and `options` fragments with the observed Console BFF contract while preserving recipe arguments and environment:

```python
"modelSource": model_source,
"options": {
    "isEnabledAuth": True,
    "isEnabledLogging": True,
    "auth": {
        "type": request.authentication_type,
        "serviceAccountId": str(request.service_account_id),
    },
    "logGroupId": str(request.log_group_id),
    "publishApiGateway": True,
},
"servingOptions": {
    "dynamicalOptions": {
        "args": [*translated.arguments, *translated.environment],
        "loraModules": [],
    },
    "cacheOptions": {},
},
```

Keep the existing top-level `gpuCount`; the browser and backend already preserve the selected value.

### Task 4: Expose access and logging choices in the deployment UI

**Files:**
- Modify: `frontend/src/types.ts`
- Modify: `frontend/src/api.ts`
- Create: `frontend/src/components/ErrorDialog.tsx`
- Modify: `frontend/src/pages/DeployPage.tsx`
- Modify: `frontend/src/styles.css`

- [x] **Step 1: Extend TypeScript contracts and response validation**

Add to `CloudOptions`:

```ts
serviceAccounts: Array<{ id: string; name: string; description: string }>;
logGroups: Array<{ id: string; name: string; description: string; isDefault: boolean }>;
```

Add to `CreateInferenceInput`:

```ts
service_account_id: string;
authentication_type: "AUTHENTICATION_TYPE_KEY_BASED" | "AUTHENTICATION_TYPE_TOKEN_BASED";
log_group_id: string;
```

In `getCloudOptions`, validate both arrays and every required primitive before casting the response.

- [x] **Step 2: Add a native accessible modal component**

Create `ErrorDialog.tsx` using the browser's native modal behavior:

```tsx
import { TriangleAlert, X } from "lucide-react";
import { useEffect, useId, useRef } from "react";

type ErrorDialogProps = {
  message: string | null;
  onClose: () => void;
};

export function ErrorDialog({ message, onClose }: ErrorDialogProps) {
  const dialogRef = useRef<HTMLDialogElement>(null);
  const titleId = useId();

  useEffect(() => {
    const dialog = dialogRef.current;
    if (!dialog) return;
    if (message && !dialog.open) dialog.showModal();
    if (!message && dialog.open) dialog.close();
  }, [message]);

  return (
    <dialog aria-labelledby={titleId} className="error-dialog" onCancel={onClose} ref={dialogRef}>
      <button aria-label="Закрыть сообщение об ошибке" className="error-dialog__close" onClick={onClose} title="Закрыть" type="button">
        <X aria-hidden="true" size={18} />
      </button>
      <TriangleAlert aria-hidden="true" className="error-dialog__icon" size={28} />
      <p className="kicker">Создание не выполнено</p>
      <h2 id={titleId}>Cloud.ru вернул ошибку</h2>
      <p>{message}</p>
      <button className="primary-button" onClick={onClose} type="button">Закрыть</button>
    </dialog>
  );
}
```

Style `::backdrop`, the cream panel, warning accent, close button, focus state, and mobile width consistently with the existing editorial/pastel interface.

- [x] **Step 3: Add deployment state and deterministic defaults**

In `DeployPage`, add:

```ts
const [serviceAccountId, setServiceAccountId] = useState("");
const [authenticationType, setAuthenticationType] = useState<
  "AUTHENTICATION_TYPE_KEY_BASED" | "AUTHENTICATION_TYPE_TOKEN_BASED"
>("AUTHENTICATION_TYPE_KEY_BASED");
const [logGroupId, setLogGroupId] = useState("");
const [createError, setCreateError] = useState<string | null>(null);
```

When options arrive, preserve a still-valid group selection; otherwise select the entry with `isDefault: true` or leave the field empty. Never automatically select a service account.

- [x] **Step 4: Add the “Доступ и логи” form section**

Render required selects/radios after scaling and before runtime confirmation. Include account descriptions in option labels only when non-empty. Disable submit when either required ID is empty, and add clear inline validation before the POST.

Pass the selected values to `createInference`:

```ts
service_account_id: serviceAccountId,
authentication_type: authenticationType,
log_group_id: logGroupId,
```

- [x] **Step 5: Route only post-submit failures into the modal**

Keep loading errors and client-side validation in the existing inline `Notice`. Keep `confirmation_required` in the warning stack. For `create_uncertain`, retain the current session marker/banner and also set `createError`; for other errors after the POST, clear the attempt marker and set `createError` to the safe `ApiError.message` or fallback text.

Render once near the page root:

```tsx
<ErrorDialog message={createError} onClose={() => setCreateError(null)} />
```

### Task 5: Filter deleted runs and compact list actions

**Files:**
- Modify: `frontend/src/pages/InferencesPage.tsx`
- Modify: `frontend/src/styles.css`

- [x] **Step 1: Add the local deleted-status filter**

```ts
const [showDeleted, setShowDeleted] = useState(false);
const visibleRuns = runs.filter((run) => {
  const status = run.status.trim().toUpperCase();
  return showDeleted || (status !== "DELETED" && !status.endsWith("_DELETED"));
});
```

Render a labelled checkbox switch whenever the request has completed. Use `visibleRuns` for cards, keep the switch available when every returned run is deleted, and show a compact filtered-empty message rather than the first-use empty illustration in that case.

- [x] **Step 2: Convert all three actions to icon-only controls**

Keep endpoint copy as its existing icon button. Remove visible text from edit/delete buttons and add both `aria-label` and `title`, for example:

```tsx
<button
  aria-label={`Редактировать ${run.name} в Cloud.ru Console`}
  onClick={() => setConsoleAction(`Редактирование «${run.name}» выполняется только в Cloud.ru Console.`)}
  title="Редактировать в Cloud.ru Console"
  type="button"
>
  <Pencil aria-hidden="true" size={17} />
</button>
```

Apply the same pattern to delete and add `title` to copy. Keep mutation behavior absent.

- [x] **Step 3: Tighten action and filter styling**

Make `.safe-actions button` square with a visible focus ring and preserve disabled states. Add the switch and filtered-empty styles without changing the established card layout.

### Task 6: Update current architecture knowledge and verify

**Files:**
- Modify: `docs/architecture.md`
- Review/update anchors in: `src/vllm_cloudru/cloudru.py`, `src/vllm_cloudru/deployment.py`

- [x] **Step 1: Update architecture documentation**

Record that Cloud options aggregate runtime templates, quotas, enabled service accounts, and active IAM-ready log groups; creation uses the selected account/auth type/log group; and errors shown to the browser remain sanitized.

- [x] **Step 2: Review AICODE anchors**

Keep the Console BFF isolation and secret-safe logging anchors accurate. Add no anchor that merely repeats field names.

- [x] **Step 3: Run only the approved production checks**

Run:

```sh
uv run ruff check src/vllm_cloudru
pnpm --dir frontend build
python3 scripts/check-repo.py
```

Expected: each command exits with status `0`. Do not run pytest, Vitest, Playwright, or any real Cloud.ru request.

- [x] **Step 4: Inspect the final scoped changes**

Use `git diff --check` and a path-scoped diff/status inspection. Preserve unrelated user changes, do not stage files, and report explicitly that automated tests and live Cloud.ru creation were not run.
