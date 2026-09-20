# BDD acceptance automation implementation plan

> **For agentic workers:** execute inline in the existing workspace. Do not commit, call
> real Cloud.ru endpoints, or use production credentials.

**Goal:** Make all 27 product scenarios in `tests/acceptance/features/` executable with
pytest-bdd and verify them against the real FastAPI and React public boundaries.

**Architecture:** Browser-facing behavior runs in Chromium against the real Vite-served
React application while Playwright intercepts `/api` at the external HTTP boundary.
Payload, configuration re-resolution, uncertain creation, and secret redaction run through
the real FastAPI application with in-memory recipe/Cloud providers or `httpx.MockTransport`.

**Tech Stack:** Python 3.13, pytest 9, pytest-bdd 8, pytest-playwright, FastAPI TestClient,
React 19, Vite 8.

---

### Task 1: Shared acceptance infrastructure

**Files:**
- Create: `tests/acceptance/conftest.py`
- Create: `tests/acceptance/support.py`

- [x] Start one Vite server on an unused loopback port for the acceptance session and stop
  it reliably after pytest.
- [x] Provide isolated browser helpers for valid local credentials and JSON route responses.
- [x] Provide minimal recipe, Cloud options, Model Run, deployment request, fake recipe,
  and fake Cloud builders shared by scenarios.
- [x] Run `uv run pytest tests/acceptance --collect-only -q`; expected initial status is
  zero collected scenarios until binding modules are added.

### Task 2: Catalog and credential scenarios

**Files:**
- Create: `tests/acceptance/test_catalog.py`
- Create: `tests/acceptance/test_credentials.py`

- [x] Bind `catalog.feature` and implement `CAT-001` through `CAT-004` through browser-visible
  cards, launch availability, warnings, filters, and URL state.
- [x] Bind `credentials.feature` and implement `CFG-001` through `CFG-003` through actual
  routing and localStorage expiry/removal.
- [x] Run `uv run pytest tests/acceptance/test_catalog.py tests/acceptance/test_credentials.py -q`.

### Task 3: Deployment API scenarios

**Files:**
- Create: `tests/acceptance/test_deployment.py`

- [x] Bind `deployment.feature` and implement API-backed steps for `DEP-001` through
  `DEP-005`, `DEP-007`, `DEP-008`, and `SEC-001`.
- [x] Assert Cloud create call counts and payload fields, recipe selection re-resolution,
  no automatic lookup after uncertain creation, and absence of secrets in response/logs.
- [x] Run only these scenario names with `uv run pytest tests/acceptance/test_deployment.py -q -k`.

### Task 4: Deployment browser scenarios

**Files:**
- Modify: `tests/acceptance/test_deployment.py`

- [x] Implement Playwright steps for `DEP-006`, `DEP-009` through `DEP-015` using the real
  React deployment form and captured HTTP requests.
- [x] Verify disabled recipe options, confirmation invalidation, explicit account/auth/log
  choices, closable error dialog, non-restored success state, and stable configurator DOM.
- [x] Run `uv run pytest tests/acceptance/test_deployment.py -q`.

### Task 5: Inference-list scenarios

**Files:**
- Create: `tests/acceptance/test_inferences.py`

- [x] Bind `inferences.feature` and implement `INF-001`, `INF-002`, `SAFE-001`, and
  `SAFE-002` through rendered Model Runs and captured requests.
- [x] Verify deleted filtering is client-side and edit/delete remain accessible icon-only
  links to console guidance without mutation requests.
- [x] Run `uv run pytest tests/acceptance/test_inferences.py -q`.

### Task 6: Promote the suite and verify

**Files:**
- Modify: `tests/acceptance/features/deployment.feature`
- Modify: `docs/development.md`

- [x] Remove the obsolete DRAFT marker only after every scenario is collected and passes.
- [x] Document exact collection and execution commands.
- [x] Run `uv run pytest tests/acceptance --collect-only -q` and confirm 27 scenarios.
- [x] Run `uv run pytest tests/acceptance -q` with zero failures and zero skips.
- [x] Run `uv run pytest -q`, `uv run ruff check src/vllm_cloudru tests`,
  `pnpm --dir frontend build`, `python3 scripts/check-repo.py`, and `git diff --check`.
