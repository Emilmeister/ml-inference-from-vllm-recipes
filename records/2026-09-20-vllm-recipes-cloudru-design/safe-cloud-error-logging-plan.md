# Safe Cloud.ru HTTP Error Logging Implementation Plan

> **For agentic workers:** execute inline; do not add or run tests without separate user approval.

**Goal:** Log every non-2xx Cloud.ru HTTP response with a useful sanitized request/response snapshot.

**Architecture:** Keep logging at the shared Cloud.ru adapter boundary in `cloudru.py`. A small recursive sanitizer redacts sensitive JSON fields and known secret values before one structured error record is emitted; `_raise_for_status` covers ordinary failures and the uncertain-create branch logs before returning its special error.

**Tech Stack:** Python standard library logging/JSON, httpx, Ruff.

---

### Task 1: Sanitized boundary logging

**Files:**
- Modify: `src/vllm_cloudru/cloudru.py`

- [ ] Add a module logger and helpers that parse JSON bodies, recursively replace fields named `secret`, `token`, `password`, `authorization` or `api_key`, replace any discovered secret value inside response strings, and truncate serialized bodies.
- [ ] Emit one `cloudru_http_error` record per response containing method, URL without sensitive query values, status, safe request ID, sanitized request body and sanitized response body, including retry attempts without duplicate logging of the same response.
- [ ] Call the helper for every final non-2xx response, including the create branch that raises `CloudruCreateUncertain` before `_raise_for_status`.

### Task 2: Verification

**Files:**
- No additional files.

- [ ] Run `uv run ruff check src/vllm_cloudru` and expect exit status 0.
- [ ] Run `python3 scripts/check-repo.py --strict` and expect exit status 0.
- [ ] Run `git diff --check` and expect exit status 0.
- [ ] Do not run or modify tests under the current project test ban.
