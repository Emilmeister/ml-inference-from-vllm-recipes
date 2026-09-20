import { ArrowRight, CloudCog, ShieldCheck } from "lucide-react";
import { type FormEvent, useId, useState } from "react";
import { useNavigate } from "react-router-dom";

import { ApiError, validateCredentials } from "../api";
import { Notice } from "../components/Notice";
import { saveCredentials } from "../credentials";
import type { CredentialInput } from "../types";

const EMPTY_CREDENTIALS: CredentialInput = {
  projectId: "",
  keyId: "",
  keySecret: "",
  hfSecretId: "",
};

export function SetupPage({ onSaved }: { onSaved?: () => void }) {
  const navigate = useNavigate();
  const formId = useId();
  const [credentials, setCredentials] = useState(EMPTY_CREDENTIALS);
  const [error, setError] = useState<string | null>(null);
  const [submitting, setSubmitting] = useState(false);

  function update(field: keyof CredentialInput, value: string) {
    setCredentials((current) => ({ ...current, [field]: value }));
  }

  async function submit(event: FormEvent<HTMLFormElement>) {
    event.preventDefault();
    setError(null);
    setSubmitting(true);
    const candidate: CredentialInput = {
      projectId: credentials.projectId.trim(),
      keyId: credentials.keyId.trim(),
      keySecret: credentials.keySecret,
      ...(credentials.hfSecretId?.trim()
        ? { hfSecretId: credentials.hfSecretId.trim() }
        : {}),
    };

    try {
      await validateCredentials(candidate);
      try {
        saveCredentials(candidate);
      } catch {
        setError(
          "Cloud.ru доступен, но браузер не разрешил сохранить настройки в localStorage.",
        );
        return;
      }
      onSaved?.();
      navigate("/inferences", { replace: true });
    } catch (reason) {
      setError(
        reason instanceof ApiError
          ? reason.message
          : "Не удалось проверить подключение. Попробуйте ещё раз.",
      );
    } finally {
      setSubmitting(false);
    }
  }

  return (
    <main className="setup-page">
      <section className="setup-story" aria-labelledby={`${formId}-story-title`}>
        <div className="setup-story__eyebrow">
          <CloudCog aria-hidden="true" size={18} />
          Первый запуск
        </div>
        <div className="setup-story__copy">
          <p className="kicker">Один мост. Ноль ручных payload.</p>
          <h1 id={`${formId}-story-title`}>Рецепт готов — подключим облако.</h1>
          <p>
            Каталог vLLM подскажет конфигурацию, а вы сохраните контроль над проектом,
            масштабированием и секретами.
          </p>
        </div>
        <img
          className="setup-illustration"
          src="/illustrations/cloud-credentials.png"
          alt=""
        />
        <div className="setup-story__caption">
          <span>01</span>
          Проверка доступа перед сохранением
        </div>
      </section>

      <section className="setup-panel" aria-labelledby={`${formId}-title`}>
        <div className="setup-panel__inner">
          <div className="setup-heading">
            <span className="section-number">Настройка / 01</span>
            <h2 id={`${formId}-title`}>Доступ к Cloud.ru</h2>
            <p>Проверим сервисный аккаунт и только после успеха сохраним данные локально.</p>
          </div>

          <Notice title="Хранение в этом браузере" tone="warning">
            Данные, включая Key Secret, сохраняются в localStorage без шифрования на 7
            дней. Не используйте общий или чужой компьютер.
          </Notice>

          <form aria-describedby={`${formId}-storage-note`} autoComplete="off" onSubmit={submit}>
            <div className="field-grid">
              <label className="field field--wide">
                <span>PROJECT_ID</span>
                <input
                  autoComplete="off"
                  name="projectId"
                  onChange={(event) => update("projectId", event.target.value)}
                  placeholder="00000000-0000-0000-0000-000000000000"
                  required
                  value={credentials.projectId}
                />
              </label>

              <label className="field">
                <span>Key ID</span>
                <input
                  autoComplete="off"
                  name="keyId"
                  onChange={(event) => update("keyId", event.target.value)}
                  required
                  value={credentials.keyId}
                />
              </label>

              <label className="field">
                <span>Key Secret</span>
                <input
                  autoComplete="off"
                  name="keySecret"
                  onChange={(event) => update("keySecret", event.target.value)}
                  required
                  type="password"
                  value={credentials.keySecret}
                />
              </label>

              <label className="field field--wide">
                <span>Hugging Face Secret UUID (необязательно)</span>
                <input
                  autoComplete="off"
                  name="hfSecretId"
                  onChange={(event) => update("hfSecretId", event.target.value)}
                  value={credentials.hfSecretId ?? ""}
                />
              </label>
            </div>

            <p className="privacy-line" id={`${formId}-storage-note`}>
              <ShieldCheck aria-hidden="true" size={17} />
              Значения передаются только в теле POST-запросов Cloud-операций и не
              попадают в URL.
            </p>

            {error ? (
              <Notice id={`${formId}-error`} tone="error">
                {error}
              </Notice>
            ) : null}

            <button className="primary-button" disabled={submitting} type="submit">
              <span>{submitting ? "Проверяем…" : "Проверить подключение"}</span>
              <ArrowRight aria-hidden="true" size={20} />
            </button>
          </form>
        </div>
      </section>
    </main>
  );
}
