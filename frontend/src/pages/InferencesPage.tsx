import {
  Check,
  CircleDot,
  CloudCog,
  Copy,
  ExternalLink,
  Pencil,
  RefreshCw,
  Trash2,
} from "lucide-react";
import { useCallback, useEffect, useRef, useState } from "react";
import { Link } from "react-router-dom";

import { ApiError, getInferences } from "../api";
import { Notice } from "../components/Notice";
import type { ModelRunSummary } from "../types";

function statusLabel(status: string): string {
  return status.replace(/^(?:MODEL_RUN|MODELRUN)_STATUS_/, "").replaceAll("_", " ");
}

function resourceLabel(resource: string): string {
  return resource
    .replace("ResourceType_GPU_", "")
    .replace("_NVLINK", "")
    .replaceAll("_", " ");
}

function statusTone(status: string): "good" | "warning" | "danger" | "neutral" {
  const normalized = statusLabel(status).toUpperCase();
  if (["ERROR", "FAILED", "DELETED"].includes(normalized)) return "danger";
  if (["PENDING", "CREATING", "STARTING", "STOPPING"].includes(normalized)) return "warning";
  if (["RUNNING", "READY", "ACTIVE"].includes(normalized)) return "good";
  return "neutral";
}

export function InferencesPage() {
  const [runs, setRuns] = useState<ModelRunSummary[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [consoleAction, setConsoleAction] = useState<string | null>(null);
  const [copied, setCopied] = useState<string | null>(null);
  const [showDeleted, setShowDeleted] = useState(false);
  const requestId = useRef(0);
  const copyTimer = useRef<number | null>(null);
  const mounted = useRef(true);

  const load = useCallback(async () => {
    const currentRequest = ++requestId.current;
    setLoading(true);
    setError(null);
    try {
      const result = await getInferences();
      if (requestId.current === currentRequest) setRuns(result);
    } catch (reason) {
      if (requestId.current === currentRequest) {
        setError(reason instanceof ApiError ? reason.message : "Не удалось загрузить Model Run");
      }
    } finally {
      if (requestId.current === currentRequest) setLoading(false);
    }
  }, []);

  useEffect(() => {
    mounted.current = true;
    void load();
    return () => {
      mounted.current = false;
      requestId.current += 1;
      if (copyTimer.current !== null) window.clearTimeout(copyTimer.current);
    };
  }, [load]);

  async function copyEndpoint(run: ModelRunSummary) {
    if (!run.endpoint) return;
    try {
      await navigator.clipboard.writeText(run.endpoint);
      if (!mounted.current) return;
      if (copyTimer.current !== null) window.clearTimeout(copyTimer.current);
      setCopied(run.modelRunId);
      copyTimer.current = window.setTimeout(() => {
        if (mounted.current) setCopied(null);
      }, 1600);
    } catch {
      if (mounted.current) {
        setConsoleAction("Браузер не разрешил скопировать endpoint автоматически.");
      }
    }
  }

  const visibleRuns = runs.filter((run) => {
    const status = run.status.trim().toUpperCase();
    return showDeleted || (status !== "DELETED" && !status.endsWith("_DELETED"));
  });

  return (
    <section className="inferences-page">
      <header className="page-header">
        <div>
          <p className="kicker">Cloud.ru / Model Runs</p>
          <h1>Инференсы под рукой.</h1>
          <p>Текущие ресурсы проекта — без опасных кнопок управления жизненным циклом.</p>
        </div>
        <button className="secondary-button" disabled={loading} onClick={() => void load()} type="button">
          <RefreshCw aria-hidden="true" className={loading ? "spin" : ""} size={18} />
          Обновить
        </button>
      </header>

      {consoleAction ? (
        <Notice live title="Управление — в Cloud.ru Console" tone="info">
          <p>{consoleAction}</p>
          <a href="https://console.cloud.ru" rel="noreferrer" target="_blank">
            Открыть Cloud.ru Console <ExternalLink aria-hidden="true" size={15} />
          </a>
        </Notice>
      ) : null}
      <span className="visually-hidden" role="status">
        {copied ? "Endpoint скопирован" : ""}
      </span>

      {!error && runs.length ? (
        <div className="inference-toolbar">
          <span>{visibleRuns.length} из {runs.length}</span>
          <label className="deleted-toggle">
            <input
              checked={showDeleted}
              onChange={(event) => setShowDeleted(event.target.checked)}
              type="checkbox"
            />
            <span>Показывать удалённые</span>
          </label>
        </div>
      ) : null}

      {error ? (
        <div className="catalog-state" role="alert">
          <h2>Не удалось получить список</h2><p>{error}</p>
          <button className="secondary-button" onClick={() => void load()} type="button">Попробовать снова</button>
        </div>
      ) : loading && !runs.length ? (
        <div className="catalog-state" aria-live="polite"><span className="loader" />Запрашиваем Model Run…</div>
      ) : !runs.length ? (
        <div className="inference-empty">
          <img alt="" src="/illustrations/model-catalog.png" />
          <div>
            <p className="kicker">Пока пусто</p>
            <h2>Первый запуск начинается с рецепта.</h2>
            <p>Выберите модель, GPU и scaling — затем новый Model Run появится здесь.</p>
            <Link className="primary-button" to="/catalog"><span>Открыть каталог</span><CloudCog aria-hidden="true" size={19} /></Link>
          </div>
        </div>
      ) : !visibleRuns.length ? (
        <div className="inference-filtered-empty">
          <p className="kicker">Фильтр включён</p>
          <h2>Остались только удалённые инференсы.</h2>
          <p>Включите «Показывать удалённые», чтобы увидеть их в списке.</p>
        </div>
      ) : (
        <div className="inference-list">
          {visibleRuns.map((run) => (
            <article className="inference-row" key={run.modelRunId}>
              <div className="inference-row__identity">
                <span className="run-icon"><CircleDot aria-hidden="true" size={21} /></span>
                <div><h2>{run.name}</h2><p>{run.model || "Модель не указана"}</p></div>
              </div>
              <dl className="run-metrics">
                <div><dt>Статус</dt><dd><span className={`status-dot status-dot--${statusTone(run.status)}`} />{statusLabel(run.status)}</dd></div>
                <div><dt>GPU</dt><dd>{resourceLabel(run.resourceType) || "—"} × {run.gpuCount ?? "—"}</dd></div>
                <div><dt>Scaling</dt><dd>{run.scaling.minScale ?? "—"} → {run.scaling.maxScale ?? "—"}</dd></div>
              </dl>
              <div className="endpoint-box">
                <span>Endpoint</span>
                <code>{run.endpoint || "Ещё не доступен"}</code>
                <button
                  aria-label={copied === run.modelRunId ? `Endpoint ${run.name} скопирован` : `Скопировать endpoint ${run.name}`}
                  disabled={!run.endpoint}
                  onClick={() => void copyEndpoint(run)}
                  title={run.endpoint ? "Скопировать endpoint" : "Endpoint ещё не доступен"}
                  type="button"
                >
                  {copied === run.modelRunId ? <Check aria-hidden="true" size={17} /> : <Copy aria-hidden="true" size={17} />}
                </button>
              </div>
              <div className="safe-actions">
                <button
                  aria-label={`Редактировать ${run.name} в Cloud.ru Console`}
                  onClick={() => setConsoleAction(`Редактирование «${run.name}» выполняется только в Cloud.ru Console.`)}
                  title="Редактировать в Cloud.ru Console"
                  type="button"
                >
                  <Pencil aria-hidden="true" size={17} />
                </button>
                <button
                  aria-label={`Удалить ${run.name} в Cloud.ru Console`}
                  onClick={() => setConsoleAction(`Удаление «${run.name}» выполняется только в Cloud.ru Console.`)}
                  title="Удалить в Cloud.ru Console"
                  type="button"
                >
                  <Trash2 aria-hidden="true" size={17} />
                </button>
              </div>
            </article>
          ))}
        </div>
      )}
    </section>
  );
}
