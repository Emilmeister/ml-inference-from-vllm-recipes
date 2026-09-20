import { ArrowLeft, Check, Cpu, Eye, Rocket, TriangleAlert } from "lucide-react";
import { useEffect, useState } from "react";
import { Link, useParams, useSearchParams } from "react-router-dom";

import { ApiError, getRecipeDetail } from "../api";
import type { RecipeDetail, VllmArgument } from "../types";

const GPU_OPTIONS = [
  ["ResourceType_GPU_H100", "NVIDIA H100 80 GB"],
  ["ResourceType_GPU_A100_NVLINK", "NVIDIA A100 80 GB"],
  ["ResourceType_GPU_V100", "NVIDIA V100 32 GB"],
] as const;

function Argument({ argument }: { argument: VllmArgument }) {
  return (
    <div className="argument-row">
      <code>{argument.parameterType === "PARAMETER_TYPE_ENV" ? argument.key : `--${argument.key}`}</code>
      <code>{argument.value === true ? "flag" : JSON.stringify(argument.value)}</code>
    </div>
  );
}

export function RecipePage() {
  const { organization = "", model = "" } = useParams();
  const [searchParams, setSearchParams] = useSearchParams();
  const hfId = `${organization}/${model}`;
  const requestedResource = searchParams.get("resource");
  const resource = GPU_OPTIONS.some(([value]) => value === requestedResource)
    ? requestedResource as (typeof GPU_OPTIONS)[number][0]
    : GPU_OPTIONS[0][0];
  const [recipe, setRecipe] = useState<RecipeDetail | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    let active = true;
    setLoading(true);
    setError(null);
    setRecipe(null);
    getRecipeDetail(hfId, resource)
      .then((result) => { if (active) setRecipe(result); })
      .catch((reason) => {
        if (active) setError(reason instanceof ApiError ? reason.message : "Не удалось открыть рецепт");
      })
      .finally(() => { if (active) setLoading(false); });
    return () => { active = false; };
  }, [hfId, resource]);

  if (loading && !recipe) return <div className="catalog-state"><span className="loader" />Открываем рецепт…</div>;
  if (error || !recipe) return <div className="catalog-state" role="alert"><h1>Рецепт недоступен</h1><p>{error}</p><Link to="/catalog">Вернуться в каталог</Link></div>;

  const argumentsList = [...(recipe.translated?.arguments || []), ...(recipe.translated?.environment || [])];
  const selectedVariant = recipe.selected_variant ? recipe.variants[recipe.selected_variant] : undefined;

  return (
    <article className="recipe-page">
      <Link className="back-link" to="/catalog"><ArrowLeft aria-hidden="true" size={18} />Каталог</Link>
      <header className="recipe-hero">
        <div>
          <p className="kicker">{recipe.provider} / {recipe.hf_id}</p>
          <h1>{recipe.title}</h1>
          <p>{recipe.description || "Описание не опубликовано."}</p>
        </div>
        <div className={`support-stamp support-stamp--${recipe.launchable ? "ready" : "view"}`}>
          {recipe.launchable ? <Rocket aria-hidden="true" /> : <Eye aria-hidden="true" />}
          <strong>{recipe.launchable ? "Можно запускать" : "Только просмотр"}</strong>
          <span>{recipe.launchable ? "Один vLLM-процесс" : recipe.launch_reason}</span>
        </div>
      </header>

      <section className="recipe-layout">
        <div className="recipe-main">
          <section className="detail-card">
            <div className="section-heading"><span>01</span><h2>Профиль рецепта</h2></div>
            <dl className="detail-grid">
              <div><dt>Параметры</dt><dd>{recipe.parameter_count || "—"}</dd></div>
              <div><dt>Активные</dt><dd>{recipe.active_parameters || "—"}</dd></div>
              <div><dt>Контекст</dt><dd>{recipe.context_length?.toLocaleString("ru-RU") || "—"}</dd></div>
              <div><dt>vLLM от</dt><dd>{recipe.min_vllm_version || "—"}</dd></div>
              <div><dt>Вариант</dt><dd>{recipe.selected_variant || "default"}</dd></div>
              <div><dt>Точность</dt><dd>{selectedVariant?.precision || "—"}</dd></div>
            </dl>
          </section>

          <section className="detail-card">
            <div className="section-heading"><span>02</span><h2>Аргументы vLLM</h2></div>
            {recipe.translated ? (
              <div className="code-panel">
                <div className="argument-row argument-row--model"><code>vllm serve</code><code>{recipe.translated.model_id}</code></div>
                {argumentsList.map((argument, index) => (
                  <Argument
                    argument={argument}
                    key={`${argument.parameterType}-${argument.key}-${index}`}
                  />
                ))}
              </div>
            ) : <p className="muted-copy">Команду нельзя безопасно выразить одним Model Run.</p>}
          </section>
        </div>

        <aside className="launch-panel">
          <p className="section-number">Профиль Cloud.ru</p>
          <label className="compact-field compact-field--large">
            <span>Тип GPU</span>
            <select
              onChange={(event) => {
                const value = event.target.value;
                const next = new URLSearchParams(searchParams);
                next.set("resource", value);
                setSearchParams(next, { replace: true });
              }}
              value={resource}
            >
              {GPU_OPTIONS.map(([value, label]) => <option key={value} value={value}>{label}</option>)}
            </select>
          </label>
          <div className="hardware-note"><Cpu aria-hidden="true" size={18} /><span>Рецепт: {recipe.recommended_hardware || "общий профиль"}{recipe.recommended_gpu_count ? ` × ${recipe.recommended_gpu_count}` : ""}</span></div>
          {recipe.compatibility_warnings.map((warning) => <div className="inline-warning" key={warning}><TriangleAlert aria-hidden="true" size={17} />{warning}</div>)}
          {recipe.launchable ? (
            <Link className="primary-button" to={`/deploy?model=${encodeURIComponent(recipe.hf_id)}&resource=${encodeURIComponent(resource)}`}>
              <span>Настроить запуск</span><Check aria-hidden="true" size={19} />
            </Link>
          ) : <div className="view-only-note"><Eye aria-hidden="true" /><strong>Запуск отключён</strong><span>{recipe.launch_reason}</span></div>}
        </aside>
      </section>
    </article>
  );
}
