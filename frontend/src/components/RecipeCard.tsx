import { ArrowUpRight, Cpu, Eye, Rocket } from "lucide-react";
import { Link } from "react-router-dom";

import type { RecipeCatalogItem } from "../types";

export function recipePrecisions(recipe: RecipeCatalogItem): string[] {
  return [
    ...new Set(
      Object.values(recipe.variants)
        .map((variant) => variant.precision)
        .filter((value): value is string => Boolean(value)),
    ),
  ];
}

export function RecipeCard({
  recipe,
  resourceType,
}: {
  recipe: RecipeCatalogItem;
  resourceType: string;
}) {
  const [organization, model] = recipe.hf_id.split("/");
  const precisions = recipePrecisions(recipe);
  const status = !recipe.launchable
    ? "view"
    : recipe.compatibility_warnings.length
      ? "warning"
      : "ready";

  return (
    <article className="recipe-card">
      <div className="recipe-card__topline">
        <span>{recipe.provider}</span>
        <span className={`recipe-status recipe-status--${status}`}>
          {recipe.launchable ? <Rocket aria-hidden="true" size={14} /> : <Eye aria-hidden="true" size={14} />}
          {status === "ready" ? "Готов к запуску" : status === "warning" ? "С предупреждением" : "Только просмотр"}
        </span>
      </div>

      <div className="recipe-card__body">
        <h2>{recipe.title}</h2>
        <p className="recipe-id">{recipe.hf_id}</p>
        <p className="recipe-description">
          {recipe.description || recipe.launch_reason || "Описание будет доступно на странице рецепта."}
        </p>
      </div>

      <dl className="recipe-facts">
        <div>
          <dt>Параметры</dt>
          <dd>{recipe.parameter_count || "—"}</dd>
        </div>
        <div>
          <dt>Профиль</dt>
          <dd>
            <Cpu aria-hidden="true" size={15} />
            {recipe.recommended_hardware || "по умолчанию"}
            {recipe.recommended_gpu_count ? ` × ${recipe.recommended_gpu_count}` : ""}
          </dd>
        </div>
      </dl>

      <div className="chip-row" aria-label="Характеристики рецепта">
        {recipe.tasks.slice(0, 3).map((task) => (
          <span className="chip chip--task" key={task}>{task}</span>
        ))}
        {precisions.slice(0, 2).map((precision) => (
          <span className="chip" key={precision}>{precision}</span>
        ))}
      </div>

      {!recipe.launchable && recipe.launch_reason ? (
        <p className="recipe-card__reason">{recipe.launch_reason}</p>
      ) : recipe.compatibility_warnings.length ? (
        <p className="recipe-card__reason">{recipe.compatibility_warnings[0]}</p>
      ) : null}

      <Link
        className="card-link"
        to={`/catalog/${encodeURIComponent(organization)}/${encodeURIComponent(model)}?resource=${encodeURIComponent(resourceType)}`}
      >
        Открыть рецепт
        <ArrowUpRight aria-hidden="true" size={18} />
      </Link>
    </article>
  );
}
