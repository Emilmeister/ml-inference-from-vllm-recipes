import { RefreshCw, Search, SlidersHorizontal } from "lucide-react";
import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { Link, useSearchParams } from "react-router-dom";

import { ApiError, getRecipes } from "../api";
import { RecipeCard, recipePrecisions } from "../components/RecipeCard";
import type { RecipeCatalogItem } from "../types";

const GPU_OPTIONS = [
  ["ResourceType_GPU_H100", "H100 80 GB"],
  ["ResourceType_GPU_A100_NVLINK", "A100 80 GB"],
  ["ResourceType_GPU_V100", "V100 32 GB"],
] as const;

export function CatalogPage() {
  const [searchParams, setSearchParams] = useSearchParams();
  const [recipes, setRecipes] = useState<RecipeCatalogItem[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const requestId = useRef(0);
  const requestedResource = searchParams.get("gpu");
  const resource = GPU_OPTIONS.some(([value]) => value === requestedResource)
    ? requestedResource as (typeof GPU_OPTIONS)[number][0]
    : GPU_OPTIONS[0][0];

  const load = useCallback(async (refresh = false) => {
    const currentRequest = ++requestId.current;
    setLoading(true);
    setError(null);
    if (!refresh) setRecipes([]);
    try {
      const result = await getRecipes(resource, refresh);
      if (requestId.current === currentRequest) setRecipes(result);
    } catch (reason) {
      if (requestId.current === currentRequest) {
        setError(reason instanceof ApiError ? reason.message : "Не удалось загрузить каталог");
      }
    } finally {
      if (requestId.current === currentRequest) setLoading(false);
    }
  }, [resource]);

  useEffect(() => {
    void load();
    return () => {
      requestId.current += 1;
    };
  }, [load]);

  const tasks = useMemo(
    () => [...new Set(recipes.flatMap((recipe) => recipe.tasks))].sort(),
    [recipes],
  );
  const precisions = useMemo(
    () => [...new Set(recipes.flatMap(recipePrecisions))].sort(),
    [recipes],
  );
  const providers = useMemo(
    () => [...new Set(recipes.map((recipe) => recipe.provider).filter(Boolean))].sort(),
    [recipes],
  );
  const sizes = useMemo(
    () => [...new Set(recipes.map((recipe) => recipe.parameter_count).filter(
      (value): value is string => Boolean(value),
    ))].sort((left, right) => left.localeCompare(right, undefined, { numeric: true })),
    [recipes],
  );

  const query = (searchParams.get("q") || "").toLocaleLowerCase("ru");
  const requestedTask = searchParams.get("task") || "";
  const task = tasks.includes(requestedTask) ? requestedTask : "";
  const requestedProvider = searchParams.get("provider") || "";
  const provider = providers.includes(requestedProvider) ? requestedProvider : "";
  const requestedSize = searchParams.get("size") || "";
  const size = sizes.includes(requestedSize) ? requestedSize : "";
  const requestedPrecision = searchParams.get("precision") || "";
  const precision = precisions.includes(requestedPrecision) ? requestedPrecision : "";
  const requestedStatus = searchParams.get("status") || "";
  const status = ["ready", "warning", "view"].includes(requestedStatus)
    ? requestedStatus
    : "";
  const visible = recipes.filter((recipe) => {
    const matchesQuery = !query || `${recipe.title} ${recipe.provider} ${recipe.hf_id}`
      .toLocaleLowerCase("ru")
      .includes(query);
    const matchesTask = !task || recipe.tasks.includes(task);
    const matchesProvider = !provider || recipe.provider === provider;
    const matchesSize = !size || recipe.parameter_count === size;
    const matchesPrecision = !precision || recipePrecisions(recipe).includes(precision);
    const recipeStatus = !recipe.launchable
      ? "view"
      : recipe.compatibility_warnings.length
        ? "warning"
        : "ready";
    const matchesStatus = !status || status === recipeStatus;
    return matchesQuery
      && matchesTask
      && matchesProvider
      && matchesSize
      && matchesPrecision
      && matchesStatus;
  });

  function setFilter(name: string, value: string) {
    const next = new URLSearchParams(searchParams);
    if (value) next.set(name, value);
    else next.delete(name);
    setSearchParams(next, { replace: true });
  }

  return (
    <section className="catalog-page">
      <header className="page-header catalog-header">
        <div>
          <p className="kicker">vLLM Recipes / {recipes.length || "—"} моделей</p>
          <h1>Каталог, который знает железо.</h1>
          <p>Публичные рецепты проверены на возможность запуска одним Cloud.ru Model Run.</p>
        </div>
        <button className="secondary-button" disabled={loading} onClick={() => void load(true)} type="button">
          <RefreshCw aria-hidden="true" className={loading ? "spin" : ""} size={18} />
          Обновить каталог
        </button>
      </header>

      <div className="catalog-toolbar" aria-label="Фильтры каталога">
        <label className="search-field">
          <Search aria-hidden="true" size={20} />
          <span className="visually-hidden">Поиск</span>
          <input
            onChange={(event) => setFilter("q", event.target.value)}
            placeholder="Модель или провайдер"
            type="search"
            value={searchParams.get("q") || ""}
          />
        </label>
        <div className="filter-mark" aria-hidden="true"><SlidersHorizontal size={18} /></div>
        <label className="compact-field">
          <span>GPU-профиль</span>
          <select onChange={(event) => setFilter("gpu", event.target.value)} value={resource}>
            {GPU_OPTIONS.map(([value, label]) => <option key={value} value={value}>{label}</option>)}
          </select>
        </label>
        <label className="compact-field">
          <span>Провайдер</span>
          <select onChange={(event) => setFilter("provider", event.target.value)} value={provider}>
            <option value="">Все</option>
            {providers.map((value) => <option key={value} value={value}>{value}</option>)}
          </select>
        </label>
        <label className="compact-field">
          <span>Размер</span>
          <select onChange={(event) => setFilter("size", event.target.value)} value={size}>
            <option value="">Любой</option>
            {sizes.map((value) => <option key={value} value={value}>{value}</option>)}
          </select>
        </label>
        <label className="compact-field">
          <span>Задача</span>
          <select onChange={(event) => setFilter("task", event.target.value)} value={task}>
            <option value="">Все</option>
            {tasks.map((value) => <option key={value} value={value}>{value}</option>)}
          </select>
        </label>
        <label className="compact-field">
          <span>Точность</span>
          <select onChange={(event) => setFilter("precision", event.target.value)} value={precision}>
            <option value="">Любая</option>
            {precisions.map((value) => <option key={value} value={value}>{value}</option>)}
          </select>
        </label>
        <label className="compact-field">
          <span>Статус</span>
          <select onChange={(event) => setFilter("status", event.target.value)} value={status}>
            <option value="">Все</option>
            <option value="ready">Готовы к запуску</option>
            <option value="warning">С предупреждением</option>
            <option value="view">Только просмотр</option>
          </select>
        </label>
      </div>

      {error ? (
        <div className="catalog-state" role="alert">
          <h2>Каталог не загрузился</h2><p>{error}</p>
          <button className="secondary-button" onClick={() => void load()} type="button">Попробовать снова</button>
        </div>
      ) : loading && !recipes.length ? (
        <div className="catalog-state" aria-live="polite"><span className="loader" />Сканируем опубликованные рецепты…</div>
      ) : visible.length ? (
        <>
          <p className="result-count" aria-live="polite">Показано {visible.length} из {recipes.length}</p>
          <div className="recipe-grid">
            {visible.map((recipe) => (
              <RecipeCard key={recipe.hf_id} recipe={recipe} resourceType={resource} />
            ))}
          </div>
        </>
      ) : (
        <div className="catalog-empty">
          <img alt="" src="/illustrations/model-catalog.png" />
          <div><h2>Ничего не совпало</h2><p>Сбросьте часть фильтров или вернитесь ко всему каталогу.</p>
            <Link className="secondary-button" to="/catalog">Сбросить фильтры</Link>
          </div>
        </div>
      )}
    </section>
  );
}
