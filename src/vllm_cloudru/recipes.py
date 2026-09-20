import asyncio
from time import monotonic
from typing import Any
from urllib.parse import unquote, urlsplit

import httpx
from pydantic import TypeAdapter, ValidationError

from vllm_cloudru.models import (
    RecipeCatalogItem,
    RecipeConfigurationSelection,
    RecipeDetail,
    RecipeSummary,
)
from vllm_cloudru.recipe_configuration import (
    RecipeConfigurationError,
    compose_configuration,
    strategy_option,
    unavailable_configuration,
    unavailable_strategy_reason,
    variant_option,
)
from vllm_cloudru.translator import UnsupportedRecipe, translate_rendering

RECIPES_BASE_URL = "https://recipes.vllm.ai"
CLOUD_TO_RECIPE_HARDWARE = {"ResourceType_GPU_H100": "h100"}
SUPPORTED_RESOURCE_TYPES = frozenset(
    {
        "ResourceType_GPU_V100",
        "ResourceType_GPU_A100_NVLINK",
        "ResourceType_GPU_H100",
    }
)

_SUMMARY_LIST = TypeAdapter(list[RecipeSummary])
_NO_PROFILE_WARNING = (
    "Для выбранного типа оборудования нет соответствующего профиля vLLM Recipes; "
    "используется рекомендованная команда рецепта."
)
_CATALOG_SCAN_COOLDOWN_SECONDS = 30


class RecipeNotFound(LookupError):
    pass


class RecipeDataError(RuntimeError):
    pass


class RecipeResourceTypeError(ValueError):
    pass


class RecipeRefreshThrottled(RuntimeError):
    pass


def _published_path(value: Any, field: str) -> str:
    if not isinstance(value, str):
        raise RecipeDataError(f"{field} должен быть строковым относительным путём")

    parsed = urlsplit(value)
    if (
        not value.startswith("/")
        or parsed.scheme
        or parsed.netloc
        or parsed.query
        or parsed.fragment
        or "?" in value
        or "#" in value
        or ".." in unquote(parsed.path).split("/")
    ):
        raise RecipeDataError(
            f"{field} содержит небезопасный опубликованный путь: {value!r}"
        )
    return value


def _object(value: Any, field: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise RecipeDataError(f"{field} должен быть JSON-объектом")
    return value


class RecipeClient:
    def __init__(self, client: httpx.AsyncClient | None = None) -> None:
        self._owns_client = client is None
        self._client = client or httpx.AsyncClient(
            base_url=RECIPES_BASE_URL,
            follow_redirects=False,
            timeout=20,
        )
        self._models: tuple[RecipeSummary, ...] | None = None
        self._taxonomy: dict[str, Any] | None = None
        self._details: dict[tuple[str, str, str, bool], RecipeDetail] = {}
        self._catalogs: dict[str, tuple[RecipeCatalogItem, ...]] = {}
        self._generation = 0
        self._last_catalog_scan: dict[str, float] = {}
        self._refresh_lock = asyncio.Lock()
        self._catalog_lock = asyncio.Lock()

    async def aclose(self) -> None:
        if self._owns_client:
            await self._client.aclose()

    async def _get_json(self, path: str) -> Any:
        published_path = _published_path(path, "Путь рецепта")
        response = await self._client.get(
            f"{RECIPES_BASE_URL}{published_path}",
            follow_redirects=False,
        )
        response.raise_for_status()
        try:
            return response.json()
        except ValueError as error:
            raise RecipeDataError("vLLM Recipes вернул некорректный JSON") from error

    async def list_models(self, refresh: bool = False) -> list[RecipeSummary]:
        if self._models is not None and not refresh:
            return list(self._models)

        async with self._refresh_lock:
            if self._models is not None and not refresh:
                return list(self._models)
            try:
                payload = await self._get_json("/models.json")
                if not isinstance(payload, list):
                    raise RecipeDataError(
                        "Индекс vLLM Recipes должен быть JSON-массивом"
                    )
                models = _SUMMARY_LIST.validate_python(payload)
                for model in models:
                    _published_path(model.url, f"url модели {model.hf_id!r}")
                    _published_path(model.json_path, f"json модели {model.hf_id!r}")
            except (httpx.HTTPError, ValidationError, RecipeDataError):
                if refresh and self._models is not None:
                    return list(self._models)
                raise

            self._models = tuple(models)
            self._generation += 1
            self._details.clear()
            self._catalogs.clear()
            return list(self._models)

    async def list_catalog(
        self,
        refresh: bool = False,
        resource_type: str = "ResourceType_GPU_H100",
    ) -> list[RecipeCatalogItem]:
        if resource_type not in SUPPORTED_RESOURCE_TYPES:
            raise RecipeResourceTypeError("Неизвестный тип GPU-ресурса")

        observed_generation = self._generation
        cached = self._catalogs.get(resource_type)
        if cached is not None and not refresh:
            return [item.model_copy(deep=True) for item in cached]

        async with self._catalog_lock:
            cached = self._catalogs.get(resource_type)
            if refresh and self._generation != observed_generation:
                refresh = False
            if cached is not None and not refresh:
                return [item.model_copy(deep=True) for item in cached]

            now = monotonic()
            last_scan = self._last_catalog_scan.get(resource_type)
            if (
                last_scan is not None
                and now - last_scan < _CATALOG_SCAN_COOLDOWN_SECONDS
            ):
                if cached is not None:
                    return [item.model_copy(deep=True) for item in cached]
                raise RecipeRefreshThrottled(
                    "Повторное сканирование каталога временно ограничено"
                )
            self._last_catalog_scan[resource_type] = now

            generation_before_refresh = self._generation
            models = await self.list_models(refresh=refresh)
            if (
                refresh
                and self._generation == generation_before_refresh
                and cached is not None
            ):
                return [item.model_copy(deep=True) for item in cached]
            scan_generation = self._generation
            semaphore = asyncio.Semaphore(8)

            async def inspect(summary: RecipeSummary) -> RecipeCatalogItem:
                async with semaphore:
                    detail = await self.get_detail(
                        summary.hf_id,
                        resource_type,
                        include_configuration=False,
                    )
                return RecipeCatalogItem(
                    **summary.model_dump(by_alias=False),
                    description=detail.description,
                    tasks=detail.tasks,
                    parameter_count=detail.parameter_count,
                    variants=detail.variants,
                    recommended_hardware=detail.recommended_hardware,
                    recommended_gpu_count=detail.recommended_gpu_count,
                    compatibility_warnings=detail.compatibility_warnings,
                    launchable=detail.launchable,
                    launch_reason=detail.launch_reason,
                )

            catalog = tuple(await asyncio.gather(*(inspect(model) for model in models)))
            if self._generation == scan_generation:
                self._catalogs[resource_type] = catalog
            return [item.model_copy(deep=True) for item in catalog]

    async def _find_summary(self, hf_id: str) -> RecipeSummary:
        for model in await self.list_models():
            if model.hf_id == hf_id:
                return model
        raise RecipeNotFound(
            f"Модель {hf_id!r} не найдена в опубликованном индексе vLLM Recipes"
        )

    async def _fetch_model(self, summary: RecipeSummary) -> dict[str, Any]:
        recipe = _object(await self._get_json(summary.json_path), "Рецепт модели")
        published_hf_id = recipe.get("hf_id")
        if not isinstance(published_hf_id, str) or published_hf_id != summary.hf_id:
            raise RecipeDataError("hf_id рецепта не совпадает с опубликованным индексом")

        raw_recommended = recipe.get("recommended_command")
        if raw_recommended is None:
            return recipe
        recommended = _object(raw_recommended, "recommended_command")
        by_hardware = recommended.get("by_hardware", {})
        if not isinstance(by_hardware, dict):
            raise RecipeDataError(
                "recommended_command.by_hardware должен быть JSON-объектом"
            )
        for hardware, path in by_hardware.items():
            if not isinstance(hardware, str):
                raise RecipeDataError(
                    "Ключи recommended_command.by_hardware должны быть строками"
                )
            _published_path(path, f"recommended_command.by_hardware.{hardware}")
        return recipe

    async def get_model(self, hf_id: str) -> dict[str, Any]:
        return await self._fetch_model(await self._find_summary(hf_id))

    async def _get_taxonomy(self) -> dict[str, Any]:
        if self._taxonomy is None:
            self._taxonomy = _object(await self._get_json("/taxonomy.json"), "taxonomy")
        return self._taxonomy

    async def _linked_variant_recipe(
        self,
        root_recipe: dict[str, Any],
        variant: dict[str, Any],
    ) -> tuple[dict[str, Any], str]:
        path = variant.get("json")
        if path is None:
            return root_recipe, ""
        published_path = _published_path(path, "variants.json")
        recipe = _object(await self._get_json(published_path), "Рецепт варианта")
        published_hf_id = recipe.get("hf_id")
        expected_hf_id = variant.get("model_id")
        if not isinstance(published_hf_id, str) or (
            isinstance(expected_hf_id, str) and published_hf_id != expected_hf_id
        ):
            raise RecipeDataError(
                "hf_id рецепта варианта не совпадает с опубликованным variant.model_id"
            )
        return recipe, published_path

    async def _preferred_rendering(
        self,
        recipe: dict[str, Any],
        fallback_path: str,
        resource_type: str,
    ) -> tuple[dict[str, Any], str, list[str]]:
        recommended = _object(recipe.get("recommended_command"), "recommended_command")
        by_hardware = recommended.get("by_hardware", {})
        if not isinstance(by_hardware, dict):
            raise RecipeDataError(
                "recommended_command.by_hardware должен быть JSON-объектом"
            )
        hardware = CLOUD_TO_RECIPE_HARDWARE.get(resource_type)
        path = by_hardware.get(hardware) if hardware else None
        if path is None:
            return recommended, fallback_path, [_NO_PROFILE_WARNING]
        published_path = _published_path(path, f"by_hardware.{hardware}")
        rendering = _object(
            await self._get_json(published_path),
            f"Профиль оборудования {hardware}",
        )
        return rendering, published_path, []

    async def _strategy_choices(
        self,
        base_rendering: dict[str, Any],
        base_path: str,
        requested: str | None,
    ) -> tuple[
        dict[str, Any] | None,
        str | None,
        str,
        list[Any],
    ]:
        raw_key = base_rendering.get("strategy")
        current_key = raw_key if isinstance(raw_key, str) else "recommended"
        candidates: dict[str, tuple[dict[str, Any] | None, str, bool, str | None]] = {}

        try:
            translate_rendering(base_rendering)
            current_available = True
            current_reason = None
        except UnsupportedRecipe as error:
            current_available = False
            current_reason = str(error)
        candidates[current_key] = (
            base_rendering,
            base_path,
            current_available,
            current_reason,
        )

        alternatives = base_rendering.get("alternatives", {})
        if not isinstance(alternatives, dict):
            raise RecipeDataError("alternatives должен быть JSON-объектом")
        for key, raw_path in alternatives.items():
            if not isinstance(key, str):
                raise RecipeDataError("Ключ alternatives должен быть строкой")
            path = _published_path(raw_path, f"alternatives.{key}")
            if key.startswith("kv_store_"):
                continue
            if not key.startswith("single_node_"):
                candidates[key] = (None, path, False, unavailable_strategy_reason(key))
                continue
            rendering = _object(
                await self._get_json(path),
                f"Стратегия {key}",
            )
            try:
                translate_rendering(rendering)
                candidates[key] = (rendering, path, True, None)
            except UnsupportedRecipe as error:
                candidates[key] = (rendering, path, False, str(error))

        selected_key = requested
        if selected_key is not None:
            candidate = candidates.get(selected_key)
            if candidate is None:
                raise RecipeConfigurationError(f"Неизвестная стратегия: {selected_key}")
            if not candidate[2]:
                raise RecipeConfigurationError(
                    candidate[3] or "Выбранная стратегия недоступна"
                )
        else:
            selected_key = next(
                (key for key, candidate in candidates.items() if candidate[2]),
                None,
            )

        options = [
            strategy_option(
                key,
                candidate[0],
                selected=key == selected_key,
                available=candidate[2],
                reason=candidate[3],
            )
            for key, candidate in candidates.items()
        ]
        if selected_key is None:
            return None, None, base_path, options
        selected = candidates[selected_key]
        return selected[0], selected_key, selected[1], options

    async def get_detail(
        self,
        hf_id: str,
        resource_type: str,
        selection: RecipeConfigurationSelection | None = None,
        *,
        include_configuration: bool = True,
    ) -> RecipeDetail:
        if resource_type not in SUPPORTED_RESOURCE_TYPES:
            raise RecipeResourceTypeError("Неизвестный тип GPU-ресурса")

        requested = selection or RecipeConfigurationSelection()
        cache_key = (
            hf_id,
            resource_type,
            requested.model_dump_json(exclude_none=True),
            include_configuration,
        )
        cached = self._details.get(cache_key)
        if cached is not None:
            return cached.model_copy(deep=True)

        summary = await self._find_summary(hf_id)
        detail_generation = self._generation
        recipe = await self._fetch_model(summary)
        meta = _object(recipe.get("meta", {}), "meta")
        model = _object(recipe.get("model", {}), "model")
        variants = _object(recipe.get("variants", {}), "variants")
        raw_recommended = recipe.get("recommended_command")
        if raw_recommended is None:
            detail = RecipeDetail(
                hf_id=summary.hf_id,
                title=summary.title,
                provider=summary.provider,
                description=meta.get("description"),
                tasks=meta.get("tasks", []),
                parameter_count=model.get("parameter_count"),
                active_parameters=model.get("active_parameters"),
                context_length=model.get("context_length"),
                min_vllm_version=model.get("min_vllm_version"),
                variants=variants,
                recommended_hardware=None,
                recommended_gpu_count=None,
                selected_variant=None,
                rendering_path=summary.json_path,
                rendering={},
                translated=None,
                launchable=False,
                launch_reason="Для этого рецепта не опубликована команда vLLM serve",
            )
            if self._generation == detail_generation:
                self._details[cache_key] = detail
            return detail.model_copy(deep=True)

        recommended = _object(raw_recommended, "recommended_command")
        # AICODE-NOTE: Follow only paths published by the recipe index/document;
        # hardware, variant and strategy URLs must never be synthesized from an HF ID.
        configuration = None
        if include_configuration:
            published_variants = variants or {"default": {}}
            states: dict[str, dict[str, Any]] = {}
            for variant_key, raw_variant in published_variants.items():
                if not isinstance(variant_key, str) or not isinstance(raw_variant, dict):
                    raise RecipeDataError("variants должен содержать JSON-объекты")
                variant_recipe, linked_path = await self._linked_variant_recipe(
                    recipe, raw_variant
                )
                variant_recommended = variant_recipe.get("recommended_command")
                if variant_recommended is None:
                    states[variant_key] = {
                        "recipe": variant_recipe,
                        "metadata": raw_variant,
                        "linked": bool(linked_path),
                        "rendering": None,
                        "reason": "Для варианта не опубликована команда vLLM serve",
                        "strategy_options": [],
                    }
                    continue
                fallback_path = linked_path or summary.json_path
                base, base_path, variant_warnings = await self._preferred_rendering(
                    variant_recipe,
                    fallback_path,
                    resource_type,
                )
                selected_rendering, strategy_key, strategy_path, strategy_options = (
                    await self._strategy_choices(base, base_path, None)
                )
                states[variant_key] = {
                    "recipe": variant_recipe,
                    "metadata": raw_variant,
                    "linked": bool(linked_path),
                    "base": base,
                    "base_path": base_path,
                    "rendering": selected_rendering,
                    "strategy_key": strategy_key,
                    "strategy_path": strategy_path,
                    "strategy_options": strategy_options,
                    "warnings": variant_warnings,
                    "reason": (
                        None
                        if selected_rendering is not None
                        else "Вариант требует нескольких процессов или узлов"
                    ),
                }

            requested_variant = requested.variant
            if requested_variant is not None and requested_variant not in states:
                raise RecipeConfigurationError(
                    f"Неизвестный вариант: {requested_variant}"
                )
            recommended_variant = recommended.get("variant")
            source_variant_key = (
                recommended_variant
                if isinstance(recommended_variant, str)
                and recommended_variant in published_variants
                else next(iter(published_variants))
            )
            variant_key = requested_variant
            if variant_key is None:
                if (
                    isinstance(recommended_variant, str)
                    and recommended_variant in states
                    and states[recommended_variant].get("rendering") is not None
                ):
                    variant_key = recommended_variant
                else:
                    variant_key = next(
                        (
                            key
                            for key, state in states.items()
                            if state.get("rendering") is not None
                        ),
                        next(iter(states)),
                    )
            state = states[variant_key]
            if requested_variant is not None and state.get("rendering") is None:
                raise RecipeConfigurationError(
                    state.get("reason") or "Выбранный вариант недоступен"
                )

            variant_options = [
                variant_option(
                    key,
                    candidate["metadata"],
                    selected=key == variant_key,
                    available=candidate.get("rendering") is not None,
                    reason=candidate.get("reason"),
                )
                for key, candidate in states.items()
            ]
            rendering = state.get("rendering")
            if rendering is not None:
                if requested.strategy is not None:
                    rendering, strategy_key, rendering_path, strategy_options = (
                        await self._strategy_choices(
                            state["base"],
                            state["base_path"],
                            requested.strategy,
                        )
                    )
                    if rendering is None or strategy_key is None:
                        raise RecipeConfigurationError("Выбранная стратегия недоступна")
                else:
                    strategy_key = state["strategy_key"]
                    rendering_path = state["strategy_path"]
                    strategy_options = state["strategy_options"]
                rendering, configuration = compose_configuration(
                    rendering=rendering,
                    recipe=state["recipe"],
                    variant_key=variant_key,
                    variant_metadata=state["metadata"],
                    strategy_key=strategy_key,
                    requested=requested,
                    variant_options=variant_options,
                    strategy_options=strategy_options,
                    taxonomy=await self._get_taxonomy(),
                    apply_variant_overrides=(
                        not state["linked"] and variant_key != source_variant_key
                    ),
                    source_variant_metadata=published_variants[source_variant_key],
                )
                warnings = state["warnings"]
            else:
                rendering = state.get("base") or recommended
                rendering_path = state.get("base_path") or summary.json_path
                warnings = state.get("warnings") or [_NO_PROFILE_WARNING]
                configuration = unavailable_configuration(
                    rendering=rendering,
                    recipe=state["recipe"],
                    variant_key=variant_key,
                    variant_options=variant_options,
                    strategy_options=state.get("strategy_options", []),
                    taxonomy=await self._get_taxonomy(),
                    reason=state.get("reason")
                    or "Конфигурация не представима одним Model Run",
                )
        else:
            rendering, rendering_path, warnings = await self._preferred_rendering(
                recipe,
                summary.json_path,
                resource_type,
            )

        hardware_profile = rendering.get("hardware_profile")
        if not isinstance(hardware_profile, dict):
            hardware_profile = {}
        recommended_hardware = hardware_profile.get("display_name")
        if not isinstance(recommended_hardware, str):
            recommended_hardware = None
        recommended_gpu_count = hardware_profile.get("gpu_count")
        if not isinstance(recommended_gpu_count, int) or isinstance(
            recommended_gpu_count, bool
        ):
            recommended_gpu_count = None

        try:
            translated = translate_rendering(rendering)
            launchable = True
            launch_reason = None
        except UnsupportedRecipe as error:
            translated = None
            launchable = False
            launch_reason = str(error)

        detail = RecipeDetail(
            hf_id=summary.hf_id,
            title=summary.title,
            provider=summary.provider,
            description=meta.get("description"),
            tasks=meta.get("tasks", []),
            parameter_count=model.get("parameter_count"),
            active_parameters=model.get("active_parameters"),
            context_length=model.get("context_length"),
            min_vllm_version=model.get("min_vllm_version"),
            variants=variants,
            recommended_hardware=recommended_hardware,
            recommended_gpu_count=recommended_gpu_count,
            selected_variant=(
                configuration.selection.variant
                if configuration is not None
                else rendering.get("variant")
            ),
            rendering_path=rendering_path,
            rendering=rendering,
            translated=translated,
            configuration=configuration,
            launchable=launchable,
            launch_reason=launch_reason,
            compatibility_warnings=warnings,
        )
        if self._generation == detail_generation:
            self._details[cache_key] = detail
        return detail.model_copy(deep=True)
