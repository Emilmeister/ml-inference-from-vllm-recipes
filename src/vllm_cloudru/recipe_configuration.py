from __future__ import annotations

from copy import deepcopy
from typing import Any

from vllm_cloudru.models import (
    RecipeConfigurationGroup,
    RecipeConfigurationOption,
    RecipeConfigurationSelection,
    ResolvedRecipeConfiguration,
)


class RecipeConfigurationError(ValueError):
    pass


_STRATEGY_LABELS = {
    "single_node_tp": "Tensor Parallel",
    "single_node_tep": "Tensor + Expert Parallel",
    "single_node_dep": "Data + Expert Parallel",
    "multi_node_tp": "Multi-node Tensor Parallel",
    "multi_node_tep": "Multi-node Tensor + Expert Parallel",
    "multi_node_dep": "Multi-node Data + Expert Parallel",
    "multi_node_tp_pp": "Multi-node TP + Pipeline",
    "multi_node_tp_dp": "Multi-node TP + Data Parallel",
    "pd_cluster": "Prefill/Decode Disaggregation",
}


def display_label(key: str) -> str:
    return _STRATEGY_LABELS.get(key, key.replace("_", " ").title())


def _option(
    key: str,
    label: str,
    *,
    selected: bool = False,
    description: str | None = None,
    available: bool = True,
    unavailable_reason: str | None = None,
) -> RecipeConfigurationOption:
    return RecipeConfigurationOption(
        key=key,
        label=label,
        selected=selected,
        description=description,
        available=available,
        unavailable_reason=unavailable_reason,
    )


def unavailable_strategy_reason(key: str) -> str:
    if key.startswith("multi_node_"):
        return "Cloud.ru ML Inference запускает один Model Run без multi-node orchestration"
    if key == "pd_cluster":
        return "Режим требует отдельных prefill и decode процессов"
    if key.startswith("kv_store_"):
        return "Режим требует внешнего KV-сервиса и дополнительных процессов"
    return "Опубликованная конфигурация не представима одним Model Run"


def variant_option(
    key: str,
    metadata: dict[str, Any],
    *,
    selected: bool,
    available: bool,
    reason: str | None = None,
) -> RecipeConfigurationOption:
    precision = metadata.get("precision")
    published_label = metadata.get("label")
    label = (
        published_label
        if isinstance(published_label, str)
        else str(precision).upper()
        if isinstance(precision, str)
        else display_label(key)
    )
    minimum = metadata.get("vram_minimum_gb")
    if isinstance(minimum, (int, float)) and not isinstance(minimum, bool):
        label = f"{label} · {minimum:g} GB"
    description = metadata.get("description")
    return _option(
        key,
        label,
        selected=selected,
        description=description if isinstance(description, str) else None,
        available=available,
        unavailable_reason=reason,
    )


def strategy_option(
    key: str,
    rendering: dict[str, Any] | None,
    *,
    selected: bool,
    available: bool,
    reason: str | None = None,
) -> RecipeConfigurationOption:
    strategy_spec = rendering.get("strategy_spec") if rendering else None
    label = None
    description = None
    if isinstance(strategy_spec, dict):
        raw_label = strategy_spec.get("display_name")
        raw_description = strategy_spec.get("description")
        label = raw_label if isinstance(raw_label, str) else None
        description = raw_description.strip() if isinstance(raw_description, str) else None
    return _option(
        key,
        label or display_label(key),
        selected=selected,
        description=description,
        available=available,
        unavailable_reason=reason,
    )


def _hardware_override(
    value: dict[str, Any], rendering: dict[str, Any]
) -> dict[str, Any]:
    overrides = value.get("hardware_overrides")
    if not isinstance(overrides, dict):
        return {}
    profile = rendering.get("hardware_profile", {})
    if not isinstance(profile, dict):
        profile = {}
    generation = profile.get("generation") or profile.get("gpu_generation")
    if isinstance(generation, list):
        generation = generation[0] if generation else None
    brand = profile.get("brand")
    hardware = rendering.get("hardware")
    keys = [
        hardware if isinstance(hardware, str) else None,
        generation if isinstance(generation, str) else None,
        brand.lower() if isinstance(brand, str) else None,
        "default",
    ]
    for key in keys:
        candidate = overrides.get(key) if key else None
        if isinstance(candidate, dict):
            return candidate
    return {}


def _payload(
    value: Any,
    rendering: dict[str, Any],
) -> tuple[list[str], dict[str, str]] | None:
    if isinstance(value, list):
        return (value, {}) if all(isinstance(item, str) for item in value) else None
    if not isinstance(value, dict):
        return None
    override = _hardware_override(value, rendering)
    raw_args = override.get("args", value.get("args", []))
    raw_env = override.get("env", value.get("env", {}))
    if not isinstance(raw_args, list) or not all(
        isinstance(item, str) for item in raw_args
    ):
        return None
    if not isinstance(raw_env, dict) or not all(
        isinstance(key, str) and isinstance(item, str)
        for key, item in raw_env.items()
    ):
        return None
    return raw_args, raw_env


def _argument_sequences(feature: dict[str, Any]) -> list[list[str]]:
    sequences: list[list[str]] = []

    def collect(value: Any) -> None:
        if not isinstance(value, dict):
            if isinstance(value, list) and all(isinstance(item, str) for item in value):
                sequences.append(value)
            return
        raw_args = value.get("args")
        if isinstance(raw_args, list) and all(isinstance(item, str) for item in raw_args):
            sequences.append(raw_args)
        overrides = value.get("hardware_overrides", {})
        if isinstance(overrides, dict):
            for override in overrides.values():
                collect(override)

    collect(feature)
    modes = feature.get("modes")
    if isinstance(modes, dict):
        for raw_mode in modes.values():
            collect(raw_mode)
    return [sequence for sequence in sequences if sequence]


def _environment_keys(feature: dict[str, Any]) -> set[str]:
    keys: set[str] = set()

    def collect(value: Any) -> None:
        if not isinstance(value, dict):
            return
        raw_env = value.get("env")
        if isinstance(raw_env, dict):
            keys.update(key for key in raw_env if isinstance(key, str))
        overrides = value.get("hardware_overrides", {})
        if isinstance(overrides, dict):
            for override in overrides.values():
                collect(override)

    collect(feature)
    modes = feature.get("modes")
    if isinstance(modes, dict):
        for raw_mode in modes.values():
            collect(raw_mode)
    return keys


def _remove_sequence(values: list[str], sequence: list[str]) -> list[str]:
    result = list(values)
    while sequence and len(sequence) <= len(result):
        try:
            start = next(
                index
                for index in range(len(result) - len(sequence) + 1)
                if result[index : index + len(sequence)] == sequence
            )
        except StopIteration:
            break
        del result[start : start + len(sequence)]
    return result


def _mode_available(
    mode: Any,
    variant: str,
    rendering: dict[str, Any],
) -> tuple[bool, str | None]:
    if isinstance(mode, dict):
        variants = mode.get("variants")
        if isinstance(variants, list) and variant not in variants:
            return False, "Режим не опубликован для выбранного варианта"
        hardware = mode.get("hardware")
        if isinstance(hardware, dict):
            profile = rendering.get("hardware_profile", {})
            if not isinstance(profile, dict):
                profile = {}
            generation = profile.get("generation") or profile.get("gpu_generation")
            if isinstance(generation, list):
                generation = generation[0] if generation else None
            hardware_id = rendering.get("hardware")
            brand = profile.get("brand")
            gates = [
                hardware.get(hardware_id) if isinstance(hardware_id, str) else None,
                hardware.get(generation) if isinstance(generation, str) else None,
                hardware.get(brand.lower()) if isinstance(brand, str) else None,
            ]
            if "unsupported" in gates:
                return False, "Режим не поддерживает выбранный тип GPU"
    payload = _payload(mode, rendering)
    if payload is None or not any(payload):
        return False, "Режим не содержит поддерживаемые args или env"
    return True, None


def _feature_available(
    feature: dict[str, Any],
    variant: str,
    strategy: str,
    rendering: dict[str, Any],
) -> tuple[bool, str | None]:
    if feature.get("companion") or feature.get("companions"):
        return False, "Опция требует дополнительного процесса"
    variants = feature.get("variants")
    if isinstance(variants, list) and variant not in variants:
        return False, "Опция не опубликована для выбранного варианта"
    strategies = feature.get("strategies")
    if isinstance(strategies, list) and strategy not in strategies:
        return False, "Опция не опубликована для выбранной стратегии"
    modes = feature.get("modes")
    if isinstance(modes, dict) and modes:
        if any(_mode_available(mode, variant, rendering)[0] for mode in modes.values()):
            return True, None
        return False, "Для выбранного варианта и GPU нет доступного режима"
    payload = _payload(feature, rendering)
    if payload is not None and any(payload):
        return True, None
    return False, "В рецепте нет поддерживаемых args или env для этой опции"


def _kv_groups(
    taxonomy: dict[str, Any],
    requested_key: str | None,
    has_mooncake: bool,
) -> tuple[str, list[RecipeConfigurationOption], list[str], dict[str, str]]:
    raw_options = taxonomy.get("kv_offload", {})
    if not isinstance(raw_options, dict):
        raw_options = {}
    options = [_option("off", "Off")]
    args: list[str] = []
    env: dict[str, str] = {}
    for key, raw in raw_options.items():
        if not isinstance(key, str) or not isinstance(raw, dict):
            continue
        label = raw.get("label") or raw.get("display_name") or display_label(key)
        description = raw.get("description")
        available = key == "simple" and not raw.get("companion")
        reason = None if available else (
            "Требуется отдельная инфраструктура или неподдерживаемая настройка хранилища"
        )
        options.append(
            _option(
                key,
                str(label),
                description=description if isinstance(description, str) else None,
                available=available,
                unavailable_reason=reason,
            )
        )
    if has_mooncake and not any(option.key == "mooncake" for option in options):
        options.append(
            _option(
                "mooncake",
                "Mooncake",
                available=False,
                unavailable_reason="Mooncake требует дополнительных KV-процессов",
            )
        )

    selected = requested_key or "off"
    match = next((option for option in options if option.key == selected), None)
    if match is None:
        raise RecipeConfigurationError(f"Неизвестный KV offload: {selected}")
    if not match.available:
        raise RecipeConfigurationError(match.unavailable_reason or "KV offload недоступен")
    match.selected = True
    if selected != "off":
        raw = raw_options[selected]
        raw_args = raw.get("args", [])
        raw_env = raw.get("env", {})
        if not isinstance(raw_args, list) or not all(isinstance(item, str) for item in raw_args):
            raise RecipeConfigurationError("KV offload содержит некорректные аргументы")
        if not isinstance(raw_env, dict) or not all(
            isinstance(key, str) and isinstance(value, str)
            for key, value in raw_env.items()
        ):
            raise RecipeConfigurationError("KV offload содержит некорректное окружение")
        args = raw_args
        env = raw_env
    return selected, options, args, env


def compose_configuration(
    *,
    rendering: dict[str, Any],
    recipe: dict[str, Any],
    variant_key: str,
    variant_metadata: dict[str, Any],
    strategy_key: str,
    requested: RecipeConfigurationSelection,
    variant_options: list[RecipeConfigurationOption],
    strategy_options: list[RecipeConfigurationOption],
    taxonomy: dict[str, Any],
    apply_variant_overrides: bool = True,
    source_variant_metadata: dict[str, Any] | None = None,
) -> tuple[dict[str, Any], ResolvedRecipeConfiguration]:
    composed = deepcopy(rendering)
    argv = composed.get("argv")
    if not isinstance(argv, list) or not all(isinstance(value, str) for value in argv):
        raise RecipeConfigurationError("Выбранная конфигурация не содержит vLLM argv")

    if apply_variant_overrides:
        source_metadata = source_variant_metadata or {}
        source_args = source_metadata.get("extra_args", [])
        if isinstance(source_args, list) and all(
            isinstance(value, str) for value in source_args
        ):
            argv = _remove_sequence(argv, source_args)
        source_env = source_metadata.get(
            "extra_env", source_metadata.get("env", {})
        )
        if isinstance(source_env, dict):
            for key in source_env:
                if isinstance(key, str):
                    composed_env = composed.get("env")
                    if isinstance(composed_env, dict):
                        composed_env.pop(key, None)
        model_id = variant_metadata.get("model_id")
        if isinstance(model_id, str) and model_id:
            argv[2] = model_id
        extra_args = variant_metadata.get("extra_args", [])
        if not isinstance(extra_args, list) or not all(
            isinstance(value, str) for value in extra_args
        ):
            raise RecipeConfigurationError("Вариант содержит некорректные extra_args")
        argv = _remove_sequence(argv, extra_args)
        argv.extend(extra_args)

    raw_env = composed.get("env", {})
    if not isinstance(raw_env, dict):
        raise RecipeConfigurationError("Выбранная конфигурация содержит некорректный env")
    env = dict(raw_env)
    variant_env = (
        variant_metadata.get("extra_env", variant_metadata.get("env", {}))
        if apply_variant_overrides
        else {}
    )
    if not isinstance(variant_env, dict) or not all(
        isinstance(key, str) and isinstance(value, str)
        for key, value in variant_env.items()
    ):
        raise RecipeConfigurationError("Вариант содержит некорректный env")
    env.update(variant_env)

    features = recipe.get("features", {})
    if not isinstance(features, dict):
        features = {}
    for raw_feature in features.values():
        if isinstance(raw_feature, dict):
            for sequence in _argument_sequences(raw_feature):
                argv = _remove_sequence(argv, sequence)
            for key in _environment_keys(raw_feature):
                env.pop(key, None)

    opt_in = recipe.get("opt_in_features", [])
    if not isinstance(opt_in, list):
        opt_in = []
    default_modes = variant_metadata.get("default_modes", {})
    if not isinstance(default_modes, dict):
        default_modes = {}
    selected_features = (
        list(requested.features)
        if requested.features is not None
        else [
            key
            for key, raw_feature in features.items()
            if (key not in opt_in or key in default_modes)
            and isinstance(raw_feature, dict)
            and _feature_available(
                raw_feature, variant_key, strategy_key, rendering
            )[0]
        ]
    )
    unknown_features = set(selected_features) - set(features)
    if unknown_features:
        raise RecipeConfigurationError(
            f"Неизвестная feature-опция: {sorted(unknown_features)[0]}"
        )

    feature_options: list[RecipeConfigurationOption] = []
    mode_groups: list[RecipeConfigurationGroup] = []
    resolved_modes: dict[str, str] = {}
    for key, raw_feature in features.items():
        if not isinstance(key, str) or not isinstance(raw_feature, dict):
            continue
        available, reason = _feature_available(
            raw_feature, variant_key, strategy_key, rendering
        )
        selected = key in selected_features
        if selected and not available:
            raise RecipeConfigurationError(reason or f"Feature {key} недоступна")
        description = raw_feature.get("description")
        feature_options.append(
            _option(
                key,
                str(raw_feature.get("label") or display_label(key)),
                selected=selected,
                description=description if isinstance(description, str) else None,
                available=available,
                unavailable_reason=reason,
            )
        )
        if not selected:
            continue

        modes = raw_feature.get("modes")
        if isinstance(modes, dict) and modes:
            requested_mode = requested.feature_modes.get(key)
            default_mode = default_modes.get(key) or raw_feature.get("default_mode")
            allowed_modes = [
                candidate
                for candidate, raw_mode in modes.items()
                if _mode_available(raw_mode, variant_key, rendering)[0]
            ]
            mode_key = requested_mode or (
                default_mode
                if isinstance(default_mode, str) and default_mode in allowed_modes
                else allowed_modes[0]
            )
            if mode_key not in modes:
                raise RecipeConfigurationError(f"Неизвестный режим {key}: {mode_key}")
            mode_options: list[RecipeConfigurationOption] = []
            for candidate, raw_mode in modes.items():
                mode_available, mode_reason = _mode_available(
                    raw_mode, variant_key, rendering
                )
                mode_label = (
                    raw_mode.get("label")
                    if isinstance(raw_mode, dict)
                    else None
                )
                mode_options.append(
                    _option(
                        candidate,
                        str(mode_label or display_label(candidate)),
                        selected=candidate == mode_key,
                        available=mode_available,
                        unavailable_reason=mode_reason,
                    )
                )
            chosen_available, chosen_reason = _mode_available(
                modes[mode_key], variant_key, rendering
            )
            if not chosen_available:
                raise RecipeConfigurationError(
                    chosen_reason or f"Режим {key}: {mode_key} недоступен"
                )
            chosen_payload = _payload(modes[mode_key], rendering)
            if chosen_payload is None:
                raise RecipeConfigurationError(f"Режим {key}: {mode_key} некорректен")
            chosen_args, chosen_env = chosen_payload
            argv.extend(chosen_args)
            env.update(chosen_env)
            resolved_modes[key] = mode_key
            mode_groups.append(
                RecipeConfigurationGroup(
                    key=f"feature_mode:{key}",
                    label=str(raw_feature.get("mode_label") or f"{display_label(key)} method"),
                    options=mode_options,
                )
            )
        else:
            feature_payload = _payload(raw_feature, rendering)
            if feature_payload is None:
                raise RecipeConfigurationError(
                    f"Feature {key} содержит некорректные args или env"
                )
            feature_args, feature_env = feature_payload
            argv.extend(feature_args)
            env.update(feature_env)

    unknown_modes = set(requested.feature_modes) - set(resolved_modes)
    if unknown_modes:
        raise RecipeConfigurationError(
            "Режим указан для невыбранной feature-опции или feature без режимов"
        )

    alternatives = rendering.get("alternatives", {})
    has_mooncake = isinstance(alternatives, dict) and any(
        "mooncake" in key for key in alternatives if isinstance(key, str)
    )
    kv_key, kv_options, kv_args, kv_env = _kv_groups(
        taxonomy, requested.kv_offload, has_mooncake
    )
    argv.extend(kv_args)
    env.update(kv_env)

    frontend = requested.frontend or (
        "rust" if env.get("VLLM_USE_RUST_FRONTEND") == "1" else "python"
    )
    env.pop("VLLM_USE_RUST_FRONTEND", None)
    if frontend == "rust":
        env["VLLM_USE_RUST_FRONTEND"] = "1"

    composed["argv"] = argv
    composed["env"] = env
    groups = [
        RecipeConfigurationGroup(key="variant", label="Variant", options=variant_options),
        RecipeConfigurationGroup(key="strategy", label="Strategy", options=strategy_options),
        RecipeConfigurationGroup(key="kv_offload", label="KV offload", options=kv_options),
        RecipeConfigurationGroup(
            key="nodes",
            label="Nodes",
            control="display",
            options=[
                _option("single_node", "Single-node", selected=True),
                _option(
                    "multi_node",
                    "Multi-node",
                    available=False,
                    unavailable_reason="Cloud.ru ML Inference не принимает multi-node recipe",
                ),
            ],
        ),
        RecipeConfigurationGroup(
            key="frontend",
            label="Frontend",
            options=[
                _option("rust", "⚡ Rust", selected=frontend == "rust"),
                _option("python", "Python", selected=frontend == "python"),
            ],
        ),
        RecipeConfigurationGroup(
            key="features",
            label="Features",
            control="multiple",
            options=feature_options,
        ),
        *mode_groups,
    ]
    selection = RecipeConfigurationSelection(
        variant=variant_key,
        strategy=strategy_key,
        kv_offload=kv_key,
        frontend=frontend,
        features=selected_features,
        feature_modes=resolved_modes,
    )
    return composed, ResolvedRecipeConfiguration(selection=selection, groups=groups)


def unavailable_configuration(
    *,
    rendering: dict[str, Any],
    recipe: dict[str, Any],
    variant_key: str,
    variant_options: list[RecipeConfigurationOption],
    strategy_options: list[RecipeConfigurationOption],
    taxonomy: dict[str, Any],
    reason: str,
) -> ResolvedRecipeConfiguration:
    for option in variant_options:
        option.selected = option.key == variant_key
        option.available = False
        option.unavailable_reason = option.unavailable_reason or reason

    strategy_key = rendering.get("strategy")
    if not isinstance(strategy_key, str):
        strategy_key = None
    for option in strategy_options:
        option.selected = option.key == strategy_key
        option.available = False
        option.unavailable_reason = option.unavailable_reason or reason

    raw_kv = taxonomy.get("kv_offload", {})
    if not isinstance(raw_kv, dict):
        raw_kv = {}
    kv_options = [
        _option("off", "Off", selected=True, available=False, unavailable_reason=reason)
    ]
    for key, raw in raw_kv.items():
        if not isinstance(key, str) or not isinstance(raw, dict):
            continue
        label = raw.get("label") or raw.get("display_name") or display_label(key)
        kv_options.append(
            _option(key, str(label), available=False, unavailable_reason=reason)
        )
    alternatives = rendering.get("alternatives", {})
    if isinstance(alternatives, dict) and any(
        isinstance(key, str) and "mooncake" in key for key in alternatives
    ):
        kv_options.append(
            _option("mooncake", "Mooncake", available=False, unavailable_reason=reason)
        )

    raw_env = rendering.get("env", {})
    frontend = (
        "rust"
        if isinstance(raw_env, dict) and raw_env.get("VLLM_USE_RUST_FRONTEND") == "1"
        else "python"
    )
    features = recipe.get("features", {})
    feature_options = []
    mode_groups: list[RecipeConfigurationGroup] = []
    if isinstance(features, dict):
        for key, raw in features.items():
            if not isinstance(key, str) or not isinstance(raw, dict):
                continue
            description = raw.get("description")
            feature_options.append(
                _option(
                    key,
                    str(raw.get("label") or display_label(key)),
                    description=description if isinstance(description, str) else None,
                    available=False,
                    unavailable_reason=reason,
                )
            )
            modes = raw.get("modes")
            if isinstance(modes, dict) and modes:
                default_mode = raw.get("default_mode")
                selected_mode = (
                    default_mode
                    if isinstance(default_mode, str) and default_mode in modes
                    else next(iter(modes))
                )
                mode_groups.append(
                    RecipeConfigurationGroup(
                        key=f"feature_mode:{key}",
                        label=str(raw.get("mode_label") or f"{display_label(key)} method"),
                        options=[
                            _option(
                                mode_key,
                                str(
                                    mode.get("label")
                                    if isinstance(mode, dict) and mode.get("label")
                                    else display_label(mode_key)
                                ),
                                selected=mode_key == selected_mode,
                                available=False,
                                unavailable_reason=reason,
                            )
                            for mode_key, mode in modes.items()
                            if isinstance(mode_key, str)
                        ],
                    )
                )

    groups = [
        RecipeConfigurationGroup(key="variant", label="Variant", options=variant_options),
        RecipeConfigurationGroup(key="strategy", label="Strategy", options=strategy_options),
        RecipeConfigurationGroup(key="kv_offload", label="KV offload", options=kv_options),
        RecipeConfigurationGroup(
            key="nodes",
            label="Nodes",
            control="display",
            options=[
                _option("single_node", "Single-node", available=False, unavailable_reason=reason),
                _option(
                    "multi_node",
                    "Multi-node",
                    selected=rendering.get("deploy_type") != "single_node",
                    available=False,
                    unavailable_reason=reason,
                ),
            ],
        ),
        RecipeConfigurationGroup(
            key="frontend",
            label="Frontend",
            control="display",
            options=[
                _option(
                    "rust",
                    "⚡ Rust",
                    selected=frontend == "rust",
                    available=False,
                    unavailable_reason=reason,
                ),
                _option(
                    "python",
                    "Python",
                    selected=frontend == "python",
                    available=False,
                    unavailable_reason=reason,
                ),
            ],
        ),
        RecipeConfigurationGroup(
            key="features",
            label="Features",
            control="multiple",
            options=feature_options,
        ),
        *mode_groups,
    ]
    return ResolvedRecipeConfiguration(
        selection=RecipeConfigurationSelection(
            variant=variant_key,
            strategy=strategy_key,
            kv_offload="off",
            frontend=frontend,
            features=[],
            feature_modes={},
        ),
        groups=groups,
    )
