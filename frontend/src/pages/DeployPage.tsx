import { ArrowLeft, Check, Cpu, Rocket, ShieldAlert, TriangleAlert } from "lucide-react";
import { type FormEvent, useEffect, useMemo, useRef, useState } from "react";
import { Link, useSearchParams } from "react-router-dom";

import { ApiError, createInference, getCloudOptions, getRecipeDetail } from "../api";
import { ErrorDialog } from "../components/ErrorDialog";
import { FrontendRecommendationDialog } from "../components/FrontendRecommendationDialog";
import { Notice } from "../components/Notice";
import { RecipeConfigurationPicker } from "../components/RecipeConfigurationPicker";
import { loadCredentials } from "../credentials";
import type {
  CloudOptions,
  CreateInferenceResult,
  RecipeConfigurationSelection,
  RecipeDetail,
} from "../types";

export const GPU_OPTIONS = [
  { value: "ResourceType_GPU_V100", label: "NVIDIA V100 · 32 GB", counts: [1, 2, 4, 8, 16] },
  { value: "ResourceType_GPU_A100_NVLINK", label: "NVIDIA A100 · 80 GB", counts: [1, 2, 4, 6, 8] },
  { value: "ResourceType_GPU_H100", label: "NVIDIA H100 · 80 GB", counts: [1, 2, 4, 6, 8] },
] as const;

type GpuChoice = {
  value: string;
  label: string;
  counts: readonly number[];
};

type AuthenticationType =
  | "AUTHENTICATION_TYPE_KEY_BASED"
  | "AUTHENTICATION_TYPE_TOKEN_BASED";

const DEPLOYMENT_ATTEMPT_KEY = "vllm-cloudru.deployment-attempt";
const DEPLOYMENT_ATTEMPT_EVENT = "vllm-cloudru:deployment-attempt";
const liveAttemptIds = new Set<string>();
const CONFIGURATION_QUERY_KEYS = [
  "variant",
  "strategy",
  "kv_offload",
  "frontend",
  "feature",
  "feature_mode",
  "features_configured",
] as const;
const DEFAULT_GPU_MEMORY_UTILIZATION = 0.97;

type DeploymentAttempt = {
  id: string;
  status: "pending" | "uncertain";
  hfId: string;
  message?: string;
};

function readAttempt(): DeploymentAttempt | null {
  try {
    const raw = window.sessionStorage.getItem(DEPLOYMENT_ATTEMPT_KEY);
    if (!raw) return null;
    const value = JSON.parse(raw) as Partial<DeploymentAttempt>;
    if (
      typeof value.id === "string"
      && (value.status === "pending" || value.status === "uncertain")
      && typeof value.hfId === "string"
    ) {
      return value as DeploymentAttempt;
    }
  } catch {
    return null;
  }
  return null;
}

function saveAttempt(value: DeploymentAttempt, notify = false): boolean {
  try {
    window.sessionStorage.setItem(DEPLOYMENT_ATTEMPT_KEY, JSON.stringify(value));
    if (notify) window.dispatchEvent(new Event(DEPLOYMENT_ATTEMPT_EVENT));
    return true;
  } catch {
    return false;
  }
}

function replaceAttempt(id: string, value: DeploymentAttempt): boolean {
  if (readAttempt()?.id !== id) return false;
  return saveAttempt(value, true);
}

function clearAttempt(id?: string, notify = true): boolean {
  try {
    if (id && readAttempt()?.id !== id) return false;
    window.sessionStorage.removeItem(DEPLOYMENT_ATTEMPT_KEY);
    if (notify) window.dispatchEvent(new Event(DEPLOYMENT_ATTEMPT_EVENT));
    return true;
  } catch {
    // An unavailable session store is handled before a create request is sent.
    return false;
  }
}

function normalizeName(value: string): string {
  let normalized = value.toLowerCase().replace(/[^a-z0-9-]+/g, "-").replace(/-+/g, "-");
  normalized = normalized.replace(/^-|-$/g, "");
  if (!normalized || !/^[a-z]/.test(normalized)) normalized = `model-${normalized}`;
  return normalized.slice(0, 63).replace(/-$/, "") || "model";
}

function warningList(error: ApiError): string[] {
  const warnings = error.details?.warnings;
  return Array.isArray(warnings)
    ? warnings.filter((warning): warning is string => typeof warning === "string")
    : [];
}

function invalidArgumentKeys(error: ApiError): string[] {
  const keys = error.details?.invalidKeys;
  return Array.isArray(keys)
    ? keys.filter((key): key is string => typeof key === "string")
    : [];
}

function configurationFromQuery(params: URLSearchParams): RecipeConfigurationSelection {
  const featureModes: Record<string, string> = {};
  params.getAll("feature_mode").forEach((item) => {
    const separator = item.indexOf(":");
    if (separator > 0 && separator < item.length - 1) {
      featureModes[item.slice(0, separator)] = item.slice(separator + 1);
    }
  });
  const featuresConfigured = params.get("features_configured") === "true";
  const frontend = params.get("frontend");
  return {
    variant: params.get("variant"),
    strategy: params.get("strategy"),
    kv_offload: params.get("kv_offload"),
    frontend: frontend === "rust" || frontend === "python" ? frontend : null,
    features: featuresConfigured ? params.getAll("feature") : undefined,
    feature_modes: featureModes,
  };
}

function queryWithConfiguration(
  params: URLSearchParams,
  selection: RecipeConfigurationSelection,
): URLSearchParams {
  const next = new URLSearchParams(params);
  CONFIGURATION_QUERY_KEYS.forEach((key) => next.delete(key));
  if (selection.variant) next.set("variant", selection.variant);
  if (selection.strategy) next.set("strategy", selection.strategy);
  if (selection.kv_offload) next.set("kv_offload", selection.kv_offload);
  if (selection.frontend) next.set("frontend", selection.frontend);
  if (selection.features !== undefined && selection.features !== null) {
    next.set("features_configured", "true");
    selection.features.forEach((feature) => next.append("feature", feature));
  }
  Object.entries(selection.feature_modes).forEach(([feature, mode]) => {
    next.append("feature_mode", `${feature}:${mode}`);
  });
  return next;
}

function normalizedArgumentKey(key: string): string {
  return key.replace(/^-+/, "").replaceAll("_", "-").toLowerCase();
}

function effectiveArguments(
  argumentsFromRecipe: NonNullable<RecipeDetail["translated"]>["arguments"],
  gpuMemoryUtilization: number,
) {
  return [
    ...argumentsFromRecipe.filter(
      (argument) => normalizedArgumentKey(argument.key) !== "gpu-memory-utilization",
    ),
    {
      key: "gpu-memory-utilization",
      value: gpuMemoryUtilization,
      parameterType: "PARAMETER_TYPE_ARG_KV_NUMERIC",
    },
  ];
}

export function DeployPage() {
  const [searchParams, setSearchParams] = useSearchParams();
  const searchKey = searchParams.toString();
  const requestedConfiguration = useMemo(
    () => configurationFromQuery(searchParams),
    [searchKey],
  );
  const configurationKey = JSON.stringify(requestedConfiguration);
  const hasConfigurationQuery = CONFIGURATION_QUERY_KEYS.some((key) => searchParams.has(key));
  const hfId = searchParams.get("model") || "";
  const validHfId = /^[^/\s]+\/[^/\s]+$/.test(hfId);
  const requestedResource = searchParams.get("resource");
  const [options, setOptions] = useState<CloudOptions | null>(null);
  const gpuOptions = useMemo<GpuChoice[]>(() => {
    if (!options) return [...GPU_OPTIONS];
    const configured = Object.entries(options.gpuOptions).map(([value, option]) => ({
      value,
      label: option.label,
      counts: option.counts,
    }));
    return configured;
  }, [options]);
  const resource = gpuOptions.some((option) => option.value === requestedResource)
    ? requestedResource as string
    : gpuOptions.find((option) => option.value === "ResourceType_GPU_H100")?.value
      ?? gpuOptions[0].value;
  const gpu = gpuOptions.find((option) => option.value === resource) ?? gpuOptions[0];

  const [recipe, setRecipe] = useState<RecipeDetail | null>(null);
  const [name, setName] = useState(() => hfId.split("/").at(-1) || "model");
  const [gpuCount, setGpuCount] = useState<number>(gpu.counts[0]);
  const [runtimeId, setRuntimeId] = useState("");
  const [minScale, setMinScale] = useState(1);
  const [maxScale, setMaxScale] = useState(1);
  const [gpuMemoryUtilization, setGpuMemoryUtilization] = useState(
    DEFAULT_GPU_MEMORY_UTILIZATION,
  );
  const [serviceAccountId, setServiceAccountId] = useState("");
  const [authenticationType, setAuthenticationType] = useState<AuthenticationType>(
    "AUTHENTICATION_TYPE_KEY_BASED",
  );
  const [logGroupId, setLogGroupId] = useState("");
  const [warnings, setWarnings] = useState<string[]>([]);
  const [warningContext, setWarningContext] = useState("");
  const [confirmed, setConfirmed] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [createError, setCreateError] = useState<string | null>(null);
  const [unsupportedArgumentKeys, setUnsupportedArgumentKeys] = useState<string[]>([]);
  const [uncertain, setUncertain] = useState<string | null>(null);
  const [canRetryAfterCheck, setCanRetryAfterCheck] = useState(false);
  const [loading, setLoading] = useState(true);
  const [recipeLoading, setRecipeLoading] = useState(true);
  const [submitting, setSubmitting] = useState(false);
  const [configurationAdjustments, setConfigurationAdjustments] = useState<string[]>([]);
  const [recommendationOpen, setRecommendationOpen] = useState(false);
  const [dismissedRecommendation, setDismissedRecommendation] = useState<string | null>(null);
  const [result, setResult] = useState<{ hfId: string; value: CreateInferenceResult } | null>(null);
  const warningsRef = useRef<HTMLDivElement>(null);
  const reconcileNextRequestRef = useRef(false);

  const confirmationContext = JSON.stringify({
    hfId,
    resource,
    gpuCount,
    runtimeId,
    minScale,
    maxScale,
    gpuMemoryUtilization,
    serviceAccountId,
    authenticationType,
    logGroupId,
    configuration: recipe?.configuration?.selection ?? requestedConfiguration,
  });
  const activeWarnings = warningContext === confirmationContext ? warnings : [];
  const normalizedName = normalizeName(name);
  const vllmArguments = effectiveArguments(
    recipe?.translated?.arguments ?? [],
    gpuMemoryUtilization,
  );
  const recommendationFingerprint = JSON.stringify({
    resource,
    renderingPath: recipe?.rendering_path,
    configuration: recipe?.configuration?.selection ?? requestedConfiguration,
  });
  const shouldRecommendPython = (
    (recipe?.configuration?.selection.frontend ?? requestedConfiguration.frontend) === "rust"
    && vllmArguments.some(
      (argument) => normalizedArgumentKey(argument.key) === "hf-overrides",
    )
  );

  useEffect(() => {
    if (!validHfId) {
      setLoading(false);
      return;
    }
    let active = true;
    getCloudOptions()
      .then((value) => { if (active) setOptions(value); })
      .catch((reason) => {
        if (active) setError(reason instanceof ApiError ? reason.message : "Не удалось получить настройки Cloud.ru");
      })
      .finally(() => { if (active) setLoading(false); });
    return () => { active = false; };
  }, [hfId, validHfId]);

  useEffect(() => {
    setName(hfId.split("/").at(-1) || "model");
    setWarnings([]);
    setWarningContext("");
    setConfirmed(false);
    setError(null);
    setCreateError(null);
    setUnsupportedArgumentKeys([]);
    setUncertain(null);
    setCanRetryAfterCheck(false);
    setResult(null);
    setServiceAccountId("");
    setAuthenticationType("AUTHENTICATION_TYPE_KEY_BASED");
    setLogGroupId("");
    setGpuMemoryUtilization(DEFAULT_GPU_MEMORY_UTILIZATION);
    setConfigurationAdjustments([]);
    setRecommendationOpen(false);
    setDismissedRecommendation(null);

    const restoreAttempt = () => {
      const attempt = readAttempt();
      if (!attempt) {
        setUncertain(null);
        setCanRetryAfterCheck(false);
        return;
      }
      if (attempt.status === "pending" && liveAttemptIds.has(attempt.id)) {
        setUncertain("Запрос создания ещё выполняется. Дождитесь подтверждённого результата.");
        setCanRetryAfterCheck(false);
        return;
      }
      const message = attempt.status === "pending"
        ? "Предыдущий запрос был прерван или страница перезагрузилась до получения ответа."
        : attempt.message || "Результат предыдущего создания не подтверждён.";
      if (attempt.status === "pending") {
        saveAttempt({ ...attempt, status: "uncertain", message });
      }
      setUncertain(message);
      setCanRetryAfterCheck(true);
    };
    restoreAttempt();
    window.addEventListener(DEPLOYMENT_ATTEMPT_EVENT, restoreAttempt);
    return () => window.removeEventListener(DEPLOYMENT_ATTEMPT_EVENT, restoreAttempt);
  }, [hfId]);

  useEffect(() => {
    if (!hfId) return;
    let active = true;
    const reconcileConfiguration = reconcileNextRequestRef.current;
    reconcileNextRequestRef.current = false;
    setError(null);
    setRecipeLoading(true);
    getRecipeDetail(hfId, resource, requestedConfiguration, reconcileConfiguration)
      .then((value) => {
        if (!active) return;
        setRecipe(value);
        if (reconcileConfiguration) {
          setConfigurationAdjustments(value.configuration_adjustments ?? []);
          if (value.configuration) {
            const next = queryWithConfiguration(
              searchParams,
              value.configuration.selection,
            );
            if (next.toString() !== searchParams.toString()) {
              setSearchParams(next, { replace: true });
            }
          }
        }
      })
      .catch((reason) => {
        if (active) setError(reason instanceof ApiError ? reason.message : "Не удалось открыть рецепт");
      })
      .finally(() => { if (active) setRecipeLoading(false); });
    return () => { active = false; };
  }, [configurationKey, hfId, resource]);

  const compatibleTemplates = useMemo(() => {
    if (!options) return [];
    return options.runtimeTemplates.filter(
      (template) => !template.gpus.length || template.gpus.some((item) => item.resourceType === resource),
    );
  }, [options, resource]);

  useEffect(() => {
    if (!compatibleTemplates.some((template) => template.id === runtimeId)) {
      setRuntimeId(compatibleTemplates[0]?.id || "");
    }
  }, [compatibleTemplates, runtimeId]);

  useEffect(() => {
    if (!options) return;
    setServiceAccountId((current) => (
      options.serviceAccounts.some((account) => account.id === current) ? current : ""
    ));
    setLogGroupId((current) => {
      if (options.logGroups.some((group) => group.id === current)) return current;
      return options.logGroups.find((group) => group.isDefault)?.id || "";
    });
  }, [options]);

  useEffect(() => {
    if (!gpu.counts.includes(gpuCount)) {
      setGpuCount(gpu.counts[0]);
      setWarnings([]);
      setWarningContext("");
      setConfirmed(false);
    }
  }, [gpu, gpuCount]);

  useEffect(() => {
    if (activeWarnings.length) warningsRef.current?.focus();
  }, [activeWarnings.length]);

  function resetConfirmation() {
    setWarnings([]);
    setWarningContext("");
    setConfirmed(false);
    setError(null);
  }

  function changeResource(value: string) {
    if (recipeLoading) return;
    const selected = gpuOptions.find((option) => option.value === value);
    if (!selected) return;
    const next = recipe?.configuration
      ? queryWithConfiguration(searchParams, recipe.configuration.selection)
      : new URLSearchParams(searchParams);
    next.set("resource", selected.value);
    ["variant", "strategy", "kv_offload"].forEach((key) => next.delete(key));
    reconcileNextRequestRef.current = true;
    setConfigurationAdjustments([]);
    setRecipeLoading(true);
    setSearchParams(next, { replace: true });
    if (!selected.counts.includes(gpuCount)) setGpuCount(selected.counts[0]);
    resetConfirmation();
  }

  function resetConfiguration() {
    if (recipeLoading) return;
    const next = new URLSearchParams(searchParams);
    CONFIGURATION_QUERY_KEYS.forEach((key) => next.delete(key));
    setConfigurationAdjustments([]);
    if (next.toString() === searchParams.toString()) {
      resetConfirmation();
      return;
    }
    setRecipeLoading(true);
    setSearchParams(next, { replace: true });
    resetConfirmation();
  }

  function changeConfiguration(group: string, option: string, selected: boolean) {
    if (recipeLoading) return;
    const next = recipe?.configuration
      ? queryWithConfiguration(searchParams, recipe.configuration.selection)
      : new URLSearchParams(searchParams);
    if (group === "variant") {
      next.set("variant", option);
      ["strategy", "kv_offload", "feature", "feature_mode", "features_configured"].forEach(
        (key) => next.delete(key),
      );
    } else if (group === "strategy") {
      next.set("strategy", option);
      ["kv_offload", "feature", "feature_mode", "features_configured"].forEach(
        (key) => next.delete(key),
      );
    } else if (group === "kv_offload" || group === "frontend") {
      next.set(group, option);
    } else if (group === "features") {
      const current = new Set(recipe?.configuration?.selection.features ?? []);
      if (selected) current.add(option);
      else current.delete(option);
      next.delete("feature");
      [...current].forEach((feature) => next.append("feature", feature));
      next.set("features_configured", "true");
      if (!selected) {
        const remainingModes = next.getAll("feature_mode").filter(
          (value) => !value.startsWith(`${option}:`),
        );
        next.delete("feature_mode");
        remainingModes.forEach((value) => next.append("feature_mode", value));
      }
    } else if (group.startsWith("feature_mode:")) {
      const feature = group.slice("feature_mode:".length);
      const remainingModes = next.getAll("feature_mode").filter(
        (value) => !value.startsWith(`${feature}:`),
      );
      next.delete("feature_mode");
      remainingModes.forEach((value) => next.append("feature_mode", value));
      next.append("feature_mode", `${feature}:${option}`);
    }
    setConfigurationAdjustments([]);
    setRecipeLoading(true);
    setSearchParams(next, { replace: true });
    resetConfirmation();
  }

  function allowRetryAfterManualCheck() {
    const attempt = readAttempt();
    if (!attempt || liveAttemptIds.has(attempt.id)) return;
    clearAttempt(attempt.id);
    setUncertain(null);
    setCanRetryAfterCheck(false);
    setError(null);
  }

  async function sendCreate(excludedArgumentKeys: string[] = []) {
    const attemptId = window.crypto.randomUUID();
    liveAttemptIds.add(attemptId);
    if (!saveAttempt({ id: attemptId, status: "pending", hfId })) {
      liveAttemptIds.delete(attemptId);
      setError("Браузер не позволяет сохранить защиту от повторного запуска. Создание отменено.");
      return;
    }

    setSubmitting(true);
    try {
      const created = await createInference({
        name: normalizedName,
        hf_id: hfId,
        resource_type: resource,
        gpu_count: gpuCount,
        runtime_template_id: runtimeId,
        min_scale: minScale,
        max_scale: maxScale,
        service_account_id: serviceAccountId,
        authentication_type: authenticationType,
        log_group_id: logGroupId,
        gpu_memory_utilization: gpuMemoryUtilization,
        ...(excludedArgumentKeys.length
          ? { excluded_argument_keys: excludedArgumentKeys }
          : {}),
        configuration: recipe?.configuration?.selection ?? requestedConfiguration,
        confirm_warnings: activeWarnings.length > 0 && confirmed,
      });
      clearAttempt(attemptId);
      setResult({ hfId, value: created });
    } catch (reason) {
      const rejectedKeys = reason instanceof ApiError ? invalidArgumentKeys(reason) : [];
      if (reason instanceof ApiError && reason.code === "confirmation_required") {
        const receivedWarnings = warningList(reason);
        setWarnings(receivedWarnings.length
          ? receivedWarnings
          : ["Cloud.ru требует явного подтверждения параметров запуска"]);
        setWarningContext(confirmationContext);
        setConfirmed(false);
        clearAttempt(attemptId);
      } else if (
        reason instanceof ApiError
        && reason.code === "unsupported_runtime_arguments"
        && rejectedKeys.length
      ) {
        clearAttempt(attemptId);
        setUnsupportedArgumentKeys(excludedArgumentKeys.length ? [] : rejectedKeys);
        setCreateError(reason.message);
      } else if (
        reason instanceof ApiError
        && (reason.code === "create_uncertain"
          || reason.code === "network_error"
          || reason.code === "invalid_response"
          || reason.status >= 500)
      ) {
        const message = reason.message || "Результат создания не подтверждён";
        replaceAttempt(attemptId, { id: attemptId, status: "uncertain", hfId, message });
        setUncertain(message);
        setCanRetryAfterCheck(true);
        setUnsupportedArgumentKeys([]);
        setCreateError(message);
      } else if (reason instanceof ApiError && reason.status > 0 && reason.status < 500) {
        clearAttempt(attemptId);
        setUnsupportedArgumentKeys([]);
        setCreateError(reason.message);
      } else {
        const message = "Результат создания не подтверждён";
        replaceAttempt(attemptId, { id: attemptId, status: "uncertain", hfId, message });
        setUncertain(message);
        setCanRetryAfterCheck(true);
        setUnsupportedArgumentKeys([]);
        setCreateError(message);
      }
    } finally {
      liveAttemptIds.delete(attemptId);
      setSubmitting(false);
    }
  }

  async function submit(event: FormEvent<HTMLFormElement>) {
    event.preventDefault();
    setError(null);
    setCreateError(null);
    setUnsupportedArgumentKeys([]);
    if (uncertain) {
      setError("Сначала проверьте список Model Run и явно разрешите новую попытку");
      return;
    }
    if (minScale > maxScale) {
      setError("Минимальный scale не может превышать максимальный");
      return;
    }
    if (!runtimeId) {
      setError("Для выбранного GPU нет активного runtime template vLLM");
      return;
    }
    if (!serviceAccountId) {
      setError("Выберите сервисный аккаунт для доступа к Model Run");
      return;
    }
    if (!logGroupId) {
      setError("Выберите log group для журналов Model Run");
      return;
    }
    const quota = options?.quotas.find((item) => item.resourceType === resource);
    if (!quota) {
      setError(`Cloud.ru не вернул данные квоты для ${gpu.label}. Создание отменено.`);
      return;
    }
    if (quota.free < gpuCount) {
      setError(`Недостаточно свободной квоты ${gpu.label}: доступно ${quota.free}, выбрано ${gpuCount}`);
      return;
    }
    if (activeWarnings.length && !confirmed) {
      setError("Подтвердите предупреждения перед созданием Model Run");
      return;
    }
    if (
      shouldRecommendPython
      && dismissedRecommendation !== recommendationFingerprint
    ) {
      setRecommendationOpen(true);
      return;
    }
    await sendCreate();
  }

  if (!hfId) {
    return <div className="catalog-state"><h1>Модель не выбрана</h1><Link to="/catalog">Открыть каталог</Link></div>;
  }

  if (result?.hfId === hfId) {
    const created = result.value;
    return (
      <section className="deploy-success">
        <img alt="" src="/illustrations/gpu-deployment.png" />
        <div>
          <p className="kicker">Model Run создан</p>
          <h1>{created.name}</h1>
          <dl><div><dt>ID</dt><dd><code>{created.model_run_id}</code></dd></div><div><dt>Статус</dt><dd>{created.status}</dd></div><div><dt>Endpoint</dt><dd>{created.endpoint || "Появится после запуска"}</dd></div></dl>
          <Link className="primary-button" to="/inferences"><span>К моим инференсам</span><Check aria-hidden="true" size={19} /></Link>
        </div>
      </section>
    );
  }

  const preview = {
    recipe: hfId,
    configuration: recipe?.configuration?.selection ?? requestedConfiguration,
    renderingPath: recipe?.rendering_path,
    modelId: recipe?.translated?.model_id,
    vllmArguments,
    vllmEnvironment: recipe?.translated?.environment ?? [],
    normalizedName,
    resourceType: resource,
    gpuCount,
    gpuMemoryUtilization,
    scaling: { minScale, maxScale },
    runtimeTemplateId: runtimeId || "не выбран",
    serviceAccount: options?.serviceAccounts.find((account) => account.id === serviceAccountId)?.name
      || "не выбран",
    authentication: authenticationType === "AUTHENTICATION_TYPE_KEY_BASED"
      ? "API-ключ"
      : "Ключи доступа",
    logGroup: options?.logGroups.find((group) => group.id === logGroupId)?.name || "не выбрана",
    hfSecretUuidConfigured: Boolean(loadCredentials()?.hfSecretId),
  };

  return (
    <section className="deploy-page">
      <ErrorDialog
        actionLabel={unsupportedArgumentKeys.length ? "Создать без этих параметров" : undefined}
        message={createError}
        onAction={unsupportedArgumentKeys.length ? () => {
          const keys = unsupportedArgumentKeys;
          setCreateError(null);
          setUnsupportedArgumentKeys([]);
          void sendCreate(keys);
        } : undefined}
        onClose={() => {
          setCreateError(null);
          setUnsupportedArgumentKeys([]);
        }}
      />
      <FrontendRecommendationDialog
        onKeepRust={() => {
          setDismissedRecommendation(recommendationFingerprint);
          setRecommendationOpen(false);
        }}
        onSwitchToPython={() => {
          setRecommendationOpen(false);
          changeConfiguration("frontend", "python", true);
        }}
        open={recommendationOpen}
      />
      <Link className="back-link" to={`/catalog/${hfId.split("/").map(encodeURIComponent).join("/")}?resource=${encodeURIComponent(resource)}`}><ArrowLeft aria-hidden="true" size={18} />К рецепту</Link>
      <header className="page-header"><div><p className="kicker">Новый Model Run / {hfId}</p><h1>Соберите запуск.</h1><p>Выберите опубликованную конфигурацию vLLM, GPU и масштабирование Cloud.ru.</p></div></header>

      {uncertain ? <Notice live title="Результат создания не подтверждён" tone="warning"><p>{uncertain}</p><p>Не повторяйте запуск автоматически. Обновите список Model Run и проверьте результат вручную.</p><div className="uncertain-actions"><Link to="/inferences">Проверить мои инференсы</Link>{canRetryAfterCheck ? <button onClick={allowRetryAfterManualCheck} type="button">Я проверил список — разрешить новую попытку</button> : null}</div></Notice> : null}
      {error ? <Notice live tone="error">{error}</Notice> : null}
      {configurationAdjustments.length ? (
        <Notice live title="Конфигурация согласована с новым GPU" tone="warning">
          <ul className="configuration-adjustments">
            {configurationAdjustments.map((adjustment) => (
              <li key={adjustment}>{adjustment}</li>
            ))}
          </ul>
        </Notice>
      ) : null}

      <form className="deploy-layout" onSubmit={submit}>
        <div className="deploy-form">
          <section className="wizard-section">
            <div className="section-heading"><span>01</span><h2>Модель и имя</h2></div>
            <p className="model-lock"><Rocket aria-hidden="true" size={18} />{recipe?.title || hfId}</p>
            <label className="field"><span>Имя Model Run</span><input maxLength={63} onChange={(event) => setName(event.target.value)} required value={name} /></label>
            <p className="normalized-name">В Cloud.ru: <code>{normalizedName}</code></p>
          </section>

          <section className="wizard-section">
            <div className="section-heading"><span>02</span><h2>Конфигурация рецепта</h2></div>
            {recipeLoading && !recipe ? <p className="configuration-loading">Собираем доступные комбинации…</p> : null}
            {recipe?.configuration ? (
              <RecipeConfigurationPicker
                configuration={recipe.configuration}
                disabled={recipeLoading}
                onChange={changeConfiguration}
                onReset={resetConfiguration}
              />
            ) : null}
            {!recipeLoading && !recipe?.configuration && hasConfigurationQuery ? (
              <button className="configuration-reset" onClick={resetConfiguration} type="button">
                Сбросить выбранную конфигурацию
              </button>
            ) : null}
          </section>

          <section className="wizard-section">
            <div className="section-heading"><span>03</span><h2>GPU</h2></div>
            <div className="choice-grid">
              {gpuOptions.map((option) => <label className={`gpu-choice${resource === option.value ? " gpu-choice--active" : ""}`} key={option.value}><input checked={resource === option.value} disabled={recipeLoading} name="gpu" onChange={() => changeResource(option.value)} type="radio" /><Cpu aria-hidden="true" size={19} /><span>{option.label}</span></label>)}
            </div>
            <label className="compact-field compact-field--large"><span>Количество GPU</span><select onChange={(event) => { setGpuCount(Number(event.target.value)); resetConfirmation(); }} value={gpuCount}>{gpu.counts.map((count) => <option key={count} value={count}>{count}</option>)}</select></label>
            <div className="gpu-memory-control">
              <div className="gpu-memory-control__heading">
                <label htmlFor="gpu-memory-utilization">GPU memory utilization</label>
                <output htmlFor="gpu-memory-utilization">{gpuMemoryUtilization.toFixed(2)}</output>
              </div>
              <input
                id="gpu-memory-utilization"
                max="0.99"
                min="0.85"
                onChange={(event) => {
                  setGpuMemoryUtilization(Number(event.target.value));
                  resetConfirmation();
                }}
                step="0.01"
                type="range"
                value={gpuMemoryUtilization}
              />
              <small><span>0.85</span><span>0.99</span></small>
            </div>
          </section>

          <section className="wizard-section">
            <div className="section-heading"><span>04</span><h2>Масштабирование</h2></div>
            <div className="scale-grid">
              <label className="field"><span>Минимум</span><input min={0} onChange={(event) => { setMinScale(Number(event.target.value)); resetConfirmation(); }} type="number" value={minScale} /></label>
              <label className="field"><span>Максимум</span><input min={1} onChange={(event) => { setMaxScale(Number(event.target.value)); resetConfirmation(); }} type="number" value={maxScale} /></label>
            </div>
          </section>

          <section className="wizard-section">
            <div className="section-heading"><span>05</span><h2>Доступ и логи</h2></div>
            <div className="access-grid">
              <label className="compact-field">
                <span>Сервисный аккаунт</span>
                <select
                  onChange={(event) => { setServiceAccountId(event.target.value); resetConfirmation(); }}
                  required
                  value={serviceAccountId}
                >
                  <option value="">Выберите аккаунт</option>
                  {options?.serviceAccounts.map((account) => (
                    <option key={account.id} value={account.id}>
                      {account.name}{account.description ? ` — ${account.description}` : ""}
                    </option>
                  ))}
                </select>
              </label>
              <label className="compact-field">
                <span>Log group</span>
                <select
                  onChange={(event) => { setLogGroupId(event.target.value); resetConfirmation(); }}
                  required
                  value={logGroupId}
                >
                  <option value="">Выберите log group</option>
                  {options?.logGroups.map((group) => (
                    <option key={group.id} value={group.id}>
                      {group.name}{group.description ? ` — ${group.description}` : ""}
                    </option>
                  ))}
                </select>
              </label>
            </div>
            <fieldset className="auth-fieldset">
              <legend>Способ доступа к endpoint</legend>
              <div className="auth-choice-grid">
                <label className={`auth-choice${authenticationType === "AUTHENTICATION_TYPE_KEY_BASED" ? " auth-choice--active" : ""}`}>
                  <input checked={authenticationType === "AUTHENTICATION_TYPE_KEY_BASED"} name="authentication" onChange={() => { setAuthenticationType("AUTHENTICATION_TYPE_KEY_BASED"); resetConfirmation(); }} type="radio" />
                  <span>API-ключ</span>
                </label>
                <label className={`auth-choice${authenticationType === "AUTHENTICATION_TYPE_TOKEN_BASED" ? " auth-choice--active" : ""}`}>
                  <input checked={authenticationType === "AUTHENTICATION_TYPE_TOKEN_BASED"} name="authentication" onChange={() => { setAuthenticationType("AUTHENTICATION_TYPE_TOKEN_BASED"); resetConfirmation(); }} type="radio" />
                  <span>Ключи доступа</span>
                </label>
              </div>
            </fieldset>
          </section>

          <section className="wizard-section">
            <div className="section-heading"><span>06</span><h2>Runtime и подтверждение</h2></div>
            <label className="compact-field compact-field--large"><span>Runtime template</span><select disabled={loading} onChange={(event) => { setRuntimeId(event.target.value); resetConfirmation(); }} value={runtimeId}><option value="">Выберите runtime</option>{compatibleTemplates.map((template) => <option key={template.id} value={template.id}>vLLM {template.version || ""} · {template.id}</option>)}</select></label>
            {activeWarnings.length ? <div aria-live="assertive" className="warning-stack" ref={warningsRef} role="alert" tabIndex={-1}><p><TriangleAlert aria-hidden="true" size={18} />Проверьте предупреждения</p><ul>{activeWarnings.map((warning) => <li key={warning}>{warning}</li>)}</ul><label className="confirm-check"><input checked={confirmed} onChange={(event) => setConfirmed(event.target.checked)} type="checkbox" /><span>Я понимаю предупреждения и всё равно хочу создать Model Run</span></label></div> : null}
          </section>
        </div>

        <aside className="payload-preview">
          <ShieldAlert aria-hidden="true" size={22} />
          <h2>Предпросмотр без секретов</h2>
          <pre>{JSON.stringify(preview, null, 2)}</pre>
          {!recipe?.launchable && recipe?.launch_reason ? <p className="launch-blocked">Запуск недоступен: {recipe.launch_reason}</p> : null}
          <button className="primary-button" disabled={submitting || loading || recipeLoading || Boolean(uncertain) || !recipe || !recipe.launchable || !serviceAccountId || !logGroupId} type="submit"><span>{submitting ? "Создаём…" : activeWarnings.length ? "Подтвердить и создать" : "Проверить и создать"}</span><Rocket aria-hidden="true" size={19} /></button>
        </aside>
      </form>
    </section>
  );
}
