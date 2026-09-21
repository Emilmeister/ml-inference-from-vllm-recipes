import { loadCredentials } from "./credentials";
import type {
  CloudCredentialsPayload,
  CloudOptions,
  CredentialInput,
  CreateInferenceInput,
  CreateInferenceResult,
  ModelRunSummary,
  RecipeCatalogItem,
  RecipeConfigurationSelection,
  RecipeDetail,
} from "./types";

const CLOUD_PREFIX = "/api/cloud/";
const VALIDATE_PATH = "/api/cloud/validate";

export class ApiError extends Error {
  constructor(
    message: string,
    readonly code: string,
    readonly status: number,
    readonly details?: Record<string, unknown>,
  ) {
    super(message);
    this.name = "ApiError";
  }
}

function serializeCredentials(credentials: CredentialInput): CloudCredentialsPayload {
  return {
    project_id: credentials.projectId,
    key_id: credentials.keyId,
    key_secret: credentials.keySecret,
    ...(credentials.hfSecretId ? { hf_secret_id: credentials.hfSecretId } : {}),
  };
}

function objectValue(value: unknown): Record<string, unknown> | null {
  return value !== null && typeof value === "object" && !Array.isArray(value)
    ? (value as Record<string, unknown>)
    : null;
}

async function responseError(response: Response): Promise<ApiError> {
  let payload: unknown;
  try {
    payload = await response.json();
  } catch {
    payload = null;
  }

  const root = objectValue(payload);
  const rawDetail = root?.detail;
  const detail = objectValue(rawDetail);
  const message =
    (typeof detail?.message === "string" && detail.message) ||
    (typeof rawDetail === "string" && rawDetail) ||
    "Cloud.ru не принял запрос";
  const code =
    (typeof detail?.code === "string" && detail.code) || `http_${response.status}`;
  return new ApiError(message, code, response.status, detail ?? undefined);
}

async function post<T>(
  path: string,
  payload: Record<string, unknown> = {},
  candidateCredentials?: CredentialInput,
): Promise<T> {
  if (candidateCredentials && path !== VALIDATE_PATH) {
    throw new ApiError(
      "Новые настройки можно использовать только для проверки подключения",
      "candidate_credentials_not_allowed",
      0,
    );
  }

  let body: Record<string, unknown> = payload;
  if (path.startsWith(CLOUD_PREFIX)) {
    const credentials = candidateCredentials ?? loadCredentials();
    if (!credentials) {
      throw new ApiError(
        "Настройки Cloud.ru отсутствуют или истекли",
        "credentials_missing",
        0,
      );
    }
    body = { ...payload, credentials: serializeCredentials(credentials) };
  }

  let response: Response;
  try {
    response = await fetch(path, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(body),
    });
  } catch {
    throw new ApiError("Не удалось связаться с приложением", "network_error", 0);
  }

  if (!response.ok) throw await responseError(response);
  try {
    return (await response.json()) as T;
  } catch {
    throw new ApiError("Приложение вернуло некорректный ответ", "invalid_response", 502);
  }
}

async function get<T>(path: string): Promise<T> {
  let response: Response;
  try {
    response = await fetch(path, { headers: { Accept: "application/json" } });
  } catch {
    throw new ApiError("Не удалось связаться с приложением", "network_error", 0);
  }
  if (!response.ok) throw await responseError(response);
  try {
    return (await response.json()) as T;
  } catch {
    throw new ApiError("Приложение вернуло некорректный ответ", "invalid_response", 502);
  }
}

export async function validateCredentials(credentials: CredentialInput): Promise<void> {
  const result = await post<{ valid?: unknown }>(VALIDATE_PATH, {}, credentials);
  if (result.valid !== true) {
    throw new ApiError("Приложение не подтвердило настройки", "invalid_response", 502);
  }
}

export function getRecipes(
  resourceType: string,
  refresh = false,
): Promise<RecipeCatalogItem[]> {
  const query = new URLSearchParams({ resource_type: resourceType });
  if (refresh) query.set("refresh", "true");
  return get<unknown>(`/api/recipes?${query}`).then((result) => {
    if (!Array.isArray(result)) {
      throw new ApiError("Каталог имеет неизвестный формат", "invalid_response", 502);
    }
    return result as RecipeCatalogItem[];
  });
}

export async function getRecipeDetail(
  hfId: string,
  resourceType: string,
  selection: RecipeConfigurationSelection = { feature_modes: {} },
  reconcileConfiguration = false,
): Promise<RecipeDetail> {
  const [organization, model, ...rest] = hfId.split("/");
  if (!organization || !model || rest.length) {
    throw new ApiError("Некорректный идентификатор модели", "invalid_model_id", 0);
  }
  const query = new URLSearchParams({ resource_type: resourceType });
  if (selection.variant) query.set("variant", selection.variant);
  if (selection.strategy) query.set("strategy", selection.strategy);
  if (selection.kv_offload) query.set("kv_offload", selection.kv_offload);
  if (selection.frontend) query.set("frontend", selection.frontend);
  if (selection.features !== undefined && selection.features !== null) {
    query.set("features_configured", "true");
    selection.features.forEach((feature) => query.append("feature", feature));
  }
  Object.entries(selection.feature_modes).forEach(([feature, mode]) => {
    query.append("feature_mode", `${feature}:${mode}`);
  });
  if (reconcileConfiguration) query.set("reconcile_configuration", "true");
  const result = await get<unknown>(
    `/api/recipes/${encodeURIComponent(organization)}/${encodeURIComponent(model)}?${query}`,
  );
  if (!result || typeof result !== "object" || Array.isArray(result)) {
    throw new ApiError("Рецепт имеет неизвестный формат", "invalid_response", 502);
  }
  return result as RecipeDetail;
}

export async function getInferences(): Promise<ModelRunSummary[]> {
  const result = await post<unknown>("/api/cloud/inferences/query");
  if (!Array.isArray(result)) {
    throw new ApiError("Список инференсов имеет неизвестный формат", "invalid_response", 502);
  }
  return result as ModelRunSummary[];
}

export async function getCloudOptions(): Promise<CloudOptions> {
  const result = await post<unknown>("/api/cloud/options");
  const root = objectValue(result);
  const runtimeTemplates = root?.runtimeTemplates;
  const gpuOptions = objectValue(root?.gpuOptions);
  const quotas = root?.quotas;
  const serviceAccounts = root?.serviceAccounts;
  const logGroups = root?.logGroups;
  const validTemplates = Array.isArray(runtimeTemplates) && runtimeTemplates.every((item) => {
    const template = objectValue(item);
    return typeof template?.id === "string"
      && typeof template.frameworkType === "string"
      && typeof template.version === "string"
      && Array.isArray(template.gpus)
      && template.gpus.every((gpu) => {
        const value = objectValue(gpu);
        return typeof value?.resourceType === "string" && typeof value.isDefault === "boolean";
      });
  });
  const validGpuOptions = gpuOptions !== null
    && Object.keys(gpuOptions).length > 0
    && Object.values(gpuOptions).every((item) => {
    const option = objectValue(item);
    return typeof option?.label === "string"
      && typeof option.memoryGb === "number"
      && Array.isArray(option.counts)
      && option.counts.every((count) => Number.isInteger(count) && count > 0);
  });
  const validQuotas = Array.isArray(quotas) && quotas.every((item) => {
    const quota = objectValue(item);
    return typeof quota?.resourceType === "string"
      && typeof quota.free === "number"
      && typeof quota.limit === "number";
  });
  const validServiceAccounts = Array.isArray(serviceAccounts)
    && serviceAccounts.every((item) => {
      const account = objectValue(item);
      return typeof account?.id === "string"
        && typeof account.name === "string"
        && typeof account.description === "string";
    });
  const validLogGroups = Array.isArray(logGroups) && logGroups.every((item) => {
    const group = objectValue(item);
    return typeof group?.id === "string"
      && typeof group.name === "string"
      && typeof group.description === "string"
      && typeof group.isDefault === "boolean";
  });
  if (
    !root
    || !validTemplates
    || !validGpuOptions
    || !validQuotas
    || !validServiceAccounts
    || !validLogGroups
  ) {
    throw new ApiError("Настройки Cloud.ru имеют неизвестный формат", "invalid_response", 502);
  }
  return result as CloudOptions;
}

export async function createInference(
  input: CreateInferenceInput,
): Promise<CreateInferenceResult> {
  const result = await post<unknown>("/api/cloud/inferences", input);
  const root = objectValue(result);
  if (
    !root
    || typeof root.model_run_id !== "string"
    || !root.model_run_id.trim()
    || typeof root.name !== "string"
    || !root.name.trim()
    || typeof root.status !== "string"
    || !root.status.trim()
    || typeof root.endpoint !== "string"
  ) {
    throw new ApiError("Результат создания имеет неизвестный формат", "invalid_response", 502);
  }
  return result as CreateInferenceResult;
}
