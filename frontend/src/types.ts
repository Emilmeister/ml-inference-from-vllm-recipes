export type CredentialInput = {
  projectId: string;
  keyId: string;
  keySecret: string;
  hfSecretId?: string;
};

export type CloudCredentialsPayload = {
  project_id: string;
  key_id: string;
  key_secret: string;
  hf_secret_id?: string;
};

export type RecipeVariant = {
  precision?: string;
  vram_minimum_gb?: number;
  description?: string;
};

export type RecipeCatalogItem = {
  hf_id: string;
  title: string;
  provider: string;
  url: string;
  json: string;
  description?: string | null;
  tasks: string[];
  parameter_count?: string | null;
  variants: Record<string, RecipeVariant>;
  recommended_hardware?: string | null;
  recommended_gpu_count?: number | null;
  compatibility_warnings: string[];
  launchable: boolean;
  launch_reason?: string | null;
};

export type VllmArgument = {
  key: string;
  value: string | number | boolean;
  parameterType: string;
};

export type RecipeConfigurationSelection = {
  variant?: string | null;
  strategy?: string | null;
  kv_offload?: string | null;
  frontend?: "rust" | "python" | null;
  features?: string[] | null;
  feature_modes: Record<string, string>;
};

export type RecipeConfigurationOption = {
  key: string;
  label: string;
  description?: string | null;
  available: boolean;
  unavailable_reason?: string | null;
  selected: boolean;
};

export type RecipeConfigurationGroup = {
  key: string;
  label: string;
  control: "single" | "multiple" | "display";
  options: RecipeConfigurationOption[];
};

export type RecipeDetail = RecipeCatalogItem & {
  active_parameters?: string | null;
  context_length?: number | null;
  min_vllm_version?: string | null;
  selected_variant?: string | null;
  rendering_path: string;
  rendering: Record<string, unknown>;
  translated?: {
    model_id: string;
    arguments: VllmArgument[];
    environment: VllmArgument[];
    recommended_gpu_count?: number | null;
    recommended_hardware?: string | null;
  } | null;
  configuration?: {
    selection: RecipeConfigurationSelection;
    groups: RecipeConfigurationGroup[];
  } | null;
};

export type ModelRunSummary = {
  modelRunId: string;
  name: string;
  model: string;
  status: string;
  resourceType: string;
  gpuCount: number | null;
  scaling: {
    minScale: number | null;
    maxScale: number | null;
  };
  endpoint: string;
};

export type CloudOptions = {
  runtimeTemplates: Array<{
    id: string;
    frameworkType: string;
    version: string;
    gpus: Array<{ resourceType: string; isDefault: boolean }>;
  }>;
  gpuOptions: Record<string, { label: string; memoryGb: number; counts: number[] }>;
  quotas: Array<{ resourceType: string; free: number; limit: number }>;
  serviceAccounts: Array<{ id: string; name: string; description: string }>;
  logGroups: Array<{ id: string; name: string; description: string; isDefault: boolean }>;
};

export type CreateInferenceInput = {
  name: string;
  hf_id: string;
  resource_type: string;
  gpu_count: number;
  runtime_template_id: string;
  min_scale: number;
  max_scale: number;
  service_account_id: string;
  authentication_type:
    | "AUTHENTICATION_TYPE_KEY_BASED"
    | "AUTHENTICATION_TYPE_TOKEN_BASED";
  log_group_id: string;
  configuration: RecipeConfigurationSelection;
  confirm_warnings: boolean;
};

export type CreateInferenceResult = {
  model_run_id: string;
  name: string;
  status: string;
  endpoint: string;
};
