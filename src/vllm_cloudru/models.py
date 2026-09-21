import re
from typing import Any, Literal, Self
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field, SecretStr, field_validator, model_validator


class CloudCredentials(BaseModel):
    model_config = ConfigDict(extra="forbid")

    project_id: str = Field(min_length=1)
    key_id: str = Field(min_length=1)
    key_secret: SecretStr = Field(min_length=1)
    hf_secret_id: str | None = Field(default=None, min_length=1)

    @field_validator("project_id", "key_id", "hf_secret_id", mode="before")
    @classmethod
    def strip_ids(cls, value: Any) -> Any:
        return value.strip() if isinstance(value, str) else value

    @field_validator("key_secret")
    @classmethod
    def reject_blank_key_secret(cls, value: SecretStr) -> SecretStr:
        if not value.get_secret_value().strip():
            raise ValueError("key_secret не может состоять только из пробелов")
        return value


class RecipeConfigurationSelection(BaseModel):
    model_config = ConfigDict(extra="forbid")

    variant: str | None = Field(default=None, pattern=r"^[a-zA-Z0-9_.-]+$")
    strategy: str | None = Field(default=None, pattern=r"^[a-zA-Z0-9_.-]+$")
    kv_offload: str | None = Field(default=None, pattern=r"^[a-zA-Z0-9_.-]+$")
    frontend: Literal["rust", "python"] | None = None
    features: list[str] | None = Field(default=None, max_length=32)
    feature_modes: dict[str, str] = Field(default_factory=dict)

    @field_validator("features")
    @classmethod
    def validate_features(cls, values: list[str] | None) -> list[str] | None:
        if values is None:
            return None
        if len(values) != len(set(values)):
            raise ValueError("features не должны повторяться")
        if any(not re.fullmatch(r"[a-zA-Z0-9_.-]+", value) for value in values):
            raise ValueError("features содержат некорректный ключ")
        return values

    @field_validator("feature_modes")
    @classmethod
    def validate_feature_modes(cls, values: dict[str, str]) -> dict[str, str]:
        pattern = re.compile(r"^[a-zA-Z0-9_.-]+$")
        if len(values) > 32 or any(
            not pattern.fullmatch(key) or not pattern.fullmatch(value)
            for key, value in values.items()
        ):
            raise ValueError("feature_modes содержат некорректный ключ")
        return values


AuthenticationType = Literal[
    "AUTHENTICATION_TYPE_KEY_BASED",
    "AUTHENTICATION_TYPE_TOKEN_BASED",
]


class CreateInferenceRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    credentials: CloudCredentials
    name: str = Field(min_length=1, max_length=63)
    hf_id: str = Field(pattern=r"^[^/\s]+/[^/\s]+$")
    resource_type: str = Field(min_length=1)
    gpu_count: int = Field(gt=0)
    runtime_template_id: str = Field(min_length=1)
    min_scale: int = Field(ge=0)
    max_scale: int = Field(ge=1)
    service_account_id: UUID
    authentication_type: AuthenticationType
    log_group_id: UUID
    gpu_memory_utilization: float = Field(default=0.97, ge=0.85, le=0.99)
    excluded_argument_keys: list[str] = Field(default_factory=list, max_length=64)
    configuration: RecipeConfigurationSelection = Field(
        default_factory=RecipeConfigurationSelection
    )
    confirm_warnings: bool = False

    @field_validator("name", "hf_id", "resource_type", "runtime_template_id", mode="before")
    @classmethod
    def strip_display_ids(cls, value: Any) -> Any:
        return value.strip() if isinstance(value, str) else value

    @field_validator("excluded_argument_keys")
    @classmethod
    def validate_excluded_argument_keys(cls, values: list[str]) -> list[str]:
        if len(values) != len(set(values)) or any(
            not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,127}", value)
            for value in values
        ):
            raise ValueError("excluded_argument_keys содержат некорректный ключ")
        return values

    @model_validator(mode="after")
    def validate_scale_range(self) -> Self:
        if self.min_scale > self.max_scale:
            raise ValueError("min_scale не может превышать max_scale")
        return self


class CreateInferenceResult(BaseModel):
    model_run_id: str
    name: str
    status: str
    endpoint: str


class ModelRunScaling(BaseModel):
    model_config = ConfigDict(populate_by_name=True, serialize_by_alias=True, extra="forbid")

    min_scale: int | None = Field(alias="minScale")
    max_scale: int | None = Field(alias="maxScale")


class ModelRunSummary(BaseModel):
    model_config = ConfigDict(populate_by_name=True, serialize_by_alias=True, extra="forbid")

    model_run_id: str = Field(alias="modelRunId")
    name: str
    model: str
    status: str
    resource_type: str = Field(alias="resourceType")
    gpu_count: int | None = Field(alias="gpuCount")
    scaling: ModelRunScaling
    endpoint: str


class RuntimeTemplateGpu(BaseModel):
    model_config = ConfigDict(populate_by_name=True, serialize_by_alias=True, extra="forbid")

    resource_type: str = Field(alias="resourceType")
    is_default: bool = Field(alias="isDefault")


class RuntimeTemplate(BaseModel):
    model_config = ConfigDict(populate_by_name=True, serialize_by_alias=True, extra="forbid")

    id: str
    framework_type: str = Field(alias="frameworkType")
    version: str
    gpus: list[RuntimeTemplateGpu]


class GpuOption(BaseModel):
    model_config = ConfigDict(populate_by_name=True, serialize_by_alias=True, extra="forbid")

    label: str
    memory_gb: int = Field(alias="memoryGb")
    counts: list[int]


class QuotaItem(BaseModel):
    model_config = ConfigDict(populate_by_name=True, serialize_by_alias=True, extra="forbid")

    resource_type: str = Field(alias="resourceType")
    free: int | float
    limit: int | float


class ServiceAccountOption(BaseModel):
    model_config = ConfigDict(extra="forbid")

    id: str
    name: str
    description: str


class LogGroupOption(BaseModel):
    model_config = ConfigDict(populate_by_name=True, serialize_by_alias=True, extra="forbid")

    id: str
    name: str
    description: str
    is_default: bool = Field(alias="isDefault")


class CloudOptions(BaseModel):
    model_config = ConfigDict(populate_by_name=True, serialize_by_alias=True, extra="forbid")

    runtime_templates: list[RuntimeTemplate] = Field(alias="runtimeTemplates")
    gpu_options: dict[str, GpuOption] = Field(alias="gpuOptions")
    quotas: list[QuotaItem]
    service_accounts: list[ServiceAccountOption] = Field(alias="serviceAccounts")
    log_groups: list[LogGroupOption] = Field(alias="logGroups")


class TranslatedRecipe(BaseModel):
    model_id: str
    arguments: list[dict[str, Any]]
    environment: list[dict[str, Any]]
    recommended_gpu_count: int | None = None
    recommended_hardware: str | None = None


class RecipeConfigurationOption(BaseModel):
    model_config = ConfigDict(extra="forbid")

    key: str
    label: str
    description: str | None = None
    available: bool = True
    unavailable_reason: str | None = None
    selected: bool = False


class RecipeConfigurationGroup(BaseModel):
    model_config = ConfigDict(extra="forbid")

    key: str
    label: str
    control: Literal["single", "multiple", "display"] = "single"
    options: list[RecipeConfigurationOption] = Field(default_factory=list)


class ResolvedRecipeConfiguration(BaseModel):
    model_config = ConfigDict(extra="forbid")

    selection: RecipeConfigurationSelection
    groups: list[RecipeConfigurationGroup] = Field(default_factory=list)


class RecipeSummary(BaseModel):
    model_config = ConfigDict(populate_by_name=True, serialize_by_alias=True, frozen=True)

    hf_id: str = Field(pattern=r"^[^/\s]+/[^/\s]+$")
    title: str
    provider: str
    url: str
    json_path: str = Field(alias="json")


class RecipeCatalogItem(RecipeSummary):
    description: str | None = None
    tasks: list[str] = Field(default_factory=list)
    parameter_count: str | None = None
    variants: dict[str, Any] = Field(default_factory=dict)
    recommended_hardware: str | None = None
    recommended_gpu_count: int | None = None
    compatibility_warnings: list[str] = Field(default_factory=list)
    launchable: bool
    launch_reason: str | None = None


class RecipeDetail(BaseModel):
    model_config = ConfigDict(extra="forbid")

    hf_id: str = Field(pattern=r"^[^/\s]+/[^/\s]+$")
    title: str
    provider: str
    description: str | None = None
    tasks: list[str] = Field(default_factory=list)
    parameter_count: str | None = None
    active_parameters: str | None = None
    context_length: int | None = None
    min_vllm_version: str | None = None
    variants: dict[str, Any] = Field(default_factory=dict)
    recommended_hardware: str | None = None
    recommended_gpu_count: int | None = None
    selected_variant: str | None = None
    rendering_path: str
    rendering: dict[str, Any]
    translated: TranslatedRecipe | None = None
    configuration: ResolvedRecipeConfiguration | None = None
    launchable: bool
    launch_reason: str | None = None
    compatibility_warnings: list[str] = Field(default_factory=list)
    configuration_adjustments: list[str] = Field(default_factory=list)
