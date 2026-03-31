from functools import lru_cache
from typing import Any

from pydantic import Field, ValidationError, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

OPENAI_API_KEY_ENV_VAR = "OPENAI_API_KEY"
OPENAI_MODEL_NAME_ENV_VAR = "OPENAI_MODEL_NAME"
SERPER_API_KEY_ENV_VAR = "SERPER_API_KEY"
DEFAULT_OPENAI_MODEL_NAME = "gpt-4o-mini"

_REQUIRED_ENV_FIELDS = (
    ("openai_api_key", OPENAI_API_KEY_ENV_VAR),
    ("serper_api_key", SERPER_API_KEY_ENV_VAR),
)


class ConfigError(RuntimeError):
    pass


class Settings(BaseSettings):
    openai_api_key: str = Field(validation_alias=OPENAI_API_KEY_ENV_VAR)
    openai_model_name: str = Field(
        default=DEFAULT_OPENAI_MODEL_NAME,
        validation_alias=OPENAI_MODEL_NAME_ENV_VAR,
    )
    serper_api_key: str = Field(validation_alias=SERPER_API_KEY_ENV_VAR)

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        env_ignore_empty=True,
        extra="ignore",
    )

    @field_validator("openai_api_key", "serper_api_key", mode="before")
    @classmethod
    def _reject_blank_required_values(cls, value: Any) -> Any:
        if isinstance(value, str):
            value = value.strip()
            return value or None
        return value

    @field_validator("openai_model_name", mode="before")
    @classmethod
    def _normalize_model_name(cls, value: Any) -> Any:
        if isinstance(value, str):
            value = value.strip()
            return value or DEFAULT_OPENAI_MODEL_NAME
        return value


def _extract_missing_env_vars(error: ValidationError) -> list[str]:
    invalid_locations: set[str] = set()
    for item in error.errors():
        if location := item.get("loc"):
            invalid_locations.add(str(location[0]))

    return [
        env_var_name
        for field_name, env_var_name in _REQUIRED_ENV_FIELDS
        if field_name in invalid_locations or env_var_name in invalid_locations
    ]


def _format_missing_env_vars(missing_env_vars: list[str]) -> str:
    label = "environment variable" if len(missing_env_vars) == 1 else "environment variables"
    return f"Missing required {label}: {', '.join(missing_env_vars)}"


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    try:
        return Settings()
    except ValidationError as error:
        missing_env_vars = _extract_missing_env_vars(error)
        if missing_env_vars:
            raise ConfigError(_format_missing_env_vars(missing_env_vars)) from error
        raise ConfigError("Invalid application configuration.") from error
