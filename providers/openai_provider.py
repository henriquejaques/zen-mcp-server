"""OpenAI model provider implementation."""

import logging
from typing import Optional

from .base import (
    ModelCapabilities,
    ModelResponse,
    ProviderType,
    create_temperature_constraint,
)
from .openai_compatible import OpenAICompatibleProvider

logger = logging.getLogger(__name__)


class OpenAIModelProvider(OpenAICompatibleProvider):
    """Official OpenAI API provider (api.openai.com)."""

    # Model configurations using ModelCapabilities objects
    SUPPORTED_MODELS = {
        "o3": ModelCapabilities(
            provider=ProviderType.OPENAI,
            model_name="o3",
            friendly_name="OpenAI (O3)",
            context_window=200_000,  # 200K tokens
            max_output_tokens=100_000,  # Updated: 100K max output tokens
            supports_extended_thinking=False,
            supports_system_prompts=True,
            supports_streaming=True,  # Confirmed: Supported
            supports_function_calling=True,  # Confirmed: Supported
            supports_json_mode=True,  # Structured outputs supported
            supports_images=True,  # Confirmed: Image input supported
            max_image_size_mb=20.0,  # Standard OpenAI limit
            supports_temperature=False,  # O3 models don't accept temperature parameter
            temperature_constraint=create_temperature_constraint("fixed"),
            description="Strong reasoning (200K context) - Logical problems, code generation, systematic analysis",
            aliases=[],
        ),
        "o3-mini": ModelCapabilities(
            provider=ProviderType.OPENAI,
            model_name="o3-mini",
            friendly_name="OpenAI (O3-mini)",
            context_window=200_000,  # 200K tokens
            max_output_tokens=100_000,  # Updated: 100K max output tokens
            supports_extended_thinking=False,
            supports_system_prompts=True,
            supports_streaming=True,  # Confirmed: Supported
            supports_function_calling=True,  # Confirmed: Supported
            supports_json_mode=True,  # Structured outputs supported
            supports_images=False,  # Confirmed: Image NOT supported
            max_image_size_mb=None,  # No image support
            supports_temperature=False,  # O3 models don't accept temperature parameter
            temperature_constraint=create_temperature_constraint("fixed"),
            description="Fast O3 variant (200K context) - Balanced performance/speed, moderate complexity",
            aliases=["o3mini", "o3-mini"],
        ),
        "o3-pro-2025-06-10": ModelCapabilities(
            provider=ProviderType.OPENAI,
            model_name="o3-pro-2025-06-10",
            friendly_name="OpenAI (O3-Pro)",
            context_window=200_000,  # 200K tokens
            max_output_tokens=100_000,  # Updated: 100K max output tokens
            supports_extended_thinking=False,
            supports_system_prompts=True,
            supports_streaming=False,  # Confirmed: Not supported
            supports_function_calling=True,  # Confirmed: Supported
            supports_json_mode=True,  # Structured outputs supported
            supports_images=True,  # Image input supported
            max_image_size_mb=20.0,  # Standard OpenAI limit
            supports_temperature=False,  # O3 models don't accept temperature parameter
            temperature_constraint=create_temperature_constraint("fixed"),
            description="Professional-grade reasoning (200K context) - EXTREMELY EXPENSIVE: Only for the most complex problems requiring universe-scale complexity analysis OR when the user explicitly asks for this model. Use sparingly for critical architectural decisions or exceptionally complex debugging that other models cannot handle.",
            aliases=["o3-pro"],
        ),
        "o4-mini": ModelCapabilities(
            provider=ProviderType.OPENAI,
            model_name="o4-mini",
            friendly_name="OpenAI (O4-mini)",
            context_window=200_000,  # 200K tokens
            max_output_tokens=100_000,  # Updated: 100K max output tokens
            supports_extended_thinking=False,
            supports_system_prompts=True,
            supports_streaming=True,  # Confirmed: Supported
            supports_function_calling=True,  # Confirmed: Supported
            supports_json_mode=True,  # Structured outputs supported
            supports_images=True,  # Confirmed: Image input supported
            max_image_size_mb=20.0,  # Standard OpenAI limit
            supports_temperature=False,  # O4 models don't accept temperature parameter
            temperature_constraint=create_temperature_constraint("fixed"),
            description="Latest reasoning model (200K context) - Optimized for shorter contexts, rapid reasoning",
            aliases=["mini", "o4mini", "o4-mini"],
        ),
        "o4-mini-deep-research": ModelCapabilities(
            provider=ProviderType.OPENAI,
            model_name="o4-mini-deep-research-2025-06-26",
            friendly_name="OpenAI (O4-mini-deep-research)",
            context_window=200_000,  # 200K tokens
            max_output_tokens=100_000,  # Confirmed: 100K max output tokens
            supports_extended_thinking=False,
            supports_system_prompts=True,
            supports_streaming=True,  # Confirmed: Supported
            supports_function_calling=True,  # Research model with function calling
            supports_json_mode=True,  # Structured outputs supported
            supports_images=True,  # Image input supported
            max_image_size_mb=20.0,  # Standard OpenAI limit
            supports_temperature=False,  # O4 models don't accept temperature parameter
            temperature_constraint=create_temperature_constraint("fixed"),
            description="Faster, more affordable deep research model (200K context) - Specialized for complex, multi-step research tasks with web search and data synthesis via MCP connectors",
            aliases=["o4-deep", "o4-mini-deep"],
        ),
        "gpt-4.1-2025-04-14": ModelCapabilities(
            provider=ProviderType.OPENAI,
            model_name="gpt-4.1-2025-04-14",
            friendly_name="OpenAI (GPT 4.1)",
            context_window=1_047_576,  # Updated: 1,047,576 tokens as shown in docs
            max_output_tokens=32_768,  # Confirmed: 32,768 max output tokens
            supports_extended_thinking=False,
            supports_system_prompts=True,
            supports_streaming=True,  # Confirmed in features
            supports_function_calling=True,  # Confirmed in features
            supports_json_mode=True,  # Structured outputs supported
            supports_images=True,  # Image input supported
            max_image_size_mb=20.0,  # Standard OpenAI limit
            supports_temperature=True,  # Regular models accept temperature parameter
            temperature_constraint=create_temperature_constraint("range"),
            description="Flagship GPT model for complex tasks (1M+ context) - Advanced reasoning with multimodal capabilities, web search, and tool use",
            aliases=["gpt4.1", "gpt-4.1"],
        ),
    }

    def __init__(self, api_key: str, **kwargs):
        """Initialize OpenAI provider with API key."""
        # Set default OpenAI base URL, allow override for regions/custom endpoints
        kwargs.setdefault("base_url", "https://api.openai.com/v1")
        super().__init__(api_key, **kwargs)

    def get_capabilities(self, model_name: str) -> ModelCapabilities:
        """Get capabilities for a specific OpenAI model."""
        # Resolve shorthand
        resolved_name = self._resolve_model_name(model_name)

        if resolved_name not in self.SUPPORTED_MODELS:
            raise ValueError(f"Unsupported OpenAI model: {model_name}")

        # Check if model is allowed by restrictions
        from utils.model_restrictions import get_restriction_service

        restriction_service = get_restriction_service()
        if not restriction_service.is_allowed(
            ProviderType.OPENAI, resolved_name, model_name
        ):
            raise ValueError(
                f"OpenAI model '{model_name}' is not allowed by restriction policy."
            )

        # Return the ModelCapabilities object directly from SUPPORTED_MODELS
        return self.SUPPORTED_MODELS[resolved_name]

    def get_provider_type(self) -> ProviderType:
        """Get the provider type."""
        return ProviderType.OPENAI

    def validate_model_name(self, model_name: str) -> bool:
        """Validate if the model name is supported and allowed."""
        resolved_name = self._resolve_model_name(model_name)

        # First check if model is supported
        if resolved_name not in self.SUPPORTED_MODELS:
            return False

        # Then check if model is allowed by restrictions
        from utils.model_restrictions import get_restriction_service

        restriction_service = get_restriction_service()
        if not restriction_service.is_allowed(
            ProviderType.OPENAI, resolved_name, model_name
        ):
            logger.debug(
                f"OpenAI model '{model_name}' -> '{resolved_name}' blocked by restrictions"
            )
            return False

        return True

    def generate_content(
        self,
        prompt: str,
        model_name: str,
        system_prompt: Optional[str] = None,
        temperature: float = 0.7,
        max_output_tokens: Optional[int] = None,
        **kwargs,
    ) -> ModelResponse:
        """Generate content using OpenAI API with proper model name resolution."""
        # Resolve model alias before making API call
        resolved_model_name = self._resolve_model_name(model_name)

        # Call parent implementation with resolved model name
        return super().generate_content(
            prompt=prompt,
            model_name=resolved_model_name,
            system_prompt=system_prompt,
            temperature=temperature,
            max_output_tokens=max_output_tokens,
            **kwargs,
        )

    def supports_thinking_mode(self, model_name: str) -> bool:
        """Check if the model supports extended thinking mode."""
        # Currently no OpenAI models support extended thinking
        # This may change with future O3 models
        return False
