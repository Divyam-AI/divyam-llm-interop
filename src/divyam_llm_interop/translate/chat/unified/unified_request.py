# Copyright 2025 Divyam.ai
# SPDX-License-Identifier: Apache-2.0

from dataclasses import dataclass, fields, field
from typing import List, Optional, Dict, Union, Any

from divyam_llm_interop.translate.chat.jsonschema.types import JSONSchema


@dataclass
class UnifiedFunctionCall:
    """Represents a function to be called."""

    name: str
    arguments: str
    unknowns: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert UnifiedFunctionCall to dictionary dynamically."""
        result = {}
        for f in fields(self):
            if f.name == "unknowns":
                # Skip unknowns
                continue
            value = getattr(self, f.name)
            result[f.name] = value

        result.update(self.unknowns)
        return result

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "UnifiedFunctionCall":
        """Create UnifiedFunctionCall from dictionary dynamically."""
        # Get all declared field names except `unknowns`
        declared_fields = {f.name for f in fields(cls) if f.name != "unknowns"}
        unknowns = {k: v for k, v in data.items() if k not in declared_fields}

        return cls(
            name=str(data.get("name")),
            arguments=str(data.get("arguments")) if "arguments" in data else "",
            unknowns=unknowns,
        )


@dataclass
class UnifiedToolCall:
    """Represents a tool call requested by the assistant."""

    id: str
    function: UnifiedFunctionCall
    type: str = "function"
    unknowns: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert UnifiedToolCall to dictionary dynamically."""
        result = {}
        for f in fields(self):
            if f.name == "unknowns":
                # Skip unknowns
                continue
            value = getattr(self, f.name)
            if isinstance(value, UnifiedFunctionCall):
                result[f.name] = value.to_dict()
            else:
                result[f.name] = value
        result.update(self.unknowns)
        return result

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "UnifiedToolCall":
        """Create UnifiedToolCall from dictionary dynamically."""
        init_kwargs = {}
        for f in fields(cls):
            if f.name not in data:
                continue
            if f.name == "function":
                init_kwargs[f.name] = UnifiedFunctionCall.from_dict(data[f.name])
            else:
                init_kwargs[f.name] = data[f.name]
        return cls(**init_kwargs)


@dataclass
class UnifiedMessage:
    """Represents a single message in the chat."""

    role: str  # 'system', 'user', 'tool', or 'assistant'
    content: Optional[str] = None

    # Optional: name of the user or system role
    name: Optional[str] = None
    tool_calls: Optional[List[UnifiedToolCall]] = None
    tool_call_id: Optional[str] = None

    # If the model refused to respond, this contains the refusal message
    refusal: Optional[str] = None

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "UnifiedMessage":
        """Create UnifiedMessage from dictionary."""
        tool_calls = None
        if data.get("tool_calls"):
            tool_calls = [UnifiedToolCall.from_dict(tc) for tc in data["tool_calls"]]

        return cls(
            role=data["role"],
            content=data["content"] if "content" in data else None,
            name=data.get("name"),
            tool_calls=tool_calls,
            tool_call_id=data.get("tool_call_id"),
            refusal=data.get("refusal"),
        )

    def to_dict(self) -> Dict[str, Any]:
        """Convert UnifiedMessage to dictionary."""
        result: Dict[str, Any] = {"role": self.role}

        # Only include optional fields if they have values
        if self.content is not None:
            result["content"] = self.content
        if self.name is not None:
            result["name"] = self.name
        if self.tool_calls is not None:
            result["tool_calls"] = [tc.to_dict() for tc in self.tool_calls]
        if self.tool_call_id is not None:
            result["tool_call_id"] = self.tool_call_id
        if self.refusal is not None:
            result["refusal"] = self.refusal

        return result


@dataclass
class UnifiedFunction:
    """Represents a function that can be called by the model."""

    name: str  # Name of the function
    description: str  # Description of what the function does
    parameters: JSONSchema  # Function parameters
    unknowns: Dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "UnifiedFunction":
        """Create UnifiedFunction from dictionary."""
        # Get all declared field names except `unknowns`
        declared_fields = {f.name for f in fields(cls) if f.name != "unknowns"}
        unknowns = {k: v for k, v in data.items() if k not in declared_fields}

        parameters = JSONSchema.from_dict(data.get("parameters", {}))
        return cls(
            name=data["name"],
            description=data["description"],
            parameters=parameters,
            unknowns=unknowns,
        )

    def to_dict(self) -> Dict[str, Any]:
        """Convert UnifiedFunction to dictionary."""
        result: Dict[str, Any] = {
            "name": self.name,
            "description": self.description,
            "parameters": self.parameters.to_dict(),
        }
        result.update(self.unknowns)
        return result


@dataclass
class UnifiedTool:
    """Represents a tool that can be called by the model (modern replacement for functions)."""

    function: UnifiedFunction

    # Always "function" for now
    type: str = "function"

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "UnifiedTool":
        """Create UnifiedTool from dictionary."""
        return cls(
            type=data.get("type", "function"),
            function=UnifiedFunction.from_dict(data["function"]),
        )

    def to_dict(self) -> Dict[str, Any]:
        """Convert UnifiedTool to dictionary."""
        return {"type": self.type, "function": self.function.to_dict()}


@dataclass
class UnifiedResponseFormatJsonSchema:
    """Represents the response format JSON schema specification."""

    name: str
    schema: "JSONSchema"  # Assuming JSONSchema has from_dict and to_dict methods
    unknowns: Dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "UnifiedResponseFormatJsonSchema":
        # Get all declared field names except `unknowns`
        declared_fields = {f.name for f in fields(cls) if f.name != "unknowns"}
        unknowns = {k: v for k, v in data.items() if k not in declared_fields}

        return cls(
            name=data["name"],
            schema=(
                data["schema"]
                if isinstance(data["schema"], JSONSchema)
                else JSONSchema.from_dict(data["schema"])
            ),
            unknowns=unknowns,
        )

    def to_dict(self) -> Dict[str, Any]:
        result: Dict[str, Any] = {
            "name": self.name,
            "schema": (
                self.schema.to_dict()
                if hasattr(self.schema, "to_dict")
                else self.schema
            ),
        }
        result.update(self.unknowns)
        return result


@dataclass
class UnifiedResponseFormat:
    """Represents the response format specification."""

    type: str  # "text", "json_object", "json_schema"
    json_schema: Optional[UnifiedResponseFormatJsonSchema] = None
    unknowns: Dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "UnifiedResponseFormat":
        declared_fields = {f.name for f in fields(cls) if f.name != "unknowns"}
        unknowns = {k: v for k, v in data.items() if k not in declared_fields}

        return cls(
            type=data["type"],
            json_schema=(
                UnifiedResponseFormatJsonSchema.from_dict(data["json_schema"])
                if data.get("json_schema") is not None
                else None
            ),
            unknowns=unknowns,
        )

    def to_dict(self) -> Dict[str, Any]:
        result: Dict[str, Any] = {"type": self.type}
        if self.json_schema is not None:
            result["json_schema"] = self.json_schema.to_dict()
        result.update(self.unknowns)
        return result


@dataclass
class UnifiedAudioConfig:
    """Represents audio configuration."""

    voice: str  # Voice to use for audio output
    format: str  # Audio format (e.g., "mp3", "opus", "aac", "flac")

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "UnifiedAudioConfig":
        """Create UnifiedAudioConfig from dictionary."""
        return cls(voice=data["voice"], format=data["format"])

    def to_dict(self) -> Dict[str, Any]:
        """Convert UnifiedAudioConfig to dictionary."""
        return {"voice": self.voice, "format": self.format}


@dataclass
class UnifiedReasoning:
    """
    Configuration options for reasoning models.
    Only supported in gpt-5 and o-series models.
    """

    effort: Optional[str] = None
    generate_summary: Optional[str] = None  # Deprecated, use summary instead
    summary: Optional[str] = None

    # Placeholder for unknown fields
    unknowns: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        data: Dict[str, Any] = {}

        if self.effort is not None:
            data["effort"] = self.effort
        if self.generate_summary is not None:
            data["generate_summary"] = self.generate_summary
        if self.summary is not None:
            data["summary"] = self.summary

        # Merge unknowns back
        data.update(self.unknowns)
        return data

    @classmethod
    def from_dict(cls, obj: Dict[str, Any]) -> "UnifiedReasoning":
        known_fields = {"effort", "generate_summary", "summary"}
        unknowns = {k: v for k, v in obj.items() if k not in known_fields}

        return cls(
            effort=obj.get("effort"),
            generate_summary=obj.get("generate_summary"),
            summary=obj.get("summary"),
            unknowns=unknowns,
        )


@dataclass
class UnifiedChatCompletionsRequestBody:
    """Represents the request body for unified completions API with function calling.
    This is based on OpenAI and will be used as a base to convert between models.

    To convert parameters of model A -> model B we do the following:
    model A -> unified model -> model B

    That way we just need to encode translation rules to and from each model
    instead of pairwise translation.
    """

    # Model name (e.g., "gpt-4", "gpt-3.5-turbo")
    model: str

    # List of messages exchanged in the chat
    messages: List[UnifiedMessage]

    # Controls randomness (0.0 to 2.0)
    temperature: Optional[float] = None

    # Nucleus sampling probability (0.0 to 1.0)
    top_p: Optional[float] = None

    # Number of completions to generate
    n: Optional[int] = None

    # Whether to stream responses
    stream: Optional[bool] = None

    # List of stop sequences (up to 4 sequences)
    stop: Optional[Union[str, List[str]]] = None

    # Max number of tokens to generate (deprecated, use max_completion_tokens)
    max_tokens: Optional[int] = None

    # Max number of completion tokens to generate (preferred over max_tokens)
    max_completion_tokens: Optional[int] = None

    # Penalty for repeating phrases (-2.0 to 2.0)
    presence_penalty: Optional[float] = None

    # Penalty for frequent tokens (-2.0 to 2.0)
    frequency_penalty: Optional[float] = None

    # Modify likelihood of specific tokens
    logit_bias: Optional[Dict[str, int]] = None

    # Optional: unique user identifier
    user: Optional[str] = None

    # The number of logprobs to return (0-20)
    logprobs: Optional[int] = None

    # Number of most likely tokens to return at each position (0-20)
    top_logprobs: Optional[int] = None

    # Modern tool calling (replaces functions)
    tools: Optional[List[UnifiedTool]] = None

    # Tool choice specification (replaces function_call)
    tool_choice: Optional[Union[str, Dict[str, Any]]] = None

    # Whether to enable parallel function calling
    parallel_tool_calls: Optional[bool] = None

    # Response format specification
    response_format: Optional[UnifiedResponseFormat] = None

    # Seed for deterministic outputs
    seed: Optional[int] = None

    # Service tier selection
    service_tier: Optional[str] = None

    # Input/output modalities
    modalities: Optional[List[str]] = None

    # Predicted outputs for reduced latency
    prediction: Optional[Dict[str, Any]] = None

    # Audio configuration
    audio: Optional[UnifiedAudioConfig] = None

    # Whether to store the conversation
    store: Optional[bool] = None

    # Additional metadata
    metadata: Optional[Dict[str, Any]] = None

    # Reasoning
    reasoning: Optional[UnifiedReasoning] = None

    # Reasoning effort. Not supported by all models
    reasoning_effort: Optional[str] = None

    # System fingerprint
    system_fingerprint: Optional[str] = None

    # Legacy fields (kept from original - echo and best_of are from Completions API)
    echo: Optional[bool] = None
    best_of: Optional[int] = None
    functions: Optional[List[UnifiedFunction]] = None
    function_call: Optional[Union[str, Dict[str, str]]] = None

    # Values that do not have fields
    unknowns: Dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "UnifiedChatCompletionsRequestBody":
        """Create UnifiedChatCompletionsRequest from dictionary."""

        # Preserve unknown fields
        declared_fields = {f.name for f in fields(cls) if f.name != "unknowns"}
        unknowns = {k: v for k, v in data.items() if k not in declared_fields}

        # Parse messages
        messages = [UnifiedMessage.from_dict(msg) for msg in data["messages"]]

        # Parse tools
        tools = None
        if data.get("tools"):
            tools = [UnifiedTool.from_dict(tool) for tool in data["tools"]]

        # Parse functions (legacy)
        functions = None
        if data.get("functions"):
            functions = [UnifiedFunction.from_dict(func) for func in data["functions"]]

        # Parse response format
        response_format = None
        if data.get("response_format"):
            response_format = UnifiedResponseFormat.from_dict(data["response_format"])

        # Parse audio config
        audio = None
        if data.get("audio"):
            audio = UnifiedAudioConfig.from_dict(data["audio"])

        # Handle legacy max_tokens
        max_tokens = None
        if data.get("max_tokens"):
            max_tokens = data["max_tokens"]
        if data.get("max_output_tokens"):
            max_tokens = data["max_output_tokens"]

        return cls(
            model=data["model"],
            messages=messages,
            temperature=data.get("temperature"),
            top_p=data.get("top_p"),
            n=data.get("n"),
            stream=data.get("stream"),
            stop=data.get("stop"),
            max_tokens=max_tokens,
            max_completion_tokens=data.get("max_completion_tokens"),
            presence_penalty=data.get("presence_penalty"),
            frequency_penalty=data.get("frequency_penalty"),
            logit_bias=data.get("logit_bias"),
            user=data.get("user"),
            logprobs=data.get("logprobs"),
            top_logprobs=data.get("top_logprobs"),
            tools=tools,
            tool_choice=data.get("tool_choice"),
            parallel_tool_calls=data.get("parallel_tool_calls"),
            response_format=response_format,
            seed=data.get("seed"),
            service_tier=data.get("service_tier"),
            modalities=data.get("modalities"),
            prediction=data.get("prediction"),
            audio=audio,
            store=data.get("store"),
            metadata=data.get("metadata"),
            echo=data.get("echo"),
            best_of=data.get("best_of"),
            functions=functions,
            function_call=data.get("function_call"),
            reasoning=data.get("reasoning"),
            reasoning_effort=data.get("reasoning_effort"),
            system_fingerprint=data.get("system_fingerprint"),
            unknowns=unknowns,
        )

    def to_dict(self, keep_unknowns: bool = False) -> Dict[str, Any]:
        """Convert UnifiedChatCompletionsRequest to dictionary."""
        result: Dict[str, Any] = {
            "model": self.model,
            "messages": [msg.to_dict() for msg in self.messages],
        }

        # Merge unknowns back
        if keep_unknowns:
            result.update(self.unknowns)

        # Add optional fields only if they have values
        optional_fields = {
            "temperature": self.temperature,
            "top_p": self.top_p,
            "n": self.n,
            "stream": self.stream,
            "stop": self.stop,
            "max_tokens": self.max_tokens,
            "max_completion_tokens": self.max_completion_tokens,
            "presence_penalty": self.presence_penalty,
            "frequency_penalty": self.frequency_penalty,
            "logit_bias": self.logit_bias,
            "user": self.user,
            "logprobs": self.logprobs,
            "top_logprobs": self.top_logprobs,
            "tool_choice": self.tool_choice,
            "parallel_tool_calls": self.parallel_tool_calls,
            "seed": self.seed,
            "service_tier": self.service_tier,
            "modalities": self.modalities,
            "prediction": self.prediction,
            "store": self.store,
            "metadata": self.metadata,
            "best_of": self.best_of,
            "function_call": self.function_call,
            "reasoning": self.reasoning,
            "reasoning_effort": self.reasoning_effort,
            "system_fingerprint": self.system_fingerprint,
        }

        for key, value in optional_fields.items():
            if value is not None:
                result[key] = value

        # Handle echo field (has default value)
        if self.echo is not None:
            result["echo"] = self.echo

        # Handle complex optional fields
        if self.tools is not None:
            result["tools"] = [tool.to_dict() for tool in self.tools]

        if self.functions is not None:
            result["functions"] = [func.to_dict() for func in self.functions]

        if self.response_format is not None:
            result["response_format"] = self.response_format.to_dict()

        if self.audio is not None:
            result["audio"] = self.audio.to_dict()

        return result


@dataclass
class UnifiedChatCompletionsRequest:
    body: UnifiedChatCompletionsRequestBody
    headers: Optional[Dict[str, str]] = None
    query_parameters: Optional[Dict[str, str]] = None
    path_parameters: Optional[Dict[str, str]] = None
