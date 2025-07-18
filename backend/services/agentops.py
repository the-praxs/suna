"""
AgentOps integration service for tracking agent conversations and tool usage.
"""

import os
import json
import ast
from typing import Optional, Any, Dict
import agentops
from agentops import StatusCode, TraceContext, tracer
from agentops.semconv import SpanKind, SpanAttributes
from opentelemetry.trace.status import Status
from utils.logger import logger
from contextlib import asynccontextmanager
import contextvars
import time
import asyncio

# Global variables for AgentOps state
_initialized = False
_current_trace_context: Optional[TraceContext] = None

# Cache for conversation traces (thread_id -> TraceContext)
_conversation_traces: Dict[str, TraceContext] = {}

# Context variable for async trace propagation
agentops_trace_context: contextvars.ContextVar[Optional[TraceContext]] = contextvars.ContextVar(
    'agentops_trace_context', 
    default=None
)


def initialize_agentops():
    """Initialize AgentOps SDK with configuration from environment variables."""
    global _initialized
    
    if _initialized:
        logger.debug("AgentOps already initialized")
        return
    
    api_key = os.getenv("AGENTOPS_API_KEY")
    if not api_key:
        logger.warning("AGENTOPS_API_KEY not found in environment variables. AgentOps will not be initialized.")
        return
    
    log_level = os.getenv("AGENTOPS_LOG_LEVEL", "INFO")
    
    try:
        agentops.init(
            api_key=api_key,
            log_level=log_level,
            instrument_llm_calls=False, # Disable automatic LLM instrumentation - we handle it manually
            auto_start_session=False,   # We'll manage traces manually
            fail_safe=True,             # Don't crash if AgentOps has issues
        )
        _initialized = True
        logger.info("AgentOps initialized successfully")
    except Exception as e:
        logger.error(f"Failed to initialize AgentOps: {str(e)}")
        _initialized = False


def start_agent_trace(
    agent_run_id: str,
    thread_id: str,
    project_id: str,
    model_name: str,
    agent_config: Optional[Dict[str, Any]] = None
) -> Optional[TraceContext]:
    """
    Start an agent run span within the conversation trace.
    
    Args:
        agent_run_id: Unique identifier for the agent run
        thread_id: Thread ID for the conversation
        project_id: Project ID
        model_name: Name of the LLM model being used
        agent_config: Optional agent configuration
        
    Returns:
        TraceContext object if successful, None otherwise
    """
    global _current_trace_context
    
    if not _initialized:
        logger.debug("AgentOps not initialized, skipping trace creation")
        return None
    
    try:
        # Get or create conversation trace
        conversation_trace = get_or_create_conversation_trace(thread_id, project_id, model_name)
        if not conversation_trace:
            logger.error("Failed to get conversation trace")
            return None
        
        # Set the conversation trace as current context
        _current_trace_context = conversation_trace
        agentops_trace_context.set(conversation_trace)
        
        # Create agent run span within the conversation trace
        logger.debug(f"Tracer initialized: {tracer.initialized}")
        
        if tracer.initialized:
            # Create span attributes
            attributes = {
                SpanAttributes.AGENTOPS_SPAN_KIND: SpanKind.AGENT,
                "agent_run_id": agent_run_id,
                "thread_id": thread_id,
                "project_id": project_id,
                "model": model_name,
            }
            
            # Add agent info if available
            if agent_config:
                attributes["agent_name"] = agent_config.get("name", "default")
                attributes["agent_id"] = agent_config.get("agent_id", "unknown")
            
            # Start agent run span
            span, _, _ = tracer.make_span(
                f"agent_run.{agent_run_id}",
                SpanKind.AGENT,
                attributes=attributes
            )
            
            # Store the span in the trace context for later access
            if hasattr(conversation_trace, '_agent_run_span'):
                # End previous span if exists
                if conversation_trace._agent_run_span:
                    conversation_trace._agent_run_span.end()
            conversation_trace._agent_run_span = span
            
            logger.info(f"Started agent run span: {agent_run_id} within conversation {thread_id}")
        
        return conversation_trace
        
    except Exception as e:
        logger.error(f"Failed to start agent run span: {str(e)}")
        return None


class ToolSpanContext:
    """Context for tool execution spans."""
    
    def __init__(self, span, start_time):
        self.span = span
        self.start_time = start_time
        
    def record_result(self, result):
        """Record the tool result to the span."""
        try:
            if hasattr(result, 'success'):
                self.span.set_attribute("tool.success", result.success)
            if hasattr(result, 'output'):
                # Limit output size to avoid huge spans
                output_str = str(result.output)[:5000]
                self.span.set_attribute("tool.output", output_str)
            
            # Add timing
            duration_ms = (time.time() - self.start_time) * 1000
            self.span.set_attribute("tool.duration_ms", duration_ms)
        except Exception as e:
            logger.error(f"Failed to record tool result: {e}")
            
    def record_error(self, error):
        """Record an error that occurred during tool execution."""
        try:
            self.span.set_status(Status(StatusCode.ERROR, str(error)))
            self.span.record_exception(error)
            self.span.set_attribute("error.message", str(error))
            self.span.set_attribute("tool.success", False)
        except Exception as e:
            logger.error(f"Failed to record tool error: {e}")


class LLMSpanContext:
    """Context for LLM call spans."""
    
    def __init__(self, span, start_time, manual_end=False):
        self.span = span
        self.start_time = start_time
        self.manual_end = manual_end
        self._ended = False
        
    def record_response(self, response):
        """Record the LLM response to the span."""
        try:
            # Update span with response data following semantic conventions
            if hasattr(response, "choices") and response.choices:
                choice = response.choices[0]
                
                # Response model (might be different from request)
                if hasattr(response, "model"):
                    self.span.set_attribute(SpanAttributes.LLM_RESPONSE_MODEL, response.model)
                
                # Finish reason
                if hasattr(choice, "finish_reason"):
                    self.span.set_attribute(SpanAttributes.LLM_RESPONSE_FINISH_REASON, choice.finish_reason)
                
                # Response ID
                if hasattr(response, "id"):
                    self.span.set_attribute(SpanAttributes.LLM_RESPONSE_ID, response.id)
                
                # System fingerprint (OpenAI specific)
                if hasattr(response, "system_fingerprint") and response.system_fingerprint:
                    self.span.set_attribute(SpanAttributes.LLM_OPENAI_RESPONSE_SYSTEM_FINGERPRINT, response.system_fingerprint)
                
                # Stop reason (for some models)
                if hasattr(choice, "stop_reason") and choice.stop_reason:
                    self.span.set_attribute(SpanAttributes.LLM_RESPONSE_STOP_REASON, choice.stop_reason)
                
                if hasattr(choice, "message"):
                    # Extract and set completion following semantic convention
                    raw_content = choice.message.content if hasattr(choice.message, "content") else str(choice.message)
                    completion = _extract_content_from_structured_format(raw_content)
                    
                    if completion:
                        # Use the semantic convention for completions - store full content without truncation
                        self.span.set_attribute(f"{SpanAttributes.LLM_COMPLETIONS}.0.role", "assistant")
                        self.span.set_attribute(f"{SpanAttributes.LLM_COMPLETIONS}.0.content", completion)
                    
                    # Track tool calls if present
                    if hasattr(choice.message, "tool_calls") and choice.message.tool_calls:
                        for i, tc in enumerate(choice.message.tool_calls):
                            prefix = f"{SpanAttributes.LLM_COMPLETIONS}.0.tool_calls.{i}"
                            self.span.set_attribute(f"{prefix}.id", tc.id if hasattr(tc, "id") else "")
                            if hasattr(tc, "function"):
                                self.span.set_attribute(f"{prefix}.name", tc.function.name)
                                self.span.set_attribute(f"{prefix}.arguments", tc.function.arguments[:500])
            
            # Add usage data following semantic conventions
            if hasattr(response, "usage") and response.usage:
                # Basic token counts
                if hasattr(response.usage, "prompt_tokens") and response.usage.prompt_tokens is not None:
                    self.span.set_attribute(SpanAttributes.LLM_USAGE_PROMPT_TOKENS, response.usage.prompt_tokens)
                if hasattr(response.usage, "completion_tokens") and response.usage.completion_tokens is not None:
                    self.span.set_attribute(SpanAttributes.LLM_USAGE_COMPLETION_TOKENS, response.usage.completion_tokens)
                if hasattr(response.usage, "total_tokens") and response.usage.total_tokens is not None:
                    self.span.set_attribute(SpanAttributes.LLM_USAGE_TOTAL_TOKENS, response.usage.total_tokens)
                else:
                    # Calculate total if not provided
                    prompt_tokens = getattr(response.usage, "prompt_tokens", 0) or 0
                    completion_tokens = getattr(response.usage, "completion_tokens", 0) or 0
                    total = prompt_tokens + completion_tokens
                    if total > 0:
                        self.span.set_attribute(SpanAttributes.LLM_USAGE_TOTAL_TOKENS, total)
                
                # Additional usage attributes
                if hasattr(response.usage, "reasoning_tokens") and response.usage.reasoning_tokens:
                    self.span.set_attribute(SpanAttributes.LLM_USAGE_REASONING_TOKENS, response.usage.reasoning_tokens)
                if hasattr(response.usage, "cache_creation_input_tokens") and response.usage.cache_creation_input_tokens:
                    self.span.set_attribute(SpanAttributes.LLM_USAGE_CACHE_CREATION_INPUT_TOKENS, response.usage.cache_creation_input_tokens)
                if hasattr(response.usage, "cache_read_input_tokens") and response.usage.cache_read_input_tokens:
                    self.span.set_attribute(SpanAttributes.LLM_USAGE_CACHE_READ_INPUT_TOKENS, response.usage.cache_read_input_tokens)
            
            
            # Set successful status
            self.span.set_status(Status(StatusCode.OK))
            
        except Exception as e:
            logger.error(f"Failed to record LLM response: {e}")
    
    def record_streaming_metrics(self, time_to_first_token: Optional[float] = None,
                                chunk_count: Optional[int] = None):
        """Record streaming-specific metrics."""
        try:
            if time_to_first_token is not None:
                self.span.set_attribute(SpanAttributes.LLM_STREAMING_TIME_TO_FIRST_TOKEN, time_to_first_token)
            
            if chunk_count is not None:
                self.span.set_attribute(SpanAttributes.LLM_STREAMING_CHUNK_COUNT, chunk_count)
            
            # Calculate total streaming duration
            duration_ms = (time.time() - self.start_time) * 1000
            self.span.set_attribute(SpanAttributes.LLM_STREAMING_DURATION, duration_ms)
            
        except Exception as e:
            logger.error(f"Failed to record streaming metrics: {e}")
            
    def record_error(self, error):
        """Record an error that occurred during the LLM call."""
        try:
            self.span.set_status(Status(StatusCode.ERROR, str(error)))
            self.span.record_exception(error)
            self.span.set_attribute("error.message", str(error))
        except Exception as e:
            logger.error(f"Failed to record LLM error: {e}")
    
    def record_tokens(self, prompt_tokens: int, completion_tokens: int, total_tokens: Optional[int] = None):
        """Record token usage to the span."""
        try:
            self.span.set_attribute(SpanAttributes.LLM_USAGE_PROMPT_TOKENS, prompt_tokens)
            self.span.set_attribute(SpanAttributes.LLM_USAGE_COMPLETION_TOKENS, completion_tokens)
            
            if total_tokens is None:
                total_tokens = prompt_tokens + completion_tokens
            self.span.set_attribute(SpanAttributes.LLM_USAGE_TOTAL_TOKENS, total_tokens)
            
            logger.info(f"✅ Recorded tokens in LLM span: prompt={prompt_tokens}, completion={completion_tokens}, total={total_tokens}")
        except Exception as e:
            logger.error(f"Failed to record tokens: {e}")
    
    async def end(self):
        """Manually end the span."""
        if not self._ended:
            try:
                logger.info("🔒 Manually ending LLM span")
                self.span.end()
                self._ended = True
                # Small delay to ensure span is sent
                await asyncio.sleep(0.01)
            except Exception as e:
                logger.error(f"Failed to manually end LLM span: {e}")


def end_agent_trace(
    trace_context: Optional[TraceContext] = None,
    status: str = "completed",
    error: Optional[str] = None
) -> None:
    """
    End an agent run span (but not the conversation trace).
    
    Args:
        trace_context: The trace context containing the agent run span.
        status: Final status of the agent run (completed, failed, stopped)
        error: Error message if status is failed
    """
    global _current_trace_context
    
    if not _initialized:
        return
    
    # Use provided context or current context
    context = trace_context or _current_trace_context
    if not context:
        logger.debug("No active AgentOps trace to end")
        return
    
    try:
        # End the agent run span if it exists
        if hasattr(context, '_agent_run_span') and context._agent_run_span:
            span = context._agent_run_span
            
            # Set status on the span
            if status == "completed":
                span.set_status(Status(StatusCode.OK))
            elif status in ["failed", "stopped"]:
                span.set_status(Status(StatusCode.ERROR, error or status))
                if error:
                    span.set_attribute("error.message", error)
                    span.record_exception(Exception(error))
            
            # End the span
            span.end()
            
            # Clear the span reference
            context._agent_run_span = None
            
            logger.info(f"Ended agent run span with status: {status}")
        else:
            logger.debug("No agent run span found to end")
        
        # Note: We do NOT end the conversation trace here
        # The conversation trace should persist across multiple agent runs
        
    except Exception as e:
        logger.error(f"Failed to end agent run span: {str(e)}")


@asynccontextmanager
async def tool_span(tool_name: str, tool_args: Optional[Dict[str, Any]] = None):
    """
    Context manager for tracking tool execution spans.
    
    Args:
        tool_name: Name of the tool being executed
        tool_args: Arguments passed to the tool
        
    Usage:
        with tool_span("web_search", {"query": "example"}):
            # Tool execution code here
            pass
    """
    # Get trace context from async context or global
    trace_context = agentops_trace_context.get() or _current_trace_context
    
    if not _initialized or not trace_context:
        yield None
        return
    
    if not tracer.initialized:
        yield None
        return
    
    # Create span attributes
    attributes = {
        # Core AgentOps attribute
        SpanAttributes.AGENTOPS_SPAN_KIND: SpanKind.TOOL,
        
        # Tool attributes
        "tool.name": tool_name,
        "tool.args": str(tool_args) if tool_args else "",
    }
    
    # Create tool span using make_span (consistent with agent spans)
    span, _, _ = tracer.make_span(
        f"tool.{tool_name}",
        SpanKind.TOOL,
        attributes=attributes
    )
    
    if not span:
        yield None
        return
    
    start_time = time.time()
    
    try:
        yield ToolSpanContext(span, start_time)
    finally:
        # Always end the span
        try:
            span.end()
        except Exception as e:
            logger.error(f"Failed to end tool span: {e}")


def _extract_content_from_structured_format(content):
    """
    Extract text content from message content.
    For strings: return as-is
    For lists: extract text from ALL items with type='text' and concatenate
    """
    # If content is a string that looks like a list, try to parse it
    if isinstance(content, str):
        if content.strip().startswith("["):
            try:
                content = json.loads(content)
            except:
                try:
                    content = ast.literal_eval(content)
                except:
                    # If parsing fails, return as-is
                    return content
    
    # If content is a list, concatenate ALL text items (not just the first)
    if isinstance(content, list) and len(content) > 0:
        text_parts = []
        for item in content:
            if isinstance(item, dict) and item.get("type") == "text":
                text = item.get("text", "")
                text_parts.append(text)
        
        if text_parts:
            return "\n".join(text_parts)
        else:
            return ""
    
    # Return content as-is (string or other)
    return str(content) if not isinstance(content, str) else content


@asynccontextmanager
async def llm_span(
    model: str,
    messages: list,
    temperature: float = 0,
    max_tokens: Optional[int] = None,
    tools: Optional[list] = None,
    manual_end: bool = False,
    **kwargs
):
    """
    Async context manager for tracking LLM calls as spans using AgentOps.
    
    This creates a proper LLM span in the current trace context that will
    show up in the AgentOps dashboard with all relevant metadata.
    
    Args:
        model: Name of the model being used
        messages: List of messages being sent to the LLM
        temperature: Temperature setting
        max_tokens: Maximum tokens
        tools: Tools available to the model
        manual_end: If True, the span won't be automatically ended when exiting the context
        **kwargs: Additional parameters
        
    Usage:
        async with llm_span("gpt-4", messages) as span_context:
            response = await litellm.acompletion(...)
            if span_context:
                span_context.record_response(response)
                
        # For manual span management:
        async with llm_span("gpt-4", messages, manual_end=True) as span_context:
            # ... do work ...
            # Later, manually end the span:
            await span_context.end()
    """
    # Get trace context from async context or global
    trace_context = agentops_trace_context.get() or _current_trace_context
    
    if not _initialized:
        logger.debug("AgentOps not initialized, skipping LLM span")
        yield None
        return
        
    if not trace_context:
        logger.warning(f"No trace context available for LLM span. Model: {model}, Messages: {len(messages)}")
        logger.debug(f"agentops_trace_context.get(): {agentops_trace_context.get()}")
        logger.debug(f"_current_trace_context: {_current_trace_context}")
        yield None
        return
    
    logger.debug(f"Creating LLM span for model {model} with trace context {trace_context}")
    
    # Extract system prompt if present
    system_prompt = None
    for msg in messages:
        if msg.get("role") == "system":
            system_prompt = msg.get("content", "")
            break
    
    # Create span attributes following OpenTelemetry Gen AI semantic conventions
    attributes = {
        # Core AgentOps attribute
        SpanAttributes.AGENTOPS_SPAN_KIND: SpanKind.LLM,
        
        # Gen AI semantic convention attributes
        SpanAttributes.LLM_SYSTEM: model.split("/")[0] if "/" in model else "openai",
        SpanAttributes.LLM_REQUEST_MODEL: model,
        SpanAttributes.LLM_REQUEST_TEMPERATURE: temperature,
        SpanAttributes.LLM_REQUEST_MAX_TOKENS: max_tokens if max_tokens else -1,
        SpanAttributes.LLM_REQUEST_STREAMING: kwargs.get("stream", False),
    }
    
    # Add additional request attributes if present in kwargs
    if "top_p" in kwargs:
        attributes[SpanAttributes.LLM_REQUEST_TOP_P] = kwargs["top_p"]
    if "top_k" in kwargs:
        attributes[SpanAttributes.LLM_REQUEST_TOP_K] = kwargs["top_k"]
    if "seed" in kwargs:
        attributes[SpanAttributes.LLM_REQUEST_SEED] = kwargs["seed"]
    if "frequency_penalty" in kwargs:
        attributes[SpanAttributes.LLM_REQUEST_FREQUENCY_PENALTY] = kwargs["frequency_penalty"]
    if "presence_penalty" in kwargs:
        attributes[SpanAttributes.LLM_REQUEST_PRESENCE_PENALTY] = kwargs["presence_penalty"]
    if "stop" in kwargs or "stop_sequences" in kwargs:
        stop_sequences = kwargs.get("stop", kwargs.get("stop_sequences", []))
        if stop_sequences:
            attributes[SpanAttributes.LLM_REQUEST_STOP_SEQUENCES] = str(stop_sequences)
    
    # Add prompts/messages following the semantic convention
    if messages:
        for i, msg in enumerate(messages):
            prefix = f"{SpanAttributes.LLM_PROMPTS}.{i}"
            if "role" in msg:
                attributes[f"{prefix}.role"] = msg["role"]
            if "content" in msg:
                raw_content = msg["content"]
                content = _extract_content_from_structured_format(raw_content)
                
                # Store the full content for all messages without truncation
                attributes[f"{prefix}.content"] = content
                
    # Add tools/functions if present
    if tools:
        for i, tool in enumerate(tools):
            if isinstance(tool, dict):
                prefix = f"{SpanAttributes.LLM_REQUEST_FUNCTIONS}.{i}"
                attributes[f"{prefix}.name"] = tool.get("name", "")
                if "description" in tool:
                    attributes[f"{prefix}.description"] = tool["description"][:200]
    
    if not tracer.initialized:
        yield None
        return
    
    # Create LLM span using make_span (consistent with agent and tool spans)
    span, _, _ = tracer.make_span(
        f"llm.{model}",
        SpanKind.LLM,
        attributes=attributes
    )
    
    if not span:
        yield None
        return
    
    start_time = time.time()
    span_context = LLMSpanContext(span, start_time, manual_end=manual_end)
    
    try:
        yield span_context
    finally:
        # Only end the span automatically if not in manual mode
        if not manual_end and not span_context._ended:
            try:
                span.end()
                # Small delay to ensure span is sent before trace context changes
                await asyncio.sleep(0.01)
            except Exception as e:
                logger.error(f"Failed to end LLM span: {e}")
        elif manual_end and not span_context._ended:
            logger.info("⏳ LLM span left open for manual ending")


def get_current_trace_context() -> Optional[TraceContext]:
    """Get the current active trace context."""
    # Check async context first, then fall back to global
    return agentops_trace_context.get() or _current_trace_context


def is_initialized() -> bool:
    """Check if AgentOps is initialized."""
    return _initialized


def record_event(name: str, level: str = "DEFAULT", message: str = "", metadata: Optional[Dict[str, Any]] = None):
    """
    Record an event in AgentOps if a trace context is available.
    
    This function records events as spans in the current trace context,
    providing parity with Langfuse event tracking.
    
    Args:
        name: Event name (e.g., "billing_limit_reached")
        level: Event level (DEFAULT, WARNING, ERROR, CRITICAL)
        message: Event message/status message
        metadata: Optional metadata dictionary
    """
    # Get trace context from async context or global
    trace_context = agentops_trace_context.get() or _current_trace_context
    
    if not _initialized or not trace_context:
        return
    
    try:
        if not tracer.initialized:
            return
            
        # Map Langfuse levels to OpenTelemetry status
        level_map = {
            "DEFAULT": StatusCode.OK,
            "WARNING": StatusCode.OK,  # Warnings are still OK status
            "ERROR": StatusCode.ERROR,
            "CRITICAL": StatusCode.ERROR
        }
        
        # Create event span
        span, _, _ = tracer.make_span(
            f"event.{name}",
            SpanKind.OPERATION,
            attributes={
                "event.name": name,
                "event.level": level,
                "event.message": message,
                "agentops.span.kind": "event"
            }
        )
        
        if span:
            # Add metadata as attributes if provided
            if metadata:
                for key, value in metadata.items():
                    # Prefix metadata keys to avoid conflicts
                    span.set_attribute(f"event.metadata.{key}", str(value))
            
            # Set span status based on level
            if level in ["ERROR", "CRITICAL"]:
                span.set_status(Status(level_map.get(level, StatusCode.UNSET), message))
            
            # End the span immediately as events are point-in-time
            span.end()
            
    except Exception as e:
        logger.error(f"Failed to record AgentOps event '{name}': {str(e)}")


async def flush_trace() -> None:
    """
    Flush any pending spans to ensure they are sent before trace ends.
    """
    if not _initialized:
        return
    
    try:
        if tracer.initialized:
            # Use the tracer's internal flush method
            tracer._flush_span_processors()
            logger.debug("Flushed pending spans")
            
            # Small delay to ensure flush completes
            await asyncio.sleep(0.1)
            
    except Exception as e:
        logger.error(f"Failed to flush trace: {str(e)}")


def get_or_create_conversation_trace(
    thread_id: str,
    project_id: str,
    model_name: str
) -> Optional[TraceContext]:
    """
    Get existing conversation trace or create a new one.
    
    Args:
        thread_id: Thread ID for the conversation
        project_id: Project ID
        model_name: Name of the LLM model being used
        
    Returns:
        TraceContext object if successful, None otherwise
    """
    global _conversation_traces
    
    if not _initialized:
        logger.debug("AgentOps not initialized, skipping trace creation")
        return None
    
    # Check if we already have a trace for this thread
    if thread_id in _conversation_traces:
        trace_context = _conversation_traces[thread_id]
        logger.debug(f"Using existing conversation trace for thread {thread_id}")
        
        # Set in async context for propagation
        agentops_trace_context.set(trace_context)
        return trace_context
    
    try:
        # Create new conversation trace
        trace_name = f"conversation_{thread_id}"
        tags = {
            "thread_id": thread_id,
            "project_id": project_id,
            "model": model_name,
            "conversation_id": thread_id
        }
        
        # Start the trace
        trace_context = agentops.start_trace(trace_name=trace_name, tags=tags)
        _conversation_traces[thread_id] = trace_context
        
        # Set in async context for propagation
        agentops_trace_context.set(trace_context)
        
        logger.info(f"Started new conversation trace for thread {thread_id}")
        return trace_context
        
    except Exception as e:
        logger.error(f"Failed to create conversation trace: {str(e)}")
        return None


def end_conversation_trace(thread_id: str) -> None:
    """
    End a conversation trace and remove it from cache.
    
    Args:
        thread_id: Thread ID for the conversation
    """
    global _conversation_traces
    
    if not _initialized:
        return
    
    if thread_id not in _conversation_traces:
        logger.debug(f"No conversation trace found for thread {thread_id}")
        return
    
    try:
        trace_context = _conversation_traces[thread_id]
        
        # End the trace
        agentops.end_trace(trace_context=trace_context, end_state=agentops.SUCCESS)
        
        # Remove from cache
        del _conversation_traces[thread_id]
        
        # Clear async context if it matches
        if agentops_trace_context.get() == trace_context:
            agentops_trace_context.set(None)
            
        logger.info(f"Ended conversation trace for thread {thread_id}")
        
    except Exception as e:
        logger.error(f"Failed to end conversation trace: {str(e)}")