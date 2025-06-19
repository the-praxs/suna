"""
AgentOps integration service for tracking agent conversations and tool usage.
"""

import os
from typing import Optional, Any, Dict
import agentops
from agentops import TraceContext
from utils.logger import logger
from contextlib import contextmanager, asynccontextmanager
import contextvars
import time

# Global variables for AgentOps state
_initialized = False
_current_trace_context: Optional[TraceContext] = None

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
            instrument_llm_calls=True,  # Automatically track LLM calls
            auto_start_session=False,   # We'll manage traces manually
            fail_safe=True,            # Don't crash if AgentOps has issues
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
    Start a new AgentOps trace for an agent run.
    
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
        # Create trace name and tags
        trace_name = f"agent_run_{agent_run_id}"
        tags = {
            "thread_id": thread_id,
            "project_id": project_id,
            "model": model_name,
            "agent_run_id": agent_run_id
        }
        
        # Add agent info if available
        if agent_config:
            tags["agent_name"] = agent_config.get("name", "default")
            tags["agent_id"] = agent_config.get("agent_id", "unknown")
        
        # Start the trace
        trace_context = agentops.start_trace(trace_name=trace_name, tags=tags)
        _current_trace_context = trace_context
        
        # Set in async context for propagation
        agentops_trace_context.set(trace_context)
        
        logger.info(f"Started AgentOps trace: {trace_name}")
        return trace_context
        
    except Exception as e:
        logger.error(f"Failed to start AgentOps trace: {str(e)}")
        return None


def end_agent_trace(
    trace_context: Optional[TraceContext] = None,
    status: str = "completed",
    error: Optional[str] = None
) -> None:
    """
    End an AgentOps trace.
    
    Args:
        trace_context: The trace context to end. If None, uses current context.
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
        # Map status to AgentOps end state
        if status == "completed":
            end_state = agentops.SUCCESS
        elif status in ["failed", "stopped"]:
            end_state = agentops.ERROR
        else:
            end_state = agentops.UNSET
        
        # Add error info if available
        if error and hasattr(context, 'span') and context.span:
            context.span.set_attribute("error.message", error)
        
        # End the trace
        agentops.end_trace(trace_context=context, end_state=end_state)
        
        # Clear current context if it matches
        if _current_trace_context == context:
            _current_trace_context = None
            # Also clear from async context
            agentops_trace_context.set(None)
            
        logger.info(f"Ended AgentOps trace with status: {status}")
        
    except Exception as e:
        logger.error(f"Failed to end AgentOps trace: {str(e)}")


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
    
    # Get the tracer from AgentOps
    from agentops.sdk.core import tracer
    from agentops.semconv import SpanKind
    
    # Create span attributes
    attributes = {
        "agentops.span_kind": SpanKind.TOOL,
        "tool.name": tool_name,
        "tool.args": str(tool_args) if tool_args else "",
    }
    
    # Start the tool span
    otel_tracer = tracer.get_tracer()
    with otel_tracer.start_as_current_span(
        f"tool.{tool_name}",
        attributes=attributes
    ) as span:
        start_time = time.time()
        
        class ToolSpanContext:
            def __init__(self, span):
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
                    from opentelemetry.trace import Status, StatusCode
                    self.span.set_status(Status(StatusCode.ERROR, str(error)))
                    self.span.record_exception(error)
                    self.span.set_attribute("error.message", str(error))
                    self.span.set_attribute("tool.success", False)
                except Exception as e:
                    logger.error(f"Failed to record tool error: {e}")
        
        yield ToolSpanContext(span)


@contextmanager
def llm_span(
    model: str,
    messages: list,
    temperature: float = 0,
    max_tokens: Optional[int] = None,
    tools: Optional[list] = None,
    **kwargs
):
    """
    Context manager for tracking LLM calls as spans using AgentOps.
    
    This creates a proper LLM span in the current trace context that will
    show up in the AgentOps dashboard with all relevant metadata.
    
    Args:
        model: Name of the model being used
        messages: List of messages being sent to the LLM
        temperature: Temperature setting
        max_tokens: Maximum tokens
        tools: Tools available to the model
        **kwargs: Additional parameters
        
    Usage:
        async with llm_span("gpt-4", messages) as span_context:
            response = await litellm.acompletion(...)
            if span_context:
                span_context.record_response(response)
    """
    # Get trace context from async context or global
    trace_context = agentops_trace_context.get() or _current_trace_context
    
    if not _initialized or not trace_context:
        yield None
        return
    
    # Extract system prompt if present
    system_prompt = None
    for msg in messages:
        if msg.get("role") == "system":
            system_prompt = msg.get("content", "")
            break
    
    # Get the tracer from AgentOps
    from agentops.sdk.core import tracer
    from agentops.semconv import SpanKind
    
    # Create span attributes
    attributes = {
        "agentops.span_kind": SpanKind.LLM,
        "model": model,
        "temperature": temperature,
        "max_tokens": max_tokens or -1,
        "has_tools": bool(tools),
        "tool_count": len(tools) if tools else 0,
        "has_system_prompt": bool(system_prompt),
        "message_count": len(messages),
        "stream": kwargs.get("stream", False),
    }
    
    # Start the LLM span
    otel_tracer = tracer.get_tracer()
    with otel_tracer.start_as_current_span(
        f"llm.{model}",
        attributes=attributes
    ) as span:
        start_time = time.time()
        
        class LLMSpanContext:
            def __init__(self, span):
                self.span = span
                self.start_time = start_time
                
            def record_response(self, response):
                """Record the LLM response to the span."""
                try:
                    # Update span with response data
                    if hasattr(response, "choices") and response.choices:
                        choice = response.choices[0]
                        if hasattr(choice, "message"):
                            # Set completion as span attribute
                            completion = choice.message.content if hasattr(choice.message, "content") else str(choice.message)
                            if completion:
                                self.span.set_attribute("completion", completion[:1000])  # Limit size
                            
                            # Track tool calls if present
                            if hasattr(choice.message, "tool_calls") and choice.message.tool_calls:
                                self.span.set_attribute("tool_calls_count", len(choice.message.tool_calls))
                                tool_names = [tc.function.name for tc in choice.message.tool_calls if hasattr(tc, "function")]
                                if tool_names:
                                    self.span.set_attribute("tool_calls", ",".join(tool_names))
                    
                    # Add usage data if available
                    if hasattr(response, "usage"):
                        self.span.set_attribute("prompt_tokens", response.usage.prompt_tokens)
                        self.span.set_attribute("completion_tokens", response.usage.completion_tokens)
                        total_tokens = response.usage.prompt_tokens + response.usage.completion_tokens
                        self.span.set_attribute("total_tokens", total_tokens)
                    
                    # Set messages as event
                    self.span.add_event(
                        "llm_call",
                        attributes={
                            "messages": str(messages)[:5000],  # Limit size
                            "system_prompt": system_prompt[:1000] if system_prompt else "",
                        }
                    )
                    
                except Exception as e:
                    logger.error(f"Failed to record LLM response: {e}")
                    
            def record_error(self, error):
                """Record an error that occurred during the LLM call."""
                try:
                    from opentelemetry.trace import Status, StatusCode
                    self.span.set_status(Status(StatusCode.ERROR, str(error)))
                    self.span.record_exception(error)
                    self.span.set_attribute("error.message", str(error))
                except Exception as e:
                    logger.error(f"Failed to record LLM error: {e}")
        
        yield LLMSpanContext(span)


def get_current_trace_context() -> Optional[TraceContext]:
    """Get the current active trace context."""
    # Check async context first, then fall back to global
    return agentops_trace_context.get() or _current_trace_context


def is_initialized() -> bool:
    """Check if AgentOps is initialized."""
    return _initialized