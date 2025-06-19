"""
AgentOps integration service for tracking agent conversations and tool usage.
"""

import os
from typing import Optional, Any, Dict
import agentops
from agentops import TraceContext
from utils.logger import logger
from contextlib import contextmanager

# Global variables for AgentOps state
_initialized = False
_current_trace_context: Optional[TraceContext] = None


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
            
        logger.info(f"Ended AgentOps trace with status: {status}")
        
    except Exception as e:
        logger.error(f"Failed to end AgentOps trace: {str(e)}")


@contextmanager
def tool_span(tool_name: str, tool_args: Optional[Dict[str, Any]] = None):
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
    if not _initialized:
        yield
        return
    
    # AgentOps v0.4.16 uses decorators for tool tracking
    # For manual span creation, we'll use the trace decorator
    # This is a simplified approach - in production, you might want
    # to use OpenTelemetry spans directly
    yield


def get_current_trace_context() -> Optional[TraceContext]:
    """Get the current active trace context."""
    return _current_trace_context


def is_initialized() -> bool:
    """Check if AgentOps is initialized."""
    return _initialized