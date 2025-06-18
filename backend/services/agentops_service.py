"""
AgentOps integration service for tracking AI agent operations.

This module provides integration with AgentOps for comprehensive tracking of:
- Chat sessions as traces
- Threads as sessions (parent spans)
- Agent, tool, and LLM calls as child spans
"""

import os
from typing import Optional, Dict, Any, List
from contextlib import contextmanager
import agentops
from agentops import TraceContext
from utils.logger import logger
from utils.config import config

# Global state for AgentOps client and trace management
_initialized = False
_trace_contexts: Dict[str, TraceContext] = {}  # Maps project_id to trace context
_session_contexts: Dict[str, Any] = {}  # Maps thread_id to session context


def initialize():
    """Initialize AgentOps with configuration from environment."""
    global _initialized
    
    if _initialized:
        logger.debug("AgentOps already initialized")
        return
    
    api_key = os.getenv("AGENTOPS_API_KEY")
    log_level = os.getenv("AGENTOPS_LOG_LEVEL", "INFO")
    
    if not api_key:
        logger.warning("AGENTOPS_API_KEY not found in environment. AgentOps tracking disabled.")
        return
    
    try:
        agentops.init(
            api_key=api_key,
            default_tags=["suna", config.ENV_MODE.value],
            auto_start_session=False,
            instrument_llm_calls=True,
            log_level=log_level
        )
        _initialized = True
        logger.info(f"AgentOps initialized successfully with log level: {log_level}")
    except Exception as e:
        logger.error(f"Failed to initialize AgentOps: {e}")
        _initialized = False


def start_chat_trace(project_id: str, metadata: Optional[Dict[str, Any]] = None) -> Optional[TraceContext]:
    """
    Start a new AgentOps trace for a chat session.
    
    Args:
        project_id: The project ID (corresponds to a chat in the frontend)
        metadata: Optional metadata to attach to the trace
        
    Returns:
        TraceContext if successful, None otherwise
    """
    if not _initialized:
        return None
    
    try:
        # Check if we already have a trace for this project
        if project_id in _trace_contexts:
            logger.debug(f"Reusing existing trace for project {project_id}")
            return _trace_contexts[project_id]
        
        # Start a new trace
        trace_name = f"chat_{project_id}"
        tags = ["chat", f"project:{project_id}"]
        
        if metadata:
            tags.extend([f"{k}:{v}" for k, v in metadata.items() if isinstance(v, (str, int, float))])
        
        trace_context = agentops.start_trace(
            trace_name=trace_name,
            tags=tags
        )
        
        _trace_contexts[project_id] = trace_context
        
        if trace_context:
            logger.info(f"Started AgentOps trace for project {project_id}")
            # Log the session URL if available
            try:
                session_url = f"https://app.agentops.ai/drilldown?session_id={trace_context.span.context.trace_id}"
                logger.info(f"AgentOps session URL: {session_url}")
            except Exception:
                pass
        
        return trace_context
        
    except Exception as e:
        logger.error(f"Failed to start AgentOps trace for project {project_id}: {e}")
        return None


def start_thread_session(thread_id: str, project_id: str, metadata: Optional[Dict[str, Any]] = None):
    """
    Start a new session (parent span) for a thread within a chat trace.
    
    Args:
        thread_id: The thread ID
        project_id: The project ID (to find the parent trace)
        metadata: Optional metadata to attach to the session
    """
    if not _initialized:
        return None
    
    try:
        # Get or create the parent trace
        trace_context = _trace_contexts.get(project_id)
        if not trace_context:
            trace_context = start_chat_trace(project_id)
            if not trace_context:
                return None
        
        # Create a session span within the trace
        with trace_context:
            session_span = agentops.tracer.start_span(
                name=f"thread_{thread_id}",
                attributes={
                    "thread.id": thread_id,
                    "project.id": project_id,
                    "span.kind": "session",
                    **(metadata or {})
                }
            )
            
            _session_contexts[thread_id] = {
                "span": session_span,
                "project_id": project_id,
                "trace_context": trace_context
            }
            
            logger.info(f"Started AgentOps session for thread {thread_id}")
            return session_span
            
    except Exception as e:
        logger.error(f"Failed to start AgentOps session for thread {thread_id}: {e}")
        return None


def track_llm_call(
    thread_id: str,
    model: str,
    messages: List[Dict[str, Any]],
    response: Optional[Dict[str, Any]] = None,
    tokens: Optional[Dict[str, int]] = None,
    error: Optional[str] = None,
    **kwargs
):
    """
    Track an LLM call as a span within a thread session.
    
    Args:
        thread_id: The thread ID
        model: The model name
        messages: The messages sent to the LLM
        response: The LLM response (if successful)
        tokens: Token usage information
        error: Error message (if failed)
        **kwargs: Additional metadata
    """
    if not _initialized or thread_id not in _session_contexts:
        return
    
    try:
        session_info = _session_contexts[thread_id]
        trace_context = session_info["trace_context"]
        
        with trace_context:
            span_name = f"llm_call_{model}"
            attributes = {
                "llm.model": model,
                "llm.messages_count": len(messages),
                "thread.id": thread_id,
                **kwargs
            }
            
            if tokens:
                attributes.update({
                    "llm.tokens.prompt": tokens.get("prompt_tokens", 0),
                    "llm.tokens.completion": tokens.get("completion_tokens", 0),
                    "llm.tokens.total": tokens.get("total_tokens", 0)
                })
            
            if error:
                attributes["error"] = error
                attributes["status"] = "error"
            else:
                attributes["status"] = "success"
            
            with agentops.tracer.start_span(name=span_name, attributes=attributes) as span:
                if response and not error:
                    # Log successful response
                    logger.debug(f"Tracked LLM call for thread {thread_id}, model {model}")
                elif error:
                    # Log error
                    span.set_status("error", error)
                    logger.warning(f"Tracked failed LLM call for thread {thread_id}: {error}")
                    
    except Exception as e:
        logger.error(f"Failed to track LLM call for thread {thread_id}: {e}")


def track_tool_call(
    thread_id: str,
    tool_name: str,
    tool_type: str,
    parameters: Optional[Dict[str, Any]] = None,
    result: Optional[Any] = None,
    error: Optional[str] = None,
    **kwargs
):
    """
    Track a tool call as a span within a thread session.
    
    Args:
        thread_id: The thread ID
        tool_name: The name of the tool
        tool_type: The type of tool (e.g., "browser", "file", "shell")
        parameters: The parameters passed to the tool
        result: The tool execution result (if successful)
        error: Error message (if failed)
        **kwargs: Additional metadata
    """
    if not _initialized or thread_id not in _session_contexts:
        return
    
    try:
        session_info = _session_contexts[thread_id]
        trace_context = session_info["trace_context"]
        
        with trace_context:
            span_name = f"tool_{tool_name}"
            attributes = {
                "tool.name": tool_name,
                "tool.type": tool_type,
                "thread.id": thread_id,
                **kwargs
            }
            
            if parameters:
                # Add flattened parameters
                for key, value in parameters.items():
                    if isinstance(value, (str, int, float, bool)):
                        attributes[f"tool.params.{key}"] = value
            
            if error:
                attributes["error"] = error
                attributes["status"] = "error"
            else:
                attributes["status"] = "success"
            
            with agentops.tracer.start_span(name=span_name, attributes=attributes) as span:
                if result and not error:
                    logger.debug(f"Tracked tool call for thread {thread_id}, tool {tool_name}")
                elif error:
                    span.set_status("error", error)
                    logger.warning(f"Tracked failed tool call for thread {thread_id}: {error}")
                    
    except Exception as e:
        logger.error(f"Failed to track tool call for thread {thread_id}: {e}")


def track_agent_event(
    thread_id: str,
    event_type: str,
    event_data: Dict[str, Any],
    **kwargs
):
    """
    Track a generic agent event as a span within a thread session.
    
    Args:
        thread_id: The thread ID
        event_type: The type of event (e.g., "agent_start", "agent_complete")
        event_data: Event-specific data
        **kwargs: Additional metadata
    """
    if not _initialized or thread_id not in _session_contexts:
        return
    
    try:
        session_info = _session_contexts[thread_id]
        trace_context = session_info["trace_context"]
        
        with trace_context:
            span_name = f"agent_{event_type}"
            attributes = {
                "agent.event_type": event_type,
                "thread.id": thread_id,
                **event_data,
                **kwargs
            }
            
            with agentops.tracer.start_span(name=span_name, attributes=attributes):
                logger.debug(f"Tracked agent event for thread {thread_id}, type {event_type}")
                
    except Exception as e:
        logger.error(f"Failed to track agent event for thread {thread_id}: {e}")


def end_thread_session(thread_id: str, status: str = "success", error: Optional[str] = None):
    """
    End a thread session (parent span).
    
    Args:
        thread_id: The thread ID
        status: The final status ("success", "error", "stopped")
        error: Error message if status is "error"
    """
    if not _initialized or thread_id not in _session_contexts:
        return
    
    try:
        session_info = _session_contexts.get(thread_id)
        if session_info and "span" in session_info:
            span = session_info["span"]
            
            # Set final status
            if status == "error" and error:
                span.set_status("error", error)
            elif status == "stopped":
                span.set_status("cancelled", "Thread stopped by user")
            else:
                span.set_status("ok")
            
            # End the span
            span.end()
            
            # Clean up
            del _session_contexts[thread_id]
            logger.info(f"Ended AgentOps session for thread {thread_id} with status: {status}")
            
    except Exception as e:
        logger.error(f"Failed to end AgentOps session for thread {thread_id}: {e}")


def end_chat_trace(project_id: str, status: str = "success"):
    """
    End a chat trace.
    
    Args:
        project_id: The project ID
        status: The final status
    """
    if not _initialized or project_id not in _trace_contexts:
        return
    
    try:
        trace_context = _trace_contexts.get(project_id)
        if trace_context:
            # End any remaining thread sessions for this project
            threads_to_end = [
                thread_id for thread_id, session in _session_contexts.items()
                if session.get("project_id") == project_id
            ]
            
            for thread_id in threads_to_end:
                end_thread_session(thread_id, status="completed")
            
            # End the trace
            agentops.end_trace(trace_context, end_state=status)
            
            # Clean up
            del _trace_contexts[project_id]
            logger.info(f"Ended AgentOps trace for project {project_id}")
            
    except Exception as e:
        logger.error(f"Failed to end AgentOps trace for project {project_id}: {e}")


def cleanup():
    """Clean up all active traces and sessions."""
    if not _initialized:
        return
    
    try:
        # End all thread sessions
        thread_ids = list(_session_contexts.keys())
        for thread_id in thread_ids:
            end_thread_session(thread_id, status="stopped")
        
        # End all traces
        project_ids = list(_trace_contexts.keys())
        for project_id in project_ids:
            end_chat_trace(project_id, status="stopped")
        
        logger.info("AgentOps cleanup completed")
        
    except Exception as e:
        logger.error(f"Error during AgentOps cleanup: {e}")


@contextmanager
def track_operation(thread_id: str, operation_name: str, **attributes):
    """
    Context manager for tracking a generic operation as a span.
    
    Usage:
        with track_operation(thread_id, "custom_operation", custom_attr="value"):
            # Your operation code here
            pass
    """
    if not _initialized or thread_id not in _session_contexts:
        yield
        return
    
    session_info = _session_contexts.get(thread_id)
    if not session_info:
        yield
        return
    
    trace_context = session_info["trace_context"]
    
    try:
        with trace_context:
            with agentops.tracer.start_span(
                name=operation_name,
                attributes={"thread.id": thread_id, **attributes}
            ) as span:
                yield span
    except Exception as e:
        logger.error(f"Failed to track operation {operation_name}: {e}")
        yield None