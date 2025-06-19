"""
AgentOps callback for LiteLLM to track LLM calls.

This module provides a custom callback that integrates LiteLLM with AgentOps,
ensuring all LLM calls are properly tracked with system prompts, tool calls,
and other metadata.
"""

from typing import Dict, Any, Optional, List
import json
from litellm.integrations.custom_logger import CustomLogger
from utils.logger import logger
import agentops
from services.agentops import get_current_trace_context, is_initialized as agentops_is_initialized


class AgentOpsLiteLLMCallback(CustomLogger):
    """Custom LiteLLM callback for AgentOps integration."""
    
    def __init__(self):
        super().__init__()
        self.pending_events = {}
        
    def log_pre_api_call(self, model: str, messages: List[Dict[str, Any]], kwargs: Dict[str, Any]) -> None:
        """Called before the LLM API call is made."""
        if not agentops_is_initialized() or not get_current_trace_context():
            return
            
        try:
            # Generate a unique ID for this call
            import uuid
            call_id = str(uuid.uuid4())
            
            # Extract system prompt if present
            system_prompt = None
            for msg in messages:
                if msg.get("role") == "system":
                    system_prompt = msg.get("content", "")
                    break
            
            # Store metadata for the completion callback
            self.pending_events[call_id] = {
                "model": model,
                "messages": messages,
                "system_prompt": system_prompt,
                "kwargs": kwargs,
                "start_time": agentops.get_current_time()
            }
            
            # Store call_id in kwargs for retrieval in post callback
            kwargs["_agentops_call_id"] = call_id
            
            logger.debug(f"AgentOps: Pre-API call tracked for model {model}")
            
        except Exception as e:
            logger.error(f"Error in AgentOps pre-API call logging: {e}")
    
    def log_success_event(self, kwargs: Dict[str, Any], response_obj: Any, start_time: float, end_time: float) -> None:
        """Called after a successful LLM API call."""
        if not agentops_is_initialized() or not get_current_trace_context():
            return
            
        try:
            # Retrieve the call ID
            call_id = kwargs.get("_agentops_call_id")
            if not call_id or call_id not in self.pending_events:
                return
                
            event_data = self.pending_events.pop(call_id)
            
            # Create LLM event with all the data
            llm_event = agentops.LLMEvent(
                init_timestamp=event_data["start_time"],
                end_timestamp=agentops.get_current_time(),
                model=event_data["model"],
                prompt=event_data["messages"],
                params={
                    "temperature": kwargs.get("temperature", 0),
                    "max_tokens": kwargs.get("max_tokens"),
                    "top_p": kwargs.get("top_p"),
                    "stream": kwargs.get("stream", False),
                    "tools": kwargs.get("tools"),
                    "tool_choice": kwargs.get("tool_choice"),
                    "system_prompt": event_data["system_prompt"]  # Include system prompt
                }
            )
            
            # Extract response data
            if hasattr(response_obj, "choices") and response_obj.choices:
                choice = response_obj.choices[0]
                if hasattr(choice, "message"):
                    llm_event.completion = choice.message.content if hasattr(choice.message, "content") else str(choice.message)
                    
                    # Track tool calls if present
                    if hasattr(choice.message, "tool_calls") and choice.message.tool_calls:
                        llm_event.tool_calls = [
                            {
                                "id": tc.id,
                                "type": tc.type,
                                "function": {
                                    "name": tc.function.name,
                                    "arguments": tc.function.arguments
                                }
                            }
                            for tc in choice.message.tool_calls
                        ]
            
            # Add usage data if available
            if hasattr(response_obj, "usage"):
                llm_event.prompt_tokens = response_obj.usage.prompt_tokens
                llm_event.completion_tokens = response_obj.usage.completion_tokens
                llm_event.cost = self._calculate_cost(
                    event_data["model"],
                    response_obj.usage.prompt_tokens,
                    response_obj.usage.completion_tokens
                )
            
            # Record the event
            agentops.record(llm_event)
            
            logger.debug(f"AgentOps: LLM call tracked successfully for model {event_data['model']}")
            
        except Exception as e:
            logger.error(f"Error in AgentOps success event logging: {e}")
    
    def log_failure_event(self, kwargs: Dict[str, Any], response_obj: Any, start_time: float, end_time: float) -> None:
        """Called after a failed LLM API call."""
        if not agentops_is_initialized() or not get_current_trace_context():
            return
            
        try:
            # Retrieve the call ID
            call_id = kwargs.get("_agentops_call_id")
            if not call_id or call_id not in self.pending_events:
                return
                
            event_data = self.pending_events.pop(call_id)
            
            # Create error event
            llm_event = agentops.LLMEvent(
                init_timestamp=event_data["start_time"],
                end_timestamp=agentops.get_current_time(),
                model=event_data["model"],
                prompt=event_data["messages"],
                params=event_data["kwargs"],
                error=str(response_obj) if response_obj else "Unknown error"
            )
            
            # Record the event
            agentops.record(llm_event)
            
            logger.debug(f"AgentOps: LLM error tracked for model {event_data['model']}")
            
        except Exception as e:
            logger.error(f"Error in AgentOps failure event logging: {e}")
    
    def _calculate_cost(self, model: str, prompt_tokens: int, completion_tokens: int) -> Optional[float]:
        """Calculate the cost of the LLM call based on model and token usage."""
        # Simplified cost calculation - you can expand this based on actual pricing
        cost_per_1k_tokens = {
            "gpt-4": {"prompt": 0.03, "completion": 0.06},
            "gpt-4-turbo": {"prompt": 0.01, "completion": 0.03},
            "gpt-3.5-turbo": {"prompt": 0.0005, "completion": 0.0015},
            "claude-3-opus": {"prompt": 0.015, "completion": 0.075},
            "claude-3-sonnet": {"prompt": 0.003, "completion": 0.015},
            "claude-3-haiku": {"prompt": 0.00025, "completion": 0.00125},
        }
        
        # Find matching model
        for model_key, costs in cost_per_1k_tokens.items():
            if model_key in model.lower():
                prompt_cost = (prompt_tokens / 1000) * costs["prompt"]
                completion_cost = (completion_tokens / 1000) * costs["completion"]
                return prompt_cost + completion_cost
                
        return None


# Create a singleton instance
agentops_callback = AgentOpsLiteLLMCallback()