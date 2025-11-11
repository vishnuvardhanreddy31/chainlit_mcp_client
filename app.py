"""
CodeCrafter AI - Intelligent Coding Assistant
An advanced AI assistant powered by Google Gemini with MCP integration.

Features:
- Natural language code assistance
- Tool calling via Model Context Protocol (MCP)
- Context-aware conversations
- Real-time streaming responses
"""

import json
import os
import traceback
from typing import Dict, List, Any, Optional
from dataclasses import dataclass

from mcp import ClientSession
import google.generativeai as genai
import chainlit as cl
from dotenv import load_dotenv

# ============================================================================
# Configuration
# ============================================================================

load_dotenv()

# Gemini Configuration
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
if not GEMINI_API_KEY:
    raise ValueError("GEMINI_API_KEY environment variable is required")

genai.configure(api_key=GEMINI_API_KEY)
GEMINI_MODEL_NAME = 'gemini-2.0-flash'

# System Prompt
SYSTEM_PROMPT = """You are CodeCrafter AI, an expert coding assistant designed to help developers excel.

Your capabilities:
- Write clean, efficient, and well-documented code
- Debug complex issues and explain solutions clearly
- Suggest best practices and design patterns
- Use available tools to enhance your assistance
- Maintain context throughout conversations

Guidelines:
- Be precise and concise in your responses
- Provide code examples when helpful
- Explain your reasoning when making suggestions
- Ask clarifying questions when requirements are unclear
- Use tools when they can provide better information
"""

# Generation Configuration
GENERATION_CONFIG = {
    "temperature": 0.7,
    "max_output_tokens": 2048,
    "top_p": 0.95,
}


# ============================================================================
# Data Models
# ============================================================================

@dataclass
class ToolCall:
    """Represents a tool call with name and arguments."""
    name: str
    input: Dict[str, Any]
    id: str


@dataclass
class AssistantResponse:
    """Represents an assistant's response with content and metadata."""
    text: str = ""
    tool_calls: List[ToolCall] = None
    stop_reason: str = "end_turn"
    
    def __post_init__(self):
        if self.tool_calls is None:
            self.tool_calls = []


# ============================================================================
# Utility Functions
# ============================================================================

def flatten_list(nested_list: List[List[Any]]) -> List[Any]:
    """Flatten a nested list into a single list."""
    return [item for sublist in nested_list for item in sublist]


def format_error_message(error: Exception, context: str = "") -> str:
    """Format an error message for user display."""
    error_type = type(error).__name__
    error_msg = str(error)
    
    user_message = f"❌ **{error_type}**"
    if context:
        user_message += f" ({context})"
    user_message += f"\n\n{error_msg}"
    
    return user_message


def convert_mcp_tools_to_gemini(mcp_tools: List[Dict]) -> List:
    """
    Convert MCP tool definitions to Gemini function declarations.
    
    Args:
        mcp_tools: List of MCP tool definitions
        
    Returns:
        List of Gemini Tool objects
    """
    gemini_tools = []
    
    for tool in mcp_tools:
        try:
            # Create function declaration
            func_decl = genai.protos.FunctionDeclaration(
                name=tool["name"],
                description=tool["description"]
            )
            
            # Parse input schema
            input_schema = tool.get("input_schema", {})
            properties = input_schema.get("properties", {})
            required = input_schema.get("required", [])
            
            if properties:
                # Build schema properties
                schema_props = {}
                for param_name, param_def in properties.items():
                    param_type = param_def.get("type", "string")
                    
                    # Map JSON Schema types to Gemini types
                    type_mapping = {
                        "string": genai.protos.Type.STRING,
                        "number": genai.protos.Type.NUMBER,
                        "integer": genai.protos.Type.INTEGER,
                        "boolean": genai.protos.Type.BOOLEAN,
                        "array": genai.protos.Type.ARRAY,
                        "object": genai.protos.Type.OBJECT,
                    }
                    
                    gemini_type = type_mapping.get(param_type, genai.protos.Type.STRING)
                    
                    schema_props[param_name] = genai.protos.Schema(
                        type=gemini_type,
                        description=param_def.get("description", "")
                    )
                
                # Set parameters
                func_decl.parameters = genai.protos.Schema(
                    type=genai.protos.Type.OBJECT,
                    properties=schema_props,
                    required=required
                )
            
            gemini_tools.append(genai.protos.Tool(function_declarations=[func_decl]))
            
        except Exception as e:
            print(f"Warning: Failed to convert tool {tool.get('name', 'unknown')}: {e}")
            continue
    
    return gemini_tools


# ============================================================================
# MCP Integration
# ============================================================================

@cl.on_mcp_connect
async def on_mcp_connect(connection, session: ClientSession):
    """
    Handle MCP connection and register available tools.
    
    This is called when an MCP server connects to the application.
    """
    try:
        # List available tools from the MCP server
        result = await session.list_tools()
        
        # Convert tools to our format
        tools = [
            {
                "name": tool.name,
                "description": tool.description,
                "input_schema": tool.inputSchema,
            }
            for tool in result.tools
        ]
        
        # Store tools in user session
        mcp_tools = cl.user_session.get("mcp_tools", {})
        mcp_tools[connection.name] = tools
        cl.user_session.set("mcp_tools", mcp_tools)
        
        # Send a message to inform user
        tool_names = [t["name"] for t in tools]
        await cl.Message(
            content=f"✅ **Connected to {connection.name}**\n\n"
                    f"Available tools: {', '.join(tool_names)}",
            author="System"
        ).send()
        
    except Exception as e:
        error_msg = format_error_message(e, "MCP Connection")
        await cl.Message(content=error_msg, author="System").send()


@cl.step(type="tool")
async def call_mcp_tool(tool_call: ToolCall) -> str:
    """
    Execute an MCP tool call.
    
    Args:
        tool_call: The tool call to execute
        
    Returns:
        Tool execution result as JSON string
    """
    current_step = cl.context.current_step
    current_step.name = tool_call.name
    current_step.input = json.dumps(tool_call.input, indent=2)
    
    try:
        # Find which MCP connection has this tool
        mcp_tools = cl.user_session.get("mcp_tools", {})
        mcp_name = None
        
        for connection_name, tools in mcp_tools.items():
            if any(tool.get("name") == tool_call.name for tool in tools):
                mcp_name = connection_name
                break
        
        if not mcp_name:
            error_result = {
                "error": f"Tool '{tool_call.name}' not found in any MCP connection"
            }
            current_step.output = json.dumps(error_result, indent=2)
            return current_step.output
        
        # Get MCP session
        mcp_session, _ = cl.context.session.mcp_sessions.get(mcp_name)
        
        if not mcp_session:
            error_result = {
                "error": f"MCP connection '{mcp_name}' not available"
            }
            current_step.output = json.dumps(error_result, indent=2)
            return current_step.output
        
        # Call the tool
        result = await mcp_session.call_tool(tool_call.name, tool_call.input)
        current_step.output = str(result)
        
        return current_step.output
        
    except Exception as e:
        error_msg = format_error_message(e, f"Tool: {tool_call.name}")
        error_result = {"error": str(e)}
        current_step.output = json.dumps(error_result, indent=2)
        print(f"Tool execution error: {traceback.format_exc()}")
        return current_step.output


# ============================================================================
# Gemini Integration
# ============================================================================

async def call_gemini_api(
    messages: List[Dict],
    tools: Optional[List] = None
) -> AssistantResponse:
    """
    Call Gemini API with conversation history and optional tools.
    
    Args:
        messages: Conversation history
        tools: Optional list of Gemini tool objects
        
    Returns:
        AssistantResponse with text and/or tool calls
    """
    try:
        # Build conversation history
        conversation_history = []
        
        for msg in messages:
            role = msg["role"]
            content = msg["content"]
            
            if role == "user":
                if isinstance(content, str):
                    conversation_history.append({
                        "role": "user",
                        "parts": [{"text": content}]
                    })
                else:
                    # Handle tool results
                    text_parts = []
                    for item in content:
                        if item.get("type") == "tool_result":
                            text_parts.append(f"Tool result: {item['content']}")
                    if text_parts:
                        conversation_history.append({
                            "role": "user",
                            "parts": [{"text": "\n".join(text_parts)}]
                        })
                        
            elif role == "assistant":
                if isinstance(content, str):
                    conversation_history.append({
                        "role": "model",
                        "parts": [{"text": content}]
                    })
        
        # Configure model
        model = genai.GenerativeModel(
            GEMINI_MODEL_NAME,
            system_instruction=SYSTEM_PROMPT
        )
        
        # Get latest message
        latest_message = ""
        if conversation_history:
            latest_msg = conversation_history[-1]
            if latest_msg["role"] == "user":
                latest_message = latest_msg["parts"][0]["text"]
        
        # Generate response
        if len(conversation_history) > 1:
            # Use chat session for history
            chat = model.start_chat(history=conversation_history[:-1])
            response = chat.send_message(
                latest_message,
                tools=tools if tools else None,
                generation_config=genai.types.GenerationConfig(**GENERATION_CONFIG)
            )
        else:
            # First message
            response = model.generate_content(
                latest_message,
                tools=tools if tools else None,
                generation_config=genai.types.GenerationConfig(**GENERATION_CONFIG)
            )
        
        # Parse response
        response_text = ""
        tool_calls = []
        
        try:
            if hasattr(response, 'text') and response.text:
                response_text = response.text
        except ValueError:
            # Response might have no text
            pass
        
        # Check for function calls
        if hasattr(response, 'candidates') and response.candidates:
            candidate = response.candidates[0]
            if hasattr(candidate, 'content') and candidate.content:
                if hasattr(candidate.content, 'parts'):
                    for part in candidate.content.parts:
                        if hasattr(part, 'function_call') and part.function_call:
                            fc = part.function_call
                            tool_calls.append(ToolCall(
                                name=fc.name,
                                input=dict(fc.args) if hasattr(fc, 'args') else {},
                                id=f"tool_{fc.name}"
                            ))
        
        return AssistantResponse(
            text=response_text,
            tool_calls=tool_calls,
            stop_reason="tool_use" if tool_calls else "end_turn"
        )
        
    except Exception as e:
        print(f"Gemini API Error: {traceback.format_exc()}")
        error_msg = format_error_message(e, "Gemini API")
        return AssistantResponse(text=error_msg, stop_reason="error")


# ============================================================================
# Chat Handlers
# ============================================================================

@cl.on_chat_start
async def on_chat_start():
    """Initialize chat session."""
    cl.user_session.set("chat_messages", [])
    
    # Send welcome message
    welcome_msg = (
        "👋 **Hello! I'm CodeCrafter AI**\n\n"
        "I'm here to help you with your coding projects. "
        "Ask me anything about programming, and I'll do my best to assist you!"
    )
    await cl.Message(content=welcome_msg, author="CodeCrafter AI").send()


@cl.on_message
async def on_message(message: cl.Message):
    """
    Handle incoming user messages.
    
    This implements a complete conversation loop with tool calling support.
    """
    try:
        # Get conversation history
        chat_messages = cl.user_session.get("chat_messages", [])
        
        # Add user message
        chat_messages.append({"role": "user", "content": message.content})
        
        # Get available tools
        mcp_tools = cl.user_session.get("mcp_tools", {})
        all_tools = flatten_list([tools for _, tools in mcp_tools.items()])
        gemini_tools = convert_mcp_tools_to_gemini(all_tools) if all_tools else None
        
        # Create response message
        response_msg = cl.Message(content="")
        await response_msg.send()
        
        # Initial API call
        response = await call_gemini_api(chat_messages, gemini_tools)
        
        # Tool calling loop
        max_iterations = 5
        iteration = 0
        
        while response.stop_reason == "tool_use" and iteration < max_iterations:
            iteration += 1
            
            # Execute tool calls
            for tool_call in response.tool_calls:
                tool_result = await call_mcp_tool(tool_call)
            
            # Add assistant message with tool call
            chat_messages.append({
                "role": "assistant",
                "content": [
                    {"type": "tool_use", **tool_call.__dict__}
                    for tool_call in response.tool_calls
                ]
            })
            
            # Add tool results
            chat_messages.append({
                "role": "user",
                "content": [
                    {
                        "type": "tool_result",
                        "tool_use_id": tc.id,
                        "content": await call_mcp_tool(tc)
                    }
                    for tc in response.tool_calls
                ]
            })
            
            # Get next response
            response = await call_gemini_api(chat_messages, gemini_tools)
        
        # Stream final response
        if response.text:
            for char in response.text:
                await response_msg.stream_token(char)
        else:
            await response_msg.stream_token("I apologize, but I couldn't generate a response. Please try again.")
        
        await response_msg.update()
        
        # Save assistant response
        chat_messages.append({"role": "assistant", "content": response.text})
        cl.user_session.set("chat_messages", chat_messages)
        
    except Exception as e:
        error_msg = format_error_message(e, "Message Processing")
        await cl.Message(content=error_msg, author="System").send()
        print(f"Message processing error: {traceback.format_exc()}")


# ============================================================================
# Main Entry Point
# ============================================================================

if __name__ == "__main__":
    print("CodeCrafter AI - Starting...")
    print(f"Using Gemini model: {GEMINI_MODEL_NAME}")
