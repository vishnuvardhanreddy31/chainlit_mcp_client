import json

from mcp import ClientSession
import google.generativeai as genai

import chainlit as cl

import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Configure Gemini
genai.configure(api_key=os.getenv("GEMINI_API_KEY"))
gemini_model = genai.GenerativeModel('gemini-2.0-flash')
SYSTEM = "You are CodeCrafter, an intelligent AI coding assistant. You help developers with their coding tasks, manage expenses, and provide comprehensive assistance using available MCP server tools. Be helpful, precise, and maintain context throughout conversations."



def flatten(xss):
    return [x for xs in xss for x in xs]

@cl.on_mcp_connect
async def on_mcp(connection, session: ClientSession):
    result = await session.list_tools()
    tools = [{
        "name": t.name,
        "description": t.description,
        "input_schema": t.inputSchema,
        } for t in result.tools]
    
    mcp_tools = cl.user_session.get("mcp_tools", {})
    mcp_tools[connection.name] = tools
    cl.user_session.set("mcp_tools", mcp_tools)


@cl.step(type="tool") 
async def call_tool(tool_use):
    tool_name = tool_use.name
    tool_input = tool_use.input
    
    current_step = cl.context.current_step
    current_step.name = tool_name
    
    # Identify which mcp is used
    mcp_tools = cl.user_session.get("mcp_tools", {})
    mcp_name = None

    for connection_name, tools in mcp_tools.items():
        if any(tool.get("name") == tool_name for tool in tools):
            mcp_name = connection_name
            break
    
    if not mcp_name:
        current_step.output = json.dumps({"error": f"Tool {tool_name} not found in any MCP connection"})
        return current_step.output
    
    mcp_session, _ = cl.context.session.mcp_sessions.get(mcp_name)
    
    if not mcp_session:
        current_step.output = json.dumps({"error": f"MCP {mcp_name} not found in any MCP connection"})
        return current_step.output
    
    try:
        current_step.output = await mcp_session.call_tool(tool_name, tool_input)
    except Exception as e:
        current_step.output = json.dumps({"error": str(e)})
    
    return current_step.output

async def call_gemini(chat_messages):
    mcp_tools = cl.user_session.get("mcp_tools", {})
    # Flatten the tools from all MCP connections
    tools = flatten([tools for _, tools in mcp_tools.items()])
    print(f"Available tools: {[tool.get('name') for tool in tools]}")
    
    # Convert tools to Gemini function calling format
    gemini_tools = []
    if tools:
        try:
            for tool in tools:
                # Clean up the input schema to only include fields Gemini supports
                input_schema = tool.get("input_schema", {"type": "object", "properties": {}})
                
                # Convert to Gemini's expected format - don't use raw schema
                # Instead, create a simple parameter structure
                parameters = {}
                required_params = input_schema.get("required", [])
                
                if "properties" in input_schema and input_schema["properties"]:
                    for prop_name, prop_def in input_schema["properties"].items():
                        param_type = prop_def.get("type", "string")
                        param_desc = prop_def.get("description", "")
                        
                        # Convert to simple parameter definition
                        parameters[prop_name] = {
                            "type": param_type,
                            "description": param_desc
                        }
                        
                        # Handle array items
                        if param_type == "array" and "items" in prop_def:
                            parameters[prop_name]["items"] = {
                                "type": prop_def["items"].get("type", "string")
                            }
                
                # Create function declaration in a simple format
                function_declaration = {
                    "name": tool["name"],
                    "description": tool["description"]
                }
                
                # Only add parameters if they exist
                if parameters:
                    function_declaration["parameters"] = {
                        "type": "object",
                        "properties": parameters
                    }
                    if required_params:
                        function_declaration["parameters"]["required"] = required_params
                
                gemini_tools.append(function_declaration)
            print(f"Successfully converted {len(gemini_tools)} tools for Gemini")
        except Exception as e:
            print(f"Error converting tools: {e}")
            import traceback
            print(f"Tool conversion traceback: {traceback.format_exc()}")
            gemini_tools = []
    
    # Prepare the conversation history for Gemini
    contents = []
    for message in chat_messages:
        if message["role"] == "user":
            if isinstance(message["content"], str):
                contents.append({
                    "role": "user",
                    "parts": [{"text": message["content"]}]
                })
            else:
                # Handle tool results
                text_parts = []
                for content_item in message["content"]:
                    if content_item.get("type") == "tool_result":
                        text_parts.append(f"Tool result: {content_item['content']}")
                if text_parts:
                    contents.append({
                        "role": "user", 
                        "parts": [{"text": "\n".join(text_parts)}]
                    })
        elif message["role"] == "assistant":
            if isinstance(message["content"], str):
                contents.append({
                    "role": "model",
                    "parts": [{"text": message["content"]}]
                })
            else:
                # Handle assistant messages with tool calls
                parts = []
                for content_item in message["content"]:
                    if hasattr(content_item, 'text'):
                        parts.append({"text": content_item.text})
                    elif hasattr(content_item, 'type') and content_item.type == 'tool_use':
                        # Add function call part - simplified approach
                        parts.append({"text": f"[Function call: {content_item.name} with args: {content_item.input}]"})
                if parts:
                    contents.append({
                        "role": "model",
                        "parts": parts
                    })
    
    try:
        # Configure the model with system instruction
        model = genai.GenerativeModel(
            'gemini-2.0-flash',
            system_instruction=SYSTEM
        )
        
        # Convert our tool declarations to the proper format for Gemini
        gemini_tool_objects = []
        if gemini_tools:
            for tool_def in gemini_tools:
                # Create a simple function object that Gemini can understand
                func_decl = genai.protos.FunctionDeclaration(
                    name=tool_def["name"],
                    description=tool_def["description"]
                )
                
                # Add parameters if they exist
                if "parameters" in tool_def and tool_def["parameters"].get("properties"):
                    properties = {}
                    for param_name, param_def in tool_def["parameters"]["properties"].items():
                        # Map string types to Gemini enums
                        param_type = genai.protos.Type.STRING
                        if param_def.get("type") == "number":
                            param_type = genai.protos.Type.NUMBER
                        elif param_def.get("type") == "integer":
                            param_type = genai.protos.Type.INTEGER  
                        elif param_def.get("type") == "boolean":
                            param_type = genai.protos.Type.BOOLEAN
                        elif param_def.get("type") == "array":
                            param_type = genai.protos.Type.ARRAY
                        elif param_def.get("type") == "object":
                            param_type = genai.protos.Type.OBJECT
                        
                        properties[param_name] = genai.protos.Schema(
                            type=param_type,
                            description=param_def.get("description", "")
                        )
                    
                    func_decl.parameters = genai.protos.Schema(
                        type=genai.protos.Type.OBJECT,
                        properties=properties,
                        required=tool_def["parameters"].get("required", [])
                    )
                
                gemini_tool_objects.append(genai.protos.Tool(function_declarations=[func_decl]))
        
        # Build conversation history for Gemini (proper format)
        conversation_history = []
        
        # Process all messages to build proper conversation context
        for content in contents:
            if content["role"] == "user":
                user_text = ""
                for part in content["parts"]:
                    user_text += part["text"] + " "
                conversation_history.append({
                    "role": "user",
                    "parts": [{"text": user_text.strip()}]
                })
            elif content["role"] == "model":
                model_text = ""
                for part in content["parts"]:
                    model_text += part["text"] + " "
                if model_text.strip():
                    conversation_history.append({
                        "role": "model", 
                        "parts": [{"text": model_text.strip()}]
                    })
        
        # Get the latest user message
        latest_message = ""
        if conversation_history:
            latest_msg = conversation_history[-1]
            if latest_msg["role"] == "user":
                latest_message = latest_msg["parts"][0]["text"]
        
        # Use chat session for history awareness
        if len(conversation_history) > 1:
            # Start chat with history (exclude the last message)
            chat = model.start_chat(history=conversation_history[:-1])
            
            # Send the latest message with tools if available
            if gemini_tool_objects:
                response = chat.send_message(
                    latest_message,
                    tools=gemini_tool_objects,
                    generation_config=genai.types.GenerationConfig(
                        temperature=0.7,
                        max_output_tokens=1024,
                    )
                )
            else:
                response = chat.send_message(
                    latest_message,
                    generation_config=genai.types.GenerationConfig(
                        temperature=0.7,
                        max_output_tokens=1024,
                    )
                )
        else:
            # First message - use generate_content
            if gemini_tool_objects:
                response = model.generate_content(
                    latest_message,
                    tools=gemini_tool_objects,
                    generation_config=genai.types.GenerationConfig(
                        temperature=0.7,
                        max_output_tokens=1024,
                    )
                )
            else:
                response = model.generate_content(
                    latest_message,
                    generation_config=genai.types.GenerationConfig(
                        temperature=0.7,
                        max_output_tokens=1024,
                    )
                )
        
        # Create response object similar to Anthropic's format
        class MockResponse:
            def __init__(self, text, function_calls=None):
                self.content = []
                if text:
                    text_content = type('obj', (object,), {'text': text, 'type': 'text'})
                    self.content.append(text_content)
                if function_calls:
                    for fc in function_calls:
                        tool_content = type('obj', (object,), {
                            'type': 'tool_use',
                            'name': fc.name,
                            'input': dict(fc.args) if hasattr(fc, 'args') else {},
                            'id': f"tool_{fc.name}"
                        })
                        self.content.append(tool_content)
                self.stop_reason = "tool_use" if function_calls else "end_turn"
        
        # Check if there are function calls in the response
        function_calls = []
        response_text = ""
        
        try:
            if hasattr(response, 'text') and response.text:
                response_text = response.text
        except ValueError:
            # Response might be blocked or have no text
            response_text = ""
            
        if hasattr(response, 'candidates') and response.candidates:
            candidate = response.candidates[0]
            if hasattr(candidate, 'content') and candidate.content and hasattr(candidate.content, 'parts'):
                for part in candidate.content.parts:
                    if hasattr(part, 'function_call') and part.function_call:
                        function_calls.append(part.function_call)
        
        return MockResponse(response_text, function_calls if function_calls else None)
        
    except Exception as e:
        error_msg = f"Error: {str(e)}"
        print(f"Gemini API Error: {e}")
        print(f"Error type: {type(e)}")
        import traceback
        print(f"Traceback: {traceback.format_exc()}")
        
        class ErrorResponse:
            def __init__(self, text):
                text_content = type('obj', (object,), {'text': text, 'type': 'text'})
                self.content = [text_content]
                self.stop_reason = "end_turn"
        
        return ErrorResponse(error_msg)

@cl.on_chat_start
async def start_chat():
    cl.user_session.set("chat_messages", [])


@cl.on_message
async def on_message(msg: cl.Message):   
    chat_messages = cl.user_session.get("chat_messages")
    chat_messages.append({"role": "user", "content": msg.content})
    response = await call_gemini(chat_messages)
    
    # Create streaming message for the response
    response_msg = cl.Message(content="")
    
    while response.stop_reason == "tool_use":
        tool_use = next(block for block in response.content if block.type == "tool_use")
        tool_result = await call_tool(tool_use)

        messages = [
            {"role": "assistant", "content": response.content},
            {
                "role": "user",
                "content": [
                    {
                        "type": "tool_result",
                        "tool_use_id": tool_use.id,
                        "content": str(tool_result),
                    }
                ],
            },
        ]

        chat_messages.extend(messages)
        response = await call_gemini(chat_messages)

    final_response = next(
        (block.text for block in response.content if hasattr(block, "text")),
        None,
    )

    # Stream the final response
    if final_response:
        for char in final_response:
            await response_msg.stream_token(char)
    
    await response_msg.send()

    chat_messages = cl.user_session.get("chat_messages")
    chat_messages.append({"role": "assistant", "content": final_response})
