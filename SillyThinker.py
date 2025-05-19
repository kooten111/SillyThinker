# server.py
import os
import json
import logging
import asyncio
from typing import Dict, List, Any, Optional, Tuple
import re

from flask import Flask, request, Response, jsonify, stream_with_context
import requests
from requests.exceptions import RequestException

# Configuration
OLLAMA_BASE_URL = os.environ.get('OLLAMA_BASE_URL', 'http://192.168.1.2:11434')
MIDDLEWARE_PORT = int(os.environ.get('MIDDLEWARE_PORT', 5000))

# Setup logging
logging.basicConfig(level=logging.INFO, 
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

app = Flask(__name__)

# -----------------------------------------------------------------------------
# Utility Functions
# -----------------------------------------------------------------------------

def extract_planning_instructions(messages: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Extract planning instructions from messages based on defined tags."""
    TAG_FORMATS = [
        {"startTag": "<PLANNING_INSTRUCTIONS>", "endTag": "</PLANNING_INSTRUCTIONS>"},
        {"startTag": "<PLANNING>", "endTag": "</PLANNING>"}
    ]

    for message in messages:
        if message.get('content') and isinstance(message['content'], str):
            for tag_format in TAG_FORMATS:
                plan_start = message['content'].find(tag_format["startTag"])
                plan_end = message['content'].find(tag_format["endTag"])

                if plan_start != -1 and plan_end != -1 and plan_end > plan_start:
                    logger.info(f"Found planning instructions with format: {tag_format['startTag']}")
                    plan_instructions = message['content'][
                        plan_start + len(tag_format["startTag"]):plan_end
                    ].strip()

                    modified_message = message.copy()
                    # Remove tags from the message content
                    before_plan = message['content'][:plan_start]
                    after_plan = message['content'][plan_end + len(tag_format["endTag"]):]
                    modified_message['content'] = (before_plan + after_plan).strip()

                    modified_messages = [m.copy() if m != message else modified_message for m in messages]

                    return {
                        "foundInstructions": True,
                        "instructions": plan_instructions,
                        "modifiedMessages": modified_messages
                    }

    logger.info('No planning instructions found in any message with any supported tag format')
    # If no instructions found, just return a copy of the original messages with the flag set to false
    return {
        "foundInstructions": False,
        "instructions": None,
        "modifiedMessages": [m.copy() for m in messages]
    }

def get_latest_user_message(messages: List[Dict[str, Any]], stopping_strings: List[str]) -> bool:
    """Helper function to handle stopping strings and find the last user message."""
    # For scenarios using specific formatting with stopping strings, 
    # we need a more sophisticated approach
    if stopping_strings and len(stopping_strings) > 0 and len(messages) > 0:
        # Get the last message which should contain the entire conversation history
        last_message = messages[-1]
        
        if last_message and 'content' in last_message:
            # Split the content by stopping strings
            parts = [last_message['content']]
            
            for stop_str in stopping_strings:
                # Create a new parts array by splitting each existing part
                new_parts = []
                for part in parts:
                    new_parts.extend(part.split(stop_str))
                parts = new_parts
            
            # Get only the last three segments which should contain:
            # [N-3]: The user turn indicator
            # [N-2]: The content of the user's last message
            # [N-1]: The model turn indicator
            # This avoids picking up OOC: markers from earlier in the conversation
            relevant_parts = parts[-3:] if len(parts) >= 3 else parts
            
            # Only check the actual user message part (should be the middle of the three parts)
            if len(relevant_parts) >= 2:  # We need at least 2 parts to have a user message
                user_message_part = relevant_parts[-2]  # The user message should be the second-to-last part
                if "OOC:" in user_message_part:
                    logger.info("Detected 'OOC:' in the latest user message (stopping string approach).")
                    return True
                
                # Additional logging to help with debugging
                logger.debug(f"Latest user message part: {user_message_part[:100]}...")
    
    return False

def contains_ooc(messages: List[Dict[str, Any]], options: Dict[str, Any]) -> bool:
    """Helper function to check if the latest user message contains "OOC:"."""
    if not messages or not isinstance(messages, list) or len(messages) == 0:
        return False
    
    # Check if we have stopping strings in the options
    stopping_strings = options.get('stop') or options.get('stopping_strings')
    if stopping_strings and len(stopping_strings) > 0:
        return get_latest_user_message(messages, stopping_strings)
    
    # If no stopping strings, use the traditional approach - look for the last user message in the array
    for i in range(len(messages) - 1, -1, -1):
        message = messages[i]
        if message.get('role') == "user":
            if message.get('content') and isinstance(message['content'], str) and "OOC:" in message['content']:
                logger.info("Detected 'OOC:' in the latest user message.")
                return True
            # We found the latest user message and it doesn't contain OOC,
            # so we can stop looking
            return False
    return False

def build_generate_prompt_messages(original_messages: List[Dict[str, Any]], 
                                  planned_response: str) -> List[Dict[str, Any]]:
    """Build messages array for the final generation request using the extracted plan."""
    # Ensure tags are removed from original messages passed to the generation prompt
    cleaned_messages = []
    for msg in original_messages:
        if msg.get('content') and isinstance(msg['content'], str):
            content = msg['content']
            tag_formats = [
                {'start': '<PLANNING_INSTRUCTIONS>', 'end': '</PLANNING_INSTRUCTIONS>'},
                {'start': '<PLANNING>', 'end': '</PLANNING>'}
            ]
            for format_tags in tag_formats:
                tag_start = content.find(format_tags['start'])
                tag_end = content.find(format_tags['end'])
                if tag_start != -1 and tag_end != -1 and tag_end > tag_start:
                    content = content[:tag_start] + content[tag_end + len(format_tags['end']):]
            cleaned_msg = msg.copy()
            cleaned_msg['content'] = content.strip()
            cleaned_messages.append(cleaned_msg)
        else:
            cleaned_messages.append(msg.copy())

    plan_holder_message = {
        "role": "system",
        "content": f"---PLAN---\n{planned_response}"
    }
    generate_instruction = {
        "role": "user",
        "content": "---GENERATION---\n"
    }
    return cleaned_messages + [plan_holder_message, generate_instruction]

# -----------------------------------------------------------------------------
# Stream Processing Functions
# -----------------------------------------------------------------------------

def stream_thinking_tags(model: str, opening: bool = True) -> str:
    """Helper to send thinking tags in the appropriate format."""
    tag = "<think>" if opening else "</think>\n"
    return json.dumps({"model": model, "response": tag, "done": False}) + '\n'

def process_planning_stream(response: requests.Response, model: str):
    """Process planning stream and yield chunks, accumulate content."""
    accumulated_content = ""
    
    for line in response.iter_lines():
        if not line:
            continue
            
        line_text = line.decode('utf-8')
        if not line_text.strip():
            continue
            
        try:
            chunk_data = json.loads(line_text)
            if chunk_data.get('message') and chunk_data['message'].get('content'):
                # Stream plan content within <think> tags
                yield json.dumps({
                    "model": model, 
                    "response": chunk_data['message']['content'], 
                    "done": False
                }) + '\n'
                accumulated_content += chunk_data['message']['content']
        except json.JSONDecodeError as e:
            logger.warning(f"Middleware: Failed to parse JSON line from plan response: {line_text}, {e}")
            # Optionally stream raw line if parsing fails
            yield json.dumps({
                "model": model, 
                "response": line_text, 
                "done": False
            }) + '\n'
    
    logger.info("Middleware: Planning response stream finished.")
    return accumulated_content.strip() or "Continue the conversation naturally."

def stream_chat_response(response: requests.Response):
    """Stream chat format response directly to client."""
    for line in response.iter_lines():
        if line:
            yield line + b'\n'

def adapt_chat_to_generate_format(response: requests.Response, model: str):
    """Adapt chat stream to generate format and stream to client."""
    logger.info("Middleware: Processing and adapting generation stream to client...")
    
    for line in response.iter_lines():
        if not line:
            continue
            
        line_text = line.decode('utf-8')
        if not line_text.strip():
            continue
            
        try:
            chunk_data = json.loads(line_text)
            if chunk_data.get('message') and chunk_data['message'].get('content'):
                # Adapt Chat message chunk to Generate format
                yield json.dumps({ 
                    "model": chunk_data.get('model', model), 
                    "response": chunk_data['message']['content'], 
                    "done": False 
                }) + '\n'
            elif chunk_data.get('done', False):
                # Handle final done chunk, adapt to Generate format including stats
                generate_chunk = { 
                    "model": chunk_data.get('model', model), 
                    "response": "", 
                    "done": True 
                }
                
                # Copy statistics fields if present
                stats_fields = [
                    "total_duration", "load_duration", "prompt_eval_count", 
                    "prompt_eval_duration", "eval_count", "eval_duration", 
                    "context", "done_reason"
                ]
                
                for key in stats_fields:
                    if key in chunk_data:
                        generate_chunk[key] = chunk_data[key]
                
                yield json.dumps(generate_chunk) + '\n'
            else:
                logger.warning(f"Middleware: Unexpected chunk in generation response: {chunk_data}")
                raw_chunk = {"model": model, "response": json.dumps(chunk_data), "done": False}
                yield json.dumps(raw_chunk) + '\n'
        except json.JSONDecodeError as e:
            logger.warning(f"Middleware: Failed to parse JSON line from generation response: {line_text}, {e}")
            error_chunk = {"model": model, "response": line_text, "done": False}
            yield json.dumps(error_chunk) + '\n'
    
    logger.info("Middleware: Generation stream finished.")

def send_instructional_message(model: str, is_generate_format: bool) -> str:
    """Send instructional message response in appropriate format."""
    message = "Please include planning instructions in your prompt using <PLANNING>...</PLANNING> or <PLANNING_INSTRUCTIONS>...</PLANNING_INSTRUCTIONS> tags."
    
    if is_generate_format:
        return json.dumps({"model": model, "response": message, "done": True}) + '\n'
    else:
        return json.dumps({ 
            "model": model, 
            "message": {"role": "assistant", "content": message}, 
            "done": True 
        }) + '\n'

# -----------------------------------------------------------------------------
# Core Processing Logic
# -----------------------------------------------------------------------------

def process_ollama_request(request_data: Dict[str, Any], is_generate_format: bool = False):
    """Central processing function that handles both chat and generate endpoints."""
    endpoint_type = "generate" if is_generate_format else "chat"
    logger.info(f"Middleware: Received POST request for /api/{endpoint_type}")

    # Extract request data based on endpoint type
    if is_generate_format:
        model = request_data.get('model')
        prompt = request_data.get('prompt')
        options = request_data.get('options', {})
        
        if not model:
            return Response(json.dumps({"error": "Model not specified"}), 
                           status=400, mimetype='application/json')
        if not isinstance(prompt, str):
            return Response(json.dumps({"error": "Prompt must be a string"}), 
                           status=400, mimetype='application/json')
        
        messages = [{"role": "user", "content": prompt}]
    else:
        model = request_data.get('model')
        messages = request_data.get('messages', [])
        options = request_data.get('options', {})
        
        if not model:
            return Response(json.dumps({"error": "Model not specified"}), 
                           status=400, mimetype='application/json')
        
        # Log stopping strings if present (helpful for debugging)
        stopping_strings = options.get('stop') or options.get('stopping_strings')
        if stopping_strings:
            logger.info(f"Stopping strings present: {stopping_strings}")

    def generate_response():
        try:
            # Extract planning instructions and check for OOC marker
            extract_result = extract_planning_instructions(messages)
            ooc_present = contains_ooc(extract_result["modifiedMessages"], options)

            # Determine execution path
            perform_planning = extract_result["foundInstructions"] and not ooc_present
            skip_instructional_message = ooc_present

            if perform_planning:
                # --- Path 1: Execute Planning + Generation ---
                logger.info("Middleware: Planning instructions found, and 'OOC:' not detected. Proceeding with planning.")
                
                # Prepare planning request
                plan_messages = extract_result["modifiedMessages"] + [{ 
                    "role": "user", 
                    "content": extract_result["instructions"] 
                }]
                
                logger.info("Middleware: Sending planning request to Ollama...")
                
                # Stream opening think tag
                yield stream_thinking_tags(model, True)
                
                # Execute planning request
                plan_request_data = {
                    "model": model,
                    "messages": plan_messages,
                    "options": {**options, "num_predict": 1500},  # Limit plan length
                    "stream": True
                }
                
                plan_response = requests.post(
                    f"{OLLAMA_BASE_URL}/api/chat",
                    json=plan_request_data,
                    headers={"Content-Type": "application/json"},
                    stream=True
                )
                
                if not plan_response.ok:
                    error_body = plan_response.text
                    raise RequestException(f"Ollama planning request failed: Status {plan_response.status_code} - {error_body}")
                
                # Process planning stream
                accumulated_content = ""
                for chunk in process_planning_stream(plan_response, model):
                    yield chunk
                    # Extract content from the chunk to accumulate
                    try:
                        chunk_data = json.loads(chunk.strip())
                        accumulated_content += chunk_data.get("response", "")
                    except json.JSONDecodeError:
                        pass
                
                planned_response = accumulated_content.strip() or "Continue the conversation naturally."
                
                # Stream closing think tag
                yield stream_thinking_tags(model, False)
                
                # Build generation messages with plan
                generate_messages = build_generate_prompt_messages(messages, planned_response)
                logger.info("Middleware: Sending generation request with plan to Ollama...")
                
                # Execute generation request
                generate_request_data = {
                    "model": model,
                    "messages": generate_messages,
                    "options": options,
                    "stream": True
                }
                
                generate_response = requests.post(
                    f"{OLLAMA_BASE_URL}/api/chat",
                    json=generate_request_data,
                    headers={"Content-Type": "application/json"},
                    stream=True
                )
                
                if not generate_response.ok:
                    error_body = generate_response.text
                    raise RequestException(f"Ollama generation request failed: Status {generate_response.status_code} - {error_body}")
                
                # Stream generation response in appropriate format
                if is_generate_format:
                    for chunk in adapt_chat_to_generate_format(generate_response, model):
                        yield chunk
                else:
                    for chunk in stream_chat_response(generate_response):
                        yield chunk
                
            elif not extract_result["foundInstructions"] and not skip_instructional_message:
                # --- Path 2: No Planning Instructions, Show Help ---
                logger.info("Middleware: No planning instructions found and 'OOC:' not detected. Returning instruction message.")
                yield send_instructional_message(model, is_generate_format)
                
            else:
                # --- Path 3: Direct Generation (Skip Planning) ---
                logger.info("Middleware: 'OOC:' detected or planning instructions not found, skipping planning. Proceeding directly to generation.")
                
                # Use the messages with planning tags already removed
                generate_messages = extract_result["modifiedMessages"]
                logger.info("Middleware: Sending direct generation request (no plan) to Ollama...")
                
                # Execute generation request
                generate_request_data = {
                    "model": model,
                    "messages": generate_messages,
                    "options": options,
                    "stream": True
                }
                
                generate_response = requests.post(
                    f"{OLLAMA_BASE_URL}/api/chat",
                    json=generate_request_data,
                    headers={"Content-Type": "application/json"},
                    stream=True
                )
                
                if not generate_response.ok:
                    error_body = generate_response.text
                    raise RequestException(f"Ollama generation request failed: Status {generate_response.status_code} - {error_body}")
                
                # Stream generation response in appropriate format
                if is_generate_format:
                    for chunk in adapt_chat_to_generate_format(generate_response, model):
                        yield chunk
                else:
                    for chunk in stream_chat_response(generate_response):
                        yield chunk
                
        except Exception as error:
            logger.error(f"Middleware: Error during {endpoint_type} process:", exc_info=True)
            yield json.dumps({ 
                "model": model, 
                "response": f"ERROR: {str(error)}", 
                "done": True 
            }) + '\n'

    return Response(stream_with_context(generate_response()), 
                    mimetype='application/json')

# -----------------------------------------------------------------------------
# API Routes
# -----------------------------------------------------------------------------

@app.route('/api/tags', methods=['GET'])
def list_models():
    """Handle listing models."""
    logger.info(f"Middleware: Received GET request for /api/tags. Forwarding to {OLLAMA_BASE_URL}/api/tags...")
    
    try:
        response = requests.get(f"{OLLAMA_BASE_URL}/api/tags")
        
        if not response.ok:
            error_body = response.text
            return Response(
                f"Error fetching models from Ollama: Status {response.status_code} - {error_body}",
                status=response.status_code
            )
        
        def generate():
            for chunk in response.iter_content(chunk_size=4096):
                yield chunk
        
        return Response(stream_with_context(generate()), 
                       mimetype='application/json')
        
    except Exception as error:
        logger.error('Middleware: Error fetching models from Ollama:', exc_info=True)
        return Response(
            f"Middleware failed to list models: {str(error)}",
            status=500
        )

@app.route('/api/chat', methods=['POST'])
def chat_completions():
    """Handle chat completions."""
    return process_ollama_request(request.json, False)

@app.route('/api/generate', methods=['POST'])
def generate_completions():
    """Handle generate completions."""
    return process_ollama_request(request.json, True)

# Start the Flask server
if __name__ == '__main__':
    logger.info(f"Python Middleware listening on port {MIDDLEWARE_PORT}")
    logger.info(f"Forwarding requests to Ollama at {OLLAMA_BASE_URL}")
    app.run(host='0.0.0.0', port=MIDDLEWARE_PORT, debug=False)