# SillyThinker
A middleware proxy for Ollama and Sillytavern (or other clients/frontends) that adds thinking/planning capabilities to any model.

## What is SillyThinker?

SillyThinker is a middleware that sits between Sillytavern and Ollama. It adds a "thinking" capability to any language model by intercepting requests and implementing a two-step process:

1. **Planning Phase**: The model first thinks about how to respond (within special `<think>` tags)
2. **Response Phase**: The model then generates a well-considered response based on the planning

As opposed to a "thinking" model this is a two stage with two separate prompts going to the model.

## Features

- Seamlessly integrates with existing Ollama deployments
- Works with any model in Ollama
- Adds planning capabilities without modifying the underlying model
- Supports both `/api/chat` and `/api/generate` endpoints
- Preserves Ollama functionality
- Special "Out-of-Character" (OOC) mode for bypassing planning when needed

## Installation

### Setup

1. Clone this repository or download `SillyThinker.py`

2. Install required dependencies:
   ```bash
   pip install flask requests
   ```

3. Set environment variables (optional):
   ```bash
   # Set the URL where Ollama is running
   export OLLAMA_BASE_URL=http://192.168.1.2:11434
   
   # Set the port for SillyThinker middleware
   export MIDDLEWARE_PORT=5000
   ```

4. Run:
   ```bash
   python SillyThinker.py
   ```

## Usage

### Basic Usage

1. Point your client application (Sillytavern probably) to the middleware instead of directly to Ollama
   ```
   http://localhost:5000/api/chat
   ```
   instead of
   ```
   http://localhost:11434/api/chat
   ```

2. Include planning instructions in your prompts using one of these formats:
   ```
   <PLANNING_INSTRUCTIONS>
   Think about how to respond to this question. Consider key points and organization.
   </PLANNING_INSTRUCTIONS>
   ```
   or
   ```
   <PLANNING>
   Consider technical accuracy, use examples, and explain complex terms.
   </PLANNING>
   ```

3. The model will first think through its response (visible to you inside `<think>` tags) and then provide a final response.

### Out-of-Character Mode

If you need to bypass the planning phase for specific messages, include "OOC:" at the beginning of your message. This is useful for giving direct instructions to the model or when planning isn't needed.

Example:
```
OOC: Please list the top 5 capital cities by population without planning.
```

## How It Works

1. When a request is sent to SillyThinker:
   - It checks for planning instructions in the messages
   - If found, it extracts them and modifies the message array
   
2. For planning instructions:
   - A separate request is sent to Ollama with the planning instructions
   - The planning response is streamed back with `<think>` tags
   - The planning response is collected
   
3. For final generation:
   - The original messages plus the planning result are sent to Ollama
   - The final response is streamed back to the client

## Configuration Options

- `OLLAMA_BASE_URL`: URL where Ollama is running (default: http://192.168.1.2:11434)
- `MIDDLEWARE_PORT`: Port for SillyThinker middleware (default: 5000)

## Sample Prompts

### Simple Planning Prompt
```
<PLANNING>
Consider pros and cons, give balanced view, use examples.
</PLANNING>

What are the advantages and disadvantages of electric vehicles?
```

### Detailed Planning Instructions
```
<PLANNING_INSTRUCTIONS>
First, explain what quantum computing is at a high level.
Then, compare to classical computing with 2-3 key differences.
Use at least one real-world application example.
Avoid unnecessary technical jargon.
</PLANNING_INSTRUCTIONS>

Can you explain quantum computing to a high school student?
```

