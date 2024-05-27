from fastapi import FastAPI, WebSocket, Request
from fastapi.responses import HTMLResponse
import httpx
import json
import logging
import os
from modal import Image, App, asgi_app
import modal

web_app = FastAPI()
app = App("llm_proxy")

image = Image.debian_slim().pip_install("httpx")

logging.basicConfig(level=logging.CRITICAL)

def parse_hume_message(messages_payload: dict) -> str:
        """
        Parses the payload of messages received from a client, extracting the latest user message
        and constructing the chat history with contextualized utterances.

        Args:
            messages_payload (dict): The payload containing messages from the chat.

        Returns:
            tuple[str, list]: A tuple containing the last user message and the constructed chat history.
        """

        messages = messages_payload["messages"]
        last_user_message = messages[-1]["message"]["content"]

        return last_user_message

@web_app.websocket("/llm_proxy")
async def llm_proxy_endpoint(websocket: WebSocket):
    """
    Handles incoming WebSocket connections and proxies the messages to a remote LLM endpoint.

    This endpoint listens for incoming text messages via WebSocket, forwards these messages
    to a specified LLM endpoint URL, and returns the responses from the LLM back to the client
    via WebSocket. This allows the FastAPI application to serve as an interface for interacting
    with a remote LLM service.

    Args:
        websocket (WebSocket): An instance of WebSocket connection.
        llm_url (str): The URL of the remote LLM endpoint to proxy the messages to.

    Workflow:
        1. Accepts an incoming WebSocket connection.
        2. Enters a loop to continuously receive messages from the WebSocket connection.
        3. For each message received, it forwards the message to the specified LLM endpoint URL.
        4. Waits for the response from the LLM endpoint and sends it back to the client via WebSocket.

    This endpoint enables the FastAPI application to act as a proxy, allowing clients to interact with
    a remote LLM service through a WebSocket interface, without directly exposing the LLM endpoint.
    """
    # Accept the incoming WebSocket connection.
    await websocket.accept()
    
    llm_url = os.getenv("LLM_URL")
    
    # Continuously listen for messages from the WebSocket connection.
    while True:
        # Wait for a text message from the WebSocket, then asynchronously receive it.
        data = await websocket.receive_text()
        logging.info(f"Received message: {data}")
        
        # Deserialize the text message (JSON format) to a Python dictionary.
        hume_socket_message = json.loads(data)
        
        message = parse_hume_message(hume_socket_message)
        logging.info(f"Last message: {message}")
                
        # Forward the received message to the specified LLM endpoint URL.
        async with httpx.AsyncClient(timeout=30.0) as client:
            response = await client.post(llm_url, json={"question": message})

        # Get the response from the LLM endpoint.
        llm_response = response.json()
        logging.info(f"LLM response: {llm_response['text']}")

        responses = []
        responses.append(json.dumps({"type": "assistant_input", "text": llm_response['text']}))
        responses.append(json.dumps({"type": "assistant_end"}))
        
        for response in responses:
            await websocket.send_text(response)

@app.function(image=image, secrets=[modal.Secret.from_name("llm_url")])
@asgi_app()
def fastapi_app():
    return web_app
