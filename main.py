from fastapi import FastAPI, WebSocket
import httpx
import json
import logging

LLM_URL = "https://flowiseai-railway-production-dd26.up.railway.app/api/v1/prediction/49824ea4-9fb4-4480-b651-611cd1c9c29e"

app = FastAPI()

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


@app.websocket("/llm_proxy")
async def llm_proxy_endpoint(websocket: WebSocket, llm_url: str = LLM_URL):
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
