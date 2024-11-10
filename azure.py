from openai import AzureOpenAI
import time
import os


def chat(system_message, prompt, model):

    return _attempt_api_call(lambda: _chat_internal(system_message, prompt, model))


def stream(system_message, prompt, model):
    if "o1" in model.lower():
        response = chat(system_message, prompt, model)
        return _simulate_stream(response)

    client = _get_azure_client()
    messages = _create_messages(system_message, prompt, model)

    parameters = {
        "model": model,
        "messages": messages,
        "stream": True
    }

    try:
        stream_response = client.chat.completions.create(**parameters)
        return (
            chunk.choices[0].delta.content
            for chunk in stream_response
            if chunk.choices and chunk.choices[0].delta.content
        )
    except Exception as e:
        raise e


def message_chat(messages, model, max_tokens=None, temp=None):
    return _attempt_api_call(lambda: _message_chat_internal(messages, model, max_tokens, temp))


def message_stream(messages, model):
    if "o1" in model.lower():
        response = message_chat(messages, model)
        return _simulate_stream(response)

    client = _get_azure_client()
    consolidated_messages = _consolidate_messages(messages, model)

    parameters = {
        "model": model,
        "messages": consolidated_messages,
        "stream": True
    }

    try:
        stream_response = client.chat.completions.create(**parameters)
        return (
            chunk.choices[0].delta.content
            for chunk in stream_response
            if chunk.choices and chunk.choices[0].delta.content
        )
    except Exception as e:
        raise e


def _get_azure_client():
    return AzureOpenAI(
        api_key=os.getenv("AZURE_API_KEY"),
        azure_endpoint="https://aidan-m1z7896m-eastus2.openai.azure.com/",
        api_version="2023-09-15-preview"
    )


def _create_messages(system_message, prompt, model):
    if "o1" in model.lower():
        user_content = f"{system_message}\n{prompt}"
        return [{"role": "user", "content": user_content}]
    else:
        return [
            {"role": "system", "content": system_message},
            {"role": "user", "content": prompt}
        ]


def _consolidate_messages(messages, model):
    if "o1" not in model.lower():
        return messages

    system_content = ""
    consolidated_messages = []

    for message in messages:
        if message["role"] == "system":
            system_content += message["content"] + "\n"
        else:
            if system_content:
                message["content"] = f"{system_content}\n{message['content']}"
                system_content = ""
            consolidated_messages.append(message)

    if system_content:
        if consolidated_messages:
            consolidated_messages[-1]["content"] = f"{system_content}\n{consolidated_messages[-1]['content']}"
        else:
            consolidated_messages.append(
                {"role": "user", "content": system_content.strip()})

    return consolidated_messages


def _attempt_api_call(func):
    while True:
        try:
            return func()
        except Exception as e:
            error_message = str(e)
            if "429" in error_message:
                # Wait for 2 seconds before retrying
                time.sleep(2)
            else:
                print(e)
            continue



def _chat_internal(system_message, prompt, model):
    client = _get_azure_client()
    messages = _create_messages(system_message, prompt, model)
    parameters = {
        "model": model,
        "messages": messages,
    }
    response = client.chat.completions.create(**parameters)
    return response.choices[0].message.content


def _message_chat_internal(messages, model, max_tokens=None, temp=None):
    client = _get_azure_client()
    consolidated_messages = _consolidate_messages(messages, model)
    parameters = {
        "model": model,
        "messages": consolidated_messages,
    }
    if max_tokens is not None:
        parameters["max_tokens"] = max_tokens
    if temp is not None:
        parameters["temperature"] = temp
    response = client.chat.completions.create(**parameters)
    return response.choices[0].message.content


def _simulate_stream(response):
    for char in response:
        yield char
        time.sleep(0.01 if char.isspace() else 0.005)
