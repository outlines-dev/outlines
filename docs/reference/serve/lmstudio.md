# Serve with LM Studio

!!! tip "Would rather not self-host?"

    If you want to get started quickly with JSON-structured generation you can call instead [.json](https://h1xbpbfsf0w.typeform.com/to/ZgBCvJHF), a [.txt](http://dottxt.co) API that guarantees valid JSON.

[LM Studio](https://lmstudio.ai/) is an application that runs local LLMs. It flexibly mixes GPU and CPU compute in hardware-constrained environments.

As of [LM Studio 0.3.4](https://lmstudio.ai/blog/lmstudio-v0.3.4), it natively supports Outlines for structured text generation, using an OpenAI-compatible endpoint.

## Setup

1. Install LM Studio by visiting their [downloads page](https://lmstudio.ai/download).
2. Enable the LM Studio [server functionality](https://lmstudio.ai/docs/basics/server).
3. Download [a model](https://lmstudio.ai/docs/basics#1-download-an-llm-to-your-computer).
4. Install Python dependencies.
```bash
pip install pydantic openai
```

## Calling the server

By default, LM Studio will serve from `http://localhost:1234`. If you are serving on a different port or host, make sure to change the `base_url` argument in `OpenAI` to the relevant location.

```python
class Testing(BaseModel):
    """
    A class representing a testing schema.
    """
    name: str
    age: int

openai_client = openai.OpenAI(
    base_url="http://0.0.0.0:1234/v1",
    api_key="dopeness"
)

# Make a request to the local LM Studio server
response = openai_client.beta.chat.completions.parse(
    model="hugging-quants/Llama-3.2-1B-Instruct-Q8_0-GGUF",
    messages=[
        {"role": "system", "content": "You are like so good at whatever you do."},
        {"role": "user", "content": "My name is Cameron and I am 28 years old. What's my name and age?"}
    ],
    response_format=Testing
)
```

You should receive a `ParsedChatCompletion[Testing]` object back:

```python
ParsedChatCompletion[Testing](
    id='chatcmpl-3hykyf0fxus7jc90k6gwlw',
    choices=[
        ParsedChoice[Testing](
            finish_reason='stop',
            index=0,
            logprobs=None,
            message=ParsedChatCompletionMessage[Testing](
                content='{ "age": 28, "name": "Cameron" }',
                refusal=None,
                role='assistant',
                function_call=None,
                tool_calls=[],
                parsed=Testing(name='Cameron', age=28)
            )
        )
    ],
    created=1728595622,
    model='lmstudio-community/Phi-3.1-mini-128k-instruct-GGUF/Phi-3.1-mini-128k-instruct-Q4_K_M.gguf',
    object='chat.completion',
    service_tier=None,
    system_fingerprint='lmstudio-community/Phi-3.1-mini-128k-instruct-GGUF/Phi-3.1-mini-128k-instruct-
Q4_K_M.gguf',
    usage=CompletionUsage(
        completion_tokens=17,
        prompt_tokens=47,
        total_tokens=64,
        completion_tokens_details=None,
        prompt_tokens_details=None
    )
)
```

You can retrieve your `Testing` object with 

```python
response.choices[0].message.parsed
```
