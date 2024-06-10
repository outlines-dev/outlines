#  _______________________________
# / Don't want to self-host?      \
# \ Try .json at http://dottxt.co /
#  -------------------------------
#        \   ^__^
#         \  (oo)\_______
#            (__)\       )\/\
#                ||----w |
#                ||     ||
#
#
# Copyright 2024- the Outlines developers
# Copyright 2023 the vLLM developers
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import argparse
import json
from typing import AsyncGenerator

import uvicorn
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, Response, StreamingResponse
from vllm.engine.arg_utils import AsyncEngineArgs
from vllm.engine.async_llm_engine import AsyncLLMEngine
from vllm.sampling_params import SamplingParams
from vllm.utils import random_uuid

from outlines.integrations.vllm import JSONLogitsProcessor, RegexLogitsProcessor

TIMEOUT_KEEP_ALIVE = 5  # seconds.
TIMEOUT_TO_PREVENT_DEADLOCK = 1  # seconds.
app = FastAPI()
engine = None


@app.get("/health")
async def health() -> Response:
    """Health check."""
    return Response(status_code=200)


@app.post("/generate")
async def generate(request: Request) -> Response:
    """Generate completion for the request.

    The request should be a JSON object with the following fields:
    - prompt: the prompt to use for the generation.
    - schema: the JSON schema to use for the generation (if regex is not provided).
    - regex: the regex to use for the generation (if schema is not provided).
    - stream: whether to stream the results or not.
    - other fields: the sampling parameters (See `SamplingParams` for details).
    """
    assert engine is not None

    request_dict = await request.json()
    prompt = request_dict.pop("prompt")
    stream = request_dict.pop("stream", False)

    json_schema = request_dict.pop("schema", None)
    regex_string = request_dict.pop("regex", None)
    whitespace_pattern = request_dict.pop("whitespace_pattern", None)
    enable_schema_optimization = request_dict.pop("enable_schema_optimization", False)
    if json_schema is not None:
        logits_processors = [
            JSONLogitsProcessor(
                json_schema,
                engine.engine,
                whitespace_pattern,
                enable_schema_optimization,
            )
        ]
    elif regex_string is not None:
        logits_processors = [RegexLogitsProcessor(regex_string, engine.engine)]
    else:
        logits_processors = []

    sampling_params = SamplingParams(
        **request_dict, logits_processors=logits_processors  # type: ignore
    )
    request_id = random_uuid()

    results_generator = engine.generate(prompt, sampling_params, request_id)  # type: ignore

    # Streaming case
    async def stream_results() -> AsyncGenerator[bytes, None]:
        async for request_output in results_generator:
            prompt = request_output.prompt
            text_outputs = [prompt + output.text for output in request_output.outputs]
            ret = {"text": text_outputs}
            yield (json.dumps(ret) + "\0").encode("utf-8")

    if stream:
        return StreamingResponse(stream_results())

    # Non-streaming case
    final_output = None
    async for request_output in results_generator:
        if await request.is_disconnected():
            # Abort the request if the client disconnects.
            await engine.abort(request_id)  # type: ignore
            return Response(status_code=499)
        final_output = request_output

    assert final_output is not None
    prompt = final_output.prompt
    text_outputs = [prompt + output.text for output in final_output.outputs]
    ret = {"text": text_outputs}
    return JSONResponse(ret)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", type=str, default=None)
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--ssl-keyfile", type=str, default=None)
    parser.add_argument("--ssl-certfile", type=str, default=None)
    parser = AsyncEngineArgs.add_cli_args(parser)
    args = parser.parse_args()

    # Adds the `engine_use_ray`,  `disable_log_requests` and `max_log_len`
    # arguments
    engine_args: AsyncEngineArgs = AsyncEngineArgs.from_cli_args(args)  # type: ignore

    # Sets default for the model (`facebook/opt-125m`)
    engine = AsyncLLMEngine.from_engine_args(engine_args)

    uvicorn.run(
        app,
        host=args.host,
        port=args.port,
        log_level="debug",
        timeout_keep_alive=TIMEOUT_KEEP_ALIVE,
        ssl_keyfile=args.ssl_keyfile,
        ssl_certfile=args.ssl_certfile,
    )
