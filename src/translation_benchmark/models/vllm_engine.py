"""vLLM engine for the chat models — runs `vllm serve` as a separate process.

Instead of embedding the engine in-process, this backend launches vLLM's
OpenAI-compatible server as a subprocess and translates via
the OpenAI-compatible API using the official ``openai`` client.

Two request paths, chosen by prompt style:

- Tower/Qwen: ``/v1/chat/completions`` — tokenization and chat-template
  rendering happen SERVER-SIDE using the template shipped in the model
  directory.
- TranslateGemma: its chat template requires language codes in custom
  content-part fields, which vLLM's OpenAI request parser strips from
  "text" parts before the template runs (as of vLLM 0.22). Workaround:
  the checkpoint's own template is rendered CLIENT-SIDE (tokenizer-only
  load, no weights) and sent to ``/v1/completions`` as a raw prompt with
  ``add_special_tokens=False`` (the template already emits BOS).

The process model buys:

- process isolation: an engine crash cannot take down the harness;
- reliable GPU cleanup: terminating the server frees all GPU memory — the
  process is closed on ``unload()`` and, as a safety net, on interpreter
  exit via ``atexit``;
- the standard OpenAI request format (no vLLM Python API coupling).

Same prompts and context handling as the transformers ChatTranslator
(shared ``build_messages``). Not available for the Tier 4 seq2seq models —
vLLM does not serve those encoder-decoder architectures.
"""
from __future__ import annotations

import atexit
import socket
import subprocess
import sys
import tempfile
import time
import urllib.error
import urllib.request
from pathlib import Path

from translation_benchmark import guards
from translation_benchmark.context import ContextPair
from translation_benchmark.models.base import BaseTranslator, ModelSpec
from translation_benchmark.models.chat import build_messages
from translation_benchmark.models.paths import resolve_model_source


def _free_port() -> int:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


class VLLMChatTranslator(BaseTranslator):
    def __init__(
        self,
        spec: ModelSpec,
        device: str = "auto",
        hf_id: str | None = None,
        models_dir: str | None = None,
        quant: str | None = None,
        prompt_style: str | None = None,
        max_new_tokens: int = 256,
        gpu_memory_utilization: float = 0.9,
        max_model_len: int | None = 8192,  # subtitles need little; saves KV memory
        port: int | None = None,
        startup_timeout: float = 600.0,
        request_timeout: float = 120.0,
        **kwargs,
    ) -> None:
        super().__init__(spec, device=device, **kwargs)
        # Validate the variant up front (fails fast, no vLLM needed).
        plan = spec.resolve_quant(quant)
        if plan.runtime == "8bit":
            raise ValueError(
                "The vLLM engine does not support 8bit (bitsandbytes int8); "
                "use --quant 4bit, awq, or fp8."
            )
        self.hf_override = hf_id
        self.models_dir = models_dir
        self.quant = quant
        self.prompt_style = prompt_style
        self.max_new_tokens = max_new_tokens
        self.gpu_memory_utilization = gpu_memory_utilization
        self.max_model_len = max_model_len
        self.port = port
        self.startup_timeout = startup_timeout
        self.request_timeout = request_timeout
        self._proc: subprocess.Popen | None = None
        self._base_url = ""
        self._log_path: str | None = None
        self._client = None
        self._render_tokenizer = None  # client-side template rendering (translategemma)

    # ----- server lifecycle ------------------------------------------------

    def _server_cmd(self, port: int) -> list[str]:
        plan = self.spec.resolve_quant(self.quant)
        source, extra = resolve_model_source(
            self.hf_override or plan.hf_id, plan.dir_key, self.models_dir
        )
        cmd = [
            sys.executable, "-m", "vllm.entrypoints.openai.api_server",
            "--model", source,
            "--served-model-name", self.spec.key,
            "--host", "127.0.0.1",
            "--port", str(port),
            "--gpu-memory-utilization", str(self.gpu_memory_utilization),
        ]
        # Cap the requested context to the model's real window — some models
        # (e.g. TowerInstruct, 4096) are smaller than our default, and vLLM
        # refuses to start when --max-model-len exceeds the checkpoint's max.
        max_len = self.max_model_len
        if max_len and self.spec.approx_context_tokens:
            max_len = min(max_len, self.spec.approx_context_tokens)
        if max_len:
            cmd += ["--max-model-len", str(max_len)]
        if extra.get("cache_dir"):
            cmd += ["--download-dir", extra["cache_dir"]]
        if plan.runtime == "4bit":
            cmd += ["--quantization", "bitsandbytes"]
        # AWQ/FP8 variant repos are autodetected from the checkpoint config.
        return cmd

    def _load(self) -> None:
        import importlib.util

        if importlib.util.find_spec("vllm") is None:
            raise RuntimeError(
                "vLLM is not installed. Run: pip install 'translation-benchmark[vllm]'"
            )

        port = self.port or _free_port()
        cmd = self._server_cmd(port)

        log = tempfile.NamedTemporaryFile(
            mode="w", prefix=f"vllm-{self.spec.key}-", suffix=".log", delete=False
        )
        self._log_path = log.name
        self._proc = subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT)
        self._base_url = f"http://127.0.0.1:{port}"
        atexit.register(self._shutdown)  # close the server when exiting
        self._wait_until_ready()
        import openai

        self._client = openai.OpenAI(
            base_url=f"{self._base_url}/v1", api_key="EMPTY", timeout=self.request_timeout
        )
        if self._style() == "translategemma":
            from transformers import AutoTokenizer

            plan = self.spec.resolve_quant(self.quant)
            source, extra = resolve_model_source(
                self.hf_override or plan.hf_id, plan.dir_key, self.models_dir
            )
            self._render_tokenizer = AutoTokenizer.from_pretrained(source, **extra)

    def _style(self) -> str:
        return self.prompt_style or self.spec.prompt_style

    def _wait_until_ready(self) -> None:
        deadline = time.monotonic() + self.startup_timeout
        while time.monotonic() < deadline:
            if self._proc is None or self._proc.poll() is not None:
                raise RuntimeError(
                    f"vLLM server for {self.spec.key!r} exited during startup "
                    f"(see log: {self._log_path}):\n{self._log_tail()}"
                )
            try:
                with urllib.request.urlopen(f"{self._base_url}/health", timeout=5):
                    return
            except (urllib.error.URLError, OSError):
                time.sleep(2)
        self._shutdown()
        raise RuntimeError(
            f"vLLM server for {self.spec.key!r} not ready after "
            f"{self.startup_timeout:.0f}s (see log: {self._log_path})"
        )

    def _log_tail(self, lines: int = 15) -> str:
        if not self._log_path:
            return ""
        try:
            return "\n".join(
                Path(self._log_path).read_text(errors="replace").splitlines()[-lines:]
            )
        except OSError:
            return ""

    def _shutdown(self) -> None:
        proc, self._proc = self._proc, None
        if proc is None or proc.poll() is not None:
            return
        proc.terminate()
        try:
            proc.wait(timeout=30)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait(timeout=10)

    def unload(self) -> None:
        self._shutdown()
        self._loaded = False

    # ----- translation -----------------------------------------------------

    def _chat(self, messages: list[dict], max_tokens: int) -> str:
        import openai

        extra_body: dict = {}
        if self.spec.prompt_style == "qwen":
            extra_body["chat_template_kwargs"] = {"enable_thinking": False}
        try:
            response = self._client.chat.completions.create(
                model=self.spec.key,
                messages=messages,
                temperature=0.0,
                max_tokens=max_tokens,
                extra_body=extra_body or None,
            )
        except openai.APIStatusError as exc:
            raise RuntimeError(
                f"vLLM server rejected the request ({exc.status_code}): {exc.message}"
            ) from None
        return (response.choices[0].message.content or "").strip()

    def _complete(self, messages: list[dict], max_tokens: int) -> str:
        """Client-side template rendering + raw completion (translategemma)."""
        import openai

        prompt = self._render_tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        try:
            response = self._client.completions.create(
                model=self.spec.key,
                prompt=prompt,
                temperature=0.0,
                max_tokens=max_tokens,
                # The rendered template already starts with BOS.
                extra_body={"add_special_tokens": False},
            )
        except openai.APIStatusError as exc:
            raise RuntimeError(
                f"vLLM server rejected the request ({exc.status_code}): {exc.message}"
            ) from None
        return (response.choices[0].text or "").strip()

    def translate_batch(
        self,
        texts: list[str],
        src_lang: str,
        tgt_lang: str,
        contexts: list[list[ContextPair]] | None = None,
    ) -> list[str]:
        self.load()
        send = self._complete if self._style() == "translategemma" else self._chat
        results = []
        for i, text in enumerate(texts):
            messages = build_messages(
                self.spec,
                text,
                src_lang,
                tgt_lang,
                contexts[i] if contexts else None,
                prompt_style=self.prompt_style,
            )
            results.append(
                send(messages, guards.max_new_tokens_for(text, self.max_new_tokens))
            )
        return results
