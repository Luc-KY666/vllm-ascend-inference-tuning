"""解析 OpenAI-compatible SSE 流并计算 TTFT、TPOT、E2EL 的 P50/P95。"""

from __future__ import annotations

import argparse
import base64
import codecs
import json
import math
import os
import statistics
import sys
from pathlib import Path
from typing import Any, Mapping


SCHEMA_VERSION = "vllm-ascend-streaming-metrics-v1"
ALLOWED_FINISH_REASONS = {"stop", "length"}


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _finite(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(float(value))


class StreamDecoder:
    """对单个请求的分片 SSE 字节流进行严格解析。"""

    def __init__(self, model: str, sent_at: float):
        require(isinstance(model, str) and model, "model 必须是非空字符串")
        require(_finite(sent_at), "sent_at 必须是有限数字")
        self.model = model
        self.sent_at = float(sent_at)
        self.decoder = codecs.getincrementaldecoder("utf-8")()
        self.buffer = ""
        self.data_lines: list[str] = []
        self.content: list[str] = []
        self.first_at: float | None = None
        self.last_at: float | None = None
        self.content_events = 0
        self.usage: dict[str, int] | None = None
        self.finish_reason: str | None = None
        self.done = False

    def feed(self, chunk: bytes, received_at: float, *, final: bool = False) -> None:
        require(isinstance(chunk, bytes), "SSE chunk 必须是 bytes")
        require(_finite(received_at), "received_at 必须是有限数字")
        self.buffer += self.decoder.decode(chunk, final=final)
        while "\n" in self.buffer:
            line, self.buffer = self.buffer.split("\n", 1)
            if line.endswith("\r"):
                line = line[:-1]
            if not line:
                if self.data_lines:
                    self._event("\n".join(self.data_lines), float(received_at))
                    self.data_lines.clear()
            elif line.startswith("data:"):
                self.data_lines.append(line[5:].lstrip(" "))
        if final:
            require(not self.buffer.strip() and not self.data_lines, "SSE 末尾存在未完成事件")

    def _event(self, raw: str, received_at: float) -> None:
        require(not self.done, "收到 [DONE] 后仍有 SSE 事件")
        if raw == "[DONE]":
            self.done = True
            return
        event = json.loads(raw)
        require(isinstance(event, dict) and not event.get("error"), "SSE 事件包含错误")
        require(event.get("model") == self.model, "SSE 模型身份不匹配")
        require(event.get("object") == "chat.completion.chunk", "SSE object 不匹配")
        choices = event.get("choices")
        require(isinstance(choices, list) and len(choices) <= 1, "SSE choices 结构无效")
        for choice in choices:
            require(isinstance(choice, dict) and choice.get("index") == 0, "choice index 无效")
            delta = choice.get("delta")
            require(isinstance(delta, dict), "delta 必须是 object")
            role = delta.get("role")
            require(role in (None, "assistant"), "delta role 无效")
            content = delta.get("content")
            require(content is None or isinstance(content, str), "delta content 无效")
            if content:
                require(not self.finish_reason, "正常结束后不能继续产生内容")
                require(self.sent_at <= received_at, "内容事件时间早于请求时间")
                if self.first_at is None:
                    self.first_at = received_at
                self.last_at = received_at
                self.content_events += 1
                self.content.append(content)
            finish_reason = choice.get("finish_reason")
            if finish_reason is not None:
                require(
                    isinstance(finish_reason, str) and finish_reason in ALLOWED_FINISH_REASONS,
                    "finish_reason 不在允许范围",
                )
                self.finish_reason = finish_reason
        usage = event.get("usage")
        if usage is not None:
            require(isinstance(usage, dict), "usage 必须是 object")
            values = {key: usage.get(key) for key in ("prompt_tokens", "completion_tokens", "total_tokens")}
            require(all(isinstance(value, int) and value >= 0 for value in values.values()), "usage token 数无效")
            require(values["total_tokens"] == values["prompt_tokens"] + values["completion_tokens"], "usage 总数不一致")
            self.usage = values

    def result(self, ended_at: float) -> dict[str, Any]:
        require(_finite(ended_at), "ended_at 必须是有限数字")
        require(self.done, "缺少 [DONE]")
        require(self.finish_reason in ALLOWED_FINISH_REASONS, "缺少正常 finish_reason")
        require(self.usage is not None, "缺少 usage")
        require("".join(self.content).strip(), "响应内容为空")
        require(self.first_at is not None and self.last_at is not None, "缺少内容时间")
        require(self.sent_at <= self.first_at <= self.last_at <= ended_at, "时间顺序无效")
        completion_tokens = self.usage["completion_tokens"]
        estimable = completion_tokens > 1 and self.content_events > 1
        return {
            "model": self.model,
            "sent_at": self.sent_at,
            "first_content_at": self.first_at,
            "last_content_at": self.last_at,
            "ended_at": float(ended_at),
            "finish_reason": self.finish_reason,
            "usage": self.usage,
            "content_events": self.content_events,
            "decode_estimable": estimable,
            "ttft_ms": 1000 * (self.first_at - self.sent_at),
            "tpot_ms": (
                1000 * (self.last_at - self.first_at) / (completion_tokens - 1)
                if estimable
                else None
            ),
            "e2e_ms": 1000 * (float(ended_at) - self.sent_at),
        }


def nearest_rank(values: list[float], percentile: float) -> float:
    require(values, "分布不能为空")
    require(0 < percentile <= 1, "percentile 必须位于 (0, 1]")
    ordered = sorted(values)
    require(all(_finite(value) and value >= 0 for value in ordered), "分布包含非法值")
    rank = max(1, math.ceil(percentile * len(ordered)))
    return ordered[rank - 1]


def distribution(values: list[float]) -> dict[str, Any]:
    require(values, "分布不能为空")
    return {
        "n": len(values),
        "mean": statistics.mean(values),
        "P50": nearest_rank(values, 0.50),
        "P95": nearest_rank(values, 0.95),
    }


def _decode_chunk(chunk: Mapping[str, Any]) -> bytes:
    if "data_base64" in chunk:
        value = chunk["data_base64"]
        require(isinstance(value, str), "data_base64 必须是字符串")
        return base64.b64decode(value, validate=True)
    value = chunk.get("data")
    require(isinstance(value, str), "chunk 必须包含 data_base64 或 data")
    return value.encode("utf-8")


def decode_request(model: str, request: Mapping[str, Any]) -> dict[str, Any]:
    request_id = request.get("request_id")
    sent_at = request.get("sent_at")
    ended_at = request.get("ended_at")
    chunks = request.get("chunks")
    require(isinstance(request_id, str) and request_id, "request_id 必须是非空字符串")
    require(_finite(sent_at) and _finite(ended_at), "请求时间必须是有限数字")
    require(isinstance(chunks, list) and chunks, "chunks 不能为空")
    decoder = StreamDecoder(model, float(sent_at))
    for index, chunk in enumerate(chunks):
        require(isinstance(chunk, Mapping), f"第 {index} 个 chunk 必须是 object")
        received_at = chunk.get("received_at")
        require(_finite(received_at), f"第 {index} 个 chunk 缺少 received_at")
        decoder.feed(_decode_chunk(chunk), float(received_at), final=index == len(chunks) - 1)
    result = decoder.result(float(ended_at))
    result["request_id"] = request_id
    return result


def summarize_payload(payload: Mapping[str, Any], minimum_decode_samples: int) -> dict[str, Any]:
    require(isinstance(payload, Mapping), "输入必须是 JSON object")
    model = payload.get("model")
    requests = payload.get("requests")
    require(isinstance(model, str) and model, "model 必须是非空字符串")
    require(isinstance(requests, list) and requests, "requests 不能为空")
    require(isinstance(minimum_decode_samples, int) and minimum_decode_samples > 0, "minimum_decode_samples 无效")

    samples: list[dict[str, Any]] = []
    failures: list[dict[str, str]] = []
    for request in requests:
        if not isinstance(request, Mapping):
            failures.append({"request_id": "<unknown>", "error": "request 必须是 object"})
            continue
        request_id = str(request.get("request_id", "<unknown>"))
        try:
            samples.append(decode_request(model, request))
        except (TypeError, ValueError, json.JSONDecodeError) as exc:
            failures.append({"request_id": request_id, "error": f"{type(exc).__name__}: {exc}"})

    if failures:
        return {
            "schema_version": SCHEMA_VERSION,
            "status": "invalid",
            "model": model,
            "counts": {
                "requested": len(requests),
                "completed": len(samples),
                "failed": len(failures),
                "decode_estimable": sum(sample.get("decode_estimable") is True for sample in samples),
                "decode_unestimable": sum(sample.get("decode_estimable") is False for sample in samples),
            },
            "metrics": None,
            "throughput": None,
            "samples": samples,
            "failures": failures,
        }

    decode_samples = [sample for sample in samples if sample["decode_estimable"]]
    base = {
        "schema_version": SCHEMA_VERSION,
        "status": "success",
        "model": model,
        "counts": {
            "requested": len(requests),
            "completed": len(samples),
            "failed": 0,
            "decode_estimable": len(decode_samples),
            "decode_unestimable": len(samples) - len(decode_samples),
        },
        "samples": samples,
        "failures": [],
    }
    if len(decode_samples) < minimum_decode_samples:
        base.update(
            status="invalid",
            metrics=None,
            throughput=None,
            failures=[
                {
                    "request_id": "<aggregate>",
                    "error": (
                        f"可估计 TPOT 样本不足: {len(decode_samples)} < "
                        f"{minimum_decode_samples}"
                    ),
                }
            ],
        )
        return base

    ttft = [sample["ttft_ms"] for sample in samples]
    tpot = [sample["tpot_ms"] for sample in decode_samples]
    e2e = [sample["e2e_ms"] for sample in samples]
    first_send = min(sample["sent_at"] for sample in samples)
    last_end = max(sample["ended_at"] for sample in samples)
    window = last_end - first_send
    require(window > 0, "整体测量窗口必须为正")
    output_tokens = sum(sample["usage"]["completion_tokens"] for sample in samples)
    base["metrics"] = {
        "TTFT": {"unit": "ms", **distribution(ttft)},
        "TPOT": {"unit": "ms", **distribution(tpot)},
        "E2EL": {"unit": "ms", **distribution(e2e)},
    }
    base["throughput"] = {
        "window_seconds": window,
        "requests_per_second": len(samples) / window,
        "output_tokens_per_second": output_tokens / window,
    }
    return base


def _read_json(path: str) -> Any:
    if path == "-":
        return json.load(sys.stdin)
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def _write_json(path: str, value: Any) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_name(f".{target.name}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary, target)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    summarize = commands.add_parser("summarize", help="解析流式样本并汇总指标")
    summarize.add_argument("--input", required=True, help="流式样本 JSON，使用 - 表示 stdin")
    summarize.add_argument("--output", required=True, help="输出 JSON")
    summarize.add_argument("--minimum-decode-samples", required=True, type=int)
    args = parser.parse_args(argv)
    try:
        payload = _read_json(args.input)
        result = summarize_payload(payload, args.minimum_decode_samples)
    except (OSError, TypeError, ValueError, json.JSONDecodeError) as exc:
        result = {
            "schema_version": SCHEMA_VERSION,
            "status": "invalid",
            "metrics": None,
            "throughput": None,
            "failures": [{"request_id": "<input>", "error": f"{type(exc).__name__}: {exc}"}],
        }
    _write_json(args.output, result)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0 if result.get("status") == "success" else 3


if __name__ == "__main__":
    raise SystemExit(main())
