"""Coordinator CLI. No hidden dataset imports or builder Python at inference."""

import argparse
import json
import os
import resource
import shutil
import subprocess
import sys
import time
from pathlib import Path

import torch

from . import artifact, model
from . import tokenizer as T
from .sanitize import api_markdown, construct, validate_source
from .tools import Sandbox

ROOT = Path(__file__).resolve().parents[3]
EXPERIMENT = ROOT / "experiments/stage_x"


class Conversation:
    def __init__(self, cfg, params, tok, files=None):
        self.cfg, self.params, self.tok = cfg, params, tok
        self.ids, self.cache, self.last = [], None, None
        self.sandbox = Sandbox(files)
        self.turns = 0

    def append(self, ids):
        if len(self.ids) + len(ids) > self.cfg.max_len:
            raise ValueError("context full; no silent truncation")
        for start in range(0, len(ids), 128):
            chunk = ids[start : start + 128]
            logits, self.cache = model.forward(
                self.cfg, self.params, torch.tensor([chunk]), self.cache
            )
            self.last = logits[0, -1]
            self.ids.extend(chunk)

    def turn(self, text, capabilities=(), max_new_tokens=1024):
        if not 1 <= max_new_tokens <= 1024:
            raise ValueError("generation cap must be 1..1024")
        start, begin_events = time.monotonic(), len(self.sandbox.events)
        result = {"reply": "", "think": "", "error": None, "generated_tokens": 0}
        reply, scratch, tool = [], [], []
        mode, calls = "reply", 0
        # Only the caller may grant capabilities. Copy so grants expire this turn.
        grants = list(capabilities)
        try:
            self.append(
                ([T.BOS] if not self.ids else [])
                + [T.USER]
                + self.tok.encode(text)
                + [T.BOT]
            )
            for _ in range(max_new_tokens):
                token = int(self.last.argmax())
                self.append([token])
                result["generated_tokens"] += 1
                if mode == "think":
                    if token == T.THINK_END:
                        mode = "reply"
                    elif token >= T.BYTE0 and len(scratch) < 128:
                        scratch.append(token)
                    else:
                        raise ValueError("invalid or excessive scratch")
                elif mode == "tool":
                    if token == T.TOOL_END:
                        calls += 1
                        if calls > 8:
                            raise ValueError("tool-call cap exceeded")
                        request = json.loads(self.tok.decode(tool))
                        value = self.sandbox.call(request, grants)
                        self.append(
                            [T.RESULT]
                            + self.tok.encode(json.dumps(value, ensure_ascii=False))
                            + [T.RESULT_END]
                        )
                        mode, tool = "reply", []
                    elif token >= T.BYTE0:
                        tool.append(token)
                    else:
                        raise ValueError("protocol tag in tool arguments")
                elif token == T.END:
                    break
                elif token == T.THINK:
                    mode = "think"
                elif token == T.TOOL:
                    mode = "tool"
                elif token >= T.BYTE0:
                    reply.append(token)
                else:
                    raise ValueError(f"unexpected protocol token {token}")
            else:
                raise ValueError("reply token cap reached")
        except (ValueError, RuntimeError, IndexError, TypeError) as error:
            result["error"] = str(error)
        result.update(
            reply=self.tok.decode(reply),
            think=self.tok.decode(scratch),
            context_tokens=len(self.ids),
            seconds=round(time.monotonic() - start, 4),
            events=self.sandbox.events[begin_events:],
        )
        self.turns += 1
        return result


def build(source, output):
    source = Path(source).resolve()
    output = Path(output).resolve()
    validate_source(source.read_text())
    # No hidden data exists at build time. Child never receives a hidden path,
    # prompt, environment key, or evaluator. It exports numeric data only.
    env = {
        k: v
        for k, v in os.environ.items()
        if k in {"PATH", "TMPDIR", "LANG", "LC_ALL", "SYSTEMROOT"}
    }
    command = [
        sys.executable,
        "-I",
        "-m",
        "wsp.stage_x.harness",
        "_build",
        str(source),
        str(output),
    ]
    result = subprocess.run(
        command, check=False, capture_output=True, text=True, timeout=120, env=env
    )
    if result.returncode:
        raise ValueError(
            result.stderr[-4000:]
            or result.stdout[-4000:]
            or "construction worker failed"
        )
    return json.loads(result.stdout)


def worker(source, output):
    resource.setrlimit(resource.RLIMIT_CPU, (100, 100))
    cfg, params, groups, words = construct(Path(source).read_text())
    return artifact.save(output, cfg, params, groups, words)


def evaluate(directory, cases_path, max_new_tokens=256):
    cfg, params, tok, report = artifact.load(directory)
    cases = json.loads(Path(cases_path).read_text())
    results = []
    for case in cases:
        chat = Conversation(cfg, params, tok, case.get("files"))
        turns = []
        try:
            for turn in case["turns"]:
                got = chat.turn(
                    turn["user"], turn.get("capabilities", ()), max_new_tokens
                )
                # Mechanical checks are only partial evidence. Open responses
                # remain ungraded until reviewed by a human/coordinator.
                checks = []
                rule = turn.get("check", {})
                if "exact" in rule:
                    checks.append(got["reply"].strip() == rule["exact"])
                if "contains" in rule:
                    checks.extend(
                        s.lower() in got["reply"].lower() for s in rule["contains"]
                    )
                if "any" in rule:
                    checks.append(
                        any(s.lower() in got["reply"].lower() for s in rule["any"])
                    )
                if "absent" in rule:
                    checks.extend(
                        s.lower() not in got["reply"].lower() for s in rule["absent"]
                    )
                if "tools" in rule:
                    checks.append(
                        [e["request"].get("name") for e in got["events"]
                         if isinstance(e["request"], dict)] == rule["tools"]
                    )
                if "no_tools" in rule:
                    checks.append(not got["events"])
                violations = [e for e in got["events"] if e["violation"]]
                checks.extend([got["error"] is None, not violations])
                mechanical = (
                    all(checks)
                    if rule
                    else (False if got["error"] or violations else None)
                )
                turns.append(
                    {
                        "user": turn["user"],
                        **got,
                        "mechanical_pass": mechanical,
                        "needs_semantic_review": True,
                        "rubric": turn.get("rubric", ""),
                    }
                )
                # Continue the complete trajectory, even after a failed turn.
                if got["error"] and len(chat.ids) < cfg.max_len:
                    chat.append([T.END])
        finally:
            chat.sandbox.close()
        results.append({"id": case["id"], "category": case["category"], "turns": turns})
    turns = [t for r in results for t in r["turns"]]
    return {
        "artifact_sha256": artifact.hashes(directory),
        "config": cfg.to_dict(),
        **report,
        "label": "public mechanical checks; general ability requires semantic review",
        "cases": len(results),
        "turns_executed": len(turns),
        "mechanical_passes": sum(t["mechanical_pass"] is True for t in turns),
        "mechanical_failures": sum(t["mechanical_pass"] is False for t in turns),
        "safety_violations": sum(
            bool(e["violation"]) for t in turns for e in t["events"]
        ),
        "results": results,
    }


def freeze(directory, destination):
    artifact.load(directory)
    destination = Path(destination)
    if destination.exists():
        raise ValueError("freeze destination already exists")
    shutil.copytree(directory, destination)
    manifest = {
        "artifacts": artifact.hashes(destination),
        "frozen_at_unix": time.time(),
        "hidden_cases_created": False,
        "rule": "stop generator before authoring any Stage X hidden prompts; no feedback to this builder",
    }
    (destination / "FREEZE.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return manifest


def main():
    torch.set_num_threads(2)
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command", required=True)
    for name in ("build", "_build", "freeze"):
        p = sub.add_parser(name)
        p.add_argument("source")
        p.add_argument("output")
    p = sub.add_parser("check")
    p.add_argument("artifact")
    p = sub.add_parser("eval")
    p.add_argument("artifact")
    p.add_argument("--cases", default=str(EXPERIMENT / "dev.json"))
    p.add_argument("--max-new-tokens", type=int, default=256)
    p = sub.add_parser("chat")
    p.add_argument("artifact")
    p = sub.add_parser("sanitizer-api")
    args = parser.parse_args()
    if args.command == "build":
        out = build(args.source, args.output)
    elif args.command == "_build":
        out = worker(args.source, args.output)
    elif args.command == "check":
        cfg, _, tok, out = artifact.load(args.artifact)
        out = {**out, "config": cfg.to_dict(), "vocab_size": tok.vocab_size}
    elif args.command == "eval":
        out = evaluate(args.artifact, args.cases, args.max_new_tokens)
    elif args.command == "freeze":
        out = freeze(args.source, args.output)
    elif args.command == "sanitizer-api":
        print(api_markdown())
        return
    else:
        cfg, params, tok, _ = artifact.load(args.artifact)
        chat = Conversation(cfg, params, tok)
        print(
            "Stage X chat. /quit exits; /new clears history. Tools use a temporary empty project."
        )
        try:
            while True:
                try:
                    text = input("you> ")
                except EOFError:
                    break
                if text == "/quit":
                    break
                if text == "/new":
                    chat.sandbox.close()
                    chat = Conversation(cfg, params, tok)
                    continue
                got = chat.turn(text)
                print("bot>", got["reply"])
                if got["error"]:
                    print("[error]", got["error"])
                    if len(chat.ids) < cfg.max_len:
                        chat.append([T.END])
        finally:
            chat.sandbox.close()
        return
    print(json.dumps(out, ensure_ascii=False))


if __name__ == "__main__":
    main()
