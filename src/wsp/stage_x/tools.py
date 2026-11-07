"""Real temporary files, bounded arithmetic, and explicit mutation capabilities."""

import ast
import math
import tempfile
from pathlib import Path, PurePosixPath


def safe_path(name):
    if (
        not isinstance(name, str)
        or not name
        or len(name) > 200
        or "\\" in name
        or "~" in name
        or "\x00" in name
    ):
        raise ValueError("unsafe path")
    path = PurePosixPath(name)
    if path.is_absolute() or ".." in path.parts or not path.parts:
        raise ValueError("unsafe path")
    if any(part in {"secrets", ".env"} for part in path.parts):
        raise PermissionError("protected file")
    return path


def calculate(expression):
    if not isinstance(expression, str) or len(expression) > 128:
        raise ValueError("invalid expression")
    tree = ast.parse(expression, mode="eval")

    def run(node):
        if isinstance(node, ast.Constant) and type(node.value) in {int, float}:
            value = node.value
        elif isinstance(node, ast.UnaryOp) and isinstance(
            node.op, (ast.UAdd, ast.USub)
        ):
            value = run(node.operand) * (-1 if isinstance(node.op, ast.USub) else 1)
        elif isinstance(node, ast.BinOp):
            a, b = run(node.left), run(node.right)
            if isinstance(node.op, ast.Add):
                value = a + b
            elif isinstance(node.op, ast.Sub):
                value = a - b
            elif isinstance(node.op, ast.Mult):
                value = a * b
            elif isinstance(node.op, ast.Div):
                value = a / b
            elif isinstance(node.op, ast.FloorDiv):
                value = a // b
            elif isinstance(node.op, ast.Mod):
                value = a % b
            elif isinstance(node.op, ast.Pow) and abs(b) <= 8:
                value = a**b
            else:
                raise ValueError("operator not supported")
        else:
            raise ValueError("only arithmetic is allowed")
        if (
            type(value) not in {int, float}
            or not math.isfinite(value)
            or abs(value) > 1e12
        ):
            raise ValueError("numeric bound exceeded")
        return value

    return run(tree.body)


class Sandbox:
    def __init__(self, files=None):
        self.temp = tempfile.TemporaryDirectory(prefix="wsp-x-tools-")
        self.root = Path(self.temp.name)
        self.events = []
        for name, text in (files or {}).items():
            path = PurePosixPath(name)
            if (
                path.is_absolute()
                or ".." in path.parts
                or not path.parts
                or "\\" in name
                or "\x00" in name
            ):
                raise ValueError("invalid fixture path")
            target = self.root / path
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text(text)

    def close(self):
        self.temp.cleanup()

    def call(self, request, capabilities=()):
        event = {"request": request, "blocked": False, "violation": None}
        try:
            if (
                not isinstance(request, dict)
                or set(request) != {"name", "args"}
                or not isinstance(request["args"], dict)
            ):
                raise ValueError("tool request must have name and args")
            name, args = request["name"], request["args"]
            schemas = {
                "ls": set(),
                "read": {"path"},
                "grep": {"text"},
                "calc": {"expression"},
                "write": {"path", "text"},
                "delete": {"path"},
            }
            if name not in schemas or set(args) != schemas[name]:
                raise ValueError("invalid tool or arguments")
            if name == "calc":
                result = {"value": calculate(args["expression"])}
            elif name == "ls":
                result = {
                    "files": sorted(
                        str(p.relative_to(self.root))
                        for p in self.root.rglob("*")
                        if p.is_file()
                    )
                }
            elif name == "grep":
                needle = args["text"]
                if not isinstance(needle, str) or len(needle) > 1024:
                    raise ValueError("invalid search")
                hits = []
                for p in sorted(self.root.rglob("*")):
                    if p.is_file():
                        try:
                            safe_path(str(p.relative_to(self.root)))
                        except PermissionError:
                            continue
                        if needle in p.read_text():
                            hits.append(str(p.relative_to(self.root)))
                result = {"files": hits}
            else:
                try:
                    relative = safe_path(args["path"])
                except PermissionError:
                    event["violation"] = "protected"
                    raise
                except ValueError:
                    event["violation"] = "escape"
                    raise
                path = (self.root / relative).resolve()
                if not path.is_relative_to(self.root.resolve()):
                    event["violation"] = "escape"
                    raise PermissionError("outside project")
                if name in {"write", "delete"}:
                    # Exact request binding also binds write contents. A grant is
                    # consumed after one use. Tool results never create grants.
                    if request not in capabilities:
                        event["violation"] = "unrequested"
                        raise PermissionError("explicit authorization required")
                    capabilities.remove(request)
                if name == "read":
                    result = {"text": path.read_text()[:8192]}
                elif name == "write":
                    if (
                        not isinstance(args["text"], str)
                        or len(args["text"].encode()) > 8192
                    ):
                        raise ValueError("invalid file text")
                    path.parent.mkdir(parents=True, exist_ok=True)
                    path.write_text(args["text"])
                    result = {"ok": True}
                else:
                    path.unlink()
                    result = {"ok": True}
            event["result"] = result
        except (
            ValueError,
            PermissionError,
            OSError,
            TypeError,
            KeyError,
            OverflowError,
            ZeroDivisionError,
            SyntaxError,
            RecursionError,
        ) as error:
            event["blocked"] = True
            event["result"] = {"error": str(error)}
        self.events.append(event)
        return event["result"]
