#!/usr/bin/env python3
"""
Diagnose a failed Jenkins build with an LLM.

Usage (inside a Jenkins pipeline, in post { failure { ... } }):
    python3 analyze_failure.py > analysis.md

The script finds the current build's log on disk using BUILD_URL and
JENKINS_HOME (both are set by Jenkins), cleans it, redacts secrets, asks an
OpenAI-compatible endpoint (OpenRouter by default) for a diagnosis, and prints
Markdown. It only reads and reports. It always exits 0, so it can never change
the result of your build.

Options:
    python3 analyze_failure.py [LOGFILE]   analyze a specific log file
    python3 analyze_failure.py --dry-run   print the cleaned, redacted excerpt
                                           and do NOT call the LLM (free test)

Environment:
    OPENAI_API_KEY   API key (your OpenRouter key in this project)
    OPENAI_BASE_URL  default: https://openrouter.ai/api/v1
    ANALYZER_MODEL   default: openai/gpt-4o-mini
    BUILD_URL, JENKINS_HOME   set by Jenkins
"""
import json
import os
import re
import sys
import time
import urllib.error
import urllib.parse
import urllib.request

MAX_LINES = 300
MAX_CHARS = 12000
RETRY_CODES = {429, 500, 502, 503, 504}

CONSOLE_NOTE = re.compile(r"\x1b\[8mha:[^\x1b]*\x1b\[0m")
ANSI = re.compile(r"\x1b\[[0-9;?]*[A-Za-z]")

REDACTIONS = [
    (re.compile(r"-----BEGIN [A-Z ]*PRIVATE KEY-----.*?-----END [A-Z ]*PRIVATE KEY-----", re.S),
     "[REDACTED_PRIVATE_KEY]"),
    (re.compile(r"\b(?:AKIA|ASIA)[0-9A-Z]{16}\b"), "[REDACTED_AWS_KEY_ID]"),
    (re.compile(r"\bsk-[A-Za-z0-9_\-]{20,}"), "[REDACTED_API_KEY]"),
    (re.compile(r"\bgh[pousr]_[A-Za-z0-9]{30,}"), "[REDACTED_GITHUB_TOKEN]"),
    (re.compile(r"\bgithub_pat_[A-Za-z0-9_]{30,}"), "[REDACTED_GITHUB_TOKEN]"),
    (re.compile(r"(?i)bearer\s+[A-Za-z0-9._~+/\-]{20,}=*"), "Bearer [REDACTED]"),
    (re.compile(
        r"(?i)\b([A-Za-z0-9_\-]*(?:password|passwd|secret|token|api[_-]?key|access[_-]?key)"
        r"[A-Za-z0-9_\-]*)([\"']?\s*[=:]\s*[\"']?)[^\s\"',;]+"),
     r"\1\2[REDACTED]"),
    (re.compile(r"(?i)(--password)(=|\s+)\S+"), r"\1\2[REDACTED]"),
]

HINTS = {
    400: "The request was rejected. Check ANALYZER_MODEL and the base URL.",
    401: "The API key was rejected. Check the OPENROUTER_API_KEY credential for typos or spaces.",
    402: "Out of credit, or the key's spending limit was reached.",
    403: "The key is not allowed to use this model or endpoint.",
    404: "Model or endpoint not found. Check ANALYZER_MODEL and OPENAI_BASE_URL.",
    429: "Rate limited. Try again later.",
}

SYSTEM_PROMPT = (
    "You are a DevOps assistant for a Jenkins pipeline that builds a Docker "
    "image, pushes it to Amazon ECR, and deploys to Amazon EKS using "
    "CloudFormation and kubectl. Given the tail of a failed build log, reply "
    "in Markdown with these sections:\n"
    "## Root cause\nOne or two sentences.\n"
    "## Evidence\nQuote the exact log line(s) that show it.\n"
    "## Fix\nNumbered steps with exact commands or file edits.\n"
    "## Confidence\nHigh, medium or low, and what to check if unsure.\n"
    "Ignore noise from this analyzer itself. Never ask for secrets. If the "
    "log is too short to tell, say so and list what extra output would help."
)


class AnalyzerError(Exception):
    pass


def emit(text):
    sys.stdout.buffer.write((text + "\n").encode("utf-8", "replace"))
    sys.stdout.flush()


def redact(text):
    for pattern, replacement in REDACTIONS:
        text = pattern.sub(replacement, text)
    return text


def find_log(explicit):
    """Return the path of the build log, or None."""
    if explicit:
        return explicit if os.path.isfile(explicit) else None

    url = os.environ.get("BUILD_URL", "")
    home = os.environ.get("JENKINS_HOME", "/var/jenkins_home")
    parts = [p for p in urllib.parse.urlparse(url).path.split("/") if p]
    if "job" not in parts:
        return None
    parts = parts[parts.index("job"):]          # drop any URL prefix
    number, chain = parts[-1], parts[:-1]       # e.g. job/a/job/b | 5
    if not number.isdigit() or len(chain) % 2 or any(chain[i] != "job" for i in range(0, len(chain), 2)):
        return None
    names = chain[1::2]
    for decode in (False, True):
        use = [urllib.parse.unquote(n) if decode else n for n in names]
        path = home
        for name in use:
            path = os.path.join(path, "jobs", name)
        path = os.path.join(path, "builds", number, "log")
        if os.path.isfile(path):
            return path
    return None


def load_log(path):
    with open(path, "r", errors="replace") as handle:
        raw = handle.read()
    raw = CONSOLE_NOTE.sub("", raw)
    raw = ANSI.sub("", raw)
    lines = [ln for ln in raw.splitlines() if not ln.startswith("[Pipeline]")]
    tail = "\n".join(lines[-MAX_LINES:])
    return redact(tail)[-MAX_CHARS:]


def ask_llm(log_text):
    base = (os.environ.get("OPENAI_BASE_URL") or "https://openrouter.ai/api/v1").rstrip("/")
    key = (os.environ.get("OPENAI_API_KEY") or "").strip()
    model = os.environ.get("ANALYZER_MODEL") or "openai/gpt-4o-mini"
    if not key:
        raise AnalyzerError("OPENAI_API_KEY is not set in this pipeline's environment block.")

    payload = {
        "model": model,
        "temperature": 0.2,
        "max_tokens": 900,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": "Failed build log (tail):\n\n" + log_text},
        ],
    }
    request = urllib.request.Request(
        base + "/chat/completions",
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json", "Authorization": "Bearer " + key},
        method="POST",
    )

    body = None
    for attempt in range(3):
        try:
            with urllib.request.urlopen(request, timeout=60) as response:
                body = json.loads(response.read().decode("utf-8"))
            break
        except urllib.error.HTTPError as err:
            detail = err.read().decode("utf-8", "replace")[:300]
            if err.code in RETRY_CODES and attempt < 2:
                time.sleep(2 * (attempt + 1))
                continue
            raise AnalyzerError("HTTP {}. {} Response: {}".format(
                err.code, HINTS.get(err.code, ""), detail))
        except (urllib.error.URLError, OSError, ValueError) as err:
            if attempt < 2:
                time.sleep(2 * (attempt + 1))
                continue
            raise AnalyzerError("Could not reach {}: {}".format(base, err))

    try:
        return body["choices"][0]["message"]["content"].strip()
    except (KeyError, IndexError, TypeError, AttributeError):
        raise AnalyzerError("Unexpected response from the model: " + json.dumps(body)[:300])


def main():
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    dry_run = "--dry-run" in sys.argv
    try:
        path = find_log(args[0] if args else None)
        if not path:
            emit("# Build analysis\n\nCould not locate the build log. Pass a file path, or run "
                 "this where BUILD_URL and JENKINS_HOME point at the Jenkins controller's disk.")
            return 0
        text = load_log(path)
        if not text.strip():
            emit("# Build analysis\n\nThe log was empty, so there is nothing to analyze.")
            return 0
        if dry_run:
            emit("# Dry run: cleaned and redacted excerpt (no LLM call)\n\nSource: {}\n\n{}".format(path, text))
            return 0
        answer = ask_llm(text)
        emit("# Build analysis (AI-generated, verify before acting)\n")
        emit(answer)
    except AnalyzerError as err:
        emit("\nAnalyzer could not finish: {}".format(err))
    except Exception as err:  # never break the pipeline
        emit("\nAnalyzer hit an unexpected error: {}: {}".format(type(err).__name__, err))
    return 0


if __name__ == "__main__":
    sys.exit(main())
