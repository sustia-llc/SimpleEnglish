#!/usr/bin/env python3
"""Advisory writing checks for Claude Code hooks. Never blocks.

PostToolUse (Write|Edit on a .md file): lint the text that the tool wrote with
evals/ste_lint.py: `new_string` for Edit, the whole file for Write. Quoted
text ("...", “...”) and blockquote lines are not linted. Semicolons and
em-dashes are not reported. A trailing condition is reported only in a
sentence that starts with a command verb. When hits remain, print the
offending text and file line number for each hit (capped at MAX_HOOK_HITS)
to stderr and exit 2 so the model sees it. Exit 2 on PostToolUse is
advisory: the tool already ran. Agent-internal Markdown, such as memory
files under the Claude configuration directory or files in the Claude Code
session temp directory, is skipped. Set SIMPLE_ENGLISH_LINT_EXCLUDE to skip
more paths.

Stop: read `last_assistant_message`, check the reply register (no bold),
and return a systemMessage only when the reply
breaks it. The same problems are saved for the session, and the next
UserPromptSubmit event adds them to the model context once. Always exit 0,
so the session never loops.
"""
import fnmatch
import json
import os
import pathlib
import re
import sys
import tempfile

HERE = pathlib.Path(__file__).resolve().parent
ROOT = HERE.parent.parent
sys.path.insert(0, str(ROOT / "evals"))

MAX_HOOK_HITS = 12
CLAUDE_DIR = ".claude"
OPENERS = re.compile(r"^\s*(certainly|great question|you're absolutely right|sure[,!]|absolutely[,!])", re.I)
CLOSERS = re.compile(r"(i hope this helps|let me know if|feel free to)", re.I)
FENCE = re.compile(r"^ {0,3}(```|~~~)")
QUOTED = re.compile(r'"[^"\n]*"|“[^”\n]*”')
BLOCKQUOTE = re.compile(r"^ {0,3}>.*$", re.M)
SESSION_TMP = re.compile(r"claude-\d+")
HOOK_SKIP = {"semicolon", "em_dash"}
SLOP_ALLOW = ("harness", "load-bearing")
IMPERATIVE = re.compile(
    r"^\W*(add|apply|ask|avoid|build|call|change|check|choose|close|copy|create|delete|disable|do|"
    r"don't|edit|enable|ensure|find|fix|give|install|keep|load|look|make|move|never|open|pass|pick|"
    r"press|put|read|remove|rename|replace|restart|run|save|see|select|send|set|skip|start|stop|"
    r"test|try|turn|update|use|verify|wait|write)\b", re.I)


def load_linter():
    try:
        import ste_lint  # noqa: WPS433
        return ste_lint
    except Exception:  # noqa: BLE001
        return None


def strip_code(text):
    text = re.sub(r"```.*?```", " ", text, flags=re.S)
    return re.sub(r"`[^`]*`", " ", text)


def absolute(path, cwd=None):
    """Make the path absolute. The harness can send it relative to the session directory."""
    return pathlib.Path(cwd or ".", pathlib.Path(path).expanduser()).absolute()


def variants(path):
    """The path as written and the path with symlinks resolved. An exclusion matches either form.

    A symlink hides a directory name in both directions. `notes.md` can point into `.claude`,
    and `.claude` itself can point at a dotfiles repository. Both forms must miss for the
    file to reach the linter.
    """
    return {pathlib.Path(os.path.normpath(path)), path.resolve()}


def excluded(target):
    """True for agent-internal Markdown and for the paths the user excludes."""
    config_dirs = variants(absolute(os.environ.get("CLAUDE_CONFIG_DIR") or f"~/{CLAUDE_DIR}"))
    raw = os.environ.get("SIMPLE_ENGLISH_LINT_EXCLUDE", "").split(os.pathsep)
    patterns = [os.path.expanduser(p) for p in raw if p]
    tmp = pathlib.Path(tempfile.gettempdir()).resolve()
    for form in variants(target):
        if CLAUDE_DIR in form.parts or any(form.is_relative_to(d) for d in config_dirs):
            return True
        if form.is_relative_to(tmp) and any(SESSION_TMP.fullmatch(p) for p in form.relative_to(tmp).parts):
            return True
        if any(fnmatch.fnmatch(str(form), p) for p in patterns):
            return True
    return False


def fenced_lines(text):
    """For each line of `text`, whether it is a fence delimiter or inside a fenced block."""
    flags, fenced = [], False
    for line in text.split("\n"):
        delimiter = bool(FENCE.match(line))
        flags.append(fenced or delimiter)
        fenced ^= delimiter
    return flags


def written_text(event, text):
    """The text that the tool wrote, and the count of file lines above it.

    An Edit's `new_string` can sit inside a fence that opens above it, so the lines of
    `new_string` that the file holds in a fenced block are blank in the result.
    """
    new = (event.get("tool_input") or {}).get("new_string")
    if event.get("tool_name") != "Edit" or new is None:
        return text, 0
    start = text.find(new)
    if start < 0:
        return new, 0
    above = text.count("\n", 0, start)
    flags = fenced_lines(text)[above:]
    return "\n".join("" if fenced else line for line, fenced in zip(new.split("\n"), flags)), above


def mask_quotes(text):
    """Blank blockquote lines and replace each one-line quoted span with one word. Newlines stay."""
    return QUOTED.sub(" QUOTE ", BLOCKQUOTE.sub("", text))


def reported(hit):
    """True for a lint hit that the file check shows to the model."""
    if hit["category"] in HOOK_SKIP:
        return False
    if hit["category"] == "slop_word" and hit["text"].lower().startswith(SLOP_ALLOW):
        return False
    if hit["category"] == "trailing_condition":
        return bool(IMPERATIVE.match(hit["text"]))
    return True


def post_tool_use(event):
    path = (event.get("tool_input") or {}).get("file_path") or ""
    if not path.endswith(".md"):
        return 0
    target = absolute(path, event.get("cwd"))
    if excluded(target):
        return 0
    lint = load_linter()
    if lint is None:
        return 0
    try:
        text = target.read_text(encoding="utf-8")
    except OSError:
        return 0
    written, offset = written_text(event, text)
    detail = [h for h in lint.lint_detail(mask_quotes(written), "descriptive") if reported(h)]
    if not detail:
        return 0
    lines = [f"simple-english: the text written to {target.name} has {len(detail)} STE violations."]
    for h in detail[:MAX_HOOK_HITS]:
        lines.append(f"  line {h['line'] + offset}, {h['category']}: {h['text']}")
    if len(detail) > MAX_HOOK_HITS:
        lines.append(f"  and {len(detail) - MAX_HOOK_HITS} more hit(s).")
    lines.append("Fix these hits in the text you just wrote, then continue.")
    sys.stderr.write("\n".join(lines) + "\n")
    return 2


def stop(event):
    reply = event.get("last_assistant_message") or ""
    problems = []
    lint = load_linter()
    if lint is not None:
        c = lint.reader_check(reply)["counts"]
        for key, label in (("bold_spans", "bold span"),):
            if c[key]:
                problems.append(f"{c[key]} {label}(s)")
        slop = sum(1 for h in lint.lint_detail(strip_code(reply), "descriptive")
                   if h["category"] == "slop_word" and reported(h))
        if slop:
            problems.append(f"{slop} slop word(s)")
    if OPENERS.search(reply):
        problems.append("a filler opener")
    if CLOSERS.search(reply):
        problems.append("a filler closer")
    if problems:
        summary = "; ".join(problems)
        print(json.dumps({"systemMessage": "simple-english reply check: " + summary + "."}))
        note = pending_note(event)
        if note is not None:
            try:
                note.parent.mkdir(parents=True, exist_ok=True)
                note.write_text(summary, encoding="utf-8")
            except OSError:
                pass
    return 0


def pending_note(event):
    """The file that carries a reply-check result from Stop to the next UserPromptSubmit of the session."""
    session = re.sub(r"[^\w-]", "", str(event.get("session_id") or ""))
    if not session:
        return None
    base = os.environ.get("CLAUDE_PLUGIN_DATA") or os.path.expanduser("~/.cache/simple-english")
    return pathlib.Path(base, f"reply-check-{session}.txt")


def user_prompt_submit(event):
    note = pending_note(event)
    if note is None or not note.is_file():
        return 0
    try:
        summary = note.read_text(encoding="utf-8").strip()
        note.unlink()
    except OSError:
        return 0
    if summary:
        context = (f"simple-english reply check: your previous reply had {summary}. "
                   "Answer this prompt with no bold.")
        print(json.dumps({"hookSpecificOutput": {"hookEventName": "UserPromptSubmit", "additionalContext": context}}))
    return 0


def main():
    try:
        event = json.load(sys.stdin)
    except Exception:  # noqa: BLE001
        return 0
    try:
        name = event.get("hook_event_name", "")
        if name == "PostToolUse":
            return post_tool_use(event)
        if name == "Stop":
            return stop(event)
        if name == "UserPromptSubmit":
            return user_prompt_submit(event)
    except Exception:  # noqa: BLE001  advisory hook: a crash must never block or loop the session
        return 0
    return 0


if __name__ == "__main__":
    sys.exit(main())
