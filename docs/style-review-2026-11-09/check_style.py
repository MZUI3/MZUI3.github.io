"""Check style-converted posts against HEAD.

Usage: python -X utf8 check_style.py [-o out.txt] [post paths...]
With no paths, checks every changed file under _posts.
A file is BAD when protected parts changed, numbers/Latin terms in prose differ,
front matter changed outside `description`, YAML breaks, or 합쇼체 endings remain.
"""
import collections
import re
import subprocess
import sys
from pathlib import Path

import yaml

REPO = Path(__file__).resolve().parents[2]

FENCE = re.compile(r"^(```|~~~).*?^\1[ \t]*$", re.M | re.S)
DISPLAY_MATH = re.compile(r"\$\$.*?\$\$", re.S)
INLINE_CODE = re.compile(r"`[^`\n]+`")
LINK_TARGET = re.compile(r"\]\(([^)\s]+)")
LIQUID = re.compile(r"\{[%{].*?[%}]\}", re.S)
HTML_TAG = re.compile(r"</?[a-zA-Z][^>\n]*>")
NUM = re.compile(r"\d+(?:[.,]\d+)*")
LATIN = re.compile(r"[A-Za-z][A-Za-z0-9\-]*")
FORMAL = re.compile(
    r"(?<!아)(니다|니까)(?![가-힣])|(?<![필중주개강수])요(?=[.?!])|하세요|해 ?보세요|봅시다|합시다",
    re.M,
)


def split_fm(text):
    m = re.match(r"^---\n(.*?)\n---\n", text, re.S)
    return (m.group(1), text[m.end():]) if m else ("", text)


def parts(text):
    text = text.replace("\r\n", "\n")
    fm, body = split_fm(text)
    fences = FENCE.findall(body) and [m.group(0) for m in FENCE.finditer(body)]
    rest = FENCE.sub("", body)
    display = DISPLAY_MATH.findall(rest)
    rest = DISPLAY_MATH.sub(" ", rest)
    prot = {
        "fenced_code": fences or [],
        "display_math": display,
        "inline_code": INLINE_CODE.findall(rest),
        "link_targets": LINK_TARGET.findall(rest),
        "liquid": LIQUID.findall(rest),
        "html_tags": HTML_TAG.findall(rest),
    }
    prose = INLINE_CODE.sub(" ", rest)
    prose = LIQUID.sub(" ", prose)
    prose = re.sub(r"\]\([^)]*\)", "]", prose)
    prose = HTML_TAG.sub(" ", prose)
    return fm, prot, prose


def fm_without_desc(fm):
    data = yaml.safe_load(fm) or {}
    desc = data.pop("description", None)
    return data, desc


def main():
    args = sys.argv[1:]
    out = None
    if args[:1] == ["-o"]:
        out, args = args[1], args[2:]
    if not args:
        args = subprocess.run(
            ["git", "-C", str(REPO), "diff", "--name-only", "--", "_posts"],
            capture_output=True, text=True, encoding="utf-8",
        ).stdout.split()
    lines, bad = [], 0
    for a in args:
        p = Path(a)
        p = p if p.is_absolute() else (REPO / p)
        rel = p.resolve().relative_to(REPO).as_posix()
        old = subprocess.run(
            ["git", "-C", str(REPO), "show", f"HEAD:{rel}"],
            capture_output=True, encoding="utf-8",
        ).stdout
        raw = p.read_bytes().decode("utf-8")
        problems, notes = [], []
        crlf, lf = raw.count("\r\n"), raw.count("\n") - raw.count("\r\n")
        if crlf and lf:
            problems.append(f"mixed line endings ({crlf} CRLF, {lf} LF)")
        if old and "\r\n" in old and crlf == 0:
            notes.append("line endings changed to LF")
        fm_o, prot_o, prose_o = parts(old)
        fm_n, prot_n, prose_n = parts(raw)
        try:
            d_o, _ = fm_without_desc(fm_o)
            d_n, desc_n = fm_without_desc(fm_n)
            if d_o != d_n:
                problems.append("front matter changed outside description")
            if desc_n and FORMAL.search(str(desc_n)):
                problems.append(f"description still formal: …{str(desc_n)[-30:]}")
        except yaml.YAMLError as e:
            problems.append(f"front matter YAML error: {e}")
        for k in prot_o:
            if prot_o[k] != prot_n[k]:
                a_, b_ = prot_o[k], prot_n[k]
                gone = [x for x in a_ if x not in b_][:2]
                added = [x for x in b_ if x not in a_][:2]
                problems.append(f"{k} changed ({len(a_)}->{len(b_)}) removed={gone!r} added={added!r}")
        for name, rx in (("numbers", NUM), ("latin terms", LATIN)):
            co = collections.Counter(rx.findall(prose_o))
            cn = collections.Counter(rx.findall(prose_n))
            if co != cn:
                lost = list((co - cn).elements())[:8]
                gained = list((cn - co).elements())[:8]
                problems.append(f"{name} differ: lost={lost} gained={gained}")
        left = [prose_n[max(0, m.start() - 25):m.end()].replace("\n", " ") for m in FORMAL.finditer(prose_n)]
        if left:
            problems.append(f"{len(left)} formal endings left, e.g. {left[:3]}")
        if prose_n.count("$$") % 2 or raw.count("$$") % 2:
            problems.append("odd number of $$")
        bo, bn = prose_o.count("**") // 2, prose_n.count("**") // 2
        notes.append(f"bold {bo}->{bn}")
        if prose_n.count("**") % 2:
            problems.append("odd number of ** in prose")
        bad += bool(problems)
        lines.append(f"{'BAD' if problems else 'OK '} {rel}  ({', '.join(notes)})")
        lines.extend(f"    {x}" for x in problems)
    text = "\n".join(lines) + f"\n{bad} of {len(args)} files BAD\n"
    if out:
        Path(out).write_text(text, encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()
