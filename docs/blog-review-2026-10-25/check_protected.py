"""Compare protected markdown parts of changed posts between HEAD and the working tree.

Usage: python -X utf8 check_protected.py <repo> <out_file>
"""
import re
import subprocess
import sys

import yaml

repo, out_path = sys.argv[1], sys.argv[2]

FENCE = re.compile(r"^(```|~~~).*?^\1[ \t]*$", re.M | re.S)
DISPLAY_MATH = re.compile(r"\$\$.*?\$\$", re.S)
INLINE_MATH = re.compile(r"(?<![\\$])\$(?!\$)[^$\n]+?(?<![\\$])\$(?!\$)")
INLINE_CODE = re.compile(r"`[^`\n]+`")
LINK_TARGET = re.compile(r"\]\(([^)\s]+)")
LIQUID = re.compile(r"\{[%{].*?[%}]\}", re.S)
HTML_TAG = re.compile(r"</?[a-zA-Z][^>\n]*>")


def split_front_matter(text):
    m = re.match(r"^---\n(.*?)\n---\n", text, re.S)
    return (m.group(1), text[m.end():]) if m else ("", text)


def protected(text):
    text = text.replace("\r\n", "\n")
    fm, body = split_front_matter(text)
    fences = [m.group(0) for m in FENCE.finditer(body)]
    rest = FENCE.sub("", body)
    display = DISPLAY_MATH.findall(rest)
    rest = DISPLAY_MATH.sub("", rest)
    return {
        "front_matter": [fm],
        "fenced_code": fences,
        "display_math": display,
        "inline_math": INLINE_MATH.findall(rest),
        "inline_code": INLINE_CODE.findall(rest),
        "link_targets": LINK_TARGET.findall(rest),
        "liquid": LIQUID.findall(rest),
        "html_tags": HTML_TAG.findall(rest),
    }, fm, body


changed = subprocess.run(
    ["git", "-C", repo, "diff", "--name-only", "--", "_posts"],
    capture_output=True, text=True, encoding="utf-8",
).stdout.split()

lines = [f"changed posts: {len(changed)}"]
for path in changed:
    old = subprocess.run(
        ["git", "-C", repo, "show", f"HEAD:{path}"],
        capture_output=True, encoding="utf-8",
    ).stdout
    with open(f"{repo}/{path}", encoding="utf-8", newline="") as f:
        new_raw = f.read()
    problems = []
    lf_only = new_raw.count("\n") - new_raw.count("\r\n")
    if lf_only and "\r\n" in new_raw:
        problems.append(f"mixed line endings: {lf_only} LF-only lines")
    p_old, _, _ = protected(old)
    p_new, fm_new, body_new = protected(new_raw)
    for key in p_old:
        if p_old[key] != p_new[key]:
            a, b = p_old[key], p_new[key]
            gone = [x for x in a if x not in b][:3]
            added = [x for x in b if x not in a][:3]
            problems.append(f"{key} changed ({len(a)}->{len(b)}) removed={gone!r} added={added!r}")
    try:
        yaml.safe_load(fm_new)
    except yaml.YAMLError as e:
        problems.append(f"front matter YAML error: {e}")
    if body_new.count("$$") % 2:
        problems.append("odd number of $$")
    if re.search(r"[\x00-\x08\x0b\x0c\x0e-\x1f]", new_raw):
        problems.append("control characters present")
    lines.append(f"{'OK ' if not problems else 'BAD'} {path}")
    lines.extend(f"    {p}" for p in problems)

with open(out_path, "w", encoding="utf-8") as f:
    f.write("\n".join(lines) + "\n")
print(sum(1 for l in lines if l.startswith("BAD")), "files with problems")
