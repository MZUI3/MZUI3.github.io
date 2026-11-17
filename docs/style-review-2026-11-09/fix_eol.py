"""Normalize given files to CRLF line endings. Usage: python fix_eol.py <files...>"""
import sys
for f in sys.argv[1:]:
    b = open(f, "rb").read()
    n = b.replace(b"\r\n", b"\n").replace(b"\n", b"\r\n")
    if n != b:
        open(f, "wb").write(n)
        print("fixed", f)
