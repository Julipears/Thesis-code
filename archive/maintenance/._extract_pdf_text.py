from pathlib import Path
import re
import zlib

pdf = Path(r"C:\Users\Julia\Downloads\Thesis_Report (16).pdf").read_bytes()
out = []
streams = list(re.finditer(rb"/FlateDecode\b.*?stream\r?\n", pdf, flags=re.S))
print("candidate streams", len(streams), flush=True)
for index, match in enumerate(streams):
    print("processing", index, flush=True)
    start = match.end()
    end = pdf.find(b"endstream", start)
    if end < 0:
        continue
    raw = pdf[start:end].rstrip(b"\r\n")
    if len(raw) > 250_000:
        continue
    try:
        data = zlib.decompress(raw)
    except Exception:
        continue
    if not (b"BT" in data and (b"Tj" in data or b"TJ" in data)):
        continue
    # Scan literal PDF strings without a backtracking regex; some font streams
    # contain long runs of parentheses that make a regex approach very slow.
    i = 0
    while i < len(data):
        if data[i:i + 1] != b"(":
            i += 1
            continue
        j = i + 1
        depth = 1
        chars = bytearray()
        while j < len(data) and depth:
            c = data[j]
            if c == 92 and j + 1 < len(data):
                chars.extend(data[j:j + 2])
                j += 2
                continue
            if c == 40:
                depth += 1
            elif c == 41:
                depth -= 1
                if depth == 0:
                    break
            chars.append(c)
            j += 1
        if depth == 0:
            value = chars.decode("latin1", errors="ignore")
            value = re.sub(r"\\([\\()])", r"\1", value).replace("\\n", " ")
            tail = data[j + 1:j + 8]
            if b"Tj" in tail or b"TJ" in tail:
                if value.strip():
                    out.append(value)
            i = j + 1
        else:
            break
    # Recover text arrays (TJ), which are common in LaTeX PDFs. Preserve a
    # joined version so captions and headings become searchable.
    i = 0
    while i < len(data):
        if data[i:i + 1] != b"[":
            i += 1
            continue
        close = data.find(b"] TJ", i + 1)
        if close < 0:
            i += 1
            continue
        block = data[i + 1:close]
        vals = []
        j = 0
        while j < len(block):
            if block[j:j + 1] != b"(":
                j += 1
                continue
            k = j + 1
            depth = 1
            chars = bytearray()
            while k < len(block) and depth:
                c = block[k]
                if c == 92 and k + 1 < len(block):
                    chars.extend(block[k:k + 2])
                    k += 2
                    continue
                if c == 40:
                    depth += 1
                elif c == 41:
                    depth -= 1
                    if depth == 0:
                        break
                chars.append(c)
                k += 1
            if depth == 0:
                vals.append(chars.decode("latin1", errors="ignore"))
                j = k + 1
            else:
                break
        value = re.sub(r"\\([\\()])", r"\1", "".join(vals)).replace("\\n", " ")
        if value.strip():
            out.append(value)
        i = close + 4
    if index % 25 == 0:
        print(index, flush=True)
Path("._thesis_report_extracted.txt").write_text("\n".join(out), encoding="utf-8")
print(f"streams={len(out)}")
