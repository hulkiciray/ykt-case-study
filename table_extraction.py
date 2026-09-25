from __future__ import annotations

import hashlib
import json
import re
import warnings
from datetime import date
from pathlib import Path

import cv2
import pymupdf as fitz
import numpy as np

warnings.filterwarnings("ignore")

# pixel values at 300 dpi
LINE_TOLERANCE = 22
COLUMN_GAP = 90
INDENT_GAP = 10
MIN_COLUMN_WORDS = 3
SUM_TOLERANCE = 1  # TL, rounding in printed totals

STATEMENT_TYPES = {
    "balance_sheet": ["BİLANÇO", "FİNANSAL DURUM"],
    "comprehensive_income": ["KAPSAMLI GELİR", "GELİR TABLOSU"],
    "cash_flow": ["NAKİT AKIM"],
    "equity_changes": ["ÖZKAYNAK DEĞİŞİM"],
}
MONTHS = {"ocak": 1, "şubat": 2, "mart": 3, "nisan": 4, "mayıs": 5, "haziran": 6, "temmuz": 7,
          "ağustos": 8, "eylül": 9, "ekim": 10, "kasım": 11, "aralık": 12}


_reader = None


def get_reader():
    global _reader
    if _reader is None:
        import easyocr
        _reader = easyocr.Reader(["tr"], gpu=True, verbose=False)
    return _reader


def render_page(pdf_path, page_no, dpi):
    with fitz.open(pdf_path) as doc:
        pix = doc[page_no - 1].get_pixmap(dpi=dpi, colorspace=fitz.csGRAY)
    return np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.h, pix.w).copy()


def ocr_page(pdf_path, page_no, dpi, cache_dir):
    doc_id = hashlib.sha1(Path(pdf_path).read_bytes()).hexdigest()[:12]
    cache = Path(cache_dir) / f"{doc_id}_p{page_no:03d}_{dpi}dpi.json"
    if cache.exists():
        return json.loads(cache.read_text(encoding="utf-8"))["tokens"]
    img = render_page(pdf_path, page_no, dpi)
    words = [to_word(box, text, conf) for box, text, conf in get_reader().readtext(img)]
    cache.parent.mkdir(parents=True, exist_ok=True)
    cache.write_text(json.dumps({"page": page_no, "dpi": dpi, "tokens": words}, ensure_ascii=False),
                     encoding="utf-8")
    return words


# second read for amounts that did not parse, e.g. '(561.5' + '989)'
def reread_number(img, box):
    x0, y0, x1, y1 = box
    result = get_reader().recognize(img[y0:y1, x0:x1], allowlist="0123456789.,()-")
    text = "".join(t for _, t, _ in result)
    conf = float(np.mean([c for _, _, c in result])) if result else 0.0
    return text, conf


def to_word(box, text, conf):
    xs = [float(p[0]) for p in box]
    ys = [float(p[1]) for p in box]
    return {"text": str(text), "conf": float(conf), "x0": min(xs), "y0": min(ys), "x1": max(xs), "y1": max(ys)}


DASHES = {"-", "–", "—", "_", "~", "−"}
LETTER_AS_DIGIT = str.maketrans("OoDlI|iSB", "000111158")
SEVERE_REPAIRS = {"unbalanced_parenthesis", "irregular_grouping", "unreadable_ink"}


def parse_number(raw, decimal=False):
    s = (raw or "").strip()
    if s == "":
        return number(raw, "empty")
    if s in DASHES:
        return number(raw, "dash")
    if re.fullmatch(r"%\s*\d+([.,]\d+)?", s):
        return number(raw, "percent", float(s[1:].strip().replace(",", ".")), decimal=True)
    m = re.fullmatch(r"%\s*(\d+(?:[.,]\d+)?)\s*-\s*%?\s*(\d+(?:[.,]\d+)?)", s)
    if m:
        return number(raw, "percent_range", [float(g.replace(",", ".")) for g in m.groups()], decimal=True)

    repairs = []
    s = s.replace(" ", "")
    fixed = s.translate(LETTER_AS_DIGIT)
    if fixed != s:
        repairs.append("letter_as_digit")
        s = fixed
    negative = s.startswith("(") or s.endswith(")")
    if negative and not (s.startswith("(") and s.endswith(")")):
        repairs.append("unbalanced_parenthesis")
    s = s.strip("()-")
    if re.search(r"[.,]{2,}", s):
        repairs.append("doubled_separator")
        s = re.sub(r"[.,]{2,}", ".", s)
    if not re.fullmatch(r"\d+([.,]\d+)*", s):
        return number(raw, "unparseable", repairs=repairs)

    sign = -1 if negative else 1
    groups = re.split(r"[.,]", s)
    if len(groups) == 2 and "," in s and (decimal or groups[0] == "0"):
        return number(raw, "number", sign * float(".".join(groups)), repairs, decimal=True)
    if len(groups) == 1 and len(s) > 3:
        repairs.append("missing_thousands_separator")
    elif len(groups) > 1 and not (len(groups[0]) <= 3 and all(len(g) == 3 for g in groups[1:])):
        repairs.append("irregular_grouping")
    elif "," in s:
        repairs.append("comma_as_thousands_separator")
    value = sign * int("".join(groups))
    return number(raw, "zero" if value == 0 else "number", value, repairs)


def number(raw, kind, value=None, repairs=None, decimal=False):
    return {"raw": raw or "", "kind": kind, "value": value, "decimal": decimal, "repairs": repairs or []}


def is_amount(word):
    t = word["text"].replace(" ", "")
    return bool(re.search(r"\d[\d.,]{2,}", t) or re.fullmatch(r"\(?0,\d+\)?", t))


def tr_upper(s):
    return s.replace("i", "İ").replace("ı", "I").upper()


def tr_lower(s):
    return s.replace("I", "ı").replace("İ", "i").lower()


def is_all_caps(s):
    letters = [c for c in s if c.isalpha()]
    return len(letters) >= 3 and sum(c.isupper() for c in letters) / len(letters) > 0.8


def clean_label(raw):
    s = re.sub(r"^\s*[-~–—_]+\s*", "", raw).replace("_", " ")
    s = re.sub(r"\s+", " ", s).strip(" .:;")
    return tr_upper(s[0]) + s[1:] if s[:1].islower() else s


def parse_note_refs(text):
    fixed = text.strip().translate(str.maketrans("IlL|!iOoS", "111111005"))
    return [int(n) for n in re.findall(r"\d{1,2}", fixed) if 1 <= int(n) <= 99], fixed != text.strip()


def parse_period(header):
    low = tr_lower(header)
    years = [int(y) for y in re.findall(r"(?:19|20)\d{2}", low)]
    period = {"type": None, "start": None, "end": None, "year": years[-1] if years else None,
              "audited": "denetim" in low and "denetimden geçmemiş" not in low, "header": header}
    if not years:
        return period
    dates = []
    for day, month in re.findall(r"(\d{1,2})\s*(" + "|".join(MONTHS) + ")", low):
        try:
            dates.append(date(years[-1], MONTHS[month], int(day)).isoformat())
        except ValueError:
            pass
    if len(dates) >= 2:
        period.update(type="duration", start=dates[0], end=dates[-1])
    else:
        period.update(type="instant", end=dates[0] if dates else date(years[-1], 12, 31).isoformat())
    return period


def detect_currency(text):
    low = tr_lower(text)
    if "türk lirası" in low or re.search(r"\btl\b", low):
        currency = "TRY"
    elif "abd doları" in low:
        currency = "USD"
    elif "avro" in low:
        currency = "EUR"
    else:
        currency = None
    scale = 1_000 if re.search(r"\bbin\b", low) else 1_000_000 if "milyon" in low else 1
    return currency, scale


def yc(w):
    return (w["y0"] + w["y1"]) / 2


def line_text(line):
    return " ".join(w["text"] for w in line)


def group_lines(words, tol):
    lines = []
    for w in sorted(words, key=yc):
        near = [ln for ln in lines[-3:] if abs(np.median([yc(x) for x in ln]) - yc(w)) <= tol]
        if near:
            min(near, key=lambda ln: abs(np.median([yc(x) for x in ln]) - yc(w))).append(w)
        else:
            lines.append([w])
    return [sorted(ln, key=lambda w: w["x0"]) for ln in lines]


def overlap(col, w):
    return max(0.0, min(col["x1"], w["x1"]) - max(col["x0"], w["x0"]))


# amounts are right-aligned, so right edges cluster per column
def find_value_columns(words, gap):
    amounts = sorted((w for w in words if is_amount(w)), key=lambda w: w["x1"])
    clusters = []
    for w in amounts:
        if clusters and w["x1"] - clusters[-1][-1]["x1"] <= gap:
            clusters[-1].append(w)
        else:
            clusters.append([w])
    clusters = [c for c in clusters if len(c) >= MIN_COLUMN_WORDS]
    return [{"col_id": f"v{i}",
             "x0": float(np.percentile([w["x0"] for w in c], 5)),
             "x1": float(np.percentile([w["x1"] for w in c], 90))} for i, c in enumerate(clusters)]


def find_note_column(words, value_cols):
    if not value_cols:
        return None
    cands = [w for w in words if re.fullmatch(r"[0-9Il|iL!]{1,2}", w["text"].strip())
             and w["x1"] < value_cols[0]["x0"]]
    header = [w for w in words if tr_lower(w["text"]).startswith("dipnot")]
    if header:
        cands = [w for w in cands if abs((w["x0"] + w["x1"]) / 2 - (header[0]["x0"] + header[0]["x1"]) / 2) < 150]
    if len(cands) < MIN_COLUMN_WORDS:
        return None
    xc = float(np.median([(w["x0"] + w["x1"]) / 2 for w in cands]))
    return {"x0": xc - 60, "x1": xc + 60}


def ink_boxes(img, box, rule_width):
    x0, y0, x1, y1 = box
    ink = (img[max(0, y0):y1, max(0, x0):x1] < 128).astype(np.uint8)
    if ink.size == 0:
        return []
    n, _, stats, _ = cv2.connectedComponentsWithStats(ink, connectivity=8)
    return [(x0 + x, y0 + y, w, h) for x, y, w, h, area in stats[1:]
            if area >= 12 and not (w >= rule_width and h <= 6)]


# EasyOCR never returns a lone '-', so empty cells are checked in the pixels
def classify_empty_cell(img, box, rule_width, char_h):
    blobs = ink_boxes(img, box, rule_width)
    if not blobs:
        return "empty", 0.95
    if len(blobs) == 1:
        _, _, w, h = blobs[0]
        if w >= 1.6 * h and h <= 0.35 * char_h and w <= 1.5 * char_h:
            return "dash", 0.9
    return "unreadable", 0.3


def extract_table(img, words, page_no, dpi):
    s = dpi / 300
    lines = group_lines(words, LINE_TOLERANCE * s)
    char_h = float(np.median([w["y1"] - w["y0"] for w in words]))

    unit_idx = next((i for i, ln in enumerate(lines) if re.search(r"tutarlar|gösterilmiş", line_text(ln), re.I)), 0)
    company = line_text(lines[0]) if unit_idx > 0 else ""
    title = " ".join(line_text(ln) for ln in lines[1:unit_idx])
    unit_text = line_text(lines[unit_idx])
    currency, scale = detect_currency(unit_text)
    statement_type = next((t for t, keys in STATEMENT_TYPES.items() if any(k in tr_upper(title) for k in keys)), None)

    below = [w for ln in lines[unit_idx + 1:] for w in ln]
    cols = find_value_columns(below, COLUMN_GAP * s)
    if not cols:
        raise ValueError(f"page {page_no}: no value columns found")
    note_col = find_note_column(below, cols)
    label_limit = (note_col or cols[0])["x0"] - 5

    body_start = next((i for i in range(unit_idx + 1, len(lines))
                       if any(w["x1"] <= label_limit for w in lines[i])), len(lines))
    for col in cols:
        header_words = [w for ln in lines[unit_idx + 1:body_start] for w in ln if overlap(col, w) > 0]
        col["period"] = parse_period(" ".join(line_text(ln) for ln in group_lines(header_words, LINE_TOLERANCE * s)))
    years = [c["period"]["year"] for c in cols if c["period"]["year"]]
    for col in cols:
        col["period"]["role"] = "current" if years and col["period"]["year"] == max(years) else "prior"

    rows = []
    for ln in lines[body_start:]:
        if re.search(r"ekteki\s+dipnot", line_text(ln), re.I):
            break
        rows.append(line_to_row(ln, cols, note_col, label_limit))
    rows = merge_wrapped_labels(rows, INDENT_GAP * s)
    for r in rows:
        read_cells(r, img, cols, char_h)
        r["label"] = clean_label(r["label_raw"]) if r["label_raw"] else ""
    set_indent_levels(rows, INDENT_GAP * s)

    checks = find_hierarchy(rows, [c["col_id"] for c in cols])
    score_confidence(rows, checks)
    return {"page": page_no, "statement_type": statement_type, "company": company, "title": title,
            "currency": currency, "unit_scale": scale, "unit_text": unit_text,
            "columns": cols, "note_column_found": note_col is not None, "rows": rows, "checks": checks}


def line_to_row(line, cols, note_col, label_limit):
    label, notes, cells = [], [], {c["col_id"]: [] for c in cols}
    for w in line:
        best = max(cols, key=lambda c: overlap(c, w))
        if note_col and overlap(note_col, w) > 0.5 * (w["x1"] - w["x0"]) and not is_amount(w):
            notes.append(w)
        elif overlap(best, w) > 0.3 * (w["x1"] - w["x0"]):
            cells[best["col_id"]].append(w)
        elif w["x0"] < label_limit or not is_amount(w):
            label.append(w)
        else:
            cells[min(cols, key=lambda c: abs(c["x1"] - w["x1"]))["col_id"]].append(w)
    note_raw = " ".join(w["text"] for w in notes)
    note_refs, note_repaired = parse_note_refs(note_raw) if notes else ([], False)
    return {"label_raw": " ".join(w["text"] for w in label),
            "label_conf": float(np.mean([w["conf"] for w in label])) if label else 0.0,
            "x0": label[0]["x0"] if label else None,
            "y0": min(w["y0"] for w in line), "y1": max(w["y1"] for w in line),
            "note_raw": note_raw, "note_refs": note_refs, "note_repaired": note_repaired,
            "cell_words": cells, "flags": []}


# 'Özkaynak Yöntemiyle Değerlenen Yatırımların' + 'Kar/Zararlarındaki Paylar'
def merge_wrapped_labels(rows, indent_gap):
    out = []
    for r in rows:
        prev = out[-1] if out else None
        if (prev and prev["x0"] is not None and r["x0"] is not None
                and not any(prev["cell_words"].values()) and not prev["note_raw"]
                and not is_all_caps(prev["label_raw"]) and r["x0"] - prev["x0"] >= indent_gap):
            r["label_raw"] = prev["label_raw"] + " " + r["label_raw"]
            r["label_conf"] = (prev["label_conf"] + r["label_conf"]) / 2
            r["x0"], r["y0"] = prev["x0"], prev["y0"]
            r["flags"].append("wrapped_label_merged")
            out[-1] = r
        else:
            out.append(r)
    return out


def read_cells(row, img, cols, char_h):
    words_by_col = row.pop("cell_words")
    all_text = [w["text"] for ws in words_by_col.values() for w in ws]
    row["decimal"] = any(re.fullmatch(r"\(?0,\d+\)?", t) for t in all_text)
    row["cells"] = {}
    for col in cols:
        words = words_by_col[col["col_id"]]
        rule_width = 0.6 * (col["x1"] - col["x0"])
        if not words:
            box = (int(col["x0"]), int(row["y0"] + 2), int(col["x1"]) + 4, int(row["y1"] - 2))
            state, conf = classify_empty_cell(img, box, rule_width, char_h)
            if state == "unreadable":
                cell = number("", "unparseable", repairs=["unreadable_ink"])
            else:
                cell = number("-" if state == "dash" else "", state)
            cell.update(source="pixels", ocr_conf=conf, bbox=list(box))
            row["cells"][col["col_id"]] = cell
            continue
        text = "".join(w["text"].replace(" ", "") for w in words)
        cell = parse_number(text, row["decimal"])
        cell.update(source="page", ocr_conf=float(np.mean([w["conf"] for w in words])),
                    bbox=[min(w["x0"] for w in words), min(w["y0"] for w in words),
                          max(w["x1"] for w in words), max(w["y1"] for w in words)])
        if cell["kind"] == "unparseable" or SEVERE_REPAIRS & set(cell["repairs"]):
            band = (int(min(col["x0"], cell["bbox"][0]) - 12), int(cell["bbox"][1] - 2),
                    int(max(col["x1"], cell["bbox"][2]) + 12), int(cell["bbox"][3] + 2))
            blobs = ink_boxes(img, band, rule_width)
            if blobs:
                box = (min(b[0] for b in blobs) - 4, min(b[1] for b in blobs) - 4,
                       max(b[0] + b[2] for b in blobs) + 4, max(b[1] + b[3] for b in blobs) + 4)
                text2, conf2 = reread_number(img, box)
                second = parse_number(text2, row["decimal"])
                if second["kind"] in ("number", "zero") and not SEVERE_REPAIRS & set(second["repairs"]):
                    second["repairs"].append(f"reread_replaced:{text}")
                    second.update(source="reread", ocr_conf=conf2, bbox=list(box))
                    cell = second
        row["cells"][col["col_id"]] = cell
    for cell in row["cells"].values():
        cell["num"] = cell["value"] if cell["kind"] in ("number", "zero") else 0 if cell["kind"] == "dash" else None
        cell["arithmetic_ok"] = None


def set_indent_levels(rows, gap):
    xs = sorted({round(r["x0"]) for r in rows if r["x0"] is not None})
    levels, level = {}, 0
    for i, x in enumerate(xs):
        if i and x - xs[i - 1] > gap:
            level += 1
        levels[x] = level
    for r in rows:
        r["indent"] = levels[round(r["x0"])] if r["x0"] is not None else -1


# Balance sheet: parent first, items indented below it.
# Income statement: totals come after their parts (Satış + Maliyet = BRÜT KAR).
# So relations are decided by arithmetic over all period columns, not by layout.
def find_hierarchy(rows, col_ids):
    n = len(rows)
    checks = []
    for r in rows:
        r.update(row_type="item", parent=None, children=[])

    def vec(i):
        vals = [rows[i]["cells"][c]["num"] for c in col_ids]
        if rows[i]["decimal"] or all(v is None for v in vals) or not any(vals):
            return None
        return [v or 0 for v in vals]

    def sums_to(total, parts):
        return all(abs(total[k] - sum(p[k] for p in parts)) <= SUM_TOLERANCE for k in range(len(total)))

    # unlabelled rows only restate a total, they are never a component
    def is_free(i):
        return vec(i) is not None and rows[i]["parent"] is None and rows[i]["label"] != ""

    def indented_children(t):
        kids = []
        for k in range(t + 1, n):
            if 0 <= rows[k]["indent"] <= rows[t]["indent"]:
                break
            kids.append(k)
        kids = [k for k in kids if vec(k) is not None and rows[k]["indent"] >= 0]
        top = min((rows[k]["indent"] for k in kids), default=None)
        return [k for k in kids if rows[k]["indent"] == top]

    def link(parent, kids, row_type, relation):
        rows[parent]["row_type"] = row_type
        rows[parent]["children"] = rows[parent]["children"] + kids
        rows[parent]["only_repeated"] = relation == "single_component"
        for k in kids:
            rows[k]["parent"] = parent
        for ci, c in enumerate(col_ids):
            expected = vec(parent)[ci]
            computed = sum(vec(k)[ci] for k in kids)
            ok = abs(expected - computed) <= SUM_TOLERANCE
            checks.append({"relation": relation, "column": c, "parent_row": parent, "component_rows": kids,
                           "expected": expected, "computed": computed, "difference": expected - computed,
                           "passed": ok})
            for j in [parent] + kids:
                cell = rows[j]["cells"][c]
                cell["arithmetic_ok"] = ok if cell["arithmetic_ok"] is None else cell["arithmetic_ok"] and ok

    def prefix_match(t, candidates):
        acc = [0.0] * len(col_ids)
        for k, i in enumerate(candidates):
            acc = [a + b for a, b in zip(acc, vec(i))]
            if k >= 1 and sums_to(vec(t), [acc]):
                return candidates[:k + 1]
        return None

    # 1. indented children whose sum matches
    for t in range(n):
        kids = indented_children(t) if vec(t) is not None and rows[t]["indent"] >= 0 else []
        if kids and sums_to(vec(t), [vec(k) for k in kids]):
            link(t, kids, "parent", "sum_of_following")

    # 2. next row repeats the same values -> its only component
    for t in range(n - 1):
        if vec(t) and vec(t + 1) and rows[t + 1]["label"] and not rows[t]["children"] \
                and rows[t + 1]["parent"] is None and sums_to(vec(t), [vec(t + 1)]):
            link(t, [t + 1], "parent", "single_component")

    # 3. running totals (rows before) or same-level parents (rows after)
    def find_totals():
        changed = True
        while changed:
            changed = False
            for t in range(n):
                # 'kapanış bakiyesi' followed by 'net defter değeri' can still be a total
                if vec(t) is None or (rows[t]["children"] and not rows[t].get("only_repeated")):
                    continue
                before = prefix_match(t, [i for i in range(t - 1, -1, -1) if is_free(i)][:14])
                after = prefix_match(t, [i for i in range(t + 1, n) if is_free(i)][:14])
                if before:
                    link(t, sorted(before), "total", "sum_of_preceding")
                    changed = True
                elif after:
                    link(t, sorted(after), "parent", "sum_of_following")
                    changed = True

    find_totals()

    # 4. layout-only links, their failed checks show OCR errors
    for t in range(n):
        if vec(t) is not None and not rows[t]["children"] and rows[t]["indent"] >= 0:
            kids = [k for k in indented_children(t) if rows[k]["parent"] is None]
            if kids:
                link(t, kids, "parent", "sum_of_following_unconfirmed")
    find_totals()

    for r in rows:
        if r["label"] and not r["decimal"] and all(r["cells"][c]["kind"] == "empty" for c in col_ids):
            r["row_type"] = "section_header"
    return checks


# value: 0.4 ocr + 0.2 clean format + 0.4 arithmetic
# row:   0.3 label ocr + 0.4 values + 0.3 structure
def score_confidence(rows, checks):
    arith_score = {True: 1.0, False: 0.0, None: 0.5}
    for i, r in enumerate(rows):
        for cell in r["cells"].values():
            fmt = 1.0 if not cell["repairs"] else 0.0 if SEVERE_REPAIRS & set(cell["repairs"]) else 0.6
            cell["confidence"] = round(0.4 * cell["ocr_conf"] + 0.2 * fmt + 0.4 * arith_score[cell["arithmetic_ok"]], 4)
        mine = [c for c in checks if c["parent_row"] == i or i in c["component_rows"]]
        structure = 0.0 if any(not c["passed"] for c in mine) else 1.0 if mine else 0.6
        label = r["label_conf"] if r["label"] else 0.6
        if r["note_repaired"]:
            label *= 0.9
        cells = np.mean([c["confidence"] for c in r["cells"].values()])
        r["confidence"] = round(float(0.3 * label + 0.4 * cells + 0.3 * structure), 4)


def extract_summary_tables(cfg, root="."):
    pdf_path = Path(root) / cfg["pdf_path"]
    dpi, threshold = cfg["dpi"], cfg["low_confidence_threshold"]
    tables = []
    for page_no in cfg["summary_pages"]:
        img = render_page(pdf_path, page_no, dpi)
        words = ocr_page(pdf_path, page_no, dpi, Path(root) / cfg["ocr_cache_dir"])
        tables.append(to_json(extract_table(img, words, page_no, dpi), threshold))

    periods = [c["period"] for t in tables for c in t["columns"]]
    checks = [c for t in tables for c in t["checks"]]
    values = [v for t in tables for r in t["rows"] for v in r["values"]]
    return {
        "document": {
            "file": pdf_path.name,
            "company": tables[0]["company"] if tables else "",
            "period_end": max((p["end"] for p in periods if p["role"] == "current"), default=None),
            "currency": tables[0]["currency"] if tables else None,
        },
        "tables": tables,
        "summary": {
            "tables": len(tables),
            "rows": sum(len(t["rows"]) for t in tables),
            "values": len(values),
            "checks_passed": sum(c["passed"] for c in checks),
            "checks_total": len(checks),
            "low_confidence_rows": [r["row_id"] for t in tables for r in t["rows"] if r["low_confidence"]],
            "low_confidence_values": [f"{r['row_id']}:{v['col_id']}" for t in tables for r in t["rows"]
                                      for v in r["values"] if v["low_confidence"]],
        },
    }


def to_json(t, threshold):
    ids = [f"p{t['page']}_r{i:02d}" for i in range(len(t["rows"]))]
    rows = []
    for i, r in enumerate(t["rows"]):
        values = []
        for c in t["columns"]:
            cell = r["cells"][c["col_id"]]
            values.append({
                "col_id": c["col_id"], "period_end": c["period"]["end"],
                "raw": cell["raw"], "kind": cell["kind"], "value": cell["value"], "repairs": cell["repairs"],
                "source": cell["source"], "ocr_conf": round(cell["ocr_conf"], 4),
                "arithmetic_ok": cell["arithmetic_ok"], "confidence": cell["confidence"],
                "low_confidence": cell["confidence"] < threshold and cell["kind"] != "empty",
            })
        rows.append({
            "row_id": ids[i], "label": r["label"], "label_raw": r["label_raw"],
            "row_type": r["row_type"], "indent": r["indent"],
            "parent_id": ids[r["parent"]] if r["parent"] is not None else None,
            "children": [ids[k] for k in r["children"]],
            "note_refs": r["note_refs"], "note_raw": r["note_raw"], "values": values,
            "confidence": r["confidence"], "low_confidence": r["confidence"] < threshold,
            "flags": r["flags"] + (["note_ref_repaired"] if r["note_repaired"] else []),
        })
    checks = [dict(c, parent_row=ids[c["parent_row"]], component_rows=[ids[k] for k in c["component_rows"]])
              for c in t["checks"]]
    passed = sum(c["passed"] for c in checks) / len(checks) if checks else 0.5
    confidence = round(0.6 * float(np.mean([r["confidence"] for r in rows])) + 0.4 * passed, 4)
    return {
        "table_id": f"p{t['page']}_{t['statement_type']}", "page": t["page"],
        "statement_type": t["statement_type"], "company": t["company"], "title": t["title"],
        "currency": t["currency"], "unit_scale": t["unit_scale"], "unit_text": t["unit_text"],
        "columns": [{"col_id": c["col_id"], "period": {k: c["period"][k] for k in
                     ("type", "start", "end", "role", "audited", "header")}} for c in t["columns"]],
        "rows": rows, "checks": checks,
        "confidence": confidence, "low_confidence": confidence < threshold,
    }
