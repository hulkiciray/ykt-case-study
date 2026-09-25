from __future__ import annotations

import re
from datetime import date
from pathlib import Path

import numpy as np

from footnote_page_finder import printed_page_number
from table_extraction import (COLUMN_GAP, LINE_TOLERANCE, MONTHS, classify_empty_cell, group_lines,
                              line_text, number, ocr_page, parse_number, render_page, tr_lower)

VALUE_KINDS = {"number", "zero", "dash", "percent", "percent_range"}


def is_value(word):
    return parse_number(word["text"])["kind"] in VALUE_KINDS


def is_year(word):
    return re.fullmatch(r"(?:19|20)\d{2}", word["text"].strip()) is not None


# a table row ends with values; paragraph lines have text after their numbers
def split_line(line):
    i = len(line)
    while i > 0 and is_value(line[i - 1]):
        i -= 1
    return line[:i], line[i:]


def find_dates(text):
    low = tr_lower(text)
    years = re.findall(r"(?:19|20)\d{2}", low)
    dates = []
    for day, month in re.findall(r"(\d{1,2})\s*(" + "|".join(MONTHS) + ")", low):
        try:
            dates.append(date(int(years[-1]), MONTHS[month], int(day)).isoformat())
        except (IndexError, ValueError):
            pass
    return dates, [int(y) for y in years]


def classify_lines(lines, value_x0):
    kinds = []
    for ln in lines:
        label, values = split_line(ln)
        if values and not all(is_year(w) for w in values):
            kinds.append("row")
        elif values or all(w["x0"] >= value_x0 - 150 for w in ln):
            kinds.append("header")
        else:
            kinds.append("text")
    return kinds


def page_blocks(lines):
    value_words = [w for ln in lines for w in split_line(ln)[1]]
    if not value_words:
        return []
    kinds = classify_lines(lines, min(w["x0"] for w in value_words))
    blocks, headers, current, prev_kind = [], [], None, None
    for i, (ln, kind) in enumerate(zip(lines, kinds)):
        if kind == "header":
            if prev_kind != "header":
                headers = []
            headers.append(ln)
            current = None
        elif kind == "row":
            if current is None:
                current = {"headers": headers, "rows": []}
                blocks.append(current)
            label = split_line(ln)[0]
            # wrapped label: '... açılış bakiyesi (Yeniden' + 'Düzenlenmiş)'
            prev = lines[i - 1] if i else None
            if prev is not None and kinds[i - 1] == "text" and line_text(prev).count("(") > line_text(prev).count(")"):
                label = prev + label
            current["rows"].append({"label_words": label, "value_words": split_line(ln)[1],
                                    "y0": min(w["y0"] for w in ln), "y1": max(w["y1"] for w in ln)})
        else:
            current = None
        prev_kind = kind
    return blocks


def block_columns(block, gap):
    words = sorted((w for r in block["rows"] for w in r["value_words"]), key=lambda w: w["x1"])
    cols = []
    for w in words:
        if cols and w["x1"] - cols[-1]["x1s"][-1] <= gap:
            cols[-1]["x1s"].append(w["x1"])
            cols[-1]["x0"] = min(cols[-1]["x0"], w["x0"])
        else:
            cols.append({"x0": w["x0"], "x1s": [w["x1"]]})
    for c in cols:
        c["x1"] = max(c.pop("x1s"))
        header = [w for ln in block["headers"] for w in ln if w["x1"] > c["x0"] - 40 and w["x0"] < c["x1"] + 40]
        c["header"] = " ".join(line_text(ln) for ln in group_lines(header, 22))
    return cols


def extract_note_rows(cfg, root, note):
    pdf_path, dpi = root / cfg["pdf_path"], cfg["dpi"]
    s = dpi / 300
    rows_out, blocks_out = [], []
    for page in note["pages"]:
        words = ocr_page(pdf_path, page, dpi, root / cfg["ocr_cache_dir"])
        lines = group_lines(words, LINE_TOLERANCE * s)
        if printed_page_number(lines) is not None:
            lines = lines[:-1]
        if note["start"] and page == note["start"]["page"]:
            lines = [ln for ln in lines if ln[0]["y0"] > note["start"]["y"]]
        if note["end"] and note["end"]["y"] and page == note["end"]["page"]:
            lines = [ln for ln in lines if ln[0]["y0"] < note["end"]["y"]]
        img = None
        char_h = float(np.median([w["y1"] - w["y0"] for w in words])) if words else 40.0

        for block in page_blocks(lines):
            cols = block_columns(block, COLUMN_GAP * s)
            block_id = f"n{note['note_number']}_b{len(blocks_out) + 1}"
            text = " ".join(line_text(r["label_words"]) for r in block["rows"]) + " " + \
                " ".join(c["header"] for c in cols)
            block_dates, block_years = find_dates(text)
            block_year = max([int(d[:4]) for d in block_dates] + block_years, default=None)
            ids = []
            for r in block["rows"]:
                label = line_text(r["label_words"]).strip(" .:;")
                values = []
                for ci, c in enumerate(cols):
                    ws = [w for w in r["value_words"] if w["x1"] > c["x0"] - 5 and w["x0"] < c["x1"] + 5]
                    if ws:
                        cell = parse_number("".join(w["text"] for w in ws))
                        cell["ocr_conf"] = float(np.mean([w["conf"] for w in ws]))
                    else:
                        img = render_page(pdf_path, page, dpi) if img is None else img
                        box = (int(c["x0"]), int(r["y0"] + 2), int(c["x1"]) + 4, int(r["y1"] - 2))
                        state, conf = classify_empty_cell(img, box, 0.6 * (c["x1"] - c["x0"]), char_h)
                        cell = number("-" if state == "dash" else "", "dash" if state == "dash" else
                                      "empty" if state == "empty" else "unparseable")
                        cell["ocr_conf"] = conf
                    cell.update(col=ci, header=c["header"], period=value_period(c["header"], label, block_year))
                    fmt = 1.0 if not cell["repairs"] and cell["kind"] != "unparseable" else 0.5
                    cell["confidence"] = round(0.6 * cell["ocr_conf"] + 0.4 * fmt, 4)
                    values.append(cell)
                row_id = f"n{note['note_number']}_p{page}_r{len(rows_out) + 1:02d}"
                label_conf = float(np.mean([w["conf"] for w in r["label_words"]])) if r["label_words"] else 0.6
                rows_out.append({
                    "row_id": row_id, "block_id": block_id, "page": page, "label": label, "values": values,
                    "confidence": round(0.3 * label_conf + 0.7 * float(np.mean([v["confidence"] for v in values])), 4),
                })
                ids.append(row_id)
            blocks_out.append({"block_id": block_id, "page": page, "year": block_year,
                               "columns": [c["header"] for c in cols], "row_ids": ids})
    return {"note_number": note["note_number"], "title": note["title"], "pages": note["pages"],
            "blocks": blocks_out, "rows": rows_out}


# period from the column header, else the row label, else the block (movement rows)
def value_period(header, label, block_year):
    for source, text in (("header", header), ("label", label)):
        dates, years = find_dates(text)
        if dates:
            return {"date": dates[-1], "year": int(dates[-1][:4]), "source": source}
        if source == "header" and years:
            return {"date": None, "year": years[-1], "source": source}
    return {"date": None, "year": block_year, "source": "block"}
