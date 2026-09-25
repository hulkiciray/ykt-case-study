from __future__ import annotations

import re
from collections import Counter
from difflib import SequenceMatcher

from table_extraction import group_lines, is_all_caps, line_text, ocr_page, parse_note_refs, tr_upper

# '11. YATIRIM ...', OCR variants like '1l. ...', '19 ÖZKAYNAKLAR', '20_ SATIŞLAR'
HEADING = re.compile(r"^(?:NOT\s*)?([0-9IlL|]{1,2})\s*[.:\-_]?\s+([^\d]{3,})$")
TOC_ENTRY = re.compile(r"^NOT\s*([0-9IlL|]{1,2})\s+(.+?)[\s.\-_]*(\d{1,3})(?:\s*-\s*(\d{1,3}))?$")
TOC_TITLE_ONLY = re.compile(r"^NOT\s*([0-9IlL|]{1,2})\s+(\D+)$")
MIN_TITLE_SIMILARITY = 0.6


def page_lines(cfg, root, page_no):
    words = ocr_page(root / cfg["pdf_path"], page_no, cfg["dpi"], root / cfg["ocr_cache_dir"])
    return group_lines(words, 22 * cfg["dpi"] / 300) if words else []


def similarity(a, b):
    clean = lambda s: re.sub(r"[^A-ZÇĞİÖŞÜ ]", "", tr_upper(s)).split()
    return SequenceMatcher(None, " ".join(clean(a)), " ".join(clean(b))).ratio()


def read_toc(lines):
    entries, pending = {}, ""
    for ln in lines:
        text = re.sub(r"\s+", " ", line_text(ln)).strip()
        if pending and not text.upper().startswith("NOT"):
            text = pending + " " + text
        elif pending:
            m = TOC_TITLE_ONLY.match(pending)
            refs = parse_note_refs(m.group(1))[0] if m else []
            if refs:
                entries.setdefault(refs[0], (m.group(2).strip(" .-_"), None, None))
        m = TOC_ENTRY.match(text)
        if m:
            refs, _ = parse_note_refs(m.group(1))
            if refs:
                first = int(m.group(3))
                entries[refs[0]] = (m.group(2).strip(" .-_"), first, int(m.group(4) or first))
            pending = ""
        else:
            pending = text if re.match(r"^NOT\s*[0-9IlL|]{1,2}\b", text) else ""
    return entries


def find_headings(lines):
    found = []
    for ln in lines:
        m = HEADING.match(line_text(ln).strip())
        if not m:
            continue
        title = m.group(2).strip()
        continued = bool(re.search(r"\(devam", title, re.I))
        title = re.sub(r"\(devam.*$", "", title, flags=re.I).strip()
        refs, _ = parse_note_refs(m.group(1))
        if refs and is_all_caps(title):
            found.append((refs[0], title, ln[0]["y0"], continued))
    return found


def printed_page_number(lines):
    if lines and re.fullmatch(r"\d{1,3}", line_text(lines[-1]).strip()):
        return int(line_text(lines[-1]))
    return None


def find_note_pages(cfg, root, n_pages):
    note = cfg["note_number"]
    pages = {p: page_lines(cfg, root, p) for p in range(1, n_pages + 1)}

    toc_page = next((p for p, lines in pages.items()
                     if any("İÇİNDEKİLER" in tr_upper(line_text(ln)) for ln in lines[:5])), None)
    toc = read_toc(pages[toc_page]) if toc_page else {}
    toc_entry = toc.get(note)

    # printed page = pdf page - offset, the most common difference over all readable folios
    folios = {p: printed_page_number(lines) for p, lines in pages.items()}
    offsets = Counter(p - f for p, f in folios.items() if f)
    offset = offsets.most_common(1)[0][0] if offsets else None
    printed = {p: p - offset if offset is not None else None for p in pages}

    # the TOC lines look like headings too
    headings = {p: find_headings(lines) if p > (toc_page or 0) else [] for p, lines in pages.items()}
    if toc_entry:
        headings = {p: [h for h in hs if h[0] != note or similarity(h[1], toc_entry[0]) >= MIN_TITLE_SIMILARITY]
                    for p, hs in headings.items()}
    flags = []

    start = next((p for p, hs in headings.items() for n, _, _, cont in hs if n == note and not cont), None)
    if start is None:
        flags.append("heading_not_found")
        if toc_entry and toc_entry[1] and offset is not None:
            first, last = toc_entry[1] + offset, toc_entry[2] + offset
            return result(note, list(range(first, last + 1)), toc_entry, None, None, printed, offset,
                          0.3, flags + ["pages_from_toc_only"])
        return result(note, [], toc_entry, None, None, printed, offset, 0.0, flags)
    heading = next(h for h in headings[start] if h[0] == note and not h[3])

    # next pages: same note number ('(devamı)') or no heading inside the TOC range
    note_pages, end_y = [start], None
    for p in range(start + 1, n_pages + 1):
        hs = headings[p]
        if hs and hs[0][0] == note:
            note_pages.append(p)
        elif not hs and toc_entry and toc_entry[2] and printed[p] is not None and printed[p] <= toc_entry[2]:
            note_pages.append(p)
        else:
            break
    last = note_pages[-1]
    later = [y for n, _, y, _ in headings[last] if n != note and (last != start or y > heading[2])]
    if later:
        end_y = min(later)

    title_sim = similarity(heading[1], toc_entry[0]) if toc_entry else 0.5
    if not toc_entry:
        toc_agree = 0.5
        flags.append("note_missing_in_toc")
    elif offset is None or toc_entry[1] is None:
        toc_agree = 0.5
    else:
        toc_agree = float(printed[note_pages[0]] == toc_entry[1] and printed[note_pages[-1]] == toc_entry[2])
        if not toc_agree:
            flags.append("pages_disagree_with_toc")
    confidence = round(0.4 + 0.3 * toc_agree + 0.3 * title_sim, 4)
    return result(note, note_pages, toc_entry, heading, end_y, printed, offset, confidence, flags)


def result(note, pages, toc_entry, heading, end_y, printed, offset, confidence, flags):
    return {
        "note_number": note,
        "title": heading[1] if heading else (toc_entry[0] if toc_entry else None),
        "pages": pages,
        "printed_pages": [printed.get(p) for p in pages],
        "start": {"page": pages[0], "y": heading[2]} if heading else None,
        "end": {"page": pages[-1], "y": end_y} if pages else None,
        "toc_entry": {"title": toc_entry[0], "printed_pages": [toc_entry[1], toc_entry[2]]} if toc_entry else None,
        "page_offset": offset,
        "confidence": confidence,
        "flags": flags,
    }
