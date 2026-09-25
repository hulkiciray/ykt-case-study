import json
import sys
from pathlib import Path

import pymupdf as fitz
import yaml

from footnote_page_finder import find_note_pages
from linking import link_note
from note_rows import extract_note_rows
from table_extraction import extract_summary_tables
from validation import validate


def main(config_path="config.yaml"):
    root = Path(config_path).resolve().parent
    cfg = yaml.safe_load(open(config_path, encoding="utf-8"))
    out_dir = root / cfg["output_dir"]
    out_dir.mkdir(exist_ok=True)

    # 1. summary tables
    result = extract_summary_tables(cfg, root)
    (out_dir / "summary_tables.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    for t in result["tables"]:
        passed = sum(c["passed"] for c in t["checks"])
        print(f"page {t['page']}: {t['statement_type']}, {len(t['rows'])} rows, "
              f"checks {passed}/{len(t['checks'])}, confidence {t['confidence']:.2f}")
    s = result["summary"]
    print(f"-> outputs/summary_tables.json ({s['values']} values, checks {s['checks_passed']}/{s['checks_total']}, "
          f"low confidence: {len(s['low_confidence_rows'])} rows, {len(s['low_confidence_values'])} values)")

    # 2. note pages
    with fitz.open(root / cfg["pdf_path"]) as doc:
        n_pages = doc.page_count
    note = find_note_pages(cfg, root, n_pages)
    (out_dir / "note_pages.json").write_text(json.dumps(note, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"note {note['note_number']} '{note['title']}': PDF pages {note['pages']} "
          f"(printed {note['printed_pages']}), confidence {note['confidence']:.2f} {note['flags'] or ''}")
    print("-> outputs/note_pages.json")

    # 3. note rows
    note_rows = extract_note_rows(cfg, root, note)
    (out_dir / "note_rows.json").write_text(json.dumps(note_rows, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"note {note['note_number']}: {len(note_rows['rows'])} rows in {len(note_rows['blocks'])} blocks")
    print("-> outputs/note_rows.json")

    # 4-5. candidates and linking (A: sentence-transformer, B: LLM judge)
    links = link_note(cfg, root, result, note_rows)
    (out_dir / "links.json").write_text(json.dumps(links, ensure_ascii=False, indent=2), encoding="utf-8")
    a, b = links["method_a"], links["method_b"]
    print(f"{len(links['items'])} summary items, {len(links['candidates'])} candidates")
    print(f"method A: {len(a['links'])} links (threshold {a['threshold']} from {a['threshold_source']})")
    print(f"method B: {len(b['links'])} links (api calls {b['calls']['api']}, cached {b['calls']['cached']}, "
          f"fallbacks {len(b['fallback_items'])})")
    if links["evaluation"]:
        for m in ("method_a", "method_b"):
            e = links["evaluation"][m]
            print(f"  {m} vs control: precision {e['precision']}, recall {e['recall']}, f1 {e['f1']}, "
                  f"relation accuracy {e['relation_accuracy']}")
    print("-> outputs/links.json")

    # 6. validation and final output
    final = validate(cfg, result, note, note_rows, links)
    (out_dir / "final_output.json").write_text(json.dumps(final, ensure_ascii=False, indent=2), encoding="utf-8")
    v = final["validation"]
    print(f"validation: {v['passed']} passed, {v['failed']} failed, {v['not_applicable']} not applicable")
    for c in v["checks"]:
        mark = {True: "ok  ", False: "FAIL", None: "n/a "}[c["passed"]]
        print(f"  {mark} {c['group']:10s} {c['check']}: {c['detail']}")
    statuses = [ln["status"] for ln in final["links"] + final["rejected_links"]]
    print(f"links: {statuses.count('accepted')} accepted, {statuses.count('low_confidence')} low confidence, "
          f"{statuses.count('rejected')} rejected")
    print("-> outputs/final_output.json")


if __name__ == "__main__":
    main(*sys.argv[1:])
