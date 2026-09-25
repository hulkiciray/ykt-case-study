from __future__ import annotations

from linking import period_match
from table_extraction import SEVERE_REPAIRS, SUM_TOLERANCE, find_hierarchy

# final link confidence, never the model score alone
LINK_WEIGHTS = {"model": 0.35, "amount": 0.25, "period": 0.15, "agreement": 0.15, "extraction": 0.10}


def check(group, name, passed, detail, items=None):
    return {"group": group, "check": name, "passed": passed, "detail": detail, "items": items or []}


def structural_checks(cfg, summary, note, links):
    out = []
    pages = [t["page"] for t in summary["tables"]]
    unknown = [t["table_id"] for t in summary["tables"] if not t["statement_type"]]
    out.append(check("structural", "tables_on_configured_pages",
                     pages == list(cfg["summary_pages"]) and not unknown,
                     f"pages {pages}; statement type unknown for {unknown or 'none'}"))

    mixed = []
    for t in summary["tables"]:
        ends = [c["period"]["end"] for c in t["columns"]]
        roles = [c["period"]["role"] for c in t["columns"]]
        if None in ends or len(set(ends)) != len(ends) or roles.count("current") != 1:
            mixed.append(t["table_id"])
    out.append(check("structural", "period_columns_distinct", not mixed,
                     "one current and distinct period ends per table", mixed))

    orphan = [r["row_id"] for t in summary["tables"] for r in t["rows"]
              if r["note_refs"] and all(v["kind"] == "empty" for v in r["values"])]
    repaired = [r["row_id"] for t in summary["tables"] for r in t["rows"] if "note_ref_repaired" in r["flags"]]
    out.append(check("structural", "note_refs_on_value_rows", not orphan,
                     f"note references on rows without values: {orphan or 'none'}; "
                     f"OCR-repaired references: {repaired or 'none'}", orphan + repaired))

    out.append(check("structural", "note_pages_found", bool(note["pages"]),
                     f"note {note['note_number']} on PDF pages {note['pages']}, confidence {note['confidence']}, "
                     f"flags {note['flags'] or 'none'}"))

    bad = [f"{ln['item_id']} -> {ln['note_row']}" for ln in links if ln["period_match"] == 0]
    out.append(check("structural", "links_between_compatible_periods", not bad if links else None,
                     "every link joins values of the same date, or a balance with the next period's opening",
                     bad))
    return out


def format_checks(summary, note_rows):
    values = [(r["row_id"], v) for t in summary["tables"] for r in t["rows"] for v in r["values"]] + \
             [(r["row_id"], v) for r in note_rows["rows"] for v in r["values"]]
    severe = [f"{rid}:{v.get('col_id', v.get('col'))}" for rid, v in values
              if v["kind"] == "unparseable" or SEVERE_REPAIRS & set(v["repairs"])]
    repaired = [f"{rid}:{v.get('col_id', v.get('col'))} {v['raw']!r}" for rid, v in values
                if v["repairs"] and v["kind"] != "unparseable"]
    out = [check("format", "numbers_parsed", not severe,
                 f"{len(values)} values, {len(repaired)} parsed after an OCR repair, {len(severe)} not parseable",
                 severe + repaired)]

    currencies = {t["currency"] for t in summary["tables"]}
    out.append(check("format", "currency_present_and_consistent", None not in currencies and len(currencies) == 1,
                     f"currencies {sorted(c or 'missing' for c in currencies)}; "
                     f"scales {sorted({t['unit_scale'] for t in summary['tables']})}"))

    no_period = [rid for rid, v in values if not (v.get("period_end") or (v.get("period") or {}).get("year"))]
    out.append(check("format", "periods_present", not no_period, "every value has a period", no_period))
    return out


def financial_checks(summary, note_sums, links):
    out = []
    sums = [c for t in summary["tables"] for c in t["checks"]]
    failed = [f"{c['parent_row']} [{c['column']}] diff {c['difference']}" for c in sums if not c["passed"]]
    out.append(check("financial", "summary_table_sums", not failed if sums else None,
                     f"{len(sums) - len(failed)}/{len(sums)} sum checks pass", failed))

    out.append(balance_sheet_identity(summary))

    failed = [f"{c['block']} {c['parent_row']} [col {c['column']}] diff {c['difference']}"
              for c in note_sums if not c["passed"]]
    out.append(check("financial", "note_table_sums", not failed if note_sums else None,
                     f"{len(note_sums) - len(failed)}/{len(note_sums)} sum checks pass in the note tables", failed))

    bad = [f"{ln['item_id']} -> {ln['note_row']}: {ln['item_value']} vs {ln['value']}"
           for ln in links if not ln["amount_ok"]]
    out.append(check("financial", "linked_amounts_agree", not bad if links else None,
                     "equals_* links have the same amount; breakdown rows are direct parts, at the item's "
                     "date, of a note total that equals the item", bad))
    return out


def balance_sheet_identity(summary):
    parts = [t for t in summary["tables"] if t["statement_type"] == "balance_sheet"]
    if len(parts) != 2:
        return check("financial", "assets_equal_liabilities_plus_equity", None, "no two-part balance sheet found")
    totals = []
    for t in parts:
        rows = {r["row_id"]: r for r in t["rows"]}
        size = lambda rid: 1 + sum(size(c) for c in rows[rid]["children"])
        totals.append(max((r for r in t["rows"] if r["row_type"] in ("total", "parent")), key=lambda r: size(r["row_id"])))
    detail, ok = [], True
    for va, vb in zip(totals[0]["values"], totals[1]["values"]):
        same = va["value"] is not None and vb["value"] is not None and abs(va["value"] - vb["value"]) <= SUM_TOLERANCE
        ok = ok and same and va["period_end"] == vb["period_end"]
        detail.append(f"{va['period_end']}: {va['value']} vs {vb['value']}")
    return check("financial", "assets_equal_liabilities_plus_equity", ok,
                 f"{totals[0]['label']} vs {totals[1]['label']}: " + "; ".join(detail))


# same solver as stage 1: opening + movements = closing, parts = total
def note_block_sums(note_rows):
    out = []
    for b in note_rows["blocks"]:
        rows = [r for r in note_rows["rows"] if r["block_id"] == b["block_id"]]
        cols = [str(i) for i in range(max((len(r["values"]) for r in rows), default=0))]
        solver_rows = []
        for r in rows:
            cells = {}
            for c in cols:
                v = r["values"][int(c)] if int(c) < len(r["values"]) else {"kind": "empty", "value": None}
                num = v["value"] if v["kind"] in ("number", "zero") else 0 if v["kind"] == "dash" else None
                cells[c] = {"num": num, "kind": v["kind"], "arithmetic_ok": None}
            solver_rows.append({"label": r["label"], "decimal": False, "indent": 0, "cells": cells})
        for c in find_hierarchy(solver_rows, cols):
            out.append({"block": b["block_id"], "parent_row": rows[c["parent_row"]]["row_id"],
                        "component_rows": [rows[k]["row_id"] for k in c["component_rows"]],
                        "relation": c["relation"], "column": c["column"], "difference": c["difference"],
                        "passed": c["passed"]})
    return out


def final_links(links_out, summary, note_rows, note_sums, threshold):
    items = {i["item_id"]: i for i in links_out["items"]}
    notes = {r["row_id"]: r for r in note_rows["rows"]}
    cell_conf = {f"{r['row_id']}:{v['col_id']}": v["confidence"] for t in summary["tables"] for r in t["rows"]
                 for v in r["values"]}
    a = {(ln["item_id"], ln["note_row"]): ln for ln in links_out["method_a"]["links"]}
    b = {(ln["item_id"], ln["note_row"]): ln for ln in links_out["method_b"]["links"]}
    parts_of = {}
    for c in note_sums:
        if c["passed"]:
            parts_of.setdefault((c["parent_row"], int(c["column"])), set()).update(c["component_rows"])

    def same_amount(x, y):
        return isinstance(x, (int, float)) and abs(abs(x) - abs(y)) <= SUM_TOLERANCE

    totals = {}
    for key in a.keys() | b.keys():
        ln = b.get(key) or a.get(key)
        cell = notes[key[1]]["values"][ln["col"]]
        if ln["relation"] != "breakdown_component" and same_amount(cell["value"], items[key[0]]["value"]):
            totals.setdefault(key[0], set()).add((key[1], ln["col"]))

    out = []
    for key in sorted(a.keys() | b.keys()):
        la, lb = a.get(key), b.get(key)
        main = lb or la
        item = items[key[0]]
        if la and lb:
            agreement = 1.0 if la["relation"] == lb["relation"] else 0.6
        else:
            agreement = 0.3
        col = main["col"]
        for v in notes[key[1]]["values"]:  # same amount in several columns: report the total column
            if "toplam" in v["header"].lower() and v["value"] == notes[key[1]]["values"][col]["value"]:
                col = v["col"]
        note_value = notes[key[1]]["values"][col]
        if main["relation"] == "breakdown_component":
            # a breakdown row must be a direct part of a linked note total, at the item's date
            same_date = item["period"]["type"] != "instant" or note_value["period"]["date"] == item["period"]["end"]
            amount_ok = same_date and any(key[1] in parts_of.get(t, ()) for t in totals.get(key[0], ()))
        else:
            amount_ok = same_amount(note_value["value"], item["value"])
        period = period_match(item["period"], note_value["period"])
        model = lb["score"] if lb else la["score"]
        extraction = (cell_conf.get(key[0], 0.5) + note_value.get("confidence", 0.5)) / 2
        conf = (LINK_WEIGHTS["model"] * model + LINK_WEIGHTS["amount"] * amount_ok + LINK_WEIGHTS["period"] * period
                + LINK_WEIGHTS["agreement"] * agreement + LINK_WEIGHTS["extraction"] * extraction)
        status = "rejected" if not amount_ok or period == 0 else "low_confidence" if conf < threshold else "accepted"
        out.append({
            "item_id": key[0], "summary_row": item["summary_row"], "summary_label": item["label"],
            "period_end": item["period"]["end"], "item_value": item["value"],
            "note_row": key[1], "note_label": notes[key[1]]["label"], "note_column": note_value["header"],
            "col": col, "value": note_value["value"], "relation": main["relation"],
            "found_by": "A+B" if la and lb else "B" if lb else "A",
            "relation_a": la["relation"] if la else None, "relation_b": lb["relation"] if lb else None,
            "llm_reason": lb.get("reason") if lb else None,
            "amount_ok": amount_ok, "period_match": period,
            "confidence": round(conf, 4), "status": status,
        })
    return out


def validate(cfg, summary, note, note_rows, links_out):
    threshold = cfg["low_confidence_threshold"]
    note_sums = note_block_sums(note_rows)
    links = final_links(links_out, summary, note_rows, note_sums, threshold)
    checks = structural_checks(cfg, summary, note, links) + format_checks(summary, note_rows) + \
        financial_checks(summary, note_sums, links)
    low_values = [f"{r['row_id']}:{v['col_id']}" for t in summary["tables"] for r in t["rows"]
                  for v in r["values"] if v["low_confidence"]]
    return {
        "document": {**summary["document"], "note_number": note["note_number"], "note_title": note["title"],
                     "note_pages": note["pages"]},
        "summary_tables": summary["tables"],
        "note": {"pages": note, "blocks": note_rows["blocks"], "rows": note_rows["rows"]},
        "links": [ln for ln in links if ln["status"] != "rejected"],
        "rejected_links": [ln for ln in links if ln["status"] == "rejected"],
        "method_comparison": {"method_a": {k: links_out["method_a"][k] for k in ("model", "threshold", "threshold_source")},
                              "method_b": {k: links_out["method_b"][k] for k in ("model", "calls", "fallback_items")},
                              **links_out["comparison"], "evaluation": links_out["evaluation"]},
        "validation": {
            "checks": checks,
            "passed": sum(c["passed"] is True for c in checks),
            "failed": sum(c["passed"] is False for c in checks),
            "not_applicable": sum(c["passed"] is None for c in checks),
        },
        "low_confidence": {
            "summary_values": low_values,
            "summary_rows": summary["summary"]["low_confidence_rows"],
            "note_rows": [r["row_id"] for r in note_rows["rows"] if r["confidence"] < threshold],
            "links": [f"{ln['item_id']} -> {ln['note_row']}" for ln in links if ln["status"] != "accepted"],
        },
    }
