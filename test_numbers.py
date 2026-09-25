from table_extraction import clean_label, parse_note_refs, parse_number, parse_period


def check(raw, kind, value, clean=None, decimal=False):
    p = parse_number(raw, decimal)
    assert (p["kind"], p["value"]) == (kind, value), (raw, p)
    if clean is not None:
        assert (not p["repairs"]) == clean, (raw, p)


def test_assignment_rules():
    check("373.992.222", "number", 373992222, clean=True)
    check("(49.167.330)", "number", -49167330, clean=True)
    check("-", "dash", None)
    check("", "empty", None)
    check("0", "zero", 0)


def test_ocr_noise():
    check("8.874,602", "number", 8874602, clean=False)
    check("7,.932.336", "number", 7932336, clean=False)
    check("227317822", "number", 227317822, clean=False)
    check("(561.5989)", "number", -5615989, clean=False)
    assert "irregular_grouping" in parse_number("(561.5989)")["repairs"]


def test_decimals_and_percent():
    check("0,058", "number", 0.058)
    check("1,191", "number", 1.191, decimal=True)
    check("1,191", "number", 1191, decimal=False)
    check("%10,5", "percent", 10.5)
    check("%2-%4", "percent_range", [2.0, 4.0])


def test_normalization():
    assert parse_note_refs("11") == ([11], False)
    assert parse_note_refs("Il") == ([11], True)
    assert clean_label("-İlişkili Taraflardan Ticari Alacaklar") == "İlişkili Taraflardan Ticari Alacaklar"
    assert clean_label("~Diğer Ticari Borçlar") == "Diğer Ticari Borçlar"
    p = parse_period("Cari Dönem Bağımsız Denetimden Geçmiş 1 Ocak- 31 Aralık 2012")
    assert (p["type"], p["start"], p["end"], p["audited"]) == ("duration", "2012-01-01", "2012-12-31", True)
    p = parse_period("Bağımsız Denetimden Geçmiş Geçmiş Dönem 31 Aralık 2011")
    assert (p["type"], p["end"]) == ("instant", "2011-12-31")


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print("ok ", name)
