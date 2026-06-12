from translation_benchmark.context import ContextBuilder


def test_keeps_most_recent_pairs():
    builder = ContextBuilder(max_pairs=3)
    for i in range(10):
        builder.push(f"src {i}", f"tgt {i}")
    pairs = builder.pairs()
    assert [pair.source for pair in pairs] == ["src 7", "src 8", "src 9"]


def test_char_budget_trims_oldest_first():
    builder = ContextBuilder(max_pairs=10, max_chars=30)
    builder.push("a" * 20, "b" * 20)  # 40 chars, over budget alone
    builder.push("short", "짧다")
    pairs = builder.pairs()
    # The newest pair always survives; the oversized old one is dropped.
    assert len(pairs) == 1
    assert pairs[0].source == "short"


def test_zero_pairs_disables_context():
    builder = ContextBuilder(max_pairs=0)
    builder.push("hello", "안녕")
    assert builder.pairs() == []
    assert builder.render("English", "Korean") == ""


def test_render_format():
    builder = ContextBuilder(max_pairs=2)
    builder.push("You're late, detective.", "늦었군, 형사.")
    block = builder.render("English", "Korean")
    assert block.splitlines() == [
        "Previous subtitle lines (English -> Korean):",
        "English: You're late, detective.",
        "Korean: 늦었군, 형사.",
    ]
