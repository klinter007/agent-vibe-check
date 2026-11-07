"""
The shareable "Wrapped" card, as a self-contained SVG (1080 x 1350, the
portrait size Instagram and LinkedIn show without cropping).

Pure function of the profile dict, so it is testable and the same card is
used in the private report and in the redacted share export. No external
fonts or images: the card renders the same offline and converts to PNG in
the browser via <canvas>.
"""

from __future__ import annotations

from datetime import datetime
from html import escape

from ..pipeline.taxonomy import CANONICAL_DIMENSIONS

W, H = 1080, 1350
FONT = "-apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Helvetica, Arial, sans-serif"
REPO_URL = "github.com/klinter007/agent-vibe-check"

SIGNAL_LABELS = {
    "specification": "Specification",
    "context": "Context",
    "verification": "Verification",
    "steering": "Steering",
    "delegation": "Delegation",
    "learning-loop": "Learning loop",
}


def wrap(text: str, max_chars: int, max_lines: int) -> list[str]:
    """Greedy word wrap; the last line gets an ellipsis if text remains."""
    words = (text or "").split()
    lines: list[str] = []
    cur = ""
    for w in words:
        nxt = f"{cur} {w}".strip()
        if len(nxt) <= max_chars:
            cur = nxt
            continue
        if cur:
            lines.append(cur)
        cur = w
        if len(lines) == max_lines:
            break
    if cur and len(lines) < max_lines:
        lines.append(cur)
    consumed = sum(len(x.split()) for x in lines)
    if consumed < len(words) and lines:
        last = lines[-1]
        lines[-1] = (last[: max_chars - 1].rstrip() + "…") if len(last) >= max_chars else last + "…"
    return lines


_CTRL_RE = __import__("re").compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")


def _clean(s) -> str:
    """Text safe for SVG/XML: no control characters."""
    return _CTRL_RE.sub("", str(s if s is not None else ""))


def _text(x, y, s, size, *, weight=400, fill="#fff", anchor="start", opacity=1.0, spacing=0) -> str:
    return (
        f'<text x="{x}" y="{y}" font-family="{FONT}" font-size="{size}" font-weight="{weight}" '
        f'fill="{fill}" text-anchor="{anchor}" fill-opacity="{opacity}" letter-spacing="{spacing}">'
        f"{escape(_clean(s))}</text>"
    )


def _pct(v) -> str:
    return "n/a" if v is None else f"{round(v * 100)}%"


def level_word(score) -> str:
    if score is None:
        return "not seen"
    if score >= 90:
        return "expert"
    if score >= 65:
        return "strong"
    if score >= 35:
        return "working"
    return "beginner"


FRICTION_LABELS = {
    "correction": "corrections",
    "loop": "loops",
    "env-failure": "tool & env failures",
    "rephrase": "rephrasing",
    "scope-drift": "scope drift",
    "revert": "reverts",
    "frustration": "frustration",
    "abandon": "giving up",
}

# Bars are drawn on the level scale, not as "x / 100": a score is an average
# of levels (beginner 20, working 50, strong 80, expert 100), so 100 only
# happens when every observation is expert.
LEVEL_TICKS = (("beginner", 20), ("working", 50), ("strong", 80), ("expert", 100))


def _text_width(s: str, size: float, weight: int = 400) -> float:
    """Rough width of system-ui text; good enough to shrink-to-fit one line."""
    factor = 0.56 if weight >= 600 else 0.52
    return len(s) * size * factor


def _fit(s: str, size: float, max_w: float, weight: int = 400, min_size: float = 18) -> float:
    while size > min_size and _text_width(s, size, weight) > max_w:
        size -= 1
    return size


def _one_line(s: str, size: float, max_w: float, weight: int = 400, min_size: float = 18) -> tuple[str, float]:
    """Shrink to fit; if still too long at min size, cut with an ellipsis."""
    size = _fit(s, size, max_w, weight, min_size)
    while _text_width(s, size, weight) > max_w and len(s) > 4:
        s = s[:-2].rstrip() + "\u2026"
    return s, size


def _bar_frac(score) -> float:
    return max(0.0, min(1.0, (score - 20) / 80))


MONO = "ui-monospace, SFMono-Regular, Menlo, Consolas, monospace"


def _mono(x, y, s, size, *, fill="#fff", anchor="start", opacity=1.0, spacing=3) -> str:
    return (f'<text x="{x}" y="{y}" font-family="{MONO}" font-size="{size}" fill="{fill}" '
            f'text-anchor="{anchor}" fill-opacity="{opacity}" letter-spacing="{spacing}">{escape(_clean(s))}</text>')


def _months_between(first: str, last: str) -> int | None:
    try:
        y0, m0 = int(first[:4]), int(first[5:7])
        y1, m1 = int(last[:4]), int(last[5:7])
    except (ValueError, IndexError):
        return None
    return max(1, (y1 - y0) * 12 + (m1 - m0) + 1)


def _duration(first: str, last: str) -> str:
    """'15 mo' for long histories, '3 wk' or '9 days' for short ones."""
    try:
        d = (datetime.fromisoformat(last[:10]) - datetime.fromisoformat(first[:10])).days + 1
    except ValueError:
        return "-"
    if d >= 60:
        return f"{_months_between(first, last)} mo"
    if d >= 14:
        return f"{round(d / 7)} wk"
    return f"{d} day" + ("s" if d != 1 else "")


def _type_code(key: str) -> str:
    from ..pipeline.profile import ARCHETYPES
    for i, a in enumerate(ARCHETYPES, start=1):
        if a.key == key:
            return f"A-{i:02d}"
    return "A-00"  # Warming Up: not enough evidence yet


def render_card_svg(profile: dict, handle: str | None = None) -> str:
    """
    A one-page "spec sheet": type, friction by term, three totals, three rules.
    Same design for every type. Monochrome, system fonts, no external assets.
    """
    stats = profile.get("stats", {})
    arch = profile.get("archetype", {})
    narr = profile.get("narrative", {}) or {}
    bg, fg = "#1b4b8f", "#ffffff"   # blueprint blue, white ink
    m = 48                        # outer margin for text
    inner_w = W - 2 * m

    parts: list[str] = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}" viewBox="0 0 {W} {H}">',
        "<defs><pattern id=\"hatch\" width=\"14\" height=\"14\" patternUnits=\"userSpaceOnUse\" "
        "patternTransform=\"rotate(45)\"><line x1=\"0\" y1=\"0\" x2=\"0\" y2=\"14\" "
        "stroke=\"#fff\" stroke-width=\"2.5\"/></pattern>"
        "<pattern id=\"grid\" width=\"30\" height=\"30\" patternUnits=\"userSpaceOnUse\">"
        "<path d=\"M30 0H0V30\" fill=\"none\" stroke=\"#fff\" stroke-opacity=\"0.07\" stroke-width=\"1\"/>"
        "</pattern></defs>",
        f'<rect width="{W}" height="{H}" fill="{bg}"/>',
        f'<rect width="{W}" height="{H}" fill="url(#grid)"/>',
        f'<rect x="3" y="3" width="{W - 6}" height="{H - 6}" fill="none" stroke="{fg}" stroke-width="3"/>',
    ]

    def hline(y):
        parts.append(f'<line x1="3" y1="{y}" x2="{W - 3}" y2="{y}" stroke="{fg}" stroke-width="2"/>')

    def vline(x, y1, y2):
        parts.append(f'<line x1="{x}" y1="{y1}" x2="{x}" y2="{y2}" stroke="{fg}" stroke-width="2"/>')

    # -- Header band -------------------------------------------------------------
    first, last = stats.get("first", ""), stats.get("last", "")
    parts.append(_mono(m, 46, "AI CODING VIBE CHECK", 21))
    parts.append(_mono(W - m, 46, f"{first[:7].replace('-', '.')} \u2192 {last[:7].replace('-', '.')}", 21, anchor="end"))
    hline(72)

    # -- Type ----------------------------------------------------------------------
    parts.append(_mono(m, 134, f"ARCHETYPE \u2014 TYPE {_type_code(arch.get('key', ''))}", 20, opacity=0.8))
    name = arch.get("name", "")
    parts.append(_text(m, 236, name, _fit(name, 104, inner_w, 800), weight=800, fill=fg))
    tagline, tsize = _one_line(arch.get("tagline", ""), 34, inner_w, weight=600)
    parts.append(_text(m, 296, tagline, tsize, weight=600, fill=fg))

    # -- Friction: big stat (left) + bars by term (right) -----------------------------
    terms = [t for t in (stats.get("terms") or []) if t.get("friction_share") is not None]
    parts.append(_mono(m, 380, "FRICTION PER SESSION", 20, opacity=0.85))
    if len(terms) >= 2:
        f0, f1 = terms[0]["friction_share"], terms[-1]["friction_share"]
        pct_line = f"{round(f0 * 100)}% \u2192 {round(f1 * 100)}%"
        if f1 > 0 and f0 / f1 >= 1.5:
            ratio = f0 / f1
            big = f"{ratio:.0f}\u00d7" if ratio >= 2 else f"{ratio:.1f}\u00d7"
            caption = "less time lost"
        elif f1 > f0 * 1.2:
            big, caption = f"{round(f1 * 100)}%", "lost now, up from before"
        else:
            big, caption = f"{round(f1 * 100)}%", "of a session lost"
        parts.append(_text(m - 4, 540, big, 150, weight=800, fill=fg))
        parts.append(_text(m, 596, caption, 30, weight=600, fill=fg))
        parts.append(_text(m, 636, pct_line, 30, weight=600, fill=fg, opacity=0.85))

        cx0, cx1, top, base = 430, W - m, 420, 760
        n = len(terms)
        gap = 26
        bw = (cx1 - cx0 - gap * (n - 1)) / n
        peak = max(t["friction_share"] for t in terms) or 1
        for i, t in enumerate(terms):
            x = cx0 + i * (bw + gap)
            h = max(6, (base - top) * t["friction_share"] / peak)
            y = base - h
            if i == n - 1:
                parts.append(f'<rect x="{x:.0f}" y="{y:.0f}" width="{bw:.0f}" height="{h:.0f}" fill="{fg}"/>')
            else:
                parts.append(f'<rect x="{x:.0f}" y="{y:.0f}" width="{bw:.0f}" height="{h:.0f}" fill="url(#hatch)" '
                             f'stroke="{fg}" stroke-width="2.5"/>')
            parts.append(_mono(x + bw / 2, y - 14, f"{round(t['friction_share'] * 100)}%", 26, anchor="middle", spacing=0))
            label = t.get("label", "").upper()
            parts.append(_mono(x + bw / 2, base + 38, label, 19, anchor="middle", spacing=1))
        parts.append(f'<line x1="{cx0}" y1="{base}" x2="{cx1}" y2="{base}" stroke="{fg}" stroke-width="2"/>')
    else:
        overall = (stats.get("overall") or {}).get("friction_share")
        if overall is not None:
            parts.append(_text(m - 4, 540, f"{round(overall * 100)}%", 150, weight=800, fill=fg))
            parts.append(_text(m, 596, "of a typical session", 30, weight=600, fill=fg))

    # -- Three totals -------------------------------------------------------------------
    hline(840)
    cells = [
        (f"{stats.get('sessions', 0):,}", "SESSIONS"),
        (_pct(stats.get("ship_rate")), "GOALS REACHED"),
        (_duration(first, last), "OF BUILDING WITH AI"),
    ]
    cw = W / 3
    for i, (big, small) in enumerate(cells):
        x = i * cw + m
        if i:
            vline(round(i * cw), 840, 980)
        parts.append(_text(x, 922, big, 70, weight=800, fill=fg))
        parts.append(_mono(x, 958, small, 18))
    hline(980)

    # -- Rules ----------------------------------------------------------------------------
    rules = [r for r in (narr.get("card_rules") or []) if isinstance(r, str) and r.strip()]
    if not rules:
        rules = [r.get("rule", "") for r in (narr.get("top_rules") or []) if isinstance(r, dict)]
    parts.append(_mono(m, 1036, "MY RULES FOR THE AGENT", 20, opacity=0.85))
    for i, r in enumerate(rules[:3]):
        y = 1100 + i * 58
        parts.append(_mono(m, y, f"{i + 1:02d}", 20, opacity=0.8, spacing=1))
        line, size = _one_line(r.strip(), 36, inner_w - 56, weight=600, min_size=24)
        parts.append(_text(m + 56, y, line, size, weight=600, fill=fg))

    # -- Footer ------------------------------------------------------------------------------
    hline(1266)
    if handle:
        parts.append(_mono(m, 1316, f"BY @{handle.lstrip('@').upper()}", 18))
    parts.append(_mono(W - m, 1316, f"What's yours? {REPO_URL}", 18, anchor="end", spacing=0))
    parts.append("</svg>")
    return "\n".join(parts)


__all__ = ["render_card_svg", "wrap", "level_word", "SIGNAL_LABELS", "FRICTION_LABELS"]
