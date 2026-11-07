"""
Share export: the public, redacted version of the profile.

Writes atlas/share/:
    card.svg     the Wrapped card (1080x1350)
    index.html   card + rules + vibes + agents compared (no projects section)
    RULES.md     paste-ready rules block

Every string passes `redact.redact_obj` (project names -> "Project A",
secrets, emails, home paths, IPs, internal hosts removed). A final
`find_leaks` scan blocks the write if anything still looks private.
"""

from __future__ import annotations

import copy
from pathlib import Path

from ..paths import get_atlas_dir
from ..redact import build_policy, find_leaks, redact_obj
from .card import render_card_svg
from .dashboard import render_html, rules_block

# Keys of profile["stats"] that name projects; never exported.
_PRIVATE_STAT_KEYS = ("top_projects",)


class ShareLeakError(RuntimeError):
    """Raised when the redacted export still contains something private."""


def redacted_profile(profile: dict, project_names: list[str], person_names: list[str] = ()) -> dict:
    return _redact(profile, build_policy(project_names, person_names=person_names))


def _redact(profile: dict, policy) -> dict:
    p = copy.deepcopy(profile)
    stats = p.get("stats", {})
    for k in _PRIVATE_STAT_KEYS:
        stats.pop(k, None)
    p.pop("rule_candidates", None)   # carry session ids
    p.pop("features", None)
    p.pop("_input_fingerprint", None)
    return redact_obj(p, policy)


def write_share(
    profile: dict,
    project_names: list[str],
    *,
    person_names: list[str] = (),
    out_dir: Path | None = None,
) -> dict[str, Path]:
    policy = build_policy(project_names, person_names=person_names)
    safe = _redact(profile, policy)
    # The final scan also checks every project name and name variant.
    leaks = find_leaks(safe, names=list(policy.project_aliases) + list(person_names))
    if leaks:
        raise ShareLeakError("share export blocked, still private: " + "; ".join(leaks[:5]))

    out_dir = out_dir or (get_atlas_dir() / "share")
    out_dir.mkdir(parents=True, exist_ok=True)
    paths = {
        "card": out_dir / "card.svg",
        "page": out_dir / "index.html",
        "rules": out_dir / "RULES.md",
    }
    paths["card"].write_text(render_card_svg(safe), encoding="utf-8")
    paths["page"].write_text(render_html([], None, None, [], safe, public=True), encoding="utf-8")
    paths["rules"].write_text(rules_block(safe) + "\n", encoding="utf-8")
    return paths


__all__ = ["write_share", "redacted_profile", "ShareLeakError"]
