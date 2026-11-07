"""
Redaction for public sharing.

Everything that leaves the machine as a "share" artifact (Wrapped-style card,
share page) passes through this module first. Input is short natural-language
text written by the LLM stages (summaries, rules, friction notes, project
names), not raw transcripts, so the patterns favour precision on prose: they
target concrete secret formats and identifiers and leave ordinary words like
"token" or "secret sauce" alone.

Public API:
    RedactionPolicy               what to scrub beyond the built-in patterns
    build_policy(...)             stable "Project A/B/..." aliases
    redact_text(text, policy)     one string
    redact_obj(obj, policy)       nested dict/list (keys untouched)
    find_leaks(obj)               final safety scan; [] means safe to write

Placeholders: [secret] [email] [path] [url] [ip] [host] [project] [person].
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Iterable
from urllib.parse import parse_qsl, urlsplit


# ---------------------------------------------------------------------------
# Policy
# ---------------------------------------------------------------------------


@dataclass
class RedactionPolicy:
    project_aliases: dict[str, str] = field(default_factory=dict)
    allowed_projects: set[str] = field(default_factory=set)
    person_names: set[str] = field(default_factory=set)
    keep_public_urls: bool = False


def _alias_label(index: int) -> str:
    """0 -> A, 25 -> Z, 26 -> AA, 27 -> AB ... (spreadsheet-column style)."""
    letters = ""
    n = index + 1
    while n:
        n, rem = divmod(n - 1, 26)
        letters = chr(ord("A") + rem) + letters
    return f"Project {letters}"


def build_policy(
    project_names: Iterable[str],
    allowed: Iterable[str] = (),
    person_names: Iterable[str] = (),
) -> RedactionPolicy:
    allowed_set = {a for a in allowed if a}
    allowed_lower = {a.lower() for a in allowed_set}
    to_alias = sorted(
        {p for p in project_names if p and p.lower() not in allowed_lower}
    )
    aliases = {name: _alias_label(i) for i, name in enumerate(to_alias)}
    # "code-acme-app" is also written "acme-app": alias distinctive tails too
    # (2+ tokens, 6+ characters), unless that tail is itself another project.
    for name, alias in list(aliases.items()):
        tokens = [t for t in re.split(r"[\-_\s.]+", name) if t]
        for k in range(1, len(tokens) - 1):
            tail = "-".join(tokens[k:])
            if len(tail) >= 6 and tail not in aliases and tail.lower() not in allowed_lower:
                aliases[tail] = alias
    return RedactionPolicy(
        project_aliases=aliases,
        allowed_projects=allowed_set,
        person_names={p for p in person_names if p and p.strip()},
    )


# ---------------------------------------------------------------------------
# Built-in patterns
# ---------------------------------------------------------------------------

# Trailing characters that are almost always sentence punctuation, not part of
# a URL or path.
_TRAILING_PUNCT = ".,;:!?)]}>'\""

_PEM_RE = re.compile(
    r"-----BEGIN [A-Z0-9 ]*PRIVATE KEY-----.*?(?:-----END [A-Z0-9 ]*PRIVATE KEY-----|\Z)",
    re.DOTALL,
)

# Concrete vendor token formats. Lookbehind keeps "task-..." etc. out.
_TOKEN_PATTERNS: list[tuple[str, re.Pattern[str]]] = [
    ("Anthropic API key", re.compile(r"(?<![A-Za-z0-9])sk-ant-[A-Za-z0-9_\-]{10,}")),
    ("OpenAI API key", re.compile(r"(?<![A-Za-z0-9])sk-(?:proj-|svcacct-|admin-)?[A-Za-z0-9_\-]{20,}")),
    ("AWS access key id", re.compile(r"(?<![A-Za-z0-9])(?:AKIA|ASIA)[0-9A-Z]{16}(?![A-Za-z0-9])")),
    ("GitHub token", re.compile(r"(?<![A-Za-z0-9])(?:gh[pousr]_[A-Za-z0-9]{20,}|github_pat_[A-Za-z0-9_]{20,})")),
    ("Slack token", re.compile(r"(?<![A-Za-z0-9])xox[abprs]-[A-Za-z0-9\-]{10,}")),
    ("Google API key", re.compile(r"(?<![A-Za-z0-9])AIza[0-9A-Za-z_\-]{30,}")),
    ("HuggingFace token", re.compile(r"(?<![A-Za-z0-9])hf_[A-Za-z0-9]{20,}")),
    ("JWT", re.compile(r"(?<![A-Za-z0-9])eyJ[A-Za-z0-9_\-]+\.eyJ[A-Za-z0-9_\-]+\.[A-Za-z0-9_\-]*")),
    ("Stripe key", re.compile(r"(?<![A-Za-z0-9])(?:sk|rk|pk)_(?:live|test)_[A-Za-z0-9]{16,}")),
    ("GitLab token", re.compile(r"(?<![A-Za-z0-9])gl(?:pat|dt|rt)-[A-Za-z0-9_\-]{20,}")),
    ("npm token", re.compile(r"(?<![A-Za-z0-9])npm_[A-Za-z0-9]{30,}")),
    ("Replicate token", re.compile(r"(?<![A-Za-z0-9])r8_[A-Za-z0-9]{20,}")),
    ("Groq/xAI key", re.compile(r"(?<![A-Za-z0-9])(?:gsk|xai)-?_?[A-Za-z0-9]{30,}")),
]

# Credentials passed as command-line flags or HTTP Basic auth.
_CLI_SECRET_RE = re.compile(
    r"(?i)((?:--(?:password|passwd|token|api-key|apikey|secret|auth-token)(?:\s+|=))|(?<=\s)-p(?=\S))"
    r"([^\s\"']{4,})"
)
_BASIC_AUTH_RE = re.compile(r"(?i)\b(basic\s+)([A-Za-z0-9+/]{12,}={0,2})")

_AWS_SECRET_RE = re.compile(
    r"(?i)(aws[_\-]?secret(?:[_\-]?access)?(?:[_\-]?key)?[\"']?\s*[:=]\s*[\"']?)"
    r"([A-Za-z0-9/+=]{40})(?![A-Za-z0-9/+=])"
)

_BEARER_RE = re.compile(r"(?i)\b(bearer\s+)([A-Za-z0-9._~+/\-]{16,}=*)")

_SECRET_KEYWORDS = (
    r"password|passwd|pwd|passphrase|secret|token|api[_\-]?key|apikey|"
    r"access[_\-]?key|private[_\-]?key|client[_\-]?secret|credentials?"
)
_ASSIGN_RE = re.compile(
    r"(?i)(?<![A-Za-z0-9])"
    r"((?:[A-Za-z0-9]+[_\-])*(?:" + _SECRET_KEYWORDS + r")(?:[_\-](?:key|token|secret|id))?"
    r"[\"']?\s*(?P<sep>[:=])\s*)"
    r"(?P<q>[\"']?)(?P<val>[^\s\"',;\[\]]+)(?P=q)"
)

_URL_RE = re.compile(
    # Any scheme: https, s3, gs, az, hdfs, ssh, git, postgres, mongodb+srv, ...
    r"(?i)\b[a-z][a-z0-9+.\-]{0,15}://[^\s<>\"'`]+"
)

_EMAIL_RE = re.compile(
    r"(?<![A-Za-z0-9._%+\-])[A-Za-z0-9._%+\-]+@[A-Za-z0-9\-]+(?:\.[A-Za-z0-9\-]+)*\.[A-Za-z]{2,}\b"
)

_PATH_RE = re.compile(
    r"(?:"
    r"(?<![\w.~/\-])/(?:Users|home|root)(?:/[^\s\"'<>`|]*)?"
    # Any other absolute path with at least two segments (/mnt/data, /opt/x/y).
    # One segment ("/clear", "/review") is a slash command, not a path.
    r"|(?<![\w.~/\-:])/[\w.\-]+/[^\s\"'<>`|]*"
    r"|(?<![\w])[A-Za-z]:[\\/]Users(?:[\\/][^\s\"'<>`|]*)?"
    r"|(?<![\w/~])~[\\/][^\s\"'<>`|]*"
    r")"
)

_OCTET = r"(?:25[0-5]|2[0-4]\d|1\d\d|[1-9]?\d)"
_IP_RE = re.compile(
    r"(?<![\w.])" + _OCTET + r"(?:\." + _OCTET + r"){3}(?::\d{1,5})?(?![\w]|\.\d)"
)

_HOST_RE = re.compile(
    r"(?i)(?<![\w.\-/@])"
    r"(?:[a-z0-9](?:[a-z0-9\-]*[a-z0-9])?\.)+"
    r"(?:internal|local|corp|lan|intranet|home\.arpa)"
    r"(?::\d{1,5})?"
    r"(?!\.[a-z0-9]|[\w\-])"
)

# TLDs treated as hosts in prose. TLDs that are also common file extensions
# (sh, md, py, rs, pl, so, zip, mov, ...) are left out on purpose, so
# "install.sh" or "README.md" are never mistaken for hosts.
_HOST_TLDS = (
    "com", "net", "org", "edu", "gov", "mil", "int", "info", "biz", "name", "pro",
    "io", "ai", "dev", "app", "cloud", "co", "xyz", "tech", "me", "gg", "run", "site", "page",
    "systems", "tools", "services", "team", "work", "studio", "space", "live", "digital",
    "network", "company", "solutions", "agency", "online", "store", "cc", "tv", "fm", "ly",
    "us", "uk", "ca", "eu", "de", "fr", "it", "es", "nl", "be", "ch", "at", "se", "dk", "fi",
    "ie", "pt", "gr", "cz", "sk", "hu", "ro", "bg", "hr", "si", "lt", "lv", "ee", "lu", "is",
    "il", "jp", "kr", "cn", "tw", "hk", "sg", "in", "au", "nz", "br", "ar", "cl", "mx", "za",
    "tr", "ae", "sa", "ru", "ua", "vn", "th", "id", "ph", "my",
)

# Domain-shaped tokens on common TLDs ("grafana.acme.io",
# "db-1.us-east-1.rds.amazonaws.com"). File names ("config.json", "app.py")
# do not match because their extensions are not in this TLD list.
_DOMAIN_RE = re.compile(
    r"(?i)(?<![\w.\-/@])"
    r"((?:[a-z0-9](?:[a-z0-9\-]*[a-z0-9])?\.)+"
    r"(?:" + "|".join(sorted(_HOST_TLDS, key=len, reverse=True)) + r"))"
    r"(?::\d{1,5})?"
    r"(?![\w\-]|\.[a-z0-9])"
)
# Hosting domains whose subdomains name a private resource.
_CLOUD_HOSTS = (
    "amazonaws.com", "cloudfront.net", "azurewebsites.net", "windows.net",
    "googleusercontent.com", "appspot.com", "run.app", "vercel.app",
    "herokuapp.com", "ngrok.io", "ngrok-free.app", "netlify.app", "fly.dev",
)


def _domain_is_public(host: str) -> bool:
    host = host.lower().rstrip(".")
    if any(host.endswith("." + c) for c in _CLOUD_HOSTS):
        return False
    return any(host == d or host.endswith("." + d) for d in _PUBLIC_DOMAINS)


def _redact_domains(text: str) -> str:
    def _sub(m: re.Match[str]) -> str:
        return m.group(0) if _domain_is_public(m.group(1)) else "[host]"
    return _DOMAIN_RE.sub(_sub, text)

# Query-parameter names that usually carry credentials.
_TOKEN_QUERY_KEYS = re.compile(
    r"(?i)^(?:.*token.*|.*secret.*|.*passw.*|.*key|key|sig|signature|auth.*|"
    r"code|session.*|sid|credential.*|x-amz-.*|access_token|api_key)$"
)

_PUBLIC_DOMAINS = (
    "github.com", "gitlab.com", "bitbucket.org", "pypi.org", "python.org",
    "npmjs.com", "readthedocs.io", "readthedocs.org", "stackoverflow.com",
    "wikipedia.org", "huggingface.co", "arxiv.org", "developer.mozilla.org",
    "anthropic.com", "openai.com", "cursor.com", "cursor.sh", "kubernetes.io",
    "docker.com", "rust-lang.org", "go.dev", "crates.io", "pytorch.org",
)


def _strip_trailing(span: str) -> tuple[str, str]:
    """Split trailing sentence punctuation off a URL/path match."""
    end = len(span)
    while end > 0 and span[end - 1] in _TRAILING_PUNCT:
        # Keep a closing paren if the URL itself opened one (wiki links).
        if span[end - 1] == ")" and span[:end].count("(") >= span[:end].count(")"):
            break
        end -= 1
    return span[:end], span[end:]


def _looks_like_secret_value(val: str, sep: str, quoted: bool) -> bool:
    """Heuristic for `key: value` / `key=value` assignments in prose."""
    if not val or val.startswith("["):
        return False
    if sep == "=" or quoted:
        return len(val) >= 3
    # "token: it expired" style prose -> only redact token-ish values.
    return any(c.isdigit() for c in val) or len(val) >= 16


def _url_is_credentialed(url: str) -> bool:
    try:
        parts = urlsplit(url)
    except ValueError:
        return True
    if "@" in (parts.netloc or ""):
        return True
    for k, _ in parse_qsl(parts.query, keep_blank_values=True):
        if _TOKEN_QUERY_KEYS.match(k):
            return True
    return False


def _url_is_public(url: str) -> bool:
    try:
        parts = urlsplit(url)
    except ValueError:
        return False
    if parts.scheme.lower() != "https":
        return False
    host = (parts.hostname or "").lower()
    if not host:
        return False
    if host.startswith("docs."):
        return True
    return any(host == d or host.endswith("." + d) for d in _PUBLIC_DOMAINS)


# ---------------------------------------------------------------------------
# Redaction
# ---------------------------------------------------------------------------


def _redact_secrets(text: str) -> str:
    text = _PEM_RE.sub("[secret]", text)
    for _, pat in _TOKEN_PATTERNS:
        text = pat.sub("[secret]", text)
    text = _AWS_SECRET_RE.sub(lambda m: m.group(1) + "[secret]", text)

    def _bearer(m: re.Match[str]) -> str:
        if any(c.isdigit() for c in m.group(2)):
            return m.group(1) + "[secret]"
        return m.group(0)

    text = _BEARER_RE.sub(_bearer, text)
    text = _BASIC_AUTH_RE.sub(lambda m: m.group(1) + "[secret]", text)
    return _CLI_SECRET_RE.sub(lambda m: m.group(1) + "[secret]", text)


def _redact_assignments(text: str) -> str:
    def _assign(m: re.Match[str]) -> str:
        if _looks_like_secret_value(m.group("val"), m.group("sep"), bool(m.group("q"))):
            q = m.group("q")
            return f"{m.group(1)}{q}[secret]{q}"
        return m.group(0)

    return _ASSIGN_RE.sub(_assign, text)


def _redact_urls(text: str, keep_public: bool) -> str:
    def _sub(m: re.Match[str]) -> str:
        url, tail = _strip_trailing(m.group(0))
        if keep_public and _url_is_public(url) and not _url_is_credentialed(url):
            return url + tail
        return "[url]" + tail

    return _URL_RE.sub(_sub, text)


def _redact_paths(text: str) -> str:
    def _sub(m: re.Match[str]) -> str:
        _, tail = _strip_trailing(m.group(0))
        return "[path]" + tail

    return _PATH_RE.sub(_sub, text)


def _name_pattern(name: str) -> str:
    """Whole-word pattern that also matches -/_/space separator variants."""
    tokens = [t for t in re.split(r"[\-_\s.]+", name.strip()) if t]
    return r"[\-_\s]?".join(re.escape(t) for t in tokens)


def _compile_names(names: Iterable[str]) -> re.Pattern[str] | None:
    pats = [_name_pattern(n) for n in sorted(set(names), key=len, reverse=True)]
    pats = [p for p in pats if p]
    if not pats:
        return None
    return re.compile(
        r"(?i)(?<![\w\[\-])(?:" + "|".join(pats) + r")(?![\w\-])"
    )


def _redact_projects(text: str, policy: RedactionPolicy) -> str:
    allowed = {a.lower() for a in policy.allowed_projects}
    targets = {
        name: (alias or "[project]")
        for name, alias in policy.project_aliases.items()
        # Names under 3 chars would scrub ordinary words ("ui", "db").
        if name and name.lower() not in allowed and len(name.strip()) >= 3
    }
    if not targets:
        return text
    lookup: list[tuple[re.Pattern[str], str]] = []
    for name in sorted(targets, key=len, reverse=True):
        pat = _compile_names([name])
        if pat is not None:
            lookup.append((pat, targets[name]))
    combined = _compile_names(targets)
    if combined is None:
        return text

    def _sub(m: re.Match[str]) -> str:
        for pat, alias in lookup:
            if pat.fullmatch(m.group(0)):
                return alias
        return "[project]"

    return combined.sub(_sub, text)


def _redact_people(text: str, names: Iterable[str]) -> str:
    pat = _compile_names(n for n in names if len(n.strip()) >= 2)
    return pat.sub("[person]", text) if pat else text


def redact_text(text: str, policy: RedactionPolicy) -> str:
    if not isinstance(text, str) or not text:
        return text
    text = _redact_secrets(text)
    # URLs before generic assignments so "?token=..." drops the whole URL.
    text = _redact_urls(text, policy.keep_public_urls)
    text = _redact_assignments(text)
    text = _EMAIL_RE.sub("[email]", text)
    text = _redact_paths(text)
    text = _IP_RE.sub("[ip]", text)
    text = _HOST_RE.sub("[host]", text)
    text = _redact_domains(text)
    text = _redact_projects(text, policy)
    text = _redact_people(text, policy.person_names)
    return text


def strip_secrets(text: str) -> str:
    """
    Remove credentials only: vendor tokens, private keys, bearer tokens,
    `password=...`-style values, and URLs that carry credentials. Paths,
    hosts and names stay, so a transcript keeps its meaning for review.
    Used on every transcript before it is sent to the LLM provider.
    """
    if not isinstance(text, str) or not text:
        return text
    text = _redact_secrets(text)

    def _url(m: re.Match[str]) -> str:
        url, tail = _strip_trailing(m.group(0))
        return ("[url]" + tail) if _url_is_credentialed(url) else m.group(0)

    text = _URL_RE.sub(_url, text)
    return _redact_assignments(text)


def redact_obj(obj: Any, policy: RedactionPolicy) -> Any:
    if isinstance(obj, str):
        return redact_text(obj, policy)
    if isinstance(obj, dict):
        return {k: redact_obj(v, policy) for k, v in obj.items()}
    if isinstance(obj, list):
        return [redact_obj(v, policy) for v in obj]
    if isinstance(obj, tuple):
        return tuple(redact_obj(v, policy) for v in obj)
    return obj


# ---------------------------------------------------------------------------
# Leak scan
# ---------------------------------------------------------------------------


_PRIVATE_IP_RE = re.compile(
    r"(?<![\w.])(?:10\.\d{1,3}|172\.(?:1[6-9]|2\d|3[01])|192\.168)\.\d{1,3}\.\d{1,3}(?![\w]|\.\d)"
)


def _mask(s: str) -> str:
    s = s.strip()
    return (s[:4] + "...") if len(s) > 4 else "..."


def _scan_string(s: str) -> list[str]:
    found: list[str] = []
    if _PEM_RE.search(s):
        found.append("PEM private key block")
    for kind, pat in _TOKEN_PATTERNS:
        for m in pat.finditer(s):
            found.append(f"{kind} ({_mask(m.group(0))})")
    for m in _AWS_SECRET_RE.finditer(s):
        found.append(f"AWS secret key ({_mask(m.group(2))})")
    for m in _BEARER_RE.finditer(s):
        if any(c.isdigit() for c in m.group(2)):
            found.append(f"bearer token ({_mask(m.group(2))})")
    for m in _ASSIGN_RE.finditer(s):
        if _looks_like_secret_value(m.group("val"), m.group("sep"), bool(m.group("q"))):
            found.append(f"credential assignment ({m.group(1).strip()} {_mask(m.group('val'))})")
    for m in _URL_RE.finditer(s):
        url, _ = _strip_trailing(m.group(0))
        if _url_is_credentialed(url):
            found.append(f"credentialed URL ({_mask(url)})")
    for m in _EMAIL_RE.finditer(s):
        found.append(f"email ({_mask(m.group(0))})")
    for m in _PATH_RE.finditer(s):
        found.append(f"path ({_mask(m.group(0))})")
    for m in _PRIVATE_IP_RE.finditer(s):
        found.append(f"private IP ({m.group(0)})")
    return found


def find_leaks(obj: Any, names: Iterable[str] = ()) -> list[str]:
    """Return descriptions of anything that still looks sensitive.

    Scans every string value and every string dict key. `names` (project or
    person names) must not appear anywhere either. Matches are masked in the
    output so the report itself does not leak.
    """
    leaks: list[str] = []
    name_pat = _compile_names(n for n in names if n and len(n.strip()) >= 3)

    def _scan(s: str) -> list[str]:
        found = _scan_string(s)
        for m in _DOMAIN_RE.finditer(s):
            if not _domain_is_public(m.group(1)):
                found.append(f"hostname ({_mask(m.group(1))})")
        if name_pat is not None:
            for m in name_pat.finditer(s):
                found.append(f"name ({_mask(m.group(0))})")
        return found

    def _walk(o: Any, where: str) -> None:
        if isinstance(o, str):
            leaks.extend(f"{where}: {d}" for d in _scan(o))
        elif isinstance(o, dict):
            for k, v in o.items():
                key_path = f"{where}.{k}" if where else str(k)
                if isinstance(k, str):
                    leaks.extend(f"{key_path} (key): {d}" for d in _scan(k))
                _walk(v, key_path)
        elif isinstance(o, (list, tuple)):
            for i, v in enumerate(o):
                _walk(v, f"{where}[{i}]")

    _walk(obj, "")
    return leaks


__all__ = [
    "strip_secrets",
    "RedactionPolicy",
    "build_policy",
    "redact_text",
    "redact_obj",
    "find_leaks",
]
