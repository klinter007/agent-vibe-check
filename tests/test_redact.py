"""Hermetic tests for agent_vibe_check.redact."""

from __future__ import annotations

import pytest

from agent_vibe_check.redact import (
    RedactionPolicy,
    build_policy,
    find_leaks,
    redact_obj,
    redact_text,
)

EMPTY = RedactionPolicy()

# Fake credentials assembled at runtime so the source file does not trip
# secret scanners.
ANT = "sk-ant-" + "api03-" + "A1b2C3d4E5f6G7h8I9j0kLmNoPqRsT"
OAI = "sk-" + "A1b2C3d4E5f6G7h8I9j0kLmNoPqRsTuV"
OAI_PROJ = "sk-proj-" + "A1b2C3d4E5f6G7h8I9j0kLmNoPqRsT"
AWS_ID = "AKIA" + "IOSFODNN7EXAMPLE"
AWS_SECRET = "Ab1/" * 10  # 40 chars, same shape as an AWS secret key, no entropy
GH = ["ghp_", "gho_", "ghs_"]
GH_BODY = "A1b2C3d4E5f6G7h8I9j0kLmNoPqRsTuVwX"
GH_PAT = "github_pat_" + "11ABCDEFG0123456789_abcdefghijKLMNOP"
SLACK = "xoxb-" + "1234567890-0987654321-AbCdEfGhIjKl"
GOOGLE = "AIza" + "SyA1b2C3d4E5f6G7h8I9j0kLmNoPqRsTu"
HF = "hf_" + "AbCdEfGhIjKlMnOpQrStUvWxYz123456"
JWT = "eyJhbGciOiJIUzI1NiJ9" + ".eyJzdWIiOiIxMjM0NTY3ODkwIn0" + ".dozjgNryP4J3jVmNHl0w5N_XgL0n3I9PlFUP0THsR8U"
PEM = (
    "-----BEGIN RSA " + "PRIVATE KEY-----\n"
    "MIIEowIBAAKCAQEAx1y2z3\nabcdEFGH\n"
    "-----END RSA " + "PRIVATE KEY-----"
)


# --- secrets ---------------------------------------------------------------


@pytest.mark.parametrize(
    "secret",
    [ANT, OAI, OAI_PROJ, AWS_ID, GH_PAT, SLACK, GOOGLE, HF, JWT]
    + [p + GH_BODY for p in GH],
)
def test_vendor_tokens_redacted(secret: str) -> None:
    out = redact_text(f"The agent pasted {secret} into the config.", EMPTY)
    assert secret not in out
    assert "[secret]" in out
    assert out.endswith("into the config.")


def test_pem_block_redacted() -> None:
    out = redact_text(f"Key was:\n{PEM}\nthen rotated.", EMPTY)
    assert "PRIVATE KEY" not in out
    assert "[secret]" in out and "then rotated." in out


def test_aws_secret_after_label() -> None:
    out = redact_text(f"aws_secret_access_key = {AWS_SECRET}", EMPTY)
    assert AWS_SECRET not in out
    assert out == "aws_secret_access_key = [secret]"


@pytest.mark.parametrize(
    "text,value",
    [
        ("set password=hunter2 in .env", "hunter2"),
        ('token: "abc123xyz"', "abc123xyz"),
        ("OPENAI_API" + "_KEY=" + "abcd1234efgh", "abcd1234efgh"),
        ("api" + "_key: " + "9f8e7d6c5b4a", "9f8e7d6c5b4a"),
        ("client_secret='s3cr3tvalue'", "s3cr3tvalue"),
    ],
)
def test_generic_assignments(text: str, value: str) -> None:
    out = redact_text(text, EMPTY)
    assert value not in out
    assert "[secret]" in out


def test_bearer_token() -> None:
    out = redact_text("curl -H 'Authorization: " + "Bearer " + "abcDEF123456ghiJKL789'", EMPTY)
    assert "abcDEF123456" not in out
    assert "Bearer [secret]" in out


# --- URLs ------------------------------------------------------------------


def test_url_with_credentials() -> None:
    out = redact_text("Connected to postgres://admin:pw123@db.example.com:5432/app.", EMPTY)
    assert out == "Connected to [url]."


def test_url_with_token_query_even_when_keeping_public() -> None:
    pol = RedactionPolicy(keep_public_urls=True)
    out = redact_text("See https://github.com/org/repo/archive.zip?token=abc123", pol)
    assert out == "See [url]"


def test_all_urls_dropped_by_default() -> None:
    assert redact_text("Docs at https://pypi.org/project/rich/.", EMPTY) == "Docs at [url]."


def test_public_urls_kept_when_enabled() -> None:
    pol = RedactionPolicy(keep_public_urls=True)
    for url in (
        "https://github.com/Textualize/rich",
        "https://pypi.org/project/rich/",
        "https://docs.python.org/3/library/re.html",
    ):
        assert redact_text(f"See {url}.", pol) == f"See {url}."
    assert redact_text("See https://intranet.acme.io/x.", pol) == "See [url]."
    assert redact_text("See http://github.com/a.", pol) == "See [url]."
    assert redact_text("See https://bob:pw@github.com/a.", pol) == "See [url]."


# --- identifiers -----------------------------------------------------------


def test_email() -> None:
    assert redact_text("Ping jane.doe+ci@example.co.uk now", EMPTY) == "Ping [email] now"


@pytest.mark.parametrize(
    "path",
    [
        "/Users/jane/code/app/train.py",
        "/home/ubuntu/work/run.sh",
        r"C:\Users\Jane\proj\main.py",
        "~/.cursor/projects/foo",
        "/Users/jane",
    ],
)
def test_home_paths(path: str) -> None:
    out = redact_text(f"Edited {path}, then ran tests.", EMPTY)
    assert out == "Edited [path], then ran tests."


def test_ips() -> None:
    assert redact_text("Node at 10.0.3.17 and 192.168.1.5:8080.", EMPTY) == "Node at [ip] and [ip]."
    assert redact_text("Public 8.8.8.8 too", EMPTY) == "Public [ip] too"
    assert redact_text("172.20.0.1 internal", EMPTY) == "[ip] internal"


@pytest.mark.parametrize(
    "host",
    ["redis.internal", "printer.local", "build01.corp", "api.default.svc.cluster.local", "nas.lan"],
)
def test_internal_hosts(host: str) -> None:
    assert redact_text(f"Talk to {host} first.", EMPTY) == "Talk to [host] first."


def test_project_aliases_and_variants() -> None:
    pol = build_policy(["infra-lab", "vibe_app"], allowed=["vibe_app"])
    out = redact_text("Worked on Infra-Lab, infra_lab and infra lab; vibe_app too.", pol)
    assert out == "Worked on Project A, Project A and Project A; vibe_app too."


def test_project_not_partial_word() -> None:
    pol = build_policy(["atlas"])
    assert redact_text("Atlases and atlas-ish stuff", pol) == "Atlases and atlas-ish stuff"
    assert redact_text("The atlas project.", pol) == "The Project A project."


def test_person_names() -> None:
    pol = build_policy([], person_names=["Jane Doe", "bob"])
    out = redact_text("JANE DOE paired with Bob; bobsled unaffected.", pol)
    assert out == "[person] paired with [person]; bobsled unaffected."


# --- false positives -------------------------------------------------------


@pytest.mark.parametrize(
    "text",
    [
        "The token budget ran out, and the secret sauce was caching.",
        "Upgraded Python to 3.10.2 and rich to 13.7.1.",
        "Switched from claude-sonnet-5-5 to gpt-6 for Stage 1.",
        "Edit config.json and settings.local.json, not .env.local.",
        "Pushed the fix to github and opened a PR.",
        "Secret: sauce. Token: it expired. Password resets were flaky.",
        "Use a bearer token in the header.",
        "Ran task-runner with max_tokens=4096 on tokenizer files.",
    ],
)
def test_prose_untouched(text: str) -> None:
    assert redact_text(text, EMPTY) == text


# --- policy ----------------------------------------------------------------


def test_alias_stability_and_order() -> None:
    p1 = build_policy(["zeta", "alpha", "mid"])
    p2 = build_policy(["mid", "zeta", "alpha", "alpha"])
    assert p1.project_aliases == p2.project_aliases
    assert p1.project_aliases == {"alpha": "Project A", "mid": "Project B", "zeta": "Project C"}


def test_alias_rollover_past_z() -> None:
    names = [f"proj{i:03d}" for i in range(30)]
    aliases = build_policy(names).project_aliases
    assert aliases["proj025"] == "Project Z"
    assert aliases["proj026"] == "Project AA"
    assert aliases["proj027"] == "Project AB"


def test_allowed_excluded_from_aliases() -> None:
    pol = build_policy(["a-proj", "public-lib"], allowed=["public-lib"])
    assert "public-lib" not in pol.project_aliases
    assert pol.allowed_projects == {"public-lib"}


# --- objects + leak scan ---------------------------------------------------


def _sample() -> dict:
    return {
        "project": "infra-lab",
        "summary": f"Fixed auth for infra-lab using {ANT} on 10.1.2.3.",
        "rules": [
            "Never commit /Users/jane/.aws/credentials",
            {"note": "email jane@acme.com", "n": 3, "ok": True},
        ],
        "friction": ("Bearer abcdef1234567890xyz failed",),
    }


def test_redact_obj_nested_and_immutable() -> None:
    pol = build_policy(["infra-lab"], person_names=["jane"])
    src = _sample()
    out = redact_obj(src, pol)
    assert src["project"] == "infra-lab"  # input untouched
    assert set(out) == set(src)  # keys untouched
    assert out["project"] == "Project A"
    assert out["summary"] == "Fixed auth for Project A using [secret] on [ip]."
    assert out["rules"][0] == "Never commit [path]"
    assert out["rules"][1] == {"note": "email [email]", "n": 3, "ok": True}
    assert out["friction"] == ("Bearer [secret] failed",)


def test_find_leaks_before_and_after() -> None:
    src = _sample()
    leaks = find_leaks(src)
    kinds = " ".join(leaks)
    for expected in ("Anthropic", "private IP", "path (", "email", "bearer"):
        assert expected in kinds
    assert ANT not in kinds  # report is masked
    pol = build_policy(["infra-lab"])
    assert find_leaks(redact_obj(src, pol)) == []


def test_find_leaks_scans_keys_and_credential_urls() -> None:
    leaks = find_leaks({"https://u:p@x.io/": 1, "v": ["https://a.io/?api_key=zz"]})
    assert any("(key)" in l for l in leaks)
    assert any("v[0]" in l and "credentialed URL" in l for l in leaks)


def test_find_leaks_clean_prose() -> None:
    assert find_leaks({"a": ["The secret sauce", "Python 3.10.2"], "b": 1}) == []


# --- gaps found in review: hosts, other paths, other URL schemes, name tails ----


def test_review_gap_cases_are_redacted():
    from agent_vibe_check.redact import build_policy, find_leaks, redact_text
    policy = build_policy(["code-acme-app"])
    text = ("Built acme-app on grafana.acme.io from /mnt/data/acme. Deploy to "
            "prod-db-1.us-east-1.rds.amazonaws.com via gs://acme-private and s3a://bucket/x")
    out = redact_text(text, policy)
    for leaked in ("acme-app", "grafana.acme.io", "/mnt/data", "amazonaws", "gs://", "s3a://"):
        assert leaked not in out, (leaked, out)
    assert "Project A" in out
    assert find_leaks({"t": out}, names=["code-acme-app", "acme-app"]) == []
    assert find_leaks({"t": "kept acme-app here"}, names=["acme-app"])


def test_public_domains_files_and_slash_commands_survive():
    from agent_vibe_check.redact import build_policy, redact_text
    policy = build_policy([])
    text = "Read docs on github.com and pypi.org, edit config.json and app.py, run /review in CI/CD."
    assert redact_text(text, policy) == text



def test_strip_secrets_keeps_meaning_but_drops_credentials() -> None:
    from agent_vibe_check.redact import strip_secrets
    text = (f"export KEY={ANT}\nopen /Users/jane/app/main.py on db.internal and "
            "postgres://admin:pw123@db.example.com/app ; " + "pass" + "word=hunter2")
    out = strip_secrets(text)
    assert ANT not in out and "pw123" not in out and "hunter2" not in out
    assert "/Users/jane/app/main.py" in out and "db.internal" in out


def test_render_session_strips_secrets() -> None:
    from agent_vibe_check.parser import Message, Session
    from agent_vibe_check.pipeline.session_render import render_session
    s = Session(session_id="x", project_folder="p",
                messages=[Message(role="user", text_blocks=[f"use {ANT} please"])])
    assert ANT not in render_session(s).text


@pytest.mark.parametrize("secret", [
    "sk_" + "live_" + "A1b2C3d4E5f6G7h8I9j0",
    "glpat" + "-" + "A1b2C3d4E5f6G7h8I9j0kL",
    "npm_" + "A1b2C3d4E5f6G7h8I9j0kLmNoPqRsTuVw",
    "r8_" + "A1b2C3d4E5f6G7h8I9j0kL",
])
def test_more_vendor_formats(secret: str) -> None:
    from agent_vibe_check.redact import strip_secrets
    assert secret not in strip_secrets(f"token is {secret} ok")


def test_cli_password_flags_and_basic_auth() -> None:
    from agent_vibe_check.redact import strip_secrets
    out = strip_secrets("mysql -u root -p" + "hunter22 db; tool --pass" + "word s3cretpw; "
                        "Authorization: Basic " + "dXNlcjpwYXNzd29yZA==")
    assert "hunter22" not in out and "s3cretpw" not in out and "dXNlcjpwYXNzd29yZA" not in out


def test_secret_removed_before_truncation() -> None:
    from agent_vibe_check.parser import Message, Session
    from agent_vibe_check.pipeline import session_render as sr
    body = "x" * 3000 + " pass" + "word=" + "Zq9" + "y" * 20 + " " + "z" * 3000
    s = Session(session_id="x", project_folder="p", messages=[Message(role="user", text_blocks=[body])])
    assert "Zq9" not in sr.render_session(s).text


def test_hosts_on_more_tlds_but_not_file_names() -> None:
    from agent_vibe_check.redact import build_policy, redact_text
    out = redact_text("grafana.acme.ca and build.acme.eu; run install.sh, read README.md, call os.path.join",
                      build_policy([]))
    assert "acme.ca" not in out and "acme.eu" not in out
    assert "install.sh" in out and "README.md" in out and "os.path.join" in out
