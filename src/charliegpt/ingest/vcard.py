"""Contacts from a vCard export (Contacts.app → File → Export → Export vCard…) → {handle: name}.

Handles are phone numbers normalized to E.164 (+15551234567) and lowercased emails, matching
how iMessage names its chats.
"""

from __future__ import annotations

import re
from pathlib import Path


def normalize_handle(raw: str) -> str:
    raw = raw.strip()
    if "@" in raw:
        return raw.lower()
    digits = re.sub(r"\D", "", raw)
    if not digits:
        return raw
    if raw.startswith("+"):
        return "+" + digits
    if len(digits) == 10:  # US number without country code
        return "+1" + digits
    if len(digits) == 11 and digits.startswith("1"):
        return "+" + digits
    return digits


def _unfold(text: str) -> list[str]:
    """RFC 6350: a line starting with a space/tab continues the previous one."""
    lines: list[str] = []
    for line in text.splitlines():
        if line[:1] in (" ", "\t") and lines:
            lines[-1] += line[1:]
        else:
            lines.append(line)
    return lines


def _unescape(v: str) -> str:
    return re.sub(r"\\([\\,;nN])", lambda m: "\n" if m.group(1) in "nN" else m.group(1), v).strip()


def parse_vcards(path: Path) -> dict[str, str]:
    contacts: dict[str, str] = {}
    card: dict[str, list[str]] = {}
    for line in _unfold(Path(path).read_text(encoding="utf-8", errors="replace")):
        if ":" not in line:
            continue
        key, value = line.split(":", 1)
        prop = key.split(";")[0].split(".")[-1].upper()  # drop params and "item1." groups
        if prop == "BEGIN":
            card = {}
        elif prop == "END":
            name = _card_name(card)
            if name:
                for h in card.get("TEL", []) + card.get("EMAIL", []):
                    contacts.setdefault(normalize_handle(h), name)
        elif prop in ("FN", "N", "ORG", "NICKNAME", "TEL", "EMAIL"):
            card.setdefault(prop, []).append(value)
    return contacts


def _card_name(card: dict[str, list[str]]) -> str | None:
    if fn := _unescape(card.get("FN", [""])[0]):
        return fn
    if n := card.get("N"):
        family, given, *_ = [_unescape(p) for p in n[0].split(";")] + ["", ""]
        if full := f"{given} {family}".strip():
            return full
    for prop in ("NICKNAME", "ORG"):
        if v := _unescape(card.get(prop, [""])[0].split(";")[0]):
            return v
    return None


def find_vcf(explicit: Path | None, folder: Path) -> Path | None:
    if explicit:
        return Path(explicit)
    return next(iter(sorted(Path(folder).glob("*.vcf"))), None) if Path(folder).is_dir() else None


def load_contacts(vcf: Path | None, overrides: dict[str, str]) -> dict[str, str]:
    """vCard contacts plus `identity.contacts` from config (config wins)."""
    contacts = parse_vcards(vcf) if vcf and Path(vcf).exists() else {}
    contacts.update({normalize_handle(k): v for k, v in overrides.items()})
    return contacts


def contacts_for(settings) -> dict[str, str]:
    return load_contacts(find_vcf(settings.identity.contacts_vcf, settings.paths.messages), settings.identity.contacts)
