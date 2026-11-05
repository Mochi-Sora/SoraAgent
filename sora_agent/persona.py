"""Prompt guidance for the configured agent name and personality.

Personality is a prompt layer, not an automatic rewrite of ``SOUL.md``.  The
persistent identity file remains authoritative and may refine these
instructions; this module only gives every fresh install a sensible tone.
"""

from __future__ import annotations

from typing import Any

BASE_GUIDANCE: dict[str, str] = {
    "professional": (
        "Use polished, precise language. Prefer complete sentences, standard "
        "grammar, and a measured tone. Avoid slang and filler."
    ),
    "friendly": (
        "Be warm and conversational. Use contractions, acknowledge the person, "
        "and keep the exchange easygoing without becoming unprofessional."
    ),
    "candid": (
        "Be direct and encouraging. State the important thing first, then the "
        "supporting detail. Do not soften a problem into vagueness."
    ),
    "efficient": (
        "Be compact and task-focused. Skip pleasantries that add no meaning, "
        "but do not become curt or dismissive."
    ),
    "neutral": "",
}

WARMTH_GUIDANCE: dict[str, str] = {
    "very_cold": "Keep warmth to a minimum; stay correct and impersonal.",
    "cold": "Stay reserved and matter-of-fact.",
    "neutral": "",
    "warm": "Add a little warmth and encouragement where it fits naturally.",
    "very_warm": (
        "Be extra warm: friendly greetings, gentle encouragement, and a sense "
        "that you are glad to help. Never let warmth crowd out accuracy or make "
        "a reply longer than it needs to be."
    ),
}


def persona_prompt(name: str, base: str, warmth: str) -> str:
    """Return the system-prompt section for one agent persona."""
    clean_name = (name or "Sora").strip() or "Sora"
    lines = [f"Your name is {clean_name}."]
    base_line = BASE_GUIDANCE.get(str(base or "").strip().lower(), "")
    warmth_line = WARMTH_GUIDANCE.get(str(warmth or "").strip().lower(), "")
    if base_line:
        lines.append(base_line)
    if warmth_line:
        lines.append(warmth_line)
    if len(lines) == 1:
        return lines[0]
    return "\n".join(lines)


def apply_persona(
    base_prompt: str, name: str, base: str, warmth: str
) -> str:
    """Append persona guidance to an existing system prompt."""
    guidance = persona_prompt(name, base, warmth)
    if not guidance:
        return base_prompt
    return f"{base_prompt.rstrip()}\n\n{guidance}"


def display_name(agent: Any) -> str:
    """Return ``icon name`` for terminal banners, or just the name."""
    name = str(getattr(agent, "name", "") or "Sora").strip() or "Sora"
    icon = str(getattr(agent, "icon", "") or "").strip()
    return f"{icon} {name}" if icon else name
