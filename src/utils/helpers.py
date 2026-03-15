"""Utility helper functions for the MenoGuide application.

Provides asset loading, chat history formatting, and other shared utilities.
"""

from __future__ import annotations

import base64
from pathlib import Path

import streamlit as st
from langchain_core.messages import AIMessage, HumanMessage

from src.config.settings import get_settings


@st.cache_data(show_spinner=False)
def load_logo_base64() -> str | None:
    """Load the application logo as a base64-encoded string.

    Returns:
        Base64 string of the logo image, or None if not found.
    """
    logo_path = get_settings().assets_dir / "menopause.png"
    try:
        return base64.b64encode(logo_path.read_bytes()).decode()
    except FileNotFoundError:
        return None


@st.cache_data(show_spinner=False)
def load_resource_images() -> dict[str, str | None]:
    """Load educational resource images as base64-encoded strings.

    Returns:
        Dictionary mapping resource names to their base64 image data.
    """
    assets_dir = get_settings().assets_dir
    resource_files = {
        "hot_flash": assets_dir / "hot_flash.jpeg",
        "nutrition": assets_dir / "nutrition.jpeg",
        "sleep": assets_dir / "sleep.jpeg",
        "wellness": assets_dir / "wellness.jpeg",
    }

    resources: dict[str, str | None] = {}
    for key, path in resource_files.items():
        try:
            resources[key] = base64.b64encode(path.read_bytes()).decode()
        except FileNotFoundError:
            resources[key] = None

    return resources


@st.cache_data(show_spinner=False)
def load_assets() -> dict[str, object]:
    """Load all static assets used by the application.

    Returns:
        Dictionary with 'logo' and 'resource_images' keys.
    """
    return {
        "logo": load_logo_base64(),
        "resource_images": load_resource_images(),
    }


def format_chat_history_for_prompt(
    chat_history: list[HumanMessage | AIMessage],
    max_turns: int = 10,
) -> str:
    """Format chat history into a string for the LLM prompt.

    Excludes the most recent message (which is the current query) and
    caps history at max_turns to prevent context window overflow.

    Args:
        chat_history: List of LangChain message objects.
        max_turns: Maximum number of message pairs to include.

    Returns:
        Formatted string of conversation history.
    """
    history_to_format = chat_history[:-1]

    if max_turns > 0:
        history_to_format = history_to_format[-(max_turns * 2):]

    formatted_lines: list[str] = []
    for msg in history_to_format:
        if isinstance(msg, HumanMessage):
            formatted_lines.append(f"User Question: {msg.content}")
        elif isinstance(msg, AIMessage):
            formatted_lines.append(f"MenoGuide Response: {msg.content}")

    return "\n".join(formatted_lines)
