"""Sidebar component with app info, menopause stages, and reset button."""

from __future__ import annotations

from collections.abc import Callable

import streamlit as st


def render_sidebar(
    wellness_tips: list[str],
    menopause_stages: dict[str, str],
    reset_callback: Callable[[], None],
) -> None:
    """Render the application sidebar.

    Args:
        wellness_tips: List of wellness tip strings.
        menopause_stages: Dict mapping stage name to description.
        reset_callback: Callback function to reset conversation.
    """
    with st.sidebar:
        # About section
        st.markdown("""
        <h4 style='font-family: "Dancing Script", cursive;'>About MenoGuide</h4>
        """, unsafe_allow_html=True)
        st.markdown("""
            <p style='font-size: 0.9rem;'>Your trusted AI companion for navigating
            menopause with confidence and ease. Get reliable information, personalized
            advice, and compassionate support.</p>
        """, unsafe_allow_html=True)

        # Menopause Stages
        st.markdown("""
        <h4 style='font-family: "Dancing Script", cursive;'>Menopause Stages</h4>
        """, unsafe_allow_html=True)
        for stage, description in menopause_stages.items():
            st.markdown(f"""
                <div class="info-card">
                    <h4 style='color: #FF69B4; margin-top: 0;'>{stage}</h4>
                    <p style='font-size: 0.9rem;'>{description}</p>
                </div>
            """, unsafe_allow_html=True)

        # Reset button
        if st.button("Reset conversation"):
            reset_callback()
