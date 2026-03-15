"""Educational resource cards tab with curated menopause information.

Displays interactive HTML resource cards in a 2x2 grid layout with
images, tips, and external reference links.
"""

from __future__ import annotations

import streamlit as st


def render_educational_cards_tab(
    resource_images: dict[str, str | None],
) -> None:
    """Render the educational cards tab with information cards.

    Args:
        resource_images: Dictionary mapping resource keys to base64 image strings.
    """
    col1, col2 = st.columns(2)

    with col1:
        st.markdown(f"""
            <div class="resource-card">
                <img src="data:image/png;base64,{resource_images.get('hot_flash', '')}"
                     alt="Hot Flash Management"
                     style="max-width:100%; height:150px; object-fit:contain; margin-bottom:10px;">
                <h4 style='font-family: "Dancing Script", cursive;'>Managing Hot Flashes</h4>
                <p>Manage hot flashes through lifestyle and cooling tips.</p>
                <ul style='text-align: left; font-size: 0.9rem;'>
                    <li>Dress in lightweight, breathable layers</li>
                    <li>Keep a cooling spray nearby</li>
                    <li>Practice paced breathing techniques</li>
                    <li>Identify and avoid personal triggers</li>
                </ul>
                <a href="https://www.menopause.org/for-women/menopause-flashes/menopause-symptoms-and-treatments/managing-hot-flashes"
                   target="_blank" style="display:inline-block; margin-top:10px; color:#ff6b6b; text-decoration:none;">
                    Learn more about managing hot flashes \u2192
                </a>
            </div>
        """, unsafe_allow_html=True)

        st.markdown(f"""
            <div class="resource-card">
                <img src="data:image/png;base64,{resource_images.get('sleep', '')}"
                     alt="Sleep Management"
                     style="max-width:100%; height:150px; object-fit:contain; margin-bottom:10px;">
                <h4 style='font-family: "Dancing Script", cursive;'>Menopause and Sleep</h4>
                <p>Discover how to improve your sleep quality during menopause.</p>
                <ul style='text-align: left; font-size: 0.9rem;'>
                    <li>Maintain a consistent sleep schedule</li>
                    <li>Create a cool, comfortable sleeping environment</li>
                    <li>Limit caffeine and alcohol before bedtime</li>
                    <li>Consider cognitive behavioral therapy for insomnia</li>
                </ul>
                <a href="https://www.sleepfoundation.org/women-sleep/menopause-and-sleep"
                   target="_blank" style="display:inline-block; margin-top:10px; color:#ff6b6b; text-decoration:none;">
                    Read sleep strategies for menopause \u2192
                </a>
            </div>
        """, unsafe_allow_html=True)

    with col2:
        st.markdown(f"""
            <div class="resource-card">
                <img src="data:image/png;base64,{resource_images.get('nutrition', '')}"
                     alt="Nutrition Guide"
                     style="max-width:100%; height:150px; object-fit:contain; margin-bottom:10px;">
                <h4 style='font-family: "Dancing Script", cursive;'>Nutrition During Menopause</h4>
                <p>Optimal food choices to support your health during this transition.</p>
                <ul style='text-align: left; font-size: 0.9rem;'>
                    <li>Focus on calcium and vitamin D rich foods</li>
                    <li>Incorporate plant-based proteins with phytoestrogens</li>
                    <li>Choose complex carbohydrates over simple sugars</li>
                    <li>Stay well hydrated throughout the day</li>
                </ul>
                <a href="https://www.eatright.org/health/wellness/healthful-habits/eating-right-during-menopause"
                   target="_blank" style="display:inline-block; margin-top:10px; color:#ff6b6b; text-decoration:none;">
                    Explore menopause nutrition guides \u2192
                </a>
            </div>
        """, unsafe_allow_html=True)

        st.markdown(f"""
            <div class="resource-card">
                <img src="data:image/png;base64,{resource_images.get('wellness', '')}"
                     alt="Emotional Wellness"
                     style="max-width:100%; height:150px; object-fit:contain; margin-bottom:10px;">
                <h4 style='font-family: "Dancing Script", cursive;'>Emotional Wellness</h4>
                <p>Support strategies for managing mood changes during menopause.</p>
                <ul style='text-align: left; font-size: 0.9rem;'>
                    <li>Prioritize regular mindfulness practice</li>
                    <li>Build a supportive community network</li>
                    <li>Consider speaking with a therapist familiar with menopause</li>
                    <li>Explore stress reduction techniques that work for you</li>
                </ul>
                <a href="https://www.menopause.org/for-women/menopause-flashes/menopause-symptoms-and-treatments/depression-mood-swings-anxiety"
                   target="_blank" style="display:inline-block; margin-top:10px; color:#ff6b6b; text-decoration:none;">
                    Learn about emotional support resources \u2192
                </a>
            </div>
        """, unsafe_allow_html=True)
