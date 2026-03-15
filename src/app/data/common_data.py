"""Static menopause data for the MenoGuide Streamlit application.

Contains curated menopause stages information and wellness tips
displayed in the sidebar and educational components.
"""

from __future__ import annotations

# Menopause stages information (dict mapping stage name to description)
menopause_stages: dict[str, str] = {
    "Perimenopause": (
        "The transitional period before menopause that can last 4-8 years. "
        "Hormonal fluctuations begin, and periods may become irregular."
    ),
    "Menopause": (
        "Officially diagnosed after 12 consecutive months without a menstrual "
        "period. The average age is 51 in the United States."
    ),
    "Postmenopause": (
        "The years following menopause when many symptoms gradually ease, but "
        "new health considerations may arise due to lower estrogen levels."
    ),
}

# Wellness tips for the sidebar
wellness_tips: list[str] = [
    "Stay hydrated by drinking 8-10 glasses of water daily",
    "Practice deep breathing for 5 minutes when feeling anxious",
    "Consider adding soy foods to your diet for their phytoestrogens",
    "Aim for 7-8 hours of quality sleep each night",
    "Regular weight-bearing exercise helps maintain bone density",
]
