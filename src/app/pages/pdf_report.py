"""PDF Report tab — generates downloadable symptom reports.

Reads symptom data and AI recommendations stored in session state
by the Symptom Tracker tab, then builds a styled PDF using FPDF2.
Paper Section IV-D: Clinical report generation.
"""

from __future__ import annotations

import base64
from datetime import datetime
from io import BytesIO

import matplotlib
import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st
from fpdf import FPDF

matplotlib.use("Agg")


def render_pdf_report_tab() -> None:
    """Render the PDF report generation tab."""
    st.markdown(
        '<h4 style=\'font-family: "Dancing Script", cursive;\'>'
        "Download Your Symptom Report</h4>",
        unsafe_allow_html=True,
    )

    # Check if symptom data exists in session state
    if "latest_symptom_data" not in st.session_state or not st.session_state.latest_symptom_data:
        st.info(
            "📋 No symptom data available yet. "
            "Please complete the **Symptom Tracker** tab first to generate a report."
        )
        return

    data = st.session_state.latest_symptom_data
    active_symptoms = data.get("active_symptoms", [])
    additional_info = data.get("additional_info", "")
    current_treatments = data.get("current_treatments", "")
    recommendations = data.get("recommendations", "")

    if not active_symptoms:
        st.warning("No active symptoms recorded. Please rate at least one symptom in the Symptom Tracker.")
        return

    # Show summary of what will be in the report
    st.markdown("**Report Preview:**")
    cols = st.columns(2)
    with cols[0]:
        st.markdown("**Symptoms Recorded:**")
        for symptom, value in active_symptoms:
            level = "🟢 Mild" if value <= 3 else "🟡 Moderate" if value <= 6 else "🔴 Severe"
            st.markdown(f"- {symptom}: **{value}/10** ({level})")
    with cols[1]:
        st.markdown("**Additional Info:**")
        st.markdown(additional_info or "*None provided*")
        st.markdown("**Current Treatments:**")
        st.markdown(current_treatments or "*None provided*")

    st.divider()

    # Generate and offer download
    try:
        pdf_bytes = _create_report_pdf(
            active_symptoms,
            additional_info,
            current_treatments,
            recommendations,
        )
        st.download_button(
            label="📄 Download PDF Report",
            data=pdf_bytes,
            file_name=f"MenoGuide_Report_{datetime.now().strftime('%Y-%m-%d')}.pdf",
            mime="application/pdf",
            use_container_width=True,
        )
    except Exception as e:
        st.error(f"Failed to generate PDF: {e}")


def _create_report_pdf(
    active_symptoms: list[tuple[str, int]],
    additional_info: str,
    current_treatments: str,
    recommendations: str,
) -> bytes:
    """Create a PDF report with symptom chart and recommendations.

    Args:
        active_symptoms: List of (symptom_name, severity) tuples.
        additional_info: Additional notes from user.
        current_treatments: Current treatments text.
        recommendations: LLM-generated recommendations text.

    Returns:
        PDF bytes.
    """
    # Generate chart image
    df = pd.DataFrame(active_symptoms, columns=["Symptom", "Rating"])
    fig, ax = plt.subplots(figsize=(9, 3.5))
    bars = ax.barh(df["Symptom"], df["Rating"], color="#FF69B4")
    ax.set_xlim(0, 10)
    ax.set_title("Symptom Severity (0-10 scale)")
    ax.set_xlabel("Severity")
    ax.bar_label(bars)
    plt.tight_layout(pad=1.2)
    plt.subplots_adjust(left=0.25, right=0.9)

    img_buf = BytesIO()
    plt.savefig(img_buf, format="png", bbox_inches="tight", dpi=150)
    plt.close()
    img_buf.seek(0)

    # Build PDF using FPDF2
    pdf = FPDF()
    pdf.add_page()
    pdf.set_auto_page_break(auto=True, margin=30)

    # Header
    pdf.set_font("Helvetica", "B", 18)
    pdf.cell(0, 12, "MenoGuide: Personal Symptom Report", align="C", new_x="LMARGIN", new_y="NEXT")
    pdf.set_font("Helvetica", "", 10)
    pdf.cell(
        0, 6,
        f"Generated on {datetime.now().strftime('%Y-%m-%d')}",
        align="C", new_x="LMARGIN", new_y="NEXT",
    )
    pdf.ln(6)

    # Symptom chart image
    img_buf.seek(0)
    pdf.image(img_buf, x=10, w=190)
    pdf.ln(6)

    # Symptom table
    pdf.set_font("Helvetica", "B", 14)
    pdf.cell(0, 10, "Symptom Assessment", new_x="LMARGIN", new_y="NEXT")
    pdf.ln(2)

    pdf.set_font("Helvetica", "B", 9)
    pdf.set_fill_color(240, 240, 240)
    pdf.cell(80, 7, "Symptom", border=1, fill=True)
    pdf.cell(30, 7, "Severity", border=1, fill=True, align="C")
    pdf.cell(80, 7, "Level", border=1, fill=True, align="C", new_x="LMARGIN", new_y="NEXT")

    pdf.set_font("Helvetica", "", 9)
    for symptom, value in active_symptoms:
        level = "Mild" if value <= 3 else "Moderate" if value <= 6 else "Severe"
        pdf.cell(80, 7, symptom, border=1)
        pdf.cell(30, 7, str(value), border=1, align="C")
        pdf.cell(80, 7, level, border=1, align="C", new_x="LMARGIN", new_y="NEXT")

    pdf.ln(6)

    # Additional info
    pdf.set_font("Helvetica", "B", 12)
    pdf.cell(0, 8, "Additional Information", new_x="LMARGIN", new_y="NEXT")
    pdf.set_font("Helvetica", "", 10)
    pdf.multi_cell(0, 5, f"Notes: {additional_info or 'None provided'}")
    pdf.multi_cell(0, 5, f"Current treatments: {current_treatments or 'None provided'}")
    pdf.ln(4)

    # Recommendations
    if recommendations:
        pdf.set_font("Helvetica", "B", 14)
        pdf.cell(0, 10, "Personalized Recommendations", new_x="LMARGIN", new_y="NEXT")
        pdf.set_font("Helvetica", "", 10)
        pdf.ln(2)

        # Clean markdown markers
        clean_text = recommendations
        for marker in ["**", "__", "###", "##", "#"]:
            clean_text = clean_text.replace(marker, "")

        for line in clean_text.split("\n"):
            line = line.strip()
            if not line:
                pdf.ln(3)
                continue
            pdf.multi_cell(0, 5, line)

    pdf.ln(8)

    # Footer disclaimer
    pdf.set_font("Helvetica", "I", 8)
    pdf.multi_cell(
        0, 4,
        "These recommendations are for informational purposes only and do not "
        "constitute medical advice. Always consult a healthcare provider before "
        "making changes to your regimen.",
        align="C",
    )
    pdf.cell(
        0, 6,
        f"\u00a9 MenoGuide {datetime.now().year}",
        align="C",
    )

    return pdf.output()
