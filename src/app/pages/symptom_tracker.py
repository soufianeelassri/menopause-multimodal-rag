"""Symptom tracker tab with severity sliders and AI recommendations.

Collects 7 symptom severity ratings (0-10), generates personalized
recommendations using the LLM, and stores data in session state
for the PDF Report tab to consume.
"""

from __future__ import annotations

import streamlit as st
from langchain_google_genai import ChatGoogleGenerativeAI


def render_symptom_tracker_tab(llm: ChatGoogleGenerativeAI) -> None:
    """Render the symptom tracker tab with form and LLM-generated recommendations.

    Args:
        llm: Gemini LLM instance for generating recommendations.
    """
    with st.form("symptom_tracker_form"):
        st.markdown(
            '<h4 style=\'font-family: "Dancing Script", cursive;\'>'
            "Rate your symptoms (0 = None, 10 = Severe)</h4>",
            unsafe_allow_html=True,
        )

        symptoms = {
            "Hot flashes": "Sudden feelings of warmth spreading throughout your body",
            "Night sweats": "Excessive sweating during sleep that may soak your nightclothes or bedding",
            "Sleep difficulties": "Problems falling asleep or staying asleep",
            "Mood changes": "Irritability, anxiety, or mood swings",
            "Fatigue": "Feeling tired or lacking energy",
            "Joint pain": "Aches or stiffness in your joints",
            "Brain fog": "Difficulty concentrating or remembering things",
        }

        symptom_values: dict[str, int] = {}
        symptom_names = list(symptoms.keys())

        for i in range(0, len(symptom_names), 2):
            col1, col2 = st.columns(2)
            with col1:
                name1 = symptom_names[i]
                symptom_values[name1] = st.slider(
                    name1, 0, 10, 0, help=symptoms[name1]
                )
            if i + 1 < len(symptom_names):
                with col2:
                    name2 = symptom_names[i + 1]
                    symptom_values[name2] = st.slider(
                        name2, 0, 10, 0, help=symptoms[name2]
                    )

        additional_info = st.text_area(
            "Additional notes or symptoms not listed above:"
        )
        current_treatments = st.text_area(
            "Current treatments or lifestyle changes you're trying:"
        )

        submitted = st.form_submit_button("Get Personalized Recommendations")

    if submitted:
        active_symptoms = [
            (s, v) for s, v in symptom_values.items() if v > 0
        ]

        if not active_symptoms and not additional_info:
            st.warning(
                "Please rate at least one symptom or provide additional information."
            )
            return

        symptom_text = "\n".join(
            [f"- {s}: {v}/10" for s, v in active_symptoms]
        )
        prompt = f"""
Please provide personalized recommendations for a woman experiencing the following menopause symptoms:

Symptoms:
{symptom_text}

Additional information: {additional_info or "None provided"}
Current treatments/lifestyle changes: {current_treatments or "None provided"}

Please provide:
1. A brief interpretation of these symptoms
2. 3-5 specific, evidence-based recommendations
3. Lifestyle adjustments
4. When to consult a healthcare provider

Use a compassionate, supportive tone.
        """

        try:
            with st.spinner("Generating personalized recommendations..."):
                response = llm.invoke(prompt)

            # Store data in session state for the PDF Report tab
            st.session_state.latest_symptom_data = {
                "active_symptoms": active_symptoms,
                "additional_info": additional_info,
                "current_treatments": current_treatments,
                "recommendations": response.content,
            }

            st.markdown(
                """
                <div style="background-color: #f8f9fa; border-radius: 10px; padding: 10px;
                     margin-top: 20px; margin-bottom: 20px; border-left: 5px solid #FF69B4;">
                    <h3 style='font-family: "Dancing Script", cursive; text-align: center;'>
                        Your Personalized Recommendations
                    </h3>
                </div>
                """,
                unsafe_allow_html=True,
            )
            st.markdown(response.content, unsafe_allow_html=True)

            st.markdown(
                """
                <div style="background-color: #edf7ed; border-radius: 10px; padding: 10px;
                     margin-top: 20px; margin-bottom: 20px;">
                    <p style="font-style: italic; text-align: center;">
                        ⚠️ These recommendations are for informational purposes only
                        and do not constitute medical advice. Always consult a healthcare
                        provider before making changes to your regimen.
                    </p>
                </div>
                """,
                unsafe_allow_html=True,
            )

            st.success("✅ Your data is ready! Go to the **PDF Report** tab to download your report.")

        except Exception as e:
            st.error(f"Error generating recommendations: {e}")
            st.info("Please try again later or contact support.")
