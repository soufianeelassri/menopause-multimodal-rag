"""Multimodal captioning for tables and images via Gemini 2.0 Flash.

Generates medically-specific captions and summaries for RAG retrieval.
Implements rate limiting (14 req/min) to respect Gemini API quotas.
"""

from __future__ import annotations

import time
from typing import Any

from langchain_core.messages import HumanMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI

from src.config.settings import Settings, get_settings
from src.utils.logging import get_logger

logger = get_logger(__name__)

# ---------------------------------------------------------------------------
# Prompt Templates (preserved verbatim from original models/prompt_templates.py)
# ---------------------------------------------------------------------------

TABLE_SUMMARIZER_PROMPT = """You are an expert medical data analyst specializing in menopause research.
Your task is to create an extremely detailed and comprehensive summary of the following table
from a peer-reviewed menopause research paper.

CRITICAL REQUIREMENTS:
1. Extract ALL numerical data - every single number, percentage, p-value, confidence interval,
   odds ratio, hazard ratio, and statistical measure present in the table
2. Preserve the EXACT statistical values - do not round or approximate
3. Include ALL row and column headers with their complete labels
4. Capture every footnote, annotation, and special symbol
5. Note the units of measurement for every variable
6. Include sample sizes (n) for every group/subgroup
7. Document any missing data indicators (NA, NR, -, etc.)

DETAILED ANALYSIS REQUIREMENTS:
- Identify and describe the study design reflected in the table (cross-sectional, longitudinal, RCT, etc.)
- Note the statistical tests used (chi-square, t-test, ANOVA, regression, etc.)
- Extract all p-values and indicate significance levels (* p<0.05, ** p<0.01, *** p<0.001)
- Report all confidence intervals with their exact bounds
- Describe any dose-response relationships visible in the data
- Note any subgroup analyses (by age, BMI, ethnicity, menopausal status, etc.)

MENOPAUSE-SPECIFIC CONTEXT:
- Identify which menopausal stage(s) are referenced (perimenopause, menopause, postmenopause)
- Note any hormone levels reported (estradiol, FSH, LH, progesterone, testosterone)
- Document symptom severity scores and scales used (MRS, Kupperman, Greene Climacteric Scale)
- Capture any treatment-related data (HRT types, dosages, duration, routes of administration)
- Note body composition measurements (BMI, waist circumference, body fat percentage)
- Include cardiovascular risk markers if present (blood pressure, lipids, glucose)
- Document bone density measurements (T-scores, Z-scores, BMD values)
- Capture quality of life scores and psychological assessment results

TREATMENT INFORMATION:
- If treatment data is present, extract: drug names, dosages, formulations
- Note administration routes (oral, transdermal, vaginal)
- Document treatment duration and follow-up periods
- Extract efficacy outcomes and adverse event rates
- Note any comparison between treatment groups

METHODOLOGY DETAILS:
- Sample recruitment method and eligibility criteria reflected in the table
- Any adjustment variables in multivariate analyses
- Model specifications (logistic regression, Cox proportional hazards, etc.)
- Reference groups used for comparisons

Table content:
{data}

Additional context from metadata:
{metadata}

Provide an exhaustive summary that captures EVERY piece of information in this table.
A reader should be able to reconstruct the complete table from your summary alone.
Begin with a one-sentence overview of what the table presents, then systematically describe
every section, column, and row with all associated values."""

IMAGE_SUMMARIZER_PROMPT = """You are an expert medical figure analyst specializing in menopause research visualization.
Your task is to provide an extremely detailed description and analysis of the following figure
from a peer-reviewed menopause research paper.

FIGURE TYPE IDENTIFICATION:
First, identify the type of figure:
- Bar chart / Grouped bar chart
- Line graph / Time series
- Scatter plot / Correlation plot
- Box plot / Violin plot
- Forest plot (meta-analysis)
- Kaplan-Meier survival curve
- Heat map / Correlation matrix
- Flow diagram / CONSORT diagram
- Histological/microscopy image
- Anatomical diagram
- Hormone level curve
- Other (specify)

AXES AND SCALES:
- X-axis: label, units, range, tick marks, categories
- Y-axis: label, units, range, tick marks, scale type (linear/log)
- Secondary y-axis if present
- Any axis breaks or discontinuities

LEGEND AND ANNOTATIONS:
- All legend entries with their visual markers (colors, patterns, symbols)
- Any text annotations, arrows, or callouts
- Statistical significance markers (*, **, ***, ns)
- Reference lines (mean, median, threshold values)
- Error bars: type (SD, SEM, 95% CI) and approximate values

QUANTITATIVE DATA EXTRACTION:
- Extract approximate values for all data points visible
- Note peaks, troughs, and inflection points
- Identify trends (increasing, decreasing, U-shaped, J-shaped)
- Estimate effect sizes where possible
- Note any outliers or anomalous data points

STATISTICAL INDICATORS:
- P-values displayed on the figure
- Confidence bands or intervals
- R-squared values or correlation coefficients
- Sample sizes per group
- Any test statistics reported

TEMPORAL ANALYSIS (if applicable):
- Time points measured
- Duration of study period
- Any seasonal or cyclical patterns
- Rate of change between time points

MENOPAUSE-SPECIFIC VISUALIZATION:
- Hormone level trajectories (estrogen, FSH, LH patterns)
- Symptom severity over time or across groups
- Treatment response curves
- Menopausal transition stages depicted
- Age-related changes or distributions
- Body composition changes
- Bone density trends
- Cardiovascular risk factor visualization
- Quality of life score distributions

CLINICAL SIGNIFICANCE:
- What clinical message does this figure convey?
- How does it support the paper's conclusions?
- Are there clinically meaningful differences visible?
- What implications for patient care can be drawn?

MULTI-PANEL INTEGRATION (if applicable):
- Relationship between panels (A, B, C, etc.)
- How panels complement each other
- Overall narrative across panels

Figure metadata:
{metadata}

Provide a comprehensive description that would allow someone who cannot see the figure
to fully understand its content, trends, and implications for menopause research.
Include approximate numerical values wherever possible."""


class TableCaptioner:
    """Generates detailed summaries of medical tables via Gemini 2.0 Flash.

    Args:
        settings: Application settings. Uses defaults if not provided.
    """

    def __init__(self, settings: Settings | None = None) -> None:
        self._settings = settings or get_settings()
        self._llm = ChatGoogleGenerativeAI(
            model=self._settings.llm_model_name,
            google_api_key=self._settings.gemini_api_key,
            temperature=0.1,
        )
        self._prompt = ChatPromptTemplate.from_template(TABLE_SUMMARIZER_PROMPT)
        self._chain = self._prompt | self._llm
        self._min_interval = 60.0 / self._settings.gemini_rate_limit
        self._last_call_time = 0.0

    def _rate_limit(self) -> None:
        """Enforce rate limiting between API calls."""
        elapsed = time.time() - self._last_call_time
        if elapsed < self._min_interval:
            time.sleep(self._min_interval - elapsed)
        self._last_call_time = time.time()

    def generate_summary(
        self,
        table_content: str,
        metadata: dict[str, Any] | None = None,
    ) -> str:
        """Generate a detailed natural language summary of a table.

        Args:
            table_content: HTML or markdown table content.
            metadata: Additional context about the table's source.

        Returns:
            Detailed summary string.
        """
        self._rate_limit()
        metadata_str = str(metadata) if metadata else "No additional metadata"

        try:
            result = self._chain.invoke({
                "data": table_content,
                "metadata": metadata_str,
            })
            return result.content
        except Exception as e:
            logger.error("table_summary_error", error=str(e))
            return f"[Table summary unavailable: {e}]"

    def summarize_tables(
        self,
        table_elements: list[dict[str, Any]],
    ) -> list[str]:
        """Summarize a batch of table elements.

        Args:
            table_elements: List of dicts with 'content' and 'metadata' keys.

        Returns:
            List of summary strings, one per table.
        """
        summaries: list[str] = []

        for i, table in enumerate(table_elements):
            logger.info(
                "summarizing_table",
                index=i + 1,
                total=len(table_elements),
            )
            summary = self.generate_summary(
                table["content"],
                table.get("metadata"),
            )
            summaries.append(summary)

        logger.info("table_summarization_complete", count=len(summaries))
        return summaries


class ImageCaptioner:
    """Generates detailed captions of medical images via Gemini 2.0 Flash.

    Args:
        settings: Application settings. Uses defaults if not provided.
    """

    def __init__(self, settings: Settings | None = None) -> None:
        self._settings = settings or get_settings()
        self._llm = ChatGoogleGenerativeAI(
            model=self._settings.llm_model_name,
            google_api_key=self._settings.gemini_api_key,
            temperature=0.1,
        )
        self._min_interval = 60.0 / self._settings.gemini_rate_limit
        self._last_call_time = 0.0

    def _rate_limit(self) -> None:
        """Enforce rate limiting between API calls."""
        elapsed = time.time() - self._last_call_time
        if elapsed < self._min_interval:
            time.sleep(self._min_interval - elapsed)
        self._last_call_time = time.time()

    def generate_caption(
        self,
        image_base64: str,
        metadata: dict[str, Any] | None = None,
    ) -> str:
        """Generate a detailed medical caption for an image.

        Args:
            image_base64: Base64-encoded image string.
            metadata: Additional context about the image's source.

        Returns:
            Detailed caption string.
        """
        self._rate_limit()
        metadata_str = str(metadata) if metadata else "No additional metadata"
        prompt_text = IMAGE_SUMMARIZER_PROMPT.format(metadata=metadata_str)

        try:
            message = HumanMessage(
                content=[
                    {"type": "text", "text": prompt_text},
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/png;base64,{image_base64}"
                        },
                    },
                ]
            )
            result = self._llm.invoke([message])
            return result.content
        except Exception as e:
            logger.error("image_caption_error", error=str(e))
            return f"[Image caption unavailable: {e}]"

    def caption_images(
        self,
        image_elements: list[dict[str, Any]],
    ) -> tuple[list[str], list[str]]:
        """Caption a batch of image elements.

        Args:
            image_elements: List of dicts with 'content' (base64) and 'metadata' keys.

        Returns:
            Tuple of (captions, base64_strings).
        """
        captions: list[str] = []
        base64_list: list[str] = []

        for i, image in enumerate(image_elements):
            logger.info(
                "captioning_image",
                index=i + 1,
                total=len(image_elements),
            )
            caption = self.generate_caption(
                image["content"],
                image.get("metadata"),
            )
            captions.append(caption)
            base64_list.append(image["content"])

        logger.info("image_captioning_complete", count=len(captions))
        return captions, base64_list
