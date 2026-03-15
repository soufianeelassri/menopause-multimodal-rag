"""CSS styles for the MenoGuide Streamlit application.

Matches the original MenoMind pink/purple accessible theme.
"""

from __future__ import annotations


def load_css_styles() -> str:
    """Return the CSS styles wrapped in <style> tags."""
    return """
    <style>
        /* Import Google Font */
        @import url('https://fonts.googleapis.com/css2?family=Dancing+Script:wght@400;700&display=swap');

        /* Remove blank space at top and bottom */
        .block-container {
            padding-top: 3rem;
            padding-bottom: 7rem !important;
            overflow-y: auto;
        }

        /* Create a gradient fade effect for text near the input */
        .main .block-container::after {
            content: "";
            position: fixed;
            bottom: 5rem;
            left: 0;
            right: 0;
            height: 40px;
            background: linear-gradient(to bottom, transparent, white);
            pointer-events: none;
            z-index: 99;
        }

        /* Main content styling */
        .main-content {
            background-color: #fcf7fa;
            border-radius: 15px;
            padding: 20px;
            box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
            margin-bottom: 0px;
        }

        /* Chat message styling */
        .chat-message {
            padding: 15px;
            border-radius: 15px;
            margin-bottom: 10px;
            position: relative;
        }

        .user-message {
            background-color: #f0f7ff;
            border-left: 5px solid #7c89ff;
        }

        .assistant-message {
            background-color: #fff0f9;
            border-left: 5px solid #FF69B4;
        }

        /* Card styling */
        .info-card {
            background-color: white;
            border: 1px solid #FF69B4;
            border-radius: 10px;
            padding: 10px;
            margin: 10px 0;
            border-left: 4px solid #FF69B4;
        }

        div.stButton > button:first-child {
            background-color: #f8f2f6;
            border: 1px solid #000000;
            border-radius: 20px;
            padding: 8px 16px;
            margin: 5px 0;
            font-size: 0.85rem;
            color: black;
            cursor: pointer;
            transition: all 0.2s;
        }

        div.stButton > button:first-child:hover {
            background-color: #ffcce6;
            border: 1px solid #000000;
            color: #000000;
        }

        .stFormSubmitButton > button {
            background-color: #f8f2f6;
            border: 1px solid #000000;
            border-radius: 20px;
            padding: 8px 16px;
            margin: 5px 0;
            font-size: 0.85rem;
            color: black;
            cursor: pointer;
            transition: all 0.2s;
        }

        .stFormSubmitButton > button:hover {
            background-color: #ffcce6;
            border: 1px solid #000000;
            color: #000000;
        }

        /* Tab styling */
        .stTabs [data-baseweb="tab-list"] {
            gap: 8px;
        }

        .stTabs [data-baseweb="tab"] {
            background-color: #fcf0f6;
            border-radius: 8px 8px 0 0;
            padding: 10px 16px;
            border: none;
        }

        .stTabs [aria-selected="true"] {
            background-color: #ff9fce !important;
            color: white !important;
        }

        /* Resource card styling */
        .resource-card {
            background-color: white;
            border: 1px solid #FF69B4;
            border-radius: 10px;
            box-shadow: 0 2px 5px rgba(0, 0, 0, 0.1);
            border-left: 4px solid #FF69B4;
            padding: 10px;
            margin: 10px 10px 10px 10px;
            text-align: center;
            transition: transform 0.2s;
        }

        .resource-card:hover {
            transform: translateY(-5px);
        }

        /* Basic fixed chat input */
        .stChatInput {
            position: fixed;
            bottom: 2.5rem;
            left: calc(50% + 110px);
            transform: translateX(-50%);
            width: min(800px, calc(100% - 400px));
            background-color: white;
            z-index: 999;
            margin: 0 auto;
        }

        /* When sidebar is collapsed */
        [data-testid="stSidebar"][aria-expanded="false"] ~ .main .stChatInput {
            left: 50%;
            width: min(800px, calc(100% - 100px));
        }

        /* Footer styling */
        .fixed-bottom-warning {
            position: fixed;
            background-color: white;
            bottom: 0;
            left: calc(50% + 110px);
            transform: translateX(-50%);
            width: 100%;
            padding: 15px;
            margin: 0 auto !important;
            text-align: center;
            font-size: 10px;
            font-weight: 600;
            font-family: "Dancing Script", cursive;
            z-index: 998;
        }

        /* When sidebar is collapsed - adjust footer */
        [data-testid="stSidebar"][aria-expanded="false"] ~ .main .fixed-bottom-warning {
            left: 50%;
        }

        /* Ensure chat messages container scrolls properly */
        [data-testid="stChatMessageContainer"] {
            overflow-y: auto;
            padding-bottom: 2rem;
        }

        /* Logo and slogan styling */
        .logo-container {
            text-align: center;
            padding: 20px 0;
            margin-bottom: 20px;
        }

        .logo-image {
            max-width: 200px;
            height: auto;
            margin-bottom: 10px;
        }

        .slogan {
            font-family: "Dancing Script", cursive;
            color: #FF69B4;
            font-size: 1.5rem;
            margin: 10px 0;
            font-weight: 600;
        }
    </style>
    """
