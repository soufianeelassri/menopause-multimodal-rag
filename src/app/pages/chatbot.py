"""RAG chatbot tab with hybrid retrieval and streaming responses.

Implements the conversational interface with retrieval settings toggles,
3-class query routing, and source citation display.
"""

from __future__ import annotations

import gc
from datetime import datetime

import streamlit as st
from langchain_chroma import Chroma
from langchain_core.messages import AIMessage, HumanMessage
from langchain_google_genai import ChatGoogleGenerativeAI

from src.config.settings import get_settings
from src.generation.classifier import QueryClass, classify_query
from src.generation.generator import ResponseGenerator
from src.generation.prompts import (
    ANSWER_PROMPT_TEMPLATE,
    DIRECT_RESPONSE_TEMPLATE,
    OUT_OF_SCOPE_RESPONSE,
)
from src.retrieval.bm25_index import BM25Index
from src.retrieval.hybrid import hybrid_retrieve
from src.retrieval.reranker import DocumentReranker
from src.retrieval.repacker import DocumentRepacker
from src.utils.helpers import format_chat_history_for_prompt


def render_chat_tab(
    llm: ChatGoogleGenerativeAI,
    vectorstore: Chroma,
    bm25_index: BM25Index,
) -> None:
    """Render the chat tab with message history and input.

    Args:
        llm: Gemini LLM instance.
        vectorstore: ChromaDB vector store.
        bm25_index: BM25 sparse index.
    """
    settings = get_settings()
    user_avatar = "\U0001f469\u200d\U0001f9b0"  # woman emoji
    assistant_avatar = "\U0001f338"  # cherry blossom emoji

    # Add retrieval settings to session state if not present
    if "use_reranking" not in st.session_state:
        st.session_state.use_reranking = settings.reranking_enabled
    if "use_repacking" not in st.session_state:
        st.session_state.use_repacking = settings.repacking_enabled
    if "repacking_method" not in st.session_state:
        st.session_state.repacking_method = "similarity"

    # Retrieval settings row
    settings_cols = st.columns(4)

    with settings_cols[0]:
        st.markdown("""<h4 style='font-family: "Dancing Script", cursive;'>
                    Retrieval Settings
                    </h4>""", unsafe_allow_html=True)

    with settings_cols[1]:
        st.toggle(
            "Use Reranking", key="use_reranking",
            help="Reranking improves retrieval quality by reordering results using a cross-encoder model",
        )

    with settings_cols[2]:
        st.toggle(
            "Use Document Repacking", key="use_repacking",
            help="Repacking groups similar documents together for better context",
        )

    with settings_cols[3]:
        if st.session_state.use_repacking:
            st.selectbox(
                "Repacking Method",
                ["similarity", "token_limit"],
                key="repacking_method",
                help="Similarity: Group by semantic similarity, Token Limit: Group by token count",
            )

    # Display all messages from history
    for msg_data in st.session_state.messages:
        role = msg_data["role"]
        avatar = user_avatar if role == "user" else assistant_avatar

        with st.chat_message(role, avatar=avatar):
            st.markdown(
                f"<div style='font-size: 14px;'>{msg_data['content']}</div>",
                unsafe_allow_html=True,
            )
            if role == "assistant" and "elapsed" in msg_data:
                st.markdown(
                    f"<div style='font-size:10px; margin-top:4px;'>"
                    f"\u23f1\ufe0f Response time: {msg_data['elapsed']:.2f} seconds"
                    f"</div>",
                    unsafe_allow_html=True,
                )

    # Input box for user questions
    user_input = st.chat_input(
        "Ask questions about menopause, symptoms, treatments, "
        "lifestyle changes, or emotional support."
    )

    if user_input:
        # Add user message to chat history
        st.session_state.chat_history.append(HumanMessage(content=user_input))

        # Process query
        start_time = datetime.now()
        conversation_history_str = format_chat_history_for_prompt(
            st.session_state.chat_history
        )

        # Classify query (3-class)
        try:
            query_class = classify_query(user_input, llm, settings)
        except Exception:
            query_class = QueryClass.RAG_REQUIRED

        # Retrieve context if needed
        context = ""
        if query_class == QueryClass.RAG_REQUIRED:
            try:
                docs = hybrid_retrieve(
                    query=user_input,
                    vectorstore=vectorstore,
                    bm25_index=bm25_index,
                    settings=settings,
                )

                # Reranking
                if st.session_state.use_reranking and docs:
                    reranker = DocumentReranker(settings)
                    docs = reranker.rerank(user_input, docs)

                # Repacking
                if st.session_state.use_repacking and docs:
                    repacker = DocumentRepacker(settings)
                    if st.session_state.repacking_method == "similarity":
                        docs = repacker.repack_by_similarity(docs)
                    else:
                        docs = repacker.repack_by_token_limit(docs)

                if docs:
                    context = "\n\n".join(doc.page_content for doc in docs[:10])
            except Exception as e:
                print(f"Error retrieving context: {e}")
                context = ""

        # Display user message
        with st.chat_message("user", avatar=user_avatar):
            st.markdown(
                f"<div style='font-size: 14px;'>{user_input}</div>",
                unsafe_allow_html=True,
            )

        # Add to messages
        st.session_state.messages.append({"role": "user", "content": user_input})

        # Handle OUT_OF_SCOPE directly
        if query_class == QueryClass.OUT_OF_SCOPE:
            elapsed_time = (datetime.now() - start_time).total_seconds()
            with st.chat_message("assistant", avatar=assistant_avatar):
                st.markdown(
                    f"<div style='font-size: 14px;'>{OUT_OF_SCOPE_RESPONSE}</div>"
                    f"<div style='font-size:10px; margin-top:4px;'>"
                    f"\u23f1\ufe0f Response time: {elapsed_time:.2f} seconds</div>",
                    unsafe_allow_html=True,
                )
            st.session_state.chat_history.append(
                AIMessage(content=OUT_OF_SCOPE_RESPONSE)
            )
            st.session_state.messages.append({
                "role": "assistant",
                "content": OUT_OF_SCOPE_RESPONSE,
                "elapsed": elapsed_time,
            })
        else:
            # Build prompt and stream response
            if query_class == QueryClass.RAG_REQUIRED:
                final_prompt = ANSWER_PROMPT_TEMPLATE.format(
                    context=context,
                    question=user_input,
                    chat_history=conversation_history_str,
                )
            else:
                final_prompt = DIRECT_RESPONSE_TEMPLATE.format(
                    question=user_input,
                    chat_history=conversation_history_str,
                )

            # Stream the assistant response
            with st.chat_message("assistant", avatar=assistant_avatar):
                response_placeholder = st.empty()
                streamed_response = ""

                try:
                    for chunk in llm.stream(final_prompt):
                        chunk_text = ""
                        if hasattr(chunk, "content") and chunk.content is not None:
                            chunk_text = chunk.content
                        elif isinstance(chunk, str):
                            chunk_text = chunk

                        if chunk_text:
                            streamed_response += chunk_text
                            response_placeholder.markdown(
                                f"<div style='font-size: 14px;'>"
                                f"{streamed_response}\u258c</div>",
                                unsafe_allow_html=True,
                            )

                    elapsed_time = (datetime.now() - start_time).total_seconds()

                    # Final update
                    response_placeholder.markdown(
                        f"<div style='font-size: 14px;'>{streamed_response}</div>"
                        f"<div style='font-size:10px; margin-top:4px;'>"
                        f"\u23f1\ufe0f Response time: {elapsed_time:.2f} seconds</div>",
                        unsafe_allow_html=True,
                    )

                    st.session_state.chat_history.append(
                        AIMessage(content=streamed_response)
                    )
                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": streamed_response,
                        "elapsed": elapsed_time,
                    })

                except Exception as e:
                    st.error(f"Error streaming response: {e}")
                    error_msg = "I apologize, but I encountered an issue generating my response."
                    response_placeholder.markdown(
                        f"<div style='font-size: 14px;'>{error_msg}</div>",
                        unsafe_allow_html=True,
                    )
                    st.session_state.chat_history.append(AIMessage(content=error_msg))
                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": error_msg,
                        "elapsed": (datetime.now() - start_time).total_seconds(),
                    })

        gc.collect()

    # Chat Footer with disclaimer
    st.markdown("""
        <div class="fixed-bottom-warning">
            \u26a0\ufe0f MenoGuide is an educational tool and not a substitute for professional medical advice.
            Always consult with a healthcare provider for medical concerns.
        </div>
    """, unsafe_allow_html=True)
