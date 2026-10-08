# app.py
# Streamlit chat interface of the AuraHome support chatbot.
#
# Run from the repository root:
#     streamlit run src/app.py

import os
from pathlib import Path

import streamlit as st
from dotenv import load_dotenv

from langchain_openai import ChatOpenAI

from config import ENV_PATH, LOGO_PATH, CHAT_MODEL, TEMPERATURE, MAX_TOKENS
from rag import list_pdfs, get_vectorstore, format_sources, answer_question_stream


# -----------------------------
# Helpers
# -----------------------------
def render_svg(svg_path: str, width: int = 700):
    """Render an SVG file inside Streamlit."""
    try:
        svg = Path(svg_path).read_text(encoding="utf-8")
        st.markdown(
            f'<div style="display:flex;justify-content:center;align-items:center;margin:10px 0;">'
            f'<div style="width:{width}px;max-width:100%;">{svg}</div>'
            f"</div>",
            unsafe_allow_html=True,
        )
    except Exception as e:
        st.warning(f"ما قدرت أعرض اللوغو: {e}")


# -----------------------------
# App
# -----------------------------
def main():
    st.set_page_config(page_title="AuraHome Chatbot", page_icon="🤖")
    render_svg(LOGO_PATH, width=1500)
    st.markdown("## AuraHome Chatbot")

    load_dotenv(ENV_PATH)
    if not os.getenv("OPENAI_API_KEY"):
        st.error("ما لقيت OPENAI_API_KEY. تأكد إنه موجود في ملف .env في جذر المشروع")
        st.stop()

    pdfs = list_pdfs()

    with st.sidebar:
        st.header("ملفات data/")
        if pdfs:
            for p in pdfs:
                st.write(f"• {p.name}")
        else:
            st.caption("ما في ملفات PDF داخل data/")

        st.divider()
        if st.button("🧹 تفريغ المحادثة"):
            st.session_state.messages = []
            st.rerun()

    if not pdfs:
        st.info("حط ملفات PDF داخل مجلد data/ ثم أعد تشغيل التطبيق.")
        st.stop()

    if "messages" not in st.session_state:
        st.session_state.messages = []

    with st.spinner("تجهيز الفهرس..."):
        vectorstore = get_vectorstore()

    llm = ChatOpenAI(
        model=CHAT_MODEL,
        temperature=TEMPERATURE,
        max_tokens=MAX_TOKENS,
        streaming=True,
    )

    # Conversation so far
    for m in st.session_state.messages:
        with st.chat_message(m["role"]):
            st.markdown(m["content"])

    if len(st.session_state.messages) == 0:
        st.info("أنا أوراهوم 👋 مساعدك الذكي داخل تطبيق AuraHome. اسألني أي سؤال عن ملفات الـPDF.")

    # User input
    user_q = st.chat_input("اكتب سؤالك هنا...")

    if user_q:
        # Show user message immediately
        st.session_state.messages.append({"role": "user", "content": user_q})
        with st.chat_message("user"):
            st.markdown(user_q)

        with st.chat_message("assistant"):
            with st.spinner("عم فكّر..."):
                # IMPORTANT: pass history WITHOUT the last user message to avoid duplication,
                # because we pass `question` separately as the final HumanMessage.
                history_wo_last = st.session_state.messages[:-1]

                stream, src_docs = answer_question_stream(
                    llm=llm,
                    vectorstore=vectorstore,
                    question=user_q,
                    history=history_wo_last,
                )

                placeholder = st.empty()
                full_answer = ""

                for chunk in stream:
                    token = getattr(chunk, "content", "") or ""
                    full_answer += token
                    placeholder.markdown(full_answer)

                with st.expander("📌 من أين جاء الجواب؟"):
                    sources = format_sources(src_docs)
                    best = sources[0] if sources else None
                    st.markdown(f"- {best}" if best else "لا يوجد مصادر")

        st.session_state.messages.append({"role": "assistant", "content": full_answer})


if __name__ == "__main__":
    main()
