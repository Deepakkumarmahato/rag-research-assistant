
import streamlit as st
import faiss

from app.rag_pipeline import load_rag_components
from app.conversation import conversational_rag
from app.document_ingestion import ingest_uploaded_pdf


# --------------------------------
# 1. Page Configuration
# --------------------------------

st.set_page_config(
    page_title="RAG Research Assistant",
    page_icon="📚",
    layout="wide"
)


# --------------------------------
# 2. Application Heading
# --------------------------------

st.title("📚 RAG Research Assistant")

st.caption(
    "Ask questions about your PDF documents "
    "and get AI-powered answers."
)


# --------------------------------
# 3. Load RAG Components
# --------------------------------

@st.cache_resource
def get_rag_components():
    return load_rag_components()


try:
    model, base_index, base_metadata = get_rag_components()

except Exception as e:
    st.error(f"Failed to load RAG components: {e}")
    st.stop()


# --------------------------------
# 4. Initialize Session State
# --------------------------------

if "vector_index" not in st.session_state:

    # Each user gets an independent FAISS index
    st.session_state.vector_index = faiss.clone_index(
        base_index
    )

    st.session_state.metadata = base_metadata.copy()


if "chat_history" not in st.session_state:
    st.session_state.chat_history = []


if "ui_messages" not in st.session_state:
    st.session_state.ui_messages = []


# Use the session-specific knowledge base
index = st.session_state.vector_index
metadata = st.session_state.metadata


# --------------------------------
# 5. Sidebar
# --------------------------------

with st.sidebar:

    st.header("Settings")

    # Dynamic PDF uploader
    st.write("### Upload Documents")

    uploaded_files = st.file_uploader(
        "Upload PDF files",
        type=["pdf"],
        accept_multiple_files=True
    )

    if uploaded_files:

        for uploaded_file in uploaded_files:

            try:

                pdf_bytes = uploaded_file.getvalue()

                with st.spinner(
                    f"Processing {uploaded_file.name}..."
                ):

                    index, metadata, added_chunks = (
                        ingest_uploaded_pdf(
                            pdf_bytes=pdf_bytes,
                            filename=uploaded_file.name,
                            model=model,
                            index=index,
                            metadata=metadata
                        )
                    )

                # Save updated knowledge base
                st.session_state.vector_index = index
                st.session_state.metadata = metadata

                if added_chunks > 0:

                    st.success(
                        f"{uploaded_file.name}: "
                        f"{added_chunks} chunks added."
                    )

                else:

                    st.info(
                        f"{uploaded_file.name} "
                        "has already been uploaded."
                    )

            except Exception as e:

                st.error(
                    f"Failed to process "
                    f"{uploaded_file.name}: {e}"
                )

    st.divider()

    # Knowledge Base
    st.write("### Knowledge Base")

    st.write(
        f"Total document chunks: {len(metadata)}"
    )

    documents = sorted(
        metadata["document_name"].unique()
    )

    st.write("### Available Documents")

    for document in documents:
        st.write(f"📄 {document}")

    st.divider()

    # Clear chat
    if st.button(
        "🗑️ Clear Chat",
        use_container_width=True
    ):

        st.session_state.chat_history = []
        st.session_state.ui_messages = []

        st.rerun()


# --------------------------------
# 6. Display Previous Messages
# --------------------------------

for message in st.session_state.ui_messages:

    with st.chat_message(message["role"]):

        st.markdown(message["content"])

        if message["role"] == "assistant":

            sources = message.get("sources", [])

            if sources:

                with st.expander("📚 View Sources"):

                    for source in sources:

                        st.write(
                            f"**Document:** "
                            f"{source['document_name']}"
                        )

                        st.write(
                            f"**Page:** "
                            f"{source['page_number']}"
                        )

                        score = source.get(
                            "similarity_score",
                            source.get("score", 0.0)
                        )

                        st.write(
                            f"**Similarity:** {score:.4f}"
                        )

                        st.divider()


# --------------------------------
# 7. Chat Input
# --------------------------------

question = st.chat_input(
    "Ask a question about your documents..."
)


# --------------------------------
# 8. Generate RAG Answer
# --------------------------------

if question:

    # Display user question
    with st.chat_message("user"):
        st.markdown(question)

    st.session_state.ui_messages.append({
        "role": "user",
        "content": question
    })

    # Generate answer
    with st.chat_message("assistant"):

        with st.spinner(
            "Searching documents and generating answer..."
        ):

            try:

                answer, sources, rewritten_question = (
                    conversational_rag(
                        question=question,
                        chat_history=st.session_state.chat_history,
                        model=model,
                        index=st.session_state.vector_index,
                        metadata=st.session_state.metadata
                    )
                )

                st.markdown(answer)

                # Display source information
                if sources:

                    with st.expander("📚 View Sources"):

                        for source in sources:

                            st.write(
                                f"**Document:** "
                                f"{source['document_name']}"
                            )

                            st.write(
                                f"**Page:** "
                                f"{source['page_number']}"
                            )

                            score = source.get(
                                "similarity_score",
                                source.get("score", 0.0)
                            )

                            st.write(
                                f"**Similarity:** {score:.4f}"
                            )

                            st.divider()

                # Save assistant response
                st.session_state.ui_messages.append({
                    "role": "assistant",
                    "content": answer,
                    "sources": sources
                })

            except Exception as e:

                st.error(
                    f"Error generating answer: {e}"
                )
