import streamlit as st
import json
import pandas as pd
import time
from pathlib import Path

from src.pipeline import SpecExtractionPipeline


st.set_page_config(
    page_title="Vehicle Spec Extractor",
    page_icon="🔧",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
    .main .block-container {
        padding-top: 2rem;
        max-width: 1200px;
    }

    .header-container {
        background: linear-gradient(135deg, #1a1a2e 0%, #16213e 50%, #0f3460 100%);
        padding: 2rem 2.5rem;
        border-radius: 16px;
        margin-bottom: 2rem;
        border: 1px solid rgba(255, 255, 255, 0.05);
        box-shadow: 0 8px 32px rgba(0, 0, 0, 0.3);
    }

    .header-title {
        color: #e2e8f0;
        font-size: 2rem;
        font-weight: 700;
        margin: 0;
        letter-spacing: -0.5px;
    }

    .header-subtitle {
        color: #94a3b8;
        font-size: 1rem;
        margin-top: 0.5rem;
        font-weight: 400;
    }

    .spec-card {
        background: linear-gradient(145deg, #1e293b, #1a2332);
        border: 1px solid rgba(99, 102, 241, 0.2);
        border-radius: 12px;
        padding: 1.25rem 1.5rem;
        margin-bottom: 1rem;
        transition: all 0.3s ease;
    }

    .spec-card:hover {
        border-color: rgba(99, 102, 241, 0.5);
        box-shadow: 0 4px 20px rgba(99, 102, 241, 0.1);
        transform: translateY(-2px);
    }

    .spec-component {
        color: #818cf8;
        font-size: 1.1rem;
        font-weight: 600;
        margin-bottom: 0.5rem;
    }

    .spec-value {
        color: #34d399;
        font-size: 1.5rem;
        font-weight: 700;
    }

    .spec-meta {
        color: #94a3b8;
        font-size: 0.85rem;
        margin-top: 0.5rem;
    }

    .source-chunk {
        background: rgba(30, 41, 59, 0.7);
        border-left: 3px solid #6366f1;
        padding: 1rem 1.25rem;
        border-radius: 0 8px 8px 0;
        margin-bottom: 0.75rem;
        font-size: 0.9rem;
        color: #cbd5e1;
    }

    .source-meta {
        color: #6366f1;
        font-weight: 600;
        font-size: 0.8rem;
        margin-bottom: 0.5rem;
    }

    .status-badge {
        display: inline-block;
        padding: 0.25rem 0.75rem;
        border-radius: 20px;
        font-size: 0.8rem;
        font-weight: 600;
    }

    .status-indexed {
        background: rgba(52, 211, 153, 0.15);
        color: #34d399;
        border: 1px solid rgba(52, 211, 153, 0.3);
    }

    .status-not-indexed {
        background: rgba(251, 191, 36, 0.15);
        color: #fbbf24;
        border: 1px solid rgba(251, 191, 36, 0.3);
    }

    .stButton > button {
        background: linear-gradient(135deg, #6366f1, #8b5cf6);
        color: white;
        border: none;
        border-radius: 8px;
        padding: 0.6rem 1.5rem;
        font-weight: 600;
        transition: all 0.3s ease;
    }

    .stButton > button:hover {
        background: linear-gradient(135deg, #818cf8, #a78bfa);
        box-shadow: 0 4px 15px rgba(99, 102, 241, 0.3);
    }

    .example-chip {
        display: inline-block;
        background: rgba(99, 102, 241, 0.1);
        border: 1px solid rgba(99, 102, 241, 0.3);
        color: #a5b4fc;
        padding: 0.35rem 0.85rem;
        border-radius: 20px;
        margin: 0.25rem;
        font-size: 0.85rem;
        cursor: pointer;
        transition: all 0.2s ease;
    }

    .example-chip:hover {
        background: rgba(99, 102, 241, 0.2);
        border-color: rgba(99, 102, 241, 0.5);
    }

    .stat-card {
        background: rgba(30, 41, 59, 0.5);
        border: 1px solid rgba(255, 255, 255, 0.05);
        border-radius: 10px;
        padding: 1rem 1.25rem;
        text-align: center;
    }

    .stat-value {
        font-size: 1.8rem;
        font-weight: 700;
        color: #818cf8;
    }

    .stat-label {
        font-size: 0.8rem;
        color: #94a3b8;
        text-transform: uppercase;
        letter-spacing: 1px;
    }
</style>
""", unsafe_allow_html=True)


@st.cache_resource
def get_pipeline():
    return SpecExtractionPipeline()


st.markdown("""
<div class="header-container">
    <p class="header-title">🔧 Vehicle Specification Extractor</p>
    <p class="header-subtitle">
        RAG-powered extraction from the 2014 Ford F-150 Service Manual
        &nbsp;•&nbsp; Powered by Llama 3.3 70B + ChromaDB
    </p>
</div>
""", unsafe_allow_html=True)


with st.sidebar:
    st.markdown("### ⚙️ Pipeline Controls")

    pipeline = get_pipeline()
    is_indexed = pipeline.vector_store.is_indexed()
    chunk_count = pipeline.vector_store.collection.count() if is_indexed else 0

    if is_indexed:
        st.markdown(
            f'<span class="status-badge status-indexed">✓ Indexed — {chunk_count} chunks</span>',
            unsafe_allow_html=True
        )
    else:
        st.markdown(
            '<span class="status-badge status-not-indexed">⚠ Not indexed</span>',
            unsafe_allow_html=True
        )

    st.markdown("---")

    if st.button("📥 Index PDF", use_container_width=True):
        with st.spinner("Parsing and indexing PDF..."):
            count = pipeline.index_pdf(force_reindex=False)
            st.success(f"Indexed {count} chunks!")
            st.rerun()

    if is_indexed:
        if st.button("🔄 Re-index PDF", use_container_width=True):
            with st.spinner("Re-indexing PDF..."):
                count = pipeline.index_pdf(force_reindex=True)
                st.success(f"Re-indexed {count} chunks!")
                st.rerun()

    st.markdown("---")

    top_k = st.slider("Chunks to retrieve", 3, 15, 8, key="top_k")

    st.markdown("---")

    st.markdown("""
    ### 📖 About
    This tool uses **RAG** (Retrieval-Augmented Generation)
    to extract vehicle specifications from the service manual.

    **Tech Stack:**
    - 📄 PyMuPDF (PDF parsing)
    - 🧠 all-MiniLM-L6-v2 (embeddings)
    - 💾 ChromaDB (vector store)
    - 🤖 Llama 3.3 70B via Groq
    """)


example_queries = [
    "Wheel nut torque specification",
    "Suspension bolt torque specs",
    "Stabilizer bar link torque",
    "Shock absorber bolt torque",
    "Ball joint specifications",
    "Upper control arm bolt torque",
]

st.markdown("**Try an example query:**")
cols = st.columns(3)
selected_example = None
for i, eq in enumerate(example_queries):
    with cols[i % 3]:
        if st.button(f"🔍 {eq}", key=f"example_{i}", use_container_width=True):
            selected_example = eq

query = st.text_input(
    "🔎 Enter your query",
    value=selected_example or "",
    placeholder="e.g., What is the torque for brake caliper bolts?",
    key="query_input",
)

col1, col2, col3 = st.columns([1, 2, 1])
with col2:
    search_clicked = st.button(
        "⚡ Extract Specifications",
        use_container_width=True,
        type="primary",
    )

if search_clicked and query:
    if not is_indexed:
        st.error("⚠️ Please index the PDF first using the sidebar button.")
    else:
        with st.spinner("Searching and extracting..."):
            start = time.time()
            result = pipeline.query(query, top_k=top_k)
            elapsed = time.time() - start

        specs = result.get("specs", [])
        sources = result.get("sources", [])

        col_a, col_b, col_c = st.columns(3)
        with col_a:
            st.markdown(f"""
            <div class="stat-card">
                <div class="stat-value">{len(specs)}</div>
                <div class="stat-label">Specs Found</div>
            </div>
            """, unsafe_allow_html=True)
        with col_b:
            st.markdown(f"""
            <div class="stat-card">
                <div class="stat-value">{len(sources)}</div>
                <div class="stat-label">Sources Retrieved</div>
            </div>
            """, unsafe_allow_html=True)
        with col_c:
            st.markdown(f"""
            <div class="stat-card">
                <div class="stat-value">{elapsed:.1f}s</div>
                <div class="stat-label">Response Time</div>
            </div>
            """, unsafe_allow_html=True)

        st.markdown("---")

        tab1, tab2, tab3 = st.tabs(["📋 Specifications", "📄 Source Chunks", "📊 Raw JSON"])

        with tab1:
            if specs:
                for spec in specs:
                    st.markdown(f"""
                    <div class="spec-card">
                        <div class="spec-component">{spec.get('component', 'N/A')}</div>
                        <div class="spec-value">{spec.get('value', 'N/A')} {spec.get('unit', '')}</div>
                        <div class="spec-meta">
                            Type: {spec.get('spec_type', 'N/A')}
                            &nbsp;•&nbsp; Pages: {spec.get('source_pages', 'N/A')}
                        </div>
                        <div class="spec-meta" style="margin-top: 0.25rem;">
                            {spec.get('context', '')[:150]}
                        </div>
                    </div>
                    """, unsafe_allow_html=True)

                st.markdown("---")
                col_dl1, col_dl2, _ = st.columns([1, 1, 2])
                with col_dl1:
                    json_str = json.dumps(specs, indent=2)
                    st.download_button(
                        "📥 Download JSON",
                        json_str,
                        "extracted_specs.json",
                        "application/json",
                        use_container_width=True,
                    )
                with col_dl2:
                    df = pd.DataFrame(specs)
                    csv_str = df.to_csv(index=False)
                    st.download_button(
                        "📥 Download CSV",
                        csv_str,
                        "extracted_specs.csv",
                        "text/csv",
                        use_container_width=True,
                    )
            else:
                st.info("No specifications found. Try rephrasing your query or using different keywords.")

        with tab2:
            for i, source in enumerate(sources):
                score = source.get('score', 0)
                score_color = '#34d399' if score > 0.5 else '#fbbf24' if score > 0.3 else '#f87171'
                st.markdown(f"""
                <div class="source-chunk">
                    <div class="source-meta">
                        📖 Page {source.get('page', '?')}
                        &nbsp;•&nbsp; {source.get('section', 'Unknown')}
                        &nbsp;•&nbsp; Score: <span style="color: {score_color}">{score:.4f}</span>
                    </div>
                    {source.get('text', '')[:500]}
                </div>
                """, unsafe_allow_html=True)

        with tab3:
            st.json(result)

elif search_clicked and not query:
    st.warning("Please enter a query first.")
