"""
Simplified Streamlit UI for News Aggregation & Chatbot
"""

__import__('pysqlite3')
import sys
sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')

import streamlit as st
import asyncio
import pandas as pd
import plotly.express as px
from datetime import datetime
import os

from simple_news_aggregator import NewsAggregator, run_news_pipeline

# Page configuration
st.set_page_config(
    page_title="AI News Aggregator",
    page_icon=None,
    layout="wide",
    initial_sidebar_state="expanded"
)

# Professional CSS styling
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');

/* Base styling */
body, .stApp {
    font-family: 'Inter', sans-serif !important;
    background: #f8fafc;
    color: #1e293b;
    line-height: 1.6;
}

/* Main container */
.main .block-container {
    padding-top: 2rem;
    padding-bottom: 2rem;
    max-width: 1200px;
}

/* Headers with better typography */
h1, h2, h3, h4, h5, h6 {
    font-family: 'Inter', sans-serif !important;
    color: #1e293b;
    font-weight: 600;
    margin-bottom: 1rem;
}

h1 {
    font-size: 2.5rem;
    font-weight: 700;
    margin-bottom: 1.5rem;
}

h2 {
    font-size: 1.875rem;
    font-weight: 600;
    margin-bottom: 1.25rem;
}

h3 {
    font-size: 1.5rem;
    font-weight: 600;
    margin-bottom: 1rem;
}

/* Links */
a {
    color: #3b82f6;
    text-decoration: none;
    transition: color 0.2s ease;
}

a:hover {
    color: #1d4ed8;
    text-decoration: underline;
}

/* Sidebar styling */
[data-testid="stSidebar"] {
    background: #ffffff;
    border-right: 1px solid #e2e8f0;
    box-shadow: 0 4px 6px rgba(0, 0, 0, 0.05);
}

[data-testid="stSidebar"] .stMarkdown {
    color: #1e293b;
}

[data-testid="stSidebar"] h1,
[data-testid="stSidebar"] h2,
[data-testid="stSidebar"] h3,
[data-testid="stSidebar"] h4,
[data-testid="stSidebar"] h5,
[data-testid="stSidebar"] h6 {
    color: #1e293b;
}

/* Enhanced buttons */
.stButton > button {
    font-family: 'Inter', sans-serif !important;
    border-radius: 12px;
    font-weight: 500;
    padding: 0.5rem 1.5rem;
    transition: all 0.3s ease;
    border: none;
    box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
}

/* Primary buttons */
.stButton > button:first-child,
.stButton > button[kind="primary"] {
    background: #3b82f6;
    color: white;
    border: none;
}

.stButton > button:first-child:hover,
.stButton > button[kind="primary"]:hover {
    background: #2563eb;
    transform: translateY(-1px);
    box-shadow: 0 4px 8px rgba(59, 130, 246, 0.3);
}

/* Secondary buttons */
.stButton > button:not(:first-child):not([kind="primary"]),
.stButton > button[kind="secondary"] {
    background: #f1f5f9;
    color: #475569;
    border: 1px solid #cbd5e1;
}

.stButton > button:not(:first-child):not([kind="primary"]):hover,
.stButton > button[kind="secondary"]:hover {
    background: #e2e8f0;
    transform: translateY(-1px);
    box-shadow: 0 2px 4px rgba(0, 0, 0, 0.1);
}

/* Input fields */
.stTextInput>div>div>input,
.stTextArea>div>div>textarea,
.stSelectbox>div>div>select {
    font-family: 'Inter', sans-serif !important;
    border: 1px solid #cbd5e1;
    background: #ffffff;
    border-radius: 8px;
    color: #1e293b;
    padding: 0.75rem 1rem;
    transition: all 0.2s ease;
}

.stTextInput>div>div>input:focus,
.stTextArea>div>div>textarea:focus,
.stSelectbox>div>div>select:focus {
    border-color: #3b82f6;
    background: #ffffff;
    box-shadow: 0 0 0 3px rgba(59, 130, 246, 0.1);
    outline: none;
}

.stTextInput>div>div>input::placeholder,
.stTextArea>div>div>textarea::placeholder {
    color: #64748b;
}

/* Tabs */
[data-baseweb="tab-list"] {
    background: #f1f5f9;
    border-radius: 8px;
    padding: 0.25rem;
    margin-bottom: 2rem;
    border: 1px solid #e2e8f0;
}

[data-baseweb="tab"] {
    font-family: 'Inter', sans-serif !important;
    color: #64748b;
    padding: 0.75rem 1.5rem;
    border-radius: 6px;
    margin: 0 0.125rem;
    transition: all 0.2s ease;
    font-weight: 500;
}

[data-baseweb="tab"][aria-selected="true"] {
    color: #1e293b;
    background: #ffffff;
    font-weight: 600;
    box-shadow: 0 1px 3px rgba(0, 0, 0, 0.1);
}

[data-baseweb="tab"]:hover {
    background: #e2e8f0;
    color: #1e293b;
}

/* Enhanced news cards */
.news-card {
    background: #ffffff;
    padding: 1.5rem;
    border-radius: 12px;
    margin: 1rem 0;
    border: 1px solid #e2e8f0;
    box-shadow: 0 1px 3px rgba(0, 0, 0, 0.1);
    transition: all 0.2s ease;
    position: relative;
    overflow: hidden;
}

.news-card::before {
    content: '';
    position: absolute;
    top: 0;
    left: 0;
    right: 0;
    height: 3px;
    background: #3b82f6;
}

.news-card:hover {
    transform: translateY(-2px);
    box-shadow: 0 4px 12px rgba(0, 0, 0, 0.15);
}

.news-card h4 {
    color: #1f2937;
    font-weight: 600;
    margin-bottom: 1rem;
    font-size: 1.25rem;
    line-height: 1.4;
}

.news-card p {
    color: #6b7280;
    margin-bottom: 0.5rem;
    font-size: 0.875rem;
}

.news-card .summary {
    color: #374151;
    line-height: 1.6;
    margin: 1rem 0;
}

/* Enhanced frequency badge */
.frequency-badge {
    background: #f1f5f9;
    color: #475569;
    padding: 0.25rem 0.75rem;
    border-radius: 12px;
    font-size: 0.75rem;
    font-weight: 500;
    margin-right: 0.5rem;
    display: inline-block;
    border: 1px solid #cbd5e1;
}

/* Enhanced chat messages */
.user-question {
    background: #3b82f6;
    color: #ffffff;
    padding: 1rem 1.5rem;
    border-radius: 18px 18px 4px 18px;
    margin-bottom: 1rem;
    max-width: 80%;
    align-self: flex-end;
    float: right;
    clear: both;
    margin-left: auto;
    margin-right: 0;
    font-family: 'Inter', sans-serif;
    font-weight: 500;
    box-shadow: 0 2px 8px rgba(59, 130, 246, 0.2);
}

.assistant-answer {
    background: #ffffff;
    color: #374151;
    padding: 1rem 1.5rem;
    border-radius: 18px 18px 18px 4px;
    margin-bottom: 1.5rem;
    border: 1px solid #e2e8f0;
    max-width: 80%;
    align-self: flex-start;
    float: left;
    clear: both;
    margin-right: auto;
    margin-left: 0;
    font-family: 'Inter', sans-serif;
    box-shadow: 0 2px 8px rgba(0, 0, 0, 0.1);
}

/* Enhanced metrics */
[data-testid="metric-container"] {
    background: #ffffff;
    border: 1px solid #e2e8f0;
    padding: 1.5rem;
    border-radius: 12px;
    box-shadow: 0 1px 3px rgba(0, 0, 0, 0.1);
    transition: all 0.2s ease;
}

[data-testid="metric-container"]:hover {
    transform: translateY(-1px);
    box-shadow: 0 4px 12px rgba(0, 0, 0, 0.15);
}

/* Main title with enhanced gradient */
.main-title {
    text-align: center;
    color: #1e293b;
    margin-bottom: 2rem;
    font-weight: 700;
    font-size: 2.5rem;
    letter-spacing: -0.02em;
}

/* Attribution styling */
.attribution {
    text-align: center;
    margin-bottom: 2rem;
    color: #64748b;
    font-size: 0.875rem;
    font-weight: 500;
}

.attribution a {
    color: #3b82f6;
    text-decoration: none;
    font-weight: 600;
    transition: color 0.2s ease;
}

.attribution a:hover {
    color: #1d4ed8;
    text-decoration: underline;
}

/* Info boxes */
.stInfo {
    background: #eff6ff;
    border: 1px solid #bfdbfe;
    border-radius: 8px;
}

.stSuccess {
    background: #f0fdf4;
    border: 1px solid #bbf7d0;
    border-radius: 8px;
}

.stWarning {
    background: #fffbeb;
    border: 1px solid #fed7aa;
    border-radius: 8px;
}

.stError {
    background: #fef2f2;
    border: 1px solid #fecaca;
    border-radius: 8px;
}

/* Hide Streamlit default elements */
div[data-testid="stHeader"],
div[data-testid="stFooter"] {
    display: none;
}

/* Loading spinner */
.stSpinner {
    color: #3b82f6;
}

/* Selectbox styling */
.stSelectbox > div > div {
    background: #ffffff;
    border-radius: 8px;
    color: #1e293b;
}

/* Slider styling */
.stSlider > div > div > div {
    background: #3b82f6;
}

/* Progress bar */
.stProgress > div > div > div {
    background: #3b82f6;
}

/* Dataframe styling */
.stDataFrame {
    background: #ffffff;
    border-radius: 8px;
    border: 1px solid #e2e8f0;
    color: #1e293b;
}

/* Plotly charts */
.js-plotly-plot {
    background: #ffffff;
    border-radius: 8px;
    border: 1px solid #e2e8f0;
}
</style>
""", unsafe_allow_html=True)

def main():
    """Main application"""
    
    # Title
    st.markdown('<h1 class="main-title">AI News Aggregator</h1>', unsafe_allow_html=True)
    
    # Attribution
    st.markdown('<div class="attribution">Built by <a href="https://www.linkedin.com/in/mkortas/" target="_blank">Magda Kortas</a></div>', unsafe_allow_html=True)
    
    try:
        api_key = st.secrets["GEMINI_API_KEY"]
    except KeyError:
        st.error(" **Configuration Error**: Gemini API key not found in Streamlit secrets.")
        st.stop()
    
    # Sidebar for controls
    with st.sidebar:
        # Pipeline trigger
        if st.button("Run News Pipeline", type="primary"):
            run_pipeline()
        
        st.markdown("---")
        
        # Quick stats
        if 'highlights' in st.session_state:
            st.markdown("### Quick Stats")
            total = sum(len(articles) for articles in st.session_state.highlights.values())
            st.metric("Total Highlights", total)
            st.metric("Categories", len(st.session_state.highlights))
            
            # Show memory stats if available
            if 'session_id' in st.session_state and 'aggregator' in st.session_state:
                history = st.session_state.aggregator.get_conversation_history(st.session_state.session_id)
                if history:
                    st.metric("Conversation Memory", f"{len(history)} exchanges")
                
    
    # Main content tabs
    tab1, tab2, tab3, tab4 = st.tabs(["News Highlights", "Chat with News", "Analytics", "About"])
    
    with tab1:
        show_news_highlights()
    
    with tab2:
        show_chatbot()
    
    with tab3:
        show_analytics()
    
    with tab4:
        show_about()

def run_pipeline():
    """Run the news aggregation pipeline"""
    api_key = st.secrets["GEMINI_API_KEY"]
    
    with st.spinner("Running news aggregation pipeline..."):
        try:
            # Run the pipeline
            highlights = asyncio.run(run_news_pipeline(api_key))
            
            # Store in session state
            st.session_state.highlights = highlights
            st.session_state.aggregator = NewsAggregator(api_key)
            
            # Update the chatbot database with the new highlights
            st.session_state.aggregator.update_chatbot_database(highlights)
            
            total = sum(len(articles) for articles in highlights.values())
            st.success(f"Pipeline completed! Generated {total} highlights")
            
            # Debug: Show that RAG data is loaded
            if st.session_state.aggregator.has_news_data():
                try:
                    doc_count = st.session_state.aggregator.news_collection.count()
                    st.info(f" RAG system loaded with {doc_count} documents")
                except:
                    st.info(" RAG system loaded")
            else:
                st.warning(" RAG system not loaded properly")
            st.rerun()
            
        except Exception as e:
            st.error(f"Pipeline failed: {str(e)}")

def show_news_highlights():
    """Display news highlights"""
    if 'highlights' not in st.session_state:
        st.info("Welcome! Click 'Run News Pipeline' in the sidebar to get started.")
        return
    
    highlights = st.session_state.highlights
    
    # Category filter
    categories = list(highlights.keys())
    selected_category = st.selectbox(
        "Filter by category:",
        ["All"] + categories,
        index=0
    )
    
    # Display highlights
    if selected_category == "All":
        categories_to_show = categories
    else:
        categories_to_show = [selected_category]
    
    for category in categories_to_show:
        if category not in highlights:
            continue
            
        articles = highlights[category]
        if not articles:
            continue
        
        st.subheader(f"{category.title()} ({len(articles)} articles)")
        
        for article in articles:
            with st.container():
                st.markdown(f"""
                <div class="news-card">
                    <h4>{article.title}</h4>
                    <p>
                        <strong>Source:</strong> {article.source} | <strong>Author:</strong> {article.author}
                    </p>
                    <p class="summary">{article.summary}</p>
                    <div style="margin-top: 1.5rem; display: flex; align-items: center; justify-content: space-between;">
                        <span class="frequency-badge">{article.frequency} source(s)</span>
                        <a href="{article.url}" target="_blank" style="color: #4F46E5; text-decoration: none; font-weight: 600; padding: 0.5rem 1rem; background: rgba(79, 70, 229, 0.1); border-radius: 8px; transition: all 0.3s ease;">
                            Read full article →
                        </a>
                    </div>
                </div>
                """, unsafe_allow_html=True)

def show_chatbot():
    """Display chatbot interface"""
    st.subheader("Ask About Today's News")
    
    if 'aggregator' not in st.session_state:
        st.warning("Please run the news pipeline first to enable the chatbot.")
        return
    
    # Check if we have news data
    if not st.session_state.aggregator.has_news_data():
        st.error("No news data available. Please run the news pipeline first to get the latest news data.")
        st.info("Click 'Run News Pipeline' in the sidebar to start the aggregation process.")
        return
    
    
    # Initialize session ID for memory
    if 'session_id' not in st.session_state:
        st.session_state.session_id = f"user_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    
    # Initialize chat history
    if 'chat_history' not in st.session_state:
        st.session_state.chat_history = []
    
    # Display chat history
    for chat in st.session_state.chat_history:
        st.markdown(f"""
        <div class="user-question">
            <strong>You:</strong> {chat['question']}
        </div>
        <div class="assistant-answer">
            <strong>AI:</strong> {chat['response']}
        </div>
        """, unsafe_allow_html=True)
    
    # Chat input
    with st.form("chat_form"):
        question = st.text_input(
            "Ask me anything about today's news...",
            placeholder="e.g., What are the main sports stories today?"
        )
        
        col1, col2, col3 = st.columns([1, 1, 3])
        with col1:
            submitted = st.form_submit_button("Send")
        with col2:
            if st.form_submit_button("Clear Chat"):
                st.session_state.chat_history = []
                # Also clear the memory
                st.session_state.aggregator.clear_conversation_history(st.session_state.session_id)
                st.rerun()
        with col3:
            # Show memory status
            history = st.session_state.aggregator.get_conversation_history(st.session_state.session_id)
            if history:
                st.caption(f"Memory: {len(history)} previous exchanges")
    
    if submitted and question:
        with st.spinner("Thinking..."):
            try:
                # Use the enhanced chat method with memory
                response = st.session_state.aggregator.chat_about_news(
                    question, 
                    session_id=st.session_state.session_id,
                    include_memory=True
                )
                
                # Add to history
                st.session_state.chat_history.append({
                    'question': question,
                    'response': response
                })
                
                st.rerun()
                
            except Exception as e:
                st.error(f"Chat error: {str(e)}")

def show_analytics():
    """Display simple analytics"""
    if 'highlights' not in st.session_state:
        st.info("No data available. Run the pipeline first to see analytics.")
        return
    
    highlights = st.session_state.highlights
    
    st.subheader("News Analytics")
    
    # Prepare data
    data = []
    for category, articles in highlights.items():
        for article in articles:
            data.append({
                'Category': category.title(),
                'Title': article.title,
                'Source': article.source,
                'Frequency': article.frequency,
                'Importance': getattr(article, 'importance_score', 0)
            })
    
    if not data:
        st.info("No data to display.")
        return
    
    df = pd.DataFrame(data)
    
    # Visualizations
    col1, col2 = st.columns(2)
    
    # Average frequency by category
    avg_freq = df.groupby('Category')['Frequency'].mean().reset_index()
    fig_bar = px.bar(
        avg_freq,
        x='Category',
        y='Frequency',
        title="Average Source Frequency by Category",
        color='Frequency',
        color_continuous_scale='Blues'
    )
    fig_bar.update_layout(
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        font=dict(color='#1e293b')
    )
    st.plotly_chart(fig_bar, width='stretch')
    
    # Summary stats
    st.subheader("Summary Statistics")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Total Articles", len(df))
    with col2:
        st.metric("Unique Sources", df['Source'].nunique())
    with col3:
        st.metric("Avg Frequency", f"{df['Frequency'].mean():.1f}")
    
    # Data table
    st.subheader("Article Details")
    st.dataframe(df, width='stretch')

def show_about():
    """Display About page with system description"""
    st.subheader("About This Application")
    
    st.markdown("""
    ## AI-Powered News Aggregation & Chatbot System
    
    This is a news aggregation platform that uses artificial intelligence to extract, process, and analyse news content from Australian news outlets. The system implements natural language processing techniques, TF-IDF similarity matching, and retrieval-augmented generation (RAG) to provide news summarisation and conversational querying capabilities.
    """)
    
    st.markdown("### System Architecture")
    
    st.markdown("""
    The system is built on a modular architecture with the following components:
    
    - **News Extraction Engine**: Asynchronous RSS feed parsing and content extraction using newspaper3k
    - **AI Processing Pipeline**: Google Gemini API integration for article classification and summarization
    - **Duplicate Detection System**: TF-IDF vectorization with cosine similarity for duplicate identification
    - **Vector Database**: ChromaDB for semantic search and RAG functionality with Gemini embeddings
    - **Conversation Memory**: Session-based memory management with configurable retention
    - **Web Interface**: Streamlit-based dashboard with interactive analytics
    """)
    
    st.markdown("### AI and Machine Learning Components")
    
    st.markdown("""
    **Natural Language Processing Pipeline**
    
    The system processes news articles through an AI-powered transformation pipeline:
    
    1. **Content Extraction**: Uses newspaper3k library to extract full article content from URLs
    2. **AI-Powered Categorization**: Uses Google Gemini 2.5 Flash Lite model for automatic article classification into four categories: sports, lifestyle, music, and finance
    3. **Summarisation**: Generates concise 2-3 sentence summaries using Gemini's generative capabilities
    4. **Duplicate Detection**: 
       - Creates TF-IDF vectors from combined title and summary text
       - Calculates cosine similarity matrices
       - Applies 0.7 similarity threshold for duplicate identification
       - Merges similar articles while preserving source attribution
    """)
    
    st.markdown("### Retrieval-Augmented Generation (RAG) System")
    
    st.markdown("""
    **Vector Database Architecture**
    
    The RAG implementation uses ChromaDB with Gemini embeddings:
    - **Gemini Embeddings**: Uses Google's embedding-001 model for high-quality semantic representations
    - **ChromaDB Storage**: Persistent vector storage with cosine similarity search using latest client configuration
    - **Semantic Search**: Advanced retrieval using vector similarity for better context understanding
    - **Metadata Preservation**: Includes category, source, URL, frequency, and author information
    - **Deduplication**: Semantic similarity detection for identifying duplicate articles
    - **Cloud Compatible**: Updated to use new ChromaDB client configuration for better deployment support
    
    **Why ChromaDB with Gemini Embeddings?**
    - **Superior Semantic Understanding**: Gemini embeddings provide better context comprehension
    - **Persistent Storage**: Vector database persists between sessions for better performance
    - **Advanced Deduplication**: Semantic similarity detection for more accurate duplicate identification
    - **Scalable Architecture**: ChromaDB handles large-scale vector operations efficiently
    - **Production Ready**: Robust vector database solution for real-world applications
    
    **Conversation Memory System**
    - Session-based conversation tracking with unique identifiers
    - Context-aware response generation using conversation history
    - Automatic cleanup of expired conversations
    """)
    
    st.markdown("### Data Processing and Storage")
    
    st.markdown("""
    **Asynchronous Processing**
    - Parallel RSS feed parsing across multiple sources
    - Concurrent article content extraction using Python asyncio
    - Non-blocking in-memory operations for better performance
    
    **Data Storage**
    - **Vector Database**: ChromaDB for persistent semantic search and embeddings
    - **Session State**: Highlights maintained in memory for UI display
    - **Conversation Memory**: In-memory session-based conversation tracking
    - **Persistent Embeddings**: Vector database persists between sessions for better performance
    
    **Importance Scoring Algorithm**
    - Base score from source frequency (10x multiplier)
    - Breaking news keyword detection (+20 points)
    - Recency boost for articles within 6 hours (+15 points)
    - Top 5 articles per category selection
    """)
    
    st.markdown("### News Sources and Coverage")
    
    st.markdown("""
    The system aggregates content from Australian news outlets across four categories:
    
    - **Sports**: ABC News Sport, Sydney Morning Herald Sport, The Guardian Australia Sport
    - **Lifestyle**: ABC News Main, Sydney Morning Herald Lifestyle, The Guardian Australia Life & Style
    - **Music**: ABC News Main, Music Feeds, The Guardian Music
    - **Finance**: ABC News Main, Sydney Morning Herald Business, Australian Financial Review
    """)
    
    st.markdown("### Technical Features")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        **AI Engineering Features:**
        - Google Gemini 2.5 Flash Lite integration
        - TF-IDF vectorization and similarity search
        - Conversation memory and context awareness
        - Duplicate detection
        - Asynchronous processing pipeline
        """)
    
    with col2:
        st.markdown("""
        **Performance Optimisations:**
        - Concurrent RSS feed processing
        - In-memory vector search
        - Automatic cleanup and memory management
        - Real-time analytics and metrics
        - Responsive UI design
        """)
    
    st.markdown("### Technology Stack")
    
    st.markdown("""
    **Core Technologies:**
    - **Backend**: Python 3.12+, asyncio, ChromaDB vector database
    - **AI/ML**: Google Gemini API, scikit-learn, TF-IDF vectorization
    - **Web Scraping**: feedparser, newspaper3k, BeautifulSoup
    - **Frontend**: Streamlit, Plotly, custom CSS
    - **Deployment**: Streamlit Cloud with secrets management
    """)
    
    st.markdown("### Getting Started")
    
    st.markdown("""
    1. Click "Run News Pipeline" in the sidebar to start the aggregation process
    2. Explore News Highlights to see categorised and summarised articles
    3. Chat with the AI to ask questions about today's news
    4. View Analytics to understand news patterns and trends
    
    The system automatically handles API key management through Streamlit secrets for secure deployment.
    """)
    
    st.markdown("---")
    st.markdown("""
    <div style="text-align: center; color: #6B7280; font-size: 0.875rem;">
        Built by <a href="https://www.linkedin.com/in/mkortas/" target="_blank" style="color: #4F46E5; text-decoration: none;">Magda Kortas</a> | 
        A demonstration of AI engineering practices
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
