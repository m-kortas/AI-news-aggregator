# AI-Powered News Aggregation & Chatbot System

A news aggregation platform that uses artificial intelligence to extract, process, and analyse news content from Australian news outlets. The system implements basic NLP techniques, TF-IDF similarity matching, and retrieval-augmented generation (RAG) to provide news summarisation and conversational querying capabilities.

## System Architecture

The system is built on a simple modular architecture with the following components:

- **News Extraction Engine**: Asynchronous RSS feed parsing and content extraction using newspaper3k
- **AI Processing Pipeline**: Basic Gemini API integration for article classification and summarization
- **Duplicate Detection System**: TF-IDF vectorization with cosine similarity for duplicate identification
- **Vector Database**: ChromaDB for semantic search and RAG functionality with Gemini embeddings
- **Conversation Memory**: Simple session-based memory management with configurable retention
- **Web Interface**: Streamlit-based dashboard with basic analytics

## Technical Implementation

### AI and Machine Learning Components

**Natural Language Processing Pipeline**
The system processes news articles through basic AI-powered transformations:

1. **Content Extraction**: Uses newspaper3k library to extract full article content from URLs.

2. **AI-Powered Categorization**: Uses Google Gemini 2.5 Flash Lite model for automatic article classification into four categories: sports, lifestyle, music, and finance. Includes basic fallback mechanisms for API failures.

3. **Summarisation**: Generates 2-3 sentence summaries using Gemini's generative capabilities with content truncation for token optimisation.

4. **Duplicate Detection**: Uses TF-IDF vectorization with cosine similarity:
   - Creates TF-IDF vectors from combined title and summary text
   - Calculates cosine similarity matrices
   - Applies 0.7 similarity threshold for duplicate identification
   - Merges similar articles while preserving source attribution

**Text Processing and Normalization**
The system includes basic text preprocessing:
- Unicode character normalization
- Contraction expansion for better matching
- Possessive form handling for duplicate detection
- Punctuation and whitespace standardisation

### Retrieval-Augmented Generation (RAG) System

**Vector Database Architecture**
The RAG implementation uses ChromaDB with Gemini embeddings:
- **Gemini Embeddings**: Uses Google's embedding-001 model for high-quality semantic representations
- **ChromaDB Storage**: Persistent vector storage with cosine similarity search using latest client configuration
- **Semantic Search**: Advanced retrieval using vector similarity for better context understanding
- **Metadata Preservation**: Includes category, source, URL, frequency, and author information
- **Deduplication**: Semantic similarity detection for identifying duplicate articles
- **Cloud Compatible**: Updated to use new ChromaDB client configuration for better deployment support

**Conversation Memory System**
Implements basic memory management:
- Session-based conversation tracking with unique identifiers
- Configurable message retention (3-20 messages, 1-72 hours)
- Automatic cleanup of expired conversations
- Context-aware response generation using conversation history

**RAG Query Processing**
The system processes user queries through an advanced pipeline:
1. Context retrieval from conversation memory
2. Query embedding generation using Gemini embedding model
3. Semantic search in ChromaDB vector database
4. Context enhancement with relevant article information
5. Response generation using Gemini with retrieved context
6. Source attribution and metadata tracking

### Data Processing and Storage

**Asynchronous Processing**
The system uses Python asyncio for concurrent news extraction:
- Parallel RSS feed parsing across multiple sources
- Concurrent article content extraction
- Non-blocking database operations

**Data Storage**
Hybrid storage approach for optimal performance:
- **Vector Database**: ChromaDB for persistent semantic search and embeddings
- **Session State**: Highlights maintained in memory for UI display
- **Conversation Memory**: In-memory session-based conversation tracking
- **Persistent Embeddings**: Vector database persists between sessions for better performance

**Importance Scoring Algorithm**
Implements a simple scoring system:
- Base score from source frequency (10x multiplier)
- Breaking news keyword detection (+20 points)
- Recency boost for articles within 6 hours (+15 points)
- Top 5 articles per category selection

## News Sources and Coverage

The system aggregates content from 16 Australian news outlets across four categories (4 sources per category):

**Sports Category**
- ABC News Sport RSS (feed/45910)
- Sydney Morning Herald Sport
- The Guardian Australia Sport
- News.com.au Sport

**Lifestyle Category**
- ABC News Main RSS (feed/51120)
- Sydney Morning Herald Lifestyle
- The Guardian Australia Life & Style
- News.com.au Lifestyle

**Music Category**
- Music Feeds
- The Guardian Music (International)
- Triple J Music
- News.com.au Music

**Finance Category**
- Sydney Morning Herald Business
- Australian Financial Review (FeedBurner)
- News.com.au Business
- ABC News Business

## User Interface and Experience

**Streamlit Web Application**
The frontend is built using Streamlit with custom CSS styling:
- Modern design system with Inter font family
- Responsive layout with sidebar configuration
- Pipeline execution with progress indicators
- Basic analytics dashboard with Plotly visualizations

**Analytics Dashboard**
Provides basic insights:
- Category distribution pie charts
- Source frequency analysis
- Summary statistics and metrics
- Sortable article data tables

**Chatbot Interface**
Interactive conversational interface with:
- Message display with custom styling
- Memory status indicators
- Source attribution for responses
- Conversation history management

## Installation and Setup

### Prerequisites
- Python 3.12 or higher
- Google Gemini API key
- Internet connection for news extraction

### Dependencies
The system uses basic dependencies with version constraints:

**Core Libraries**
- pandas, numpy, scikit-learn for data processing
- asyncio for asynchronous operations
- In-memory data structures for optimal performance

**AI and ML Libraries**
- google-generativeai for Gemini API integration and embeddings
- chromadb for vector database operations
- scikit-learn for TF-IDF vectorization and similarity calculations
- Advanced semantic search with vector embeddings

**Web Scraping and Content Extraction**
- feedparser for RSS feed parsing
- newspaper3k for article content extraction
- beautifulsoup4 and requests for web scraping
- lxml for XML/HTML parsing

**User Interface**
- streamlit for web application framework
- plotly for basic visualizations

### Installation Steps

1. Clone the repository and navigate to the project directory
2. Install dependencies using pip: `pip install -r requirements.txt`
3. Obtain a Google Gemini API key from Google AI Studio
4. Set the API key as an environment variable: `export GEMINI_API_KEY="your-key"`
5. Run the application: `streamlit run app.py`

## Usage and Operation

### Web Application
1. Start the Streamlit server: `streamlit run app.py`
2. Navigate to http://localhost:8501 in your browser
3. Enter your Gemini API key in the sidebar
4. Click "Run News Pipeline" to execute the aggregation process
5. Explore news highlights, interact with the chatbot, and view analytics

### Command Line Interface
Execute the pipeline directly: `python simple_news_aggregator.py`

### Pipeline Execution Flow
1. **News Extraction**: Concurrent RSS feed parsing and content extraction
2. **AI Processing**: Article categorization and summarization
3. **Semantic Deduplication**: Vector-based similarity analysis and article merging
4. **Embedding Generation**: Create vector embeddings for all articles
5. **Vector Database Updates**: Store embeddings and metadata in ChromaDB
6. **Highlight Generation**: Importance scoring and ranking
7. **Interface Updates**: UI refresh with new data

## Performance and Scalability

**Optimisation Strategies**
- Asynchronous processing for concurrent operations
- TF-IDF vectorization with limited features (1000) for memory efficiency
- Content truncation for API token usage
- Automatic cleanup of expired conversation data

**Error Handling and Resilience**
- Basic exception handling throughout the pipeline
- Fallback mechanisms for API failures
- Graceful degradation when services are unavailable
- Basic text processing with normalization

**Memory Management**
- Configurable conversation memory limits
- Automatic cleanup of expired sessions
- Persistent vector database with efficient storage
- Optimized memory usage for cloud deployment
