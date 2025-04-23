# SocialSynth-AI

![SocialSynth-AI](logo.png)

SocialSynth-AI is an advanced content generation tool leveraging knowledge graphs and AI to create high-quality scripts for social media and other content platforms.

## Features

- **Knowledge Graph Enhanced Generation**: Automatically extracts entities and relationships from content to build a knowledge graph that guides the generation process
- **Multi-Source Content Retrieval**: Retrieves relevant content from YouTube, news sites, and blogs to enhance generated content
- **Customizable Output**: Configure tone, target audience, content type, and other parameters to tailor the output to your needs
- **Interactive Visualizations**: Visualize the knowledge graph to understand how information is connected
- **Multiple Input Options**: Generate content from topics/keywords or uploaded documents in various formats (PDF, DOCX, TXT)

## Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/SocialSynth-AI.git
cd SocialSynth-AI
```

2. Create a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Set up API keys:
Create a `.env` file in the root directory with the following keys:
```
YOUTUBE_API_KEY=your_youtube_api_key
NEWS_API_KEY=your_news_api_key
BLOG_SEARCH_API_KEY=your_google_custom_search_api_key
BLOG_SEARCH_CX=your_google_custom_search_cx
GOOGLE_API_KEY=your_google_api_key
```

5. Run the application:
```bash
streamlit run src/socialsynth/ui/streamlit_app.py
```

## Usage

1. **Choose Input Method**: Select "Topic/Keyword", "Upload Documents", or both
2. **Configure Content Settings**: Choose content type, tone, target audience, and other options
3. **Generate**: Click the "Generate Script" button to create your content
4. **View Results**: The generated script and knowledge graph visualization will be displayed
5. **Download**: Download the generated script for your use

## Project Structure

```
SocialSynth-AI/
│
├── src/                         # Source code
│   └── socialsynth/            
│       ├── knowledge_graph/     # Knowledge graph building components
│       ├── retrieval/           # Content retrieval components
│       ├── generation/          # Script generation components
│       ├── utils/               # Utility functions
│       ├── documentation/       # Documentation generation
│       └── ui/                  # Streamlit UI
│
├── output/                      # Generated content and visualizations
├── demo_enhanced_rag.py         # Demo for enhanced RAG functionality
├── simple_demo.py               # Simple demo for knowledge graph
├── requirements.txt             # Project dependencies
└── README.md                    # Project documentation
```

## Requirements

- Python 3.8+
- Libraries: streamlit, networkx, pyvis, langchain, spaCy, and others (see requirements.txt)
- API keys for external data sources (optional)

## License

MIT License

## Credits

Created by Raza Abbas
