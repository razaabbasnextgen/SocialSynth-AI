# SocialSynth-AI Visualization Fix

## Overview of Changes

The enhanced knowledge graph visualization in SocialSynth-AI has been fixed to address some issues with saving visualizations and compatibility problems. The following changes were made:

1. **Enhanced Error Handling in Visualization Code**
   - Added better error logging with full tracebacks
   - Implemented multiple fallback mechanisms for saving visualizations
   - Ensured output directories exist before saving files

2. **Updated LangChain Dependencies**
   - Fixed compatibility issues by downgrading to stable versions:
     - langchain==0.0.354
     - langchain-core==0.1.6
     - langchain-community==0.0.8

3. **Added Asyncio Compatibility Layer**
   - Installed nest-asyncio to fix asyncio-related errors
   - This prevents "RuntimeError: asyncio.run() cannot be called from a running event loop" errors

## Demo Scripts

Two demonstration scripts were created to test and showcase the visualization capabilities:

1. **test_visualization.py** - A basic test script that creates a simple graph and saves it
2. **enhanced_demo.py** - A more comprehensive demo that:
   - Creates sample documents
   - Builds a knowledge graph
   - Extracts entities and relationships
   - Visualizes the graph with the enhanced adapter
   - Shows centrality metrics and other graph statistics

## How to Run the Demos

```bash
# Install nest-asyncio if not already installed
pip install nest-asyncio

# Run the basic test
python test_visualization.py

# Run the enhanced demo
python enhanced_demo.py
```

The visualizations will be saved in the `static` directory with HTML files that can be opened in any web browser.

## Integrating with Jupyter Notebooks

When using the visualization code in Jupyter notebooks or similar environments, make sure to add the following code at the beginning of your notebook:

```python
import nest_asyncio
nest_asyncio.apply()
```

This will prevent asyncio-related errors when running code with event loops in Jupyter.

## Known Issues and Limitations

1. The knowledge graph visualization may be slow for very large graphs (more than a few hundred nodes).

2. Some LangChain integrations (like Google Generative AI) may show compatibility warnings with the downgraded LangChain versions. These warnings can generally be ignored as long as the functionality works.

3. The script does not yet optimize the visualization parameters for very large or very small graphs. You may need to adjust parameters like `height`, `width`, and physics settings for optimal display.

## Next Steps

1. Consider upgrading to more recent LangChain versions once the compatibility issues are resolved in the upstream libraries.

2. Explore alternative visualization libraries for larger graphs if performance becomes an issue. 