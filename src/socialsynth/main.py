"""
SocialSynth-AI Main Entry Point

This module serves as the main entry point for the SocialSynth-AI application.
It sets up logging and launches the Streamlit UI.
"""

import os
import sys
import logging
from pathlib import Path

# Configure root logger
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("socialsynth.log"),
        logging.StreamHandler(sys.stdout)
    ]
)

logger = logging.getLogger("socialsynth")

def setup_environment():
    """Set up the application environment"""
    # Create required directories
    os.makedirs("static", exist_ok=True)
    os.makedirs("logs", exist_ok=True)
    
    # Ensure .env file exists
    env_path = Path(".env")
    if not env_path.exists():
        logger.warning(".env file not found, creating a template")
        with open(env_path, "w") as f:
            f.write("""# SocialSynth-AI API Keys
# Add your API keys below to enable all features

# YouTube and Google API keys
YOUTUBE_API_KEY=
GOOGLE_API_KEY=

# News API key (https://newsapi.org)
NEWS_API_KEY=

# LLM API keys (only one needed)
OPENAI_API_KEY=
ANTHROPIC_API_KEY=
""")

def launch_streamlit():
    """Launch the Streamlit UI"""
    from .ui.streamlit_app import main
    main()

def run_cli():
    """Run the command-line interface version"""
    print("SocialSynth-AI CLI mode not implemented yet")
    print("Use the Streamlit UI instead by running: streamlit run app.py")

def main():
    """Main entry point function"""
    setup_environment()
    
    # Check if we should run in CLI mode
    if len(sys.argv) > 1 and sys.argv[1] == "--cli":
        run_cli()
    else:
        try:
            import streamlit
            launch_streamlit()
        except ImportError:
            logger.error("Streamlit is not installed. Install it with: pip install streamlit")
            sys.exit(1)

if __name__ == "__main__":
    main() 