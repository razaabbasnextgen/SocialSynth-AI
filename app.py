#!/usr/bin/env python3
"""
SocialSynth-AI Launcher

This script launches the SocialSynth-AI Streamlit application with the correct paths set up.
"""

import os
import sys
import subprocess
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('socialsynth')

def main():
    """Launch the Streamlit application."""
    logger.info("Starting SocialSynth-AI application")
    
    # Add the project root to Python path to ensure imports work correctly
    project_root = os.path.dirname(os.path.abspath(__file__))
    if project_root not in sys.path:
        sys.path.insert(0, project_root)
        logger.info(f"Added {project_root} to Python path")
    
    # Path to the Streamlit app
    app_path = os.path.join(project_root, "src", "socialsynth", "ui", "streamlit_app.py")
    
    # Check if the file exists
    if not os.path.exists(app_path):
        logger.error(f"Streamlit app not found at {app_path}")
        print(f"Error: Streamlit app not found at {app_path}")
        return 1
    
    # Set environment variable for Python path
    env = os.environ.copy()
    if "PYTHONPATH" in env:
        # Use os.pathsep for platform-specific path separator (: on Unix, ; on Windows)
        env["PYTHONPATH"] = f"{project_root}{os.pathsep}{env['PYTHONPATH']}"
    else:
        env["PYTHONPATH"] = project_root
    
    try:
        # Launch the Streamlit app
        logger.info(f"Launching Streamlit app from {app_path}")
        result = subprocess.run(
            ["streamlit", "run", app_path],
            env=env,
            check=True
        )
        return result.returncode
    except subprocess.CalledProcessError as e:
        logger.error(f"Error running Streamlit app: {e}")
        print(f"Error running Streamlit app: {e}")
        return e.returncode
    except FileNotFoundError:
        logger.error("Streamlit not found. Please ensure it is installed.")
        print("Error: Streamlit not found. Please ensure it is installed with 'pip install streamlit'.")
        return 1

if __name__ == "__main__":
    sys.exit(main()) 