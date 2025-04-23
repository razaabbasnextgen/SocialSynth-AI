#!/usr/bin/env python3
"""
SocialSynth-AI Documentation Generator

This script generates a PDF documentation file for the SocialSynth-AI project.
"""

import os
import logging
from datetime import datetime
import sys
import re

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("socialsynth.log"),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger("socialsynth.documentation")

# Check for required dependencies
try:
    from fpdf import FPDF
    import matplotlib.pyplot as plt
    from PIL import Image
except ImportError:
    logger.error("Required dependencies not found. Please install with: pip install fpdf matplotlib pillow")
    logger.info("Installation command: pip install fpdf matplotlib pillow")
    sys.exit(1)

class DocumentationGenerator:
    """Generate PDF documentation for SocialSynth-AI."""
    
    def __init__(self, output_file="SocialSynth-AI_Documentation.pdf"):
        """Initialize the documentation generator."""
        self.output_file = output_file
        
        # Use Unicode font with encoding set to prevent encoding issues
        self.pdf = FPDF()
        self.pdf.set_auto_page_break(auto=True, margin=15)
        self.pdf.add_page()
        
        # Set up fonts - use standard font to avoid encoding issues
        self.pdf.add_font('DejaVu', '', 'DejaVuSansCondensed.ttf', uni=True)
        self.pdf.set_font('Arial', '', 12)
    
    def sanitize_text(self, text):
        """Remove or replace special characters that might cause encoding issues."""
        # Replace em dashes with regular dashes
        text = text.replace('\u2014', '-')
        text = text.replace('\u2013', '-')
        # Replace other potentially problematic characters
        text = text.replace('\u2019', "'")  # Right single quotation mark
        text = text.replace('\u201c', '"')  # Left double quotation mark
        text = text.replace('\u201d', '"')  # Right double quotation mark
        text = text.replace('\u2022', '*')  # Bullet point
        # Remove any remaining non-ASCII characters
        text = re.sub(r'[^\x00-\x7F]+', ' ', text)
        return text
    
    def add_title(self, title, size=20):
        """Add a title to the PDF."""
        title = self.sanitize_text(title)
        self.pdf.set_font('Arial', 'B', size)
        self.pdf.set_text_color(0, 102, 204)  # Blue color
        self.pdf.cell(0, 10, title, ln=True, align='C')
        self.pdf.ln(5)
        
        # Reset font
        self.pdf.set_font('Arial', '', 12)
        self.pdf.set_text_color(0, 0, 0)  # Black color
    
    def add_section_title(self, title, size=16):
        """Add a section title to the PDF."""
        title = self.sanitize_text(title)
        self.pdf.set_font('Arial', 'B', size)
        self.pdf.set_text_color(51, 51, 153)  # Dark blue color
        self.pdf.cell(0, 10, title, ln=True)
        self.pdf.ln(2)
        
        # Reset font
        self.pdf.set_font('Arial', '', 12)
        self.pdf.set_text_color(0, 0, 0)  # Black color
    
    def add_paragraph(self, text):
        """Add a paragraph to the PDF."""
        text = self.sanitize_text(text)
        self.pdf.multi_cell(0, 6, text)
        self.pdf.ln(3)
    
    def add_bullet_points(self, items):
        """Add bullet points to the PDF."""
        for item in items:
            item = self.sanitize_text(item)
            self.pdf.cell(10, 6, "*", ln=0)  # Use * instead of • to avoid encoding issues
            self.pdf.multi_cell(0, 6, item)
    
    def add_image(self, image_path, width=160):
        """Add an image to the PDF."""
        try:
            self.pdf.image(image_path, x=25, w=width)
            self.pdf.ln(5)
        except Exception as e:
            logger.error(f"Error adding image: {e}")
    
    def create_documentation(self):
        """Create the full documentation."""
        try:
            # Title and introduction
            self.add_title("SocialSynth-AI Documentation")
            self.pdf.ln(5)
            
            creation_date = datetime.now().strftime("%Y-%m-%d")
            self.pdf.set_text_color(128, 128, 128)  # Gray
            self.pdf.cell(0, 6, f"Generated on: {creation_date}", ln=True, align='R')
            self.pdf.set_text_color(0, 0, 0)  # Reset to black
            self.pdf.ln(10)
            
            # Description
            self.add_paragraph("SocialSynth-AI is an advanced content generation platform powered by Google's Gemini 2.0 Flash model. It combines retrieval-augmented generation (RAG) with state-of-the-art AI to create engaging, factual scripts for content creators.")
            
            # Try to add logo if available
            if os.path.exists("logo.png"):
                self.add_image("logo.png", width=80)
            
            # RAG section - remove emoji characters
            self.add_section_title("Smart RAG System (Retrieval-Augmented Generation)")
            self.add_paragraph("SocialSynth uses an advanced RAG architecture to generate highly contextual and accurate scripts. It pulls data in real-time from:")
            self.add_bullet_points([
                "News sources",
                "YouTube videos",
                "User-uploaded documents"
            ])
            self.add_paragraph("It then uses Gemini 2.0 Flash, a cutting-edge language model, to synthesize that information into a hyper-contextual script - written as if it were crafted by a professional screenwriter.")
            
            # Multi-Source section - remove emoji characters
            self.add_section_title("Multi-Source Integration for Rich Context")
            self.add_paragraph("Whether the topic is science, politics, or entertainment, SocialSynth ensures that all relevant angles are covered:")
            self.add_bullet_points([
                "Every script is fact-checked",
                "Content is cross-verified",
                "Sources are diverse and relevant"
            ])
            
            # Script styles section - remove emoji characters
            self.add_section_title("Versatile Script Styles")
            self.add_paragraph("SocialSynth doesn't limit you to one format. It can generate scripts in multiple styles including:")
            self.add_bullet_points([
                "News reporting",
                "Documentary narration",
                "Comedic commentary",
                "Dramatic storytelling"
            ])
            
            # Engagement section - remove emoji characters
            self.add_section_title("Creativity + Context = Engagement")
            self.add_paragraph("The end goal? To create scripts that are not only informative but also engaging, hook-based, and designed to retain viewer attention.")
            
            # Requirements section
            self.add_section_title("System Requirements")
            self.add_bullet_points([
                "Python 3.8 or higher",
                "Streamlit",
                "Internet connection",
                "Google API key for Gemini"
            ])
            
            # How to use section
            self.add_section_title("How to Use SocialSynth-AI")
            self.add_paragraph("1. Enter your access token to unlock the application.")
            self.add_paragraph("2. Select the Script Generator tab to create content.")
            self.add_paragraph("3. Enter a topic and customize parameters (tone, length, sources).")
            self.add_paragraph("4. Click 'Generate Script' to create content with Gemini 2.0 Flash.")
            self.add_paragraph("5. Download your script and use it for your content creation.")
            
            # Knowledge Graph section
            self.add_section_title("Enhanced Knowledge Graph")
            self.add_paragraph("SocialSynth-AI now includes an enhanced knowledge graph that helps visualize relationships between concepts in your content:")
            self.add_bullet_points([
                "Automatically extracts key entities and concepts",
                "Shows relationships between different pieces of information",
                "Helps identify important connections for more comprehensive scripts"
            ])
            
            # Save the PDF
            self.pdf.output(self.output_file)
            logger.info(f"Documentation generated successfully: {self.output_file}")
            return True
        except Exception as e:
            logger.error(f"Error generating documentation: {e}")
            import traceback
            logger.error(traceback.format_exc())
            return False

def main():
    """Main function to generate documentation."""
    logger.info("Starting documentation generation")
    generator = DocumentationGenerator()
    success = generator.create_documentation()
    
    if success:
        print(f"Documentation successfully generated: {generator.output_file}")
    else:
        print("Error generating documentation. See log for details.")

if __name__ == "__main__":
    main() 