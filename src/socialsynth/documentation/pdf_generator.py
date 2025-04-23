#!/usr/bin/env python3
"""
PDF Documentation Generator

This module provides functionality to generate PDF documentation from knowledge graphs and content scripts.
It uses ReportLab to create professional-looking PDF documents with appropriate styling and sections.
"""

import os
import logging
from typing import Dict, Any, List, Optional
import networkx as nx
from datetime import datetime
import io
from PIL import Image

try:
    from reportlab.lib.pagesizes import letter
    from reportlab.lib import colors
    from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
    from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image as RLImage
    from reportlab.platypus import PageBreak, ListFlowable, ListItem
    from reportlab.lib.units import inch
except ImportError:
    logging.error("ReportLab not installed. Install with: pip install reportlab")
    raise

# Configure logging
logger = logging.getLogger("socialsynth.documentation.pdf")

class PDFDocumentGenerator:
    """
    Generates PDF documentation from knowledge graphs and content scripts.
    """
    
    def __init__(self, title: str = "SocialSynth AI Documentation"):
        """
        Initialize the PDF document generator.
        
        Args:
            title: The title for the documentation
        """
        self.title = title
        self.styles = getSampleStyleSheet()
        self._setup_styles()
    
    def _setup_styles(self):
        """Set up custom styles for the PDF document."""
        # Title style
        self.styles.add(ParagraphStyle(
            name='CustomTitle',
            parent=self.styles['Title'],
            fontSize=24,
            spaceAfter=24,
            textColor=colors.darkblue
        ))
        
        # Section header style
        self.styles.add(ParagraphStyle(
            name='SectionHeader',
            parent=self.styles['Heading2'],
            fontSize=18,
            spaceAfter=12,
            textColor=colors.darkblue
        ))
        
        # Subsection header style
        self.styles.add(ParagraphStyle(
            name='SubsectionHeader',
            parent=self.styles['Heading3'],
            fontSize=14,
            spaceAfter=8,
            textColor=colors.darkblue
        ))
        
        # Body text style
        self.styles.add(ParagraphStyle(
            name='BodyText',
            parent=self.styles['Normal'],
            fontSize=10,
            leading=14,
            spaceAfter=8
        ))
        
        # Caption style
        self.styles.add(ParagraphStyle(
            name='Caption',
            parent=self.styles['Italic'],
            fontSize=9,
            leading=12,
            alignment=1  # Center
        ))
        
        # Code style
        self.styles.add(ParagraphStyle(
            name='Code',
            parent=self.styles['Code'],
            fontSize=9,
            leading=12,
            fontName='Courier',
            spaceAfter=8
        ))
    
    def generate_pdf(
        self, 
        output_path: str, 
        graph_summary: Dict[str, Any] = None,
        graph_image_path: str = None,
        content_script: str = None,
        search_query: str = None,
        documents: List[Dict[str, Any]] = None
    ) -> str:
        """
        Generate a PDF document containing documentation.
        
        Args:
            output_path: Path to save the PDF file
            graph_summary: Summary of the knowledge graph (optional)
            graph_image_path: Path to the graph visualization image (optional)
            content_script: Generated content script (optional)
            search_query: The original search query (optional)
            documents: List of source documents used (optional)
            
        Returns:
            Path to the generated PDF file
        """
        try:
            # Create a file-like buffer for the PDF
            buffer = io.BytesIO()
            
            # Create the PDF document
            doc = SimpleDocTemplate(
                buffer,
                pagesize=letter,
                rightMargin=72,
                leftMargin=72,
                topMargin=72,
                bottomMargin=72
            )
            
            # Create the content for the PDF
            content = []
            
            # Add title
            content.append(Paragraph(self.title, self.styles['CustomTitle']))
            content.append(Spacer(1, 0.25*inch))
            
            # Add date and time
            now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            content.append(Paragraph(f"Generated on: {now}", self.styles['BodyText']))
            content.append(Spacer(1, 0.25*inch))
            
            # Add search query if provided
            if search_query:
                content.append(Paragraph("Search Query", self.styles['SectionHeader']))
                content.append(Paragraph(f"\"{search_query}\"", self.styles['BodyText']))
                content.append(Spacer(1, 0.25*inch))
            
            # Add knowledge graph section if summary is provided
            if graph_summary:
                content.append(Paragraph("Knowledge Graph", self.styles['SectionHeader']))
                
                # Add graph statistics
                content.append(Paragraph("Graph Statistics:", self.styles['SubsectionHeader']))
                
                stats_data = [
                    ["Metric", "Value"],
                    ["Total Nodes", str(graph_summary.get("node_count", 0))],
                    ["Total Edges", str(graph_summary.get("edge_count", 0))]
                ]
                
                # Add entity type counts
                entity_types = graph_summary.get("entity_types", {})
                for entity_type, count in entity_types.items():
                    stats_data.append([f"{entity_type.title()} Nodes", str(count)])
                
                # Create table for statistics
                stats_table = Table(stats_data, colWidths=[2*inch, 2*inch])
                stats_table.setStyle(TableStyle([
                    ('BACKGROUND', (0, 0), (1, 0), colors.lightgrey),
                    ('TEXTCOLOR', (0, 0), (1, 0), colors.darkblue),
                    ('ALIGN', (0, 0), (1, 0), 'CENTER'),
                    ('ALIGN', (0, 1), (0, -1), 'LEFT'),
                    ('ALIGN', (1, 1), (1, -1), 'RIGHT'),
                    ('FONTNAME', (0, 0), (1, 0), 'Helvetica-Bold'),
                    ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
                    ('GRID', (0, 0), (-1, -1), 0.25, colors.grey),
                ]))
                
                content.append(stats_table)
                content.append(Spacer(1, 0.2*inch))
                
                # Add top keywords if available
                top_keywords = graph_summary.get("top_keywords", [])
                if top_keywords:
                    content.append(Paragraph("Top Keywords:", self.styles['SubsectionHeader']))
                    
                    # Create bullet list of keywords
                    keyword_items = []
                    for keyword in top_keywords[:10]:  # Limit to top 10
                        keyword_items.append(
                            ListItem(Paragraph(keyword, self.styles['BodyText']))
                        )
                    
                    if keyword_items:
                        keyword_list = ListFlowable(
                            keyword_items,
                            bulletType='bullet',
                            start=None,
                            bulletFontName='Helvetica',
                            bulletFontSize=8,
                            leftIndent=20
                        )
                        content.append(keyword_list)
                        content.append(Spacer(1, 0.2*inch))
                
                # Add graph visualization if available
                if graph_image_path and os.path.exists(graph_image_path):
                    try:
                        # Add header for visualization
                        content.append(Paragraph("Graph Visualization:", self.styles['SubsectionHeader']))
                        
                        # Open and resize the image
                        img = Image.open(graph_image_path)
                        img_width, img_height = img.size
                        
                        # Calculate aspect ratio to fit within page width
                        max_width = 6 * inch  # Maximum width (letter page width minus margins)
                        if img_width > max_width:
                            aspect_ratio = img_height / float(img_width)
                            img_width = max_width
                            img_height = max_width * aspect_ratio
                        
                        # Add the image
                        content.append(RLImage(graph_image_path, width=img_width, height=img_height))
                        content.append(Paragraph("Knowledge Graph Visualization", self.styles['Caption']))
                        content.append(Spacer(1, 0.2*inch))
                    except Exception as e:
                        logger.error(f"Error including graph image: {e}")
                
                # Add page break after knowledge graph section
                content.append(PageBreak())
            
            # Add content script section if provided
            if content_script:
                content.append(Paragraph("Generated Content Script", self.styles['SectionHeader']))
                
                # Split the script into paragraphs and add them
                script_paragraphs = content_script.split('\n\n')
                for paragraph in script_paragraphs:
                    paragraph = paragraph.strip()
                    if paragraph:
                        content.append(Paragraph(paragraph, self.styles['BodyText']))
                        content.append(Spacer(1, 0.1*inch))
                
                content.append(PageBreak())
            
            # Add source documents section if provided
            if documents:
                content.append(Paragraph("Source Documents", self.styles['SectionHeader']))
                
                for i, doc in enumerate(documents, 1):
                    # Document title
                    doc_title = doc.get('title', f"Document {i}")
                    content.append(Paragraph(f"{i}. {doc_title}", self.styles['SubsectionHeader']))
                    
                    # Document metadata
                    metadata = []
                    if 'source' in doc:
                        metadata.append(f"Source: {doc['source']}")
                    if 'date' in doc:
                        metadata.append(f"Date: {doc['date']}")
                    
                    if metadata:
                        metadata_text = " | ".join(metadata)
                        content.append(Paragraph(metadata_text, self.styles['Italic']))
                        content.append(Spacer(1, 0.05*inch))
                    
                    # Document summary or excerpt
                    if 'summary' in doc:
                        content.append(Paragraph(doc['summary'], self.styles['BodyText']))
                    elif 'content' in doc:
                        # Show just the first few sentences
                        excerpt = ' '.join(doc['content'].split('.')[:3]) + '...'
                        content.append(Paragraph(excerpt, self.styles['BodyText']))
                    
                    content.append(Spacer(1, 0.15*inch))
            
            # Build the PDF document
            doc.build(content)
            
            # Save the PDF
            with open(output_path, 'wb') as f:
                f.write(buffer.getvalue())
            
            logger.info(f"PDF documentation generated successfully: {output_path}")
            return output_path
        
        except Exception as e:
            logger.error(f"Error generating PDF documentation: {e}")
            return ""

def create_documentation(
    output_path: str,
    title: str = "SocialSynth AI Documentation",
    graph_summary: Dict[str, Any] = None,
    graph_image_path: str = None,
    content_script: str = None,
    search_query: str = None,
    documents: List[Dict[str, Any]] = None
) -> str:
    """
    Convenience function to create PDF documentation.
    
    Args:
        output_path: Path to save the PDF file
        title: The title for the documentation
        graph_summary: Summary of the knowledge graph (optional)
        graph_image_path: Path to the graph visualization image (optional)
        content_script: Generated content script (optional)
        search_query: The original search query (optional)
        documents: List of source documents used (optional)
        
    Returns:
        Path to the generated PDF file
    """
    generator = PDFDocumentGenerator(title=title)
    return generator.generate_pdf(
        output_path=output_path,
        graph_summary=graph_summary,
        graph_image_path=graph_image_path,
        content_script=content_script,
        search_query=search_query,
        documents=documents
    ) 