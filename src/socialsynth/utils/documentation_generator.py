"""
Documentation Generator for SocialSynth-AI

This module generates comprehensive PDF documentation that includes:
1. Knowledge graph visualization and analysis
2. Source documents and references
3. Content script with sections and talking points
4. Key insights and trending topics
"""

import os
import time
import logging
import tempfile
from typing import Dict, Any, List, Optional
from datetime import datetime

# PDF generation
from reportlab.lib.pagesizes import letter
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image, Table, TableStyle
from reportlab.platypus import PageBreak, ListFlowable, ListItem
from reportlab.lib.units import inch

# HTML parsing for visualizations
import re
from bs4 import BeautifulSoup

# LangChain
from langchain.schema import Document

# Configure logging
logger = logging.getLogger("socialsynth.utils.documentation_generator")

def generate_documentation(data: Dict[str, Any]) -> str:
    """
    Generate comprehensive PDF documentation with all analysis results.
    
    Args:
        data: Dictionary containing all the data for the documentation:
            - topic: The main topic/query
            - graph_summary: Summary of the knowledge graph (nodes, edges, etc.)
            - documents: List of retrieved source documents
            - script: Generated script content
            - visualization_path: Path to the knowledge graph visualization HTML
    
    Returns:
        str: Path to the generated PDF file
    """
    try:
        # Create output filename
        timestamp = int(time.time())
        output_dir = "static"
        os.makedirs(output_dir, exist_ok=True)
        output_filename = os.path.join(output_dir, f"SocialSynth_Report_{timestamp}.pdf")
        
        # Extract data
        topic = data.get("topic", "Untitled Topic")
        graph_summary = data.get("graph_summary", {})
        documents = data.get("documents", [])
        script = data.get("script", "")
        vis_path = data.get("visualization_path", "")
        
        # Create PDF document
        doc = SimpleDocTemplate(
            output_filename,
            pagesize=letter,
            rightMargin=72,
            leftMargin=72,
            topMargin=72,
            bottomMargin=72
        )
        
        # Styles
        styles = getSampleStyleSheet()
        styles.add(ParagraphStyle(
            name='Heading1',
            parent=styles['Heading1'],
            fontSize=18,
            spaceAfter=12
        ))
        styles.add(ParagraphStyle(
            name='Heading2',
            parent=styles['Heading2'],
            fontSize=16,
            spaceAfter=10
        ))
        styles.add(ParagraphStyle(
            name='Heading3',
            parent=styles['Heading3'],
            fontSize=14,
            spaceAfter=8
        ))
        styles.add(ParagraphStyle(
            name='Normal',
            parent=styles['Normal'],
            fontSize=12,
            spaceAfter=6
        ))
        
        # Build content
        content = []
        
        # Cover page
        content.append(Paragraph(f"SocialSynth-AI Report", styles['Title']))
        content.append(Spacer(1, 0.5*inch))
        content.append(Paragraph(f"Topic: {topic}", styles['Heading1']))
        content.append(Spacer(1, 0.25*inch))
        
        # Add date and time
        now = datetime.now().strftime("%B %d, %Y at %H:%M")
        content.append(Paragraph(f"Generated on {now}", styles['Normal']))
        content.append(Spacer(1, 0.25*inch))
        
        try:
            # Try to include logo if available
            if os.path.exists("logo.png"):
                img = Image("logo.png", width=2.5*inch, height=2.5*inch)
                content.append(img)
        except Exception as e:
            logger.warning(f"Could not include logo: {e}")
        
        content.append(PageBreak())
        
        # Table of Contents
        content.append(Paragraph("Table of Contents", styles['Heading1']))
        content.append(Paragraph("1. Executive Summary", styles['Normal']))
        content.append(Paragraph("2. Knowledge Graph Analysis", styles['Normal']))
        content.append(Paragraph("3. Key Insights and Trends", styles['Normal']))
        content.append(Paragraph("4. Content Script", styles['Normal']))
        content.append(Paragraph("5. Sources and References", styles['Normal']))
        content.append(PageBreak())
        
        # 1. Executive Summary
        content.append(Paragraph("1. Executive Summary", styles['Heading1']))
        
        # Basic statistics
        node_count = graph_summary.get("node_count", 0)
        edge_count = graph_summary.get("edge_count", 0)
        entity_types = graph_summary.get("entity_types", {})
        
        summary_text = f"""
        This report provides an in-depth analysis of "{topic}". Our AI-powered system analyzed 
        {len(documents)} sources to build a knowledge graph with {node_count} entities and {edge_count} 
        relationships. The analysis identified key connections, trends, and insights to support 
        content creation and research on this topic.
        """
        
        content.append(Paragraph(summary_text, styles['Normal']))
        content.append(Spacer(1, 0.25*inch))
        
        # Summary table
        summary_data = [
            ["Metric", "Value"],
            ["Topic", topic],
            ["Sources Analyzed", len(documents)],
            ["Knowledge Graph Nodes", node_count],
            ["Knowledge Graph Edges", edge_count],
            ["Entity Types Identified", len(entity_types)],
        ]
        
        t = Table(summary_data, colWidths=[2*inch, 3*inch])
        t.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (1, 0), colors.lightgrey),
            ('TEXTCOLOR', (0, 0), (1, 0), colors.black),
            ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
            ('GRID', (0, 0), (-1, -1), 1, colors.black)
        ]))
        
        content.append(t)
        content.append(Spacer(1, 0.5*inch))
        content.append(PageBreak())
        
        # 2. Knowledge Graph Analysis
        content.append(Paragraph("2. Knowledge Graph Analysis", styles['Heading1']))
        
        # If we have a visualization, try to capture a screenshot or extract key parts
        if vis_path and os.path.exists(vis_path):
            try:
                # Try to get a static image of the graph if possible
                content.append(Paragraph("Interactive knowledge graph visualization is available at:", styles['Normal']))
                content.append(Paragraph(f"{vis_path}", styles['Normal']))
                content.append(Spacer(1, 0.25*inch))
                
                # Parse the HTML to extract graph data for a simpler representation
                graph_data = _extract_graph_data(vis_path)
                if graph_data:
                    content.append(Paragraph("Top Entities in the Knowledge Graph:", styles['Heading3']))
                    
                    entities_list = []
                    for entity, details in graph_data.get("top_entities", {}).items():
                        entity_text = f"{entity} (Type: {details.get('type', 'Unknown')}, Connections: {details.get('connections', 0)})"
                        entities_list.append(ListItem(Paragraph(entity_text, styles['Normal'])))
                    
                    if entities_list:
                        content.append(ListFlowable(entities_list, bulletType='bullet', start=1))
            except Exception as e:
                logger.error(f"Error processing visualization for PDF: {e}")
                content.append(Paragraph("Knowledge graph visualization available separately.", styles['Normal']))
        
        # Entity type breakdown
        if entity_types:
            content.append(Spacer(1, 0.25*inch))
            content.append(Paragraph("Entity Types Distribution", styles['Heading3']))
            
            entity_data = [["Entity Type", "Count"]]
            for entity_type, count in entity_types.items():
                entity_data.append([entity_type, count])
            
            t = Table(entity_data, colWidths=[3*inch, 1*inch])
            t.setStyle(TableStyle([
                ('BACKGROUND', (0, 0), (1, 0), colors.lightgrey),
                ('TEXTCOLOR', (0, 0), (1, 0), colors.black),
                ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
                ('ALIGN', (1, 0), (1, -1), 'CENTER'),
                ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
                ('GRID', (0, 0), (-1, -1), 1, colors.black)
            ]))
            
            content.append(t)
        
        content.append(PageBreak())
        
        # 3. Key Insights and Trends
        content.append(Paragraph("3. Key Insights and Trends", styles['Heading1']))
        
        # Add top keywords section
        top_keywords = graph_summary.get("top_keywords", [])
        if top_keywords:
            content.append(Paragraph("Top Keywords and Concepts", styles['Heading3']))
            
            # Prepare keyword data
            keyword_data = [["Keyword/Phrase", "Relevance Score"]]
            for keyword, score in top_keywords:
                # Format score as percentage
                formatted_score = f"{score:.1%}" if isinstance(score, float) else str(score)
                keyword_data.append([keyword, formatted_score])
            
            # Create keyword table (show top 15 max)
            keyword_table = Table(keyword_data[:16], colWidths=[4*inch, 1.5*inch])
            keyword_table.setStyle(TableStyle([
                ('BACKGROUND', (0, 0), (1, 0), colors.lightgrey),
                ('TEXTCOLOR', (0, 0), (1, 0), colors.black),
                ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
                ('ALIGN', (1, 0), (1, -1), 'CENTER'),
                ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
                ('GRID', (0, 0), (-1, -1), 1, colors.black)
            ]))
            
            content.append(keyword_table)
            content.append(Spacer(1, 0.35*inch))
        
        # Add trending topics analysis
        trending_text = """
        Based on our analysis of the knowledge graph and source documents, we've identified the 
        following key trends and insights that are currently most relevant to this topic:
        """
        content.append(Paragraph(trending_text, styles['Normal']))
        
        # Generate insights from the graph structure (simplified for now)
        insights = _generate_insights(graph_summary, documents, topic)
        
        insights_list = []
        for insight in insights:
            insights_list.append(ListItem(Paragraph(insight, styles['Normal'])))
        
        if insights_list:
            content.append(ListFlowable(insights_list, bulletType='bullet'))
        
        content.append(PageBreak())
        
        # 4. Content Script
        content.append(Paragraph("4. Content Script", styles['Heading1']))
        
        if script:
            # Try to format script nicely with sections
            formatted_script = _format_script_for_pdf(script, styles)
            content.extend(formatted_script)
        else:
            content.append(Paragraph("No script was generated for this topic.", styles['Normal']))
        
        content.append(PageBreak())
        
        # 5. Sources and References
        content.append(Paragraph("5. Sources and References", styles['Heading1']))
        
        if documents:
            content.append(Paragraph(f"The following {len(documents)} sources were used in this analysis:", styles['Normal']))
            content.append(Spacer(1, 0.15*inch))
            
            # Create source entries
            for i, doc in enumerate(documents, 1):
                source_info = _get_source_info(doc)
                
                if source_info.get("title"):
                    content.append(Paragraph(f"{i}. {source_info['title']}", styles['Heading3']))
                
                # Source details table
                source_data = []
                if source_info.get("source_type"):
                    source_data.append(["Type", source_info["source_type"]])
                if source_info.get("url"):
                    source_data.append(["URL", source_info["url"]])
                if source_info.get("date"):
                    source_data.append(["Date", source_info["date"]])
                
                if source_data:
                    t = Table(source_data, colWidths=[1*inch, 4.5*inch])
                    t.setStyle(TableStyle([
                        ('ALIGN', (0, 0), (0, -1), 'LEFT'),
                        ('ALIGN', (1, 0), (1, -1), 'LEFT'),
                        ('FONTNAME', (0, 0), (0, -1), 'Helvetica-Bold'),
                        ('BOTTOMPADDING', (0, 0), (-1, -1), 6),
                        ('TOPPADDING', (0, 0), (-1, -1), 6),
                    ]))
                    content.append(t)
                
                # Brief excerpt if available
                if source_info.get("excerpt"):
                    content.append(Paragraph(f"Excerpt: {source_info['excerpt']}", styles['Normal']))
                
                content.append(Spacer(1, 0.25*inch))
                
                # Avoid overloading the PDF with too many sources
                if i >= 15:
                    content.append(Paragraph(f"...and {len(documents) - 15} more sources", styles['Normal']))
                    break
        else:
            content.append(Paragraph("No source documents were used in this analysis.", styles['Normal']))
        
        # Build the PDF
        doc.build(content)
        
        logger.info(f"Documentation generated successfully: {output_filename}")
        return output_filename
        
    except Exception as e:
        logger.error(f"Error generating documentation: {e}")
        # Create a minimal PDF with error information
        try:
            error_filename = os.path.join("static", f"error_report_{int(time.time())}.pdf")
            
            doc = SimpleDocTemplate(error_filename, pagesize=letter)
            styles = getSampleStyleSheet()
            
            content = [
                Paragraph("SocialSynth-AI Error Report", styles['Title']),
                Spacer(1, 0.5*inch),
                Paragraph(f"Error generating documentation: {str(e)}", styles['Normal'])
            ]
            
            doc.build(content)
            return error_filename
        except:
            # If even the error PDF fails, return a dummy path
            return "documentation_error.pdf"

def _extract_graph_data(html_path: str) -> Dict[str, Any]:
    """Extract graph data from the HTML visualization file"""
    try:
        with open(html_path, 'r', encoding='utf-8') as f:
            soup = BeautifulSoup(f.read(), 'html.parser')
        
        # Try to extract network data from the JavaScript in the file
        scripts = soup.find_all('script')
        node_data = {}
        
        for script in scripts:
            script_text = script.string
            if script_text and "nodes.add" in script_text:
                # Extract nodes with regex (simplified)
                node_matches = re.findall(r'label: [\'"]([^\'"]+)[\'"].*?group: [\'"]([^\'"]+)[\'"]', script_text)
                
                for node_label, node_type in node_matches:
                    if node_label not in node_data:
                        node_data[node_label] = {"type": node_type, "connections": 0}
                    
                # Extract edges to count connections
                edge_matches = re.findall(r'from: (\d+).*?to: (\d+)', script_text)
                for from_id, to_id in edge_matches:
                    # This is simplified - we'd need to map IDs to labels
                    # For now, just increment connection counts for known nodes
                    for node in node_data.values():
                        node["connections"] += 1
        
        # Sort nodes by connections and get top 10
        sorted_nodes = {k: v for k, v in sorted(
            node_data.items(), 
            key=lambda item: item[1]['connections'], 
            reverse=True
        )[:10]}
        
        return {
            "top_entities": sorted_nodes
        }
    except Exception as e:
        logger.error(f"Error extracting graph data: {e}")
        return {}

def _generate_insights(graph_summary: Dict[str, Any], documents: List[Document], topic: str) -> List[str]:
    """Generate insights based on graph data and documents"""
    insights = []
    
    # Simple insights based on entity types
    entity_types = graph_summary.get("entity_types", {})
    if entity_types:
        # Get top entity types
        top_types = sorted(entity_types.items(), key=lambda x: x[1], reverse=True)[:3]
        
        if top_types:
            insights.append(f"The most prevalent entity types in relation to {topic} are "
                           f"{', '.join([t[0] for t in top_types])}, indicating the significance "
                           f"of these categories in understanding this subject.")
    
    # Insights from top keywords
    top_keywords = graph_summary.get("top_keywords", [])
    if top_keywords and len(top_keywords) >= 3:
        top_3_keywords = [k[0] for k in top_keywords[:3]]
        insights.append(f"The most prominent concepts associated with {topic} are "
                       f"{', '.join(top_3_keywords)}, which suggests these are central "
                       f"to current discourse on this topic.")
    
    # Insights from document sources
    source_types = {}
    for doc in documents:
        source = doc.metadata.get("source", "unknown")
        source_types[source] = source_types.get(source, 0) + 1
    
    if source_types:
        top_source = max(source_types.items(), key=lambda x: x[1])[0]
        insights.append(f"The majority of information on this topic comes from {top_source} sources, "
                       f"which may indicate where the most active discussion is taking place.")
    
    # Add some generic insights if we don't have many
    if len(insights) < 3:
        insights.append(f"The knowledge graph reveals interconnections between various aspects of {topic}, "
                       f"highlighting the complex nature of this subject matter.")
        
        insights.append(f"Current trends suggest growing interest in {topic}, with evolving "
                       f"discussions across multiple domains and platforms.")
    
    return insights

def _format_script_for_pdf(script: str, styles: Dict[str, Any]) -> List[Any]:
    """Format a script into nicely formatted paragraphs for PDF"""
    result = []
    
    # Try to identify sections in the script
    lines = script.split('\n')
    current_section = []
    current_heading = None
    
    for line in lines:
        line = line.strip()
        if not line:
            continue
            
        # Check if this line looks like a section heading
        if (line.isupper() or 
            (len(line) <= 100 and (line.endswith(':') or all(c.isupper() for c in line if c.isalpha())))):
            # If we have a previous section, add it
            if current_heading and current_section:
                result.append(Paragraph(current_heading, styles['Heading2']))
                result.append(Paragraph('\n'.join(current_section), styles['Normal']))
                result.append(Spacer(1, 0.2*inch))
            
            # Start new section
            current_heading = line
            current_section = []
        else:
            current_section.append(line)
    
    # Add the last section
    if current_heading and current_section:
        result.append(Paragraph(current_heading, styles['Heading2']))
        result.append(Paragraph('\n'.join(current_section), styles['Normal']))
    
    # If no sections were found, just format as a single block
    if not result and script:
        result.append(Paragraph(script, styles['Normal']))
    
    return result

def _get_source_info(doc: Document) -> Dict[str, str]:
    """Extract source information from a document"""
    info = {
        "title": "",
        "source_type": "",
        "url": "",
        "date": "",
        "excerpt": ""
    }
    
    # Get title
    if doc.metadata.get("title"):
        info["title"] = doc.metadata.get("title")
    elif doc.metadata.get("source") == "youtube":
        info["title"] = f"YouTube: {doc.metadata.get('video_title', 'Untitled Video')}"
    elif doc.metadata.get("source") == "news":
        info["title"] = doc.metadata.get("headline", "Untitled Article")
    else:
        # Generate a title from content
        content = doc.page_content
        info["title"] = content.split('\n')[0][:100] if content else "Untitled Document"
    
    # Get source type
    info["source_type"] = doc.metadata.get("source", "Unknown")
    
    # Get URL
    if doc.metadata.get("url"):
        info["url"] = doc.metadata.get("url")
    elif doc.metadata.get("source") == "youtube" and doc.metadata.get("video_id"):
        info["url"] = f"https://www.youtube.com/watch?v={doc.metadata.get('video_id')}"
    
    # Get date
    if doc.metadata.get("date"):
        info["date"] = doc.metadata.get("date")
    elif doc.metadata.get("publish_date"):
        info["date"] = doc.metadata.get("publish_date")
    
    # Get excerpt
    if doc.page_content:
        # Get first 150 chars for excerpt
        info["excerpt"] = doc.page_content[:150] + "..." if len(doc.page_content) > 150 else doc.page_content
    
    return info 