"""
Document parsers for different file formats
"""

import io
import re
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import List, Dict, Any, Optional, Union
from pathlib import Path

import PyPDF2
import pdfplumber
from bs4 import BeautifulSoup
from PIL import Image
import docx
from loguru import logger


@dataclass
class ParsedElement:
    """Represents a parsed document element"""
    element_type: str  # paragraph, table, heading, figure, metadata
    content: str
    metadata: Dict[str, Any]
    page_num: Optional[int] = None
    bbox: Optional[tuple] = None  # (x0, y0, x1, y1)
    parent_id: Optional[str] = None
    element_id: Optional[str] = None


@dataclass 
class DocumentStructure:
    """Complete document structure after parsing"""
    elements: List[ParsedElement]
    metadata: Dict[str, Any]
    total_pages: int
    file_path: str


class DocumentParser(ABC):
    """Abstract base class for document parsers"""
    
    @abstractmethod
    def parse(self, file_path: Union[str, Path]) -> DocumentStructure:
        """Parse document and return structured representation"""
        pass
    
    @abstractmethod
    def extract_text(self, file_path: Union[str, Path]) -> str:
        """Extract plain text from document"""
        pass


class PDFParser(DocumentParser):
    """Enhanced PDF parser with table and figure detection"""
    
    def __init__(self, extract_tables: bool = True, extract_images: bool = True):
        self.extract_tables = extract_tables
        self.extract_images = extract_images
        
    def parse(self, file_path: Union[str, Path]) -> DocumentStructure:
        """Parse PDF with structure detection"""
        file_path = Path(file_path)
        logger.info(f"Parsing PDF: {file_path}")
        
        elements = []
        metadata = {}
        
        try:
            with pdfplumber.open(file_path) as pdf:
                metadata = {
                    "title": pdf.metadata.get("Title", ""),
                    "author": pdf.metadata.get("Author", ""),
                    "creator": pdf.metadata.get("Creator", ""),
                    "creation_date": pdf.metadata.get("CreationDate", ""),
                    "total_pages": len(pdf.pages)
                }
                
                for page_num, page in enumerate(pdf.pages, 1):
                    # Extract text blocks
                    text_elements = self._extract_text_elements(page, page_num)
                    elements.extend(text_elements)
                    
                    # Extract tables
                    if self.extract_tables:
                        table_elements = self._extract_tables(page, page_num)
                        elements.extend(table_elements)
                    
                    # Extract images
                    if self.extract_images:
                        image_elements = self._extract_images(page, page_num)
                        elements.extend(image_elements)
                        
        except Exception as e:
            logger.error(f"Error parsing PDF {file_path}: {e}")
            raise
            
        return DocumentStructure(
            elements=elements,
            metadata=metadata,
            total_pages=metadata.get("total_pages", 0),
            file_path=str(file_path)
        )
    
    def _extract_text_elements(self, page, page_num: int) -> List[ParsedElement]:
        """Extract text elements with hierarchy detection"""
        elements = []
        
        # Get text with bounding boxes
        text_objects = page.extract_words(keep_blank_chars=True)
        
        if not text_objects:
            return elements
            
        # Group words into lines and paragraphs
        lines = self._group_words_to_lines(text_objects)
        paragraphs = self._group_lines_to_paragraphs(lines)
        
        for i, para in enumerate(paragraphs):
            text = " ".join([word["text"] for word in para])
            
            # Determine element type based on formatting
            element_type = self._classify_text_element(para, text)
            
            # Calculate bounding box
            bbox = self._calculate_bbox(para)
            
            element = ParsedElement(
                element_type=element_type,
                content=text.strip(),
                metadata={
                    "font_size": para[0].get("size", 0) if para else 0,
                    "font": para[0].get("fontname", "") if para else "",
                    "word_count": len(text.split()),
                    "char_count": len(text)
                },
                page_num=page_num,
                bbox=bbox,
                element_id=f"page_{page_num}_text_{i}"
            )
            
            if element.content.strip():  # Only add non-empty elements
                elements.append(element)
                
        return elements
    
    def _extract_tables(self, page, page_num: int) -> List[ParsedElement]:
        """Extract tables from page"""
        elements = []
        
        try:
            tables = page.extract_tables()
            
            for i, table in enumerate(tables):
                if not table:
                    continue
                    
                # Convert table to string representation
                table_text = self._table_to_text(table)
                
                # Get table bbox (approximate)
                bbox = page.bbox  # Placeholder - could be improved
                
                element = ParsedElement(
                    element_type="table",
                    content=table_text,
                    metadata={
                        "rows": len(table),
                        "cols": len(table[0]) if table else 0,
                        "table_data": table
                    },
                    page_num=page_num,
                    bbox=bbox,
                    element_id=f"page_{page_num}_table_{i}"
                )
                
                elements.append(element)
                
        except Exception as e:
            logger.warning(f"Error extracting tables from page {page_num}: {e}")
            
        return elements
    
    def _extract_images(self, page, page_num: int) -> List[ParsedElement]:
        """Extract images from page"""
        elements = []
        
        try:
            # This is a simplified implementation
            # In practice, you'd use libraries like pymupdf for better image extraction
            images = page.images
            
            for i, img in enumerate(images):
                element = ParsedElement(
                    element_type="figure",
                    content=f"[Image: {img.get('name', 'unnamed')}]",
                    metadata={
                        "image_data": img,
                        "width": img.get("width", 0),
                        "height": img.get("height", 0)
                    },
                    page_num=page_num,
                    bbox=(img.get("x0", 0), img.get("top", 0), 
                          img.get("x1", 0), img.get("bottom", 0)),
                    element_id=f"page_{page_num}_image_{i}"
                )
                
                elements.append(element)
                
        except Exception as e:
            logger.warning(f"Error extracting images from page {page_num}: {e}")
            
        return elements
    
    def _group_words_to_lines(self, words: List[Dict]) -> List[List[Dict]]:
        """Group words into lines based on y-coordinate"""
        if not words:
            return []
            
        # Sort by y-coordinate (top to bottom)
        words_sorted = sorted(words, key=lambda w: (w["top"], w["x0"]))
        
        lines = []
        current_line = [words_sorted[0]]
        
        for word in words_sorted[1:]:
            # Check if word is on the same line (similar y-coordinate)
            if abs(word["top"] - current_line[-1]["top"]) < 5:  # 5 pixel tolerance
                current_line.append(word)
            else:
                lines.append(current_line)
                current_line = [word]
                
        if current_line:
            lines.append(current_line)
            
        return lines
    
    def _group_lines_to_paragraphs(self, lines: List[List[Dict]]) -> List[List[Dict]]:
        """Group lines into paragraphs based on spacing"""
        if not lines:
            return []
            
        paragraphs = []
        current_para = lines[0]
        
        for line in lines[1:]:
            # Calculate spacing between lines
            prev_bottom = max(word["bottom"] for word in current_para)
            curr_top = min(word["top"] for word in line)
            spacing = curr_top - prev_bottom
            
            # If spacing is large, start new paragraph
            if spacing > 10:  # 10 pixel threshold
                paragraphs.append(current_para)
                current_para = line
            else:
                current_para.extend(line)
                
        if current_para:
            paragraphs.append(current_para)
            
        return paragraphs
    
    def _classify_text_element(self, words: List[Dict], text: str) -> str:
        """Classify text element type based on formatting and content"""
        if not words:
            return "paragraph"
            
        avg_font_size = sum(w.get("size", 12) for w in words) / len(words)
        
        # Check for headings
        if (avg_font_size > 14 or 
            any([re.match(r'^\d+\.?\s+[A-Z]', text), 
                re.match(r'^[A-Z][A-Z\s]+$', text),
                re.match(r'^\d+\.\d+', text)])):
            return "heading"
            
        # Check for list items
        if re.match(r'^\s*[•\-\*]\s+', text) or re.match(r'^\s*\d+\.\s+', text):
            return "list_item"
            
        return "paragraph"
    
    def _calculate_bbox(self, words: List[Dict]) -> tuple:
        """Calculate bounding box for group of words"""
        if not words:
            return (0, 0, 0, 0)
            
        x0 = min(w["x0"] for w in words)
        y0 = min(w["top"] for w in words)
        x1 = max(w["x1"] for w in words)
        y1 = max(w["bottom"] for w in words)
        
        return (x0, y0, x1, y1)
    
    def _table_to_text(self, table: List[List[str]]) -> str:
        """Convert table data to text representation"""
        if not table:
            return ""
            
        # Create markdown-style table
        lines = []
        for i, row in enumerate(table):
            if row:  # Skip empty rows
                clean_row = [str(cell).strip() if cell else "" for cell in row]
                lines.append(" | ".join(clean_row))
                
                # Add header separator
                if i == 0:
                    separator = " | ".join(["---"] * len(clean_row))
                    lines.append(separator)
                    
        return "\n".join(lines)
    
    def extract_text(self, file_path: Union[str, Path]) -> str:
        """Extract plain text from PDF"""
        file_path = Path(file_path)
        
        try:
            with pdfplumber.open(file_path) as pdf:
                text_parts = []
                for page in pdf.pages:
                    text = page.extract_text()
                    if text:
                        text_parts.append(text)
                        
                return "\n\n".join(text_parts)
                
        except Exception as e:
            logger.error(f"Error extracting text from PDF {file_path}: {e}")
            return ""


class HTMLParser(DocumentParser):
    """HTML parser with semantic structure extraction"""
    
    def __init__(self, extract_links: bool = True, extract_images: bool = True):
        self.extract_links = extract_links
        self.extract_images = extract_images
        
    def parse(self, file_path: Union[str, Path]) -> DocumentStructure:
        """Parse HTML with semantic structure"""
        file_path = Path(file_path)
        
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
            
        soup = BeautifulSoup(content, 'html.parser')
        
        elements = []
        metadata = self._extract_metadata(soup)
        
        # Extract structured elements
        elements.extend(self._extract_headings(soup))
        elements.extend(self._extract_paragraphs(soup))
        elements.extend(self._extract_lists(soup))
        elements.extend(self._extract_tables(soup))
        
        if self.extract_links:
            elements.extend(self._extract_links(soup))
            
        if self.extract_images:
            elements.extend(self._extract_images(soup))
            
        return DocumentStructure(
            elements=elements,
            metadata=metadata,
            total_pages=1,  # HTML is single "page"
            file_path=str(file_path)
        )
    
    def _extract_metadata(self, soup: BeautifulSoup) -> Dict[str, Any]:
        """Extract HTML metadata"""
        metadata = {}
        
        # Title
        title = soup.find('title')
        if title:
            metadata['title'] = title.get_text().strip()
            
        # Meta tags
        meta_tags = soup.find_all('meta')
        for meta in meta_tags:
            name = meta.get('name') or meta.get('property')
            content = meta.get('content')
            if name and content:
                metadata[name] = content
                
        return metadata
    
    def _extract_headings(self, soup: BeautifulSoup) -> List[ParsedElement]:
        """Extract heading elements"""
        elements = []
        
        for i, heading in enumerate(soup.find_all(['h1', 'h2', 'h3', 'h4', 'h5', 'h6'])):
            element = ParsedElement(
                element_type="heading",
                content=heading.get_text().strip(),
                metadata={
                    "tag": heading.name,
                    "level": int(heading.name[1]),
                    "id": heading.get('id', ''),
                    "class": heading.get('class', [])
                },
                element_id=f"heading_{i}"
            )
            elements.append(element)
            
        return elements
    
    def _extract_paragraphs(self, soup: BeautifulSoup) -> List[ParsedElement]:
        """Extract paragraph elements"""
        elements = []
        
        for i, para in enumerate(soup.find_all('p')):
            text = para.get_text().strip()
            if text:
                element = ParsedElement(
                    element_type="paragraph",
                    content=text,
                    metadata={
                        "tag": "p",
                        "class": para.get('class', []),
                        "word_count": len(text.split())
                    },
                    element_id=f"paragraph_{i}"
                )
                elements.append(element)
                
        return elements
    
    def _extract_lists(self, soup: BeautifulSoup) -> List[ParsedElement]:
        """Extract list elements"""
        elements = []
        
        for i, lst in enumerate(soup.find_all(['ul', 'ol'])):
            items = [li.get_text().strip() for li in lst.find_all('li')]
            content = '\n'.join(f"• {item}" for item in items if item)
            
            if content:
                element = ParsedElement(
                    element_type="list",
                    content=content,
                    metadata={
                        "tag": lst.name,
                        "item_count": len(items),
                        "class": lst.get('class', [])
                    },
                    element_id=f"list_{i}"
                )
                elements.append(element)
                
        return elements
    
    def _extract_tables(self, soup: BeautifulSoup) -> List[ParsedElement]:
        """Extract table elements"""
        elements = []
        
        for i, table in enumerate(soup.find_all('table')):
            rows = []
            for tr in table.find_all('tr'):
                cells = [td.get_text().strip() for td in tr.find_all(['td', 'th'])]
                if cells:
                    rows.append(cells)
                    
            if rows:
                # Convert to markdown-style table
                content = self._table_to_text(rows)
                
                element = ParsedElement(
                    element_type="table",
                    content=content,
                    metadata={
                        "tag": "table",
                        "rows": len(rows),
                        "cols": len(rows[0]) if rows else 0,
                        "class": table.get('class', []),
                        "table_data": rows
                    },
                    element_id=f"table_{i}"
                )
                elements.append(element)
                
        return elements
    
    def _extract_links(self, soup: BeautifulSoup) -> List[ParsedElement]:
        """Extract link elements"""
        elements = []
        
        for i, link in enumerate(soup.find_all('a', href=True)):
            text = link.get_text().strip()
            href = link['href']
            
            if text and href:
                element = ParsedElement(
                    element_type="link",
                    content=f"{text} ({href})",
                    metadata={
                        "tag": "a",
                        "href": href,
                        "text": text,
                        "class": link.get('class', [])
                    },
                    element_id=f"link_{i}"
                )
                elements.append(element)
                
        return elements
    
    def _extract_images(self, soup: BeautifulSoup) -> List[ParsedElement]:
        """Extract image elements"""
        elements = []
        
        for i, img in enumerate(soup.find_all('img')):
            src = img.get('src', '')
            alt = img.get('alt', '')
            
            element = ParsedElement(
                element_type="figure",
                content=f"[Image: {alt or src}]",
                metadata={
                    "tag": "img",
                    "src": src,
                    "alt": alt,
                    "width": img.get('width', ''),
                    "height": img.get('height', ''),
                    "class": img.get('class', [])
                },
                element_id=f"image_{i}"
            )
            elements.append(element)
            
        return elements
    
    def _table_to_text(self, rows: List[List[str]]) -> str:
        """Convert table rows to text representation"""
        if not rows:
            return ""
            
        lines = []
        for i, row in enumerate(rows):
            lines.append(" | ".join(row))
            if i == 0:  # Add header separator
                separator = " | ".join(["---"] * len(row))
                lines.append(separator)
                
        return "\n".join(lines)
    
    def extract_text(self, file_path: Union[str, Path]) -> str:
        """Extract plain text from HTML"""
        file_path = Path(file_path)
        
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
            
        soup = BeautifulSoup(content, 'html.parser')
        
        # Remove script and style elements
        for script in soup(["script", "style"]):
            script.decompose()
            
        return soup.get_text()


class DOCXParser(DocumentParser):
    """DOCX parser for Word documents"""
    
    def parse(self, file_path: Union[str, Path]) -> DocumentStructure:
        """Parse DOCX document"""
        file_path = Path(file_path)
        
        doc = docx.Document(file_path)
        elements = []
        
        for i, paragraph in enumerate(doc.paragraphs):
            text = paragraph.text.strip()
            if text:
                # Determine element type based on style
                element_type = "heading" if paragraph.style.name.startswith('Heading') else "paragraph"
                
                element = ParsedElement(
                    element_type=element_type,
                    content=text,
                    metadata={
                        "style": paragraph.style.name,
                        "level": self._get_heading_level(paragraph.style.name) if element_type == "heading" else 0
                    },
                    element_id=f"paragraph_{i}"
                )
                elements.append(element)
                
        # Extract tables
        for i, table in enumerate(doc.tables):
            rows = []
            for row in table.rows:
                cells = [cell.text.strip() for cell in row.cells]
                if any(cells):  # Skip empty rows
                    rows.append(cells)
                    
            if rows:
                content = self._table_to_text(rows)
                element = ParsedElement(
                    element_type="table",
                    content=content,
                    metadata={
                        "rows": len(rows),
                        "cols": len(rows[0]) if rows else 0,
                        "table_data": rows
                    },
                    element_id=f"table_{i}"
                )
                elements.append(element)
        
        metadata = {
            "title": doc.core_properties.title or "",
            "author": doc.core_properties.author or "",
            "created": str(doc.core_properties.created) if doc.core_properties.created else "",
            "modified": str(doc.core_properties.modified) if doc.core_properties.modified else ""
        }
        
        return DocumentStructure(
            elements=elements,
            metadata=metadata,
            total_pages=1,
            file_path=str(file_path)
        )
    
    def _get_heading_level(self, style_name: str) -> int:
        """Extract heading level from style name"""
        if 'Heading' in style_name:
            try:
                return int(style_name.split()[-1])
            except (ValueError, IndexError):
                return 1
        return 0
    
    def _table_to_text(self, rows: List[List[str]]) -> str:
        """Convert table rows to text representation"""
        if not rows:
            return ""
            
        lines = []
        for i, row in enumerate(rows):
            lines.append(" | ".join(row))
            if i == 0:  # Add header separator
                separator = " | ".join(["---"] * len(row))
                lines.append(separator)
                
        return "\n".join(lines)
    
    def extract_text(self, file_path: Union[str, Path]) -> str:
        """Extract plain text from DOCX"""
        file_path = Path(file_path)
        doc = docx.Document(file_path)
        
        text_parts = []
        for paragraph in doc.paragraphs:
            text = paragraph.text.strip()
            if text:
                text_parts.append(text)
                
        return "\n\n".join(text_parts)
