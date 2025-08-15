"""
Text and structure cleaners for preprocessing documents
"""

import re
import string
from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional, Set
from dataclasses import dataclass

import pandas as pd
from loguru import logger

from .parsers import ParsedElement, DocumentStructure
from .chunkers import Chunk
from config_manager import get_config


@dataclass
class CleaningRules:
    """Configuration for cleaning rules"""
    remove_headers_footers: bool = True
    remove_page_numbers: bool = True
    normalize_whitespace: bool = True
    remove_special_chars: bool = False
    fix_encoding: bool = True
    remove_urls: bool = False
    remove_emails: bool = False
    remove_phone_numbers: bool = False
    min_content_length: int = 20
    remove_duplicate_lines: bool = True
    merge_broken_sentences: bool = True
    
    @classmethod
    def from_config(cls):
        """从配置文件创建清洗规则"""
        config = get_config()
        cleaning_config = config.ingestion.get('cleaning', {})
        
        return cls(
            remove_headers_footers=cleaning_config.get('remove_headers_footers', True),
            remove_page_numbers=cleaning_config.get('remove_page_numbers', True),
            normalize_whitespace=cleaning_config.get('normalize_whitespace', True),
            remove_special_chars=cleaning_config.get('remove_special_chars', False),
            fix_encoding=cleaning_config.get('fix_encoding', True),
            remove_urls=cleaning_config.get('remove_urls', False),
            remove_emails=cleaning_config.get('remove_emails', False),
            min_content_length=cleaning_config.get('min_content_length', 20),
            remove_duplicate_lines=cleaning_config.get('remove_duplicate_lines', True)
        )


class DocumentCleaner(ABC):
    """Abstract base class for document cleaners"""
    
    @abstractmethod
    def clean(self, document: DocumentStructure) -> DocumentStructure:
        """Clean document and return cleaned version"""
        pass


class TextCleaner(DocumentCleaner):
    """Text cleaner for basic text preprocessing"""
    
    def __init__(self, rules: Optional[CleaningRules] = None):
        self.rules = rules or CleaningRules()
        
        # Compile regex patterns for efficiency
        self._compile_patterns()
    
    def _compile_patterns(self):
        """Compile commonly used regex patterns"""
        self.patterns = {
            'page_numbers': re.compile(r'^\s*(?:page\s*)?[\d\-]+\s*$', re.IGNORECASE),
            'headers_footers': re.compile(r'^(?:header|footer|page \d+|\d+ of \d+).*$', re.IGNORECASE),
            'whitespace': re.compile(r'\s+'),
            'urls': re.compile(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+'),
            'emails': re.compile(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'),
            'phone_numbers': re.compile(r'(\+?1[-.\s]?)?(\(?[0-9]{3}\)?[-.\s]?[0-9]{3}[-.\s]?[0-9]{4})'),
            'broken_sentences': re.compile(r'(?<=[a-z])\s+(?=[A-Z])'),
            'encoding_issues': re.compile(r'[^\x00-\x7F]+'),  # Non-ASCII characters
            'special_chars': re.compile(r'[^\w\s\.\,\!\?\;\:\-\(\)\[\]\{\}\"\']+'),
            'duplicate_spaces': re.compile(r' {2,}'),
            'line_breaks': re.compile(r'\n{3,}')
        }
    
    def clean(self, document: DocumentStructure) -> DocumentStructure:
        """Clean document elements"""
        logger.info(f"Cleaning document: {document.file_path}")
        
        cleaned_elements = []
        
        for element in document.elements:
            cleaned_element = self._clean_element(element)
            
            # Filter out elements that are too short after cleaning
            # Skip length check for figure and table elements
            if cleaned_element:
                if (element.element_type in ["figure", "table"] or 
                    len(cleaned_element.content.strip()) >= self.rules.min_content_length):
                    cleaned_elements.append(cleaned_element)
        
        # Remove duplicate elements
        if self.rules.remove_duplicate_lines:
            cleaned_elements = self._remove_duplicates(cleaned_elements)
        
        return DocumentStructure(
            elements=cleaned_elements,
            metadata=document.metadata,
            total_pages=document.total_pages,
            file_path=document.file_path
        )
    
    def _clean_element(self, element: ParsedElement) -> Optional[ParsedElement]:
        """Clean individual element"""
        content = element.content
        
        if not content or not content.strip():
            return None
        
        # Skip cleaning for certain element types
        if element.element_type in ["table", "figure"]:
            return element
        
        # Apply cleaning rules
        if self.rules.remove_headers_footers and self._is_header_footer(content):
            return None
        
        if self.rules.remove_page_numbers and self._is_page_number(content):
            return None
        
        # Clean text content
        cleaned_content = self._clean_text(content)
        
        if not cleaned_content or len(cleaned_content.strip()) < self.rules.min_content_length:
            return None
        
        # Create cleaned element
        cleaned_element = ParsedElement(
            element_type=element.element_type,
            content=cleaned_content,
            metadata=element.metadata.copy(),
            page_num=element.page_num,
            bbox=element.bbox,
            parent_id=element.parent_id,
            element_id=element.element_id
        )
        
        # Update metadata with cleaning info
        cleaned_element.metadata.update({
            'original_length': len(element.content),
            'cleaned_length': len(cleaned_content),
            'cleaning_applied': True
        })
        
        return cleaned_element
    
    def _clean_text(self, text: str) -> str:
        """Apply text cleaning rules"""
        cleaned = text
        
        # Fix encoding issues
        if self.rules.fix_encoding:
            cleaned = self._fix_encoding(cleaned)
        
        # Remove URLs
        if self.rules.remove_urls:
            cleaned = self.patterns['urls'].sub('', cleaned)
        
        # Remove emails
        if self.rules.remove_emails:
            cleaned = self.patterns['emails'].sub('', cleaned)
        
        # Remove phone numbers
        if self.rules.remove_phone_numbers:
            cleaned = self.patterns['phone_numbers'].sub('', cleaned)
        
        # Remove special characters
        if self.rules.remove_special_chars:
            cleaned = self.patterns['special_chars'].sub(' ', cleaned)
        
        # Merge broken sentences
        if self.rules.merge_broken_sentences:
            cleaned = self._merge_broken_sentences(cleaned)
        
        # Normalize whitespace
        if self.rules.normalize_whitespace:
            cleaned = self._normalize_whitespace(cleaned)
        
        return cleaned.strip()
    
    def _is_header_footer(self, text: str) -> bool:
        """Check if text is likely a header or footer"""
        text_lower = text.lower().strip()
        
        # Common header/footer patterns
        header_footer_indicators = [
            'page', 'chapter', 'section', 'confidential', 'proprietary',
            'copyright', '©', 'all rights reserved', 'footer', 'header'
        ]
        
        # Check for short text with header/footer indicators
        if len(text) < 100:
            for indicator in header_footer_indicators:
                if indicator in text_lower:
                    return True
        
        # Check pattern matching
        return bool(self.patterns['headers_footers'].match(text))
    
    def _is_page_number(self, text: str) -> bool:
        """Check if text is a page number"""
        return bool(self.patterns['page_numbers'].match(text.strip()))
    
    def _fix_encoding(self, text: str) -> str:
        """Fix common encoding issues"""
        # Common replacements for encoding issues
        replacements = {
            ''': "'",  # Right single quotation mark
            ''': "'",  # Left single quotation mark
            '"': '"',  # Left double quotation mark
            '"': '"',  # Right double quotation mark
            '–': '-',  # En dash
            '—': '-',  # Em dash
            '…': '...',  # Horizontal ellipsis
            'â€™': "'",  # Common encoding error
            'â€œ': '"',  # Common encoding error
            'â€': '"',   # Common encoding error
        }
        
        cleaned = text
        for old, new in replacements.items():
            cleaned = cleaned.replace(old, new)
        
        # Remove other non-ASCII characters if needed
        # cleaned = self.patterns['encoding_issues'].sub('', cleaned)
        
        return cleaned
    
    def _merge_broken_sentences(self, text: str) -> str:
        """Merge sentences that were broken across lines"""
        # Pattern for words that were hyphenated across lines
        text = re.sub(r'-\s*\n\s*', '', text)
        
        # Pattern for sentences broken across lines
        text = re.sub(r'(?<=[a-z])\s*\n\s*(?=[a-z])', ' ', text)
        
        return text
    
    def _normalize_whitespace(self, text: str) -> str:
        """Normalize whitespace in text"""
        # Replace multiple spaces with single space
        text = self.patterns['duplicate_spaces'].sub(' ', text)
        
        # Replace multiple line breaks with double line break
        text = self.patterns['line_breaks'].sub('\n\n', text)
        
        # Replace tabs with spaces
        text = text.replace('\t', ' ')
        
        # Remove trailing whitespace from lines
        lines = text.split('\n')
        cleaned_lines = [line.rstrip() for line in lines]
        
        return '\n'.join(cleaned_lines)
    
    def _remove_duplicates(self, elements: List[ParsedElement]) -> List[ParsedElement]:
        """Remove duplicate elements based on content"""
        seen_content = set()
        unique_elements = []
        
        for element in elements:
            # Use content hash for deduplication
            content_hash = hash(element.content.strip().lower())
            
            if content_hash not in seen_content:
                seen_content.add(content_hash)
                unique_elements.append(element)
            else:
                logger.debug(f"Removing duplicate element: {element.element_id}")
        
        return unique_elements


class StructureCleaner(DocumentCleaner):
    """Cleaner that focuses on document structure and relationships"""
    
    def __init__(
        self,
        merge_adjacent_paragraphs: bool = True,
        fix_heading_hierarchy: bool = True,
        group_list_items: bool = True,
        associate_tables_with_context: bool = True
    ):
        self.merge_adjacent_paragraphs = merge_adjacent_paragraphs
        self.fix_heading_hierarchy = fix_heading_hierarchy
        self.group_list_items = group_list_items
        self.associate_tables_with_context = associate_tables_with_context
    
    def clean(self, document: DocumentStructure) -> DocumentStructure:
        """Clean document structure"""
        logger.info(f"Cleaning document structure: {document.file_path}")
        
        elements = document.elements.copy()
        
        # Apply structure cleaning
        if self.fix_heading_hierarchy:
            elements = self._fix_heading_hierarchy(elements)
        
        if self.merge_adjacent_paragraphs:
            elements = self._merge_adjacent_paragraphs(elements)
        
        if self.group_list_items:
            elements = self._group_list_items(elements)
        
        if self.associate_tables_with_context:
            elements = self._associate_tables_with_context(elements)
        
        return DocumentStructure(
            elements=elements,
            metadata=document.metadata,
            total_pages=document.total_pages,
            file_path=document.file_path
        )
    
    def _fix_heading_hierarchy(self, elements: List[ParsedElement]) -> List[ParsedElement]:
        """Fix heading hierarchy inconsistencies"""
        heading_elements = []
        other_elements = []
        
        # Separate headings from other elements
        for element in elements:
            if element.element_type == "heading":
                heading_elements.append(element)
            else:
                other_elements.append(element)
        
        # Fix heading levels
        if heading_elements:
            fixed_headings = self._normalize_heading_levels(heading_elements)
            
            # Merge back with other elements, maintaining order
            result = []
            heading_idx = 0
            
            for element in elements:
                if element.element_type == "heading":
                    if heading_idx < len(fixed_headings):
                        result.append(fixed_headings[heading_idx])
                        heading_idx += 1
                else:
                    result.append(element)
            
            return result
        
        return elements
    
    def _normalize_heading_levels(self, headings: List[ParsedElement]) -> List[ParsedElement]:
        """Normalize heading levels to be sequential"""
        if not headings:
            return headings
        
        # Extract current levels
        levels = []
        for heading in headings:
            level = heading.metadata.get('level', 1)
            levels.append(level)
        
        # Create level mapping
        unique_levels = sorted(set(levels))
        level_mapping = {old: new for new, old in enumerate(unique_levels, 1)}
        
        # Apply mapping
        normalized_headings = []
        for heading in headings:
            old_level = heading.metadata.get('level', 1)
            new_level = level_mapping.get(old_level, 1)
            
            normalized_heading = ParsedElement(
                element_type=heading.element_type,
                content=heading.content,
                metadata={**heading.metadata, 'level': new_level, 'original_level': old_level},
                page_num=heading.page_num,
                bbox=heading.bbox,
                parent_id=heading.parent_id,
                element_id=heading.element_id
            )
            normalized_headings.append(normalized_heading)
        
        return normalized_headings
    
    def _merge_adjacent_paragraphs(self, elements: List[ParsedElement]) -> List[ParsedElement]:
        """Merge adjacent paragraphs that should be together"""
        merged_elements = []
        current_paragraph = None
        
        for element in elements:
            if element.element_type == "paragraph":
                if current_paragraph is None:
                    current_paragraph = element
                else:
                    # Check if paragraphs should be merged
                    if self._should_merge_paragraphs(current_paragraph, element):
                        # Merge content
                        merged_content = current_paragraph.content + " " + element.content
                        
                        # Create merged element
                        merged_element = ParsedElement(
                            element_type="paragraph",
                            content=merged_content,
                            metadata={
                                **current_paragraph.metadata,
                                'merged_from': [current_paragraph.element_id, element.element_id],
                                'word_count': len(merged_content.split())
                            },
                            page_num=current_paragraph.page_num,
                            bbox=current_paragraph.bbox,
                            element_id=f"merged_{current_paragraph.element_id}_{element.element_id}"
                        )
                        current_paragraph = merged_element
                    else:
                        # Don't merge, add current and start new
                        merged_elements.append(current_paragraph)
                        current_paragraph = element
            else:
                # Non-paragraph element
                if current_paragraph:
                    merged_elements.append(current_paragraph)
                    current_paragraph = None
                merged_elements.append(element)
        
        # Add final paragraph if exists
        if current_paragraph:
            merged_elements.append(current_paragraph)
        
        return merged_elements
    
    def _should_merge_paragraphs(self, para1: ParsedElement, para2: ParsedElement) -> bool:
        """Determine if two paragraphs should be merged"""
        # Don't merge if either is very long
        if len(para1.content) > 500 or len(para2.content) > 500:
            return False
        
        # Don't merge if first paragraph ends with period (complete sentence)
        if para1.content.strip().endswith('.'):
            return False
        
        # Merge if first paragraph seems incomplete
        if (para1.content.strip().endswith(',') or 
            para1.content.strip().endswith(';') or
            para1.content.strip().endswith(':') or
            not para1.content.strip().endswith(('.', '!', '?'))):
            return True
        
        return False
    
    def _group_list_items(self, elements: List[ParsedElement]) -> List[ParsedElement]:
        """Group consecutive list items into single list elements"""
        grouped_elements = []
        current_list_items = []
        
        for element in elements:
            if element.element_type == "list_item":
                current_list_items.append(element)
            else:
                # Process accumulated list items
                if current_list_items:
                    grouped_list = self._create_grouped_list(current_list_items)
                    grouped_elements.append(grouped_list)
                    current_list_items = []
                
                grouped_elements.append(element)
        
        # Process final list items
        if current_list_items:
            grouped_list = self._create_grouped_list(current_list_items)
            grouped_elements.append(grouped_list)
        
        return grouped_elements
    
    def _create_grouped_list(self, list_items: List[ParsedElement]) -> ParsedElement:
        """Create a single list element from multiple list items"""
        combined_content = "\n".join(item.content for item in list_items)
        
        return ParsedElement(
            element_type="list",
            content=combined_content,
            metadata={
                'item_count': len(list_items),
                'source_items': [item.element_id for item in list_items],
                'list_type': 'grouped'
            },
            page_num=list_items[0].page_num if list_items else None,
            element_id=f"grouped_list_{'_'.join(item.element_id for item in list_items[:3])}"
        )
    
    def _associate_tables_with_context(self, elements: List[ParsedElement]) -> List[ParsedElement]:
        """Associate tables with nearby context elements"""
        associated_elements = []
        
        for i, element in enumerate(elements):
            if element.element_type == "table":
                # Look for context before and after table
                context_before = self._find_table_context(elements, i, direction="before")
                context_after = self._find_table_context(elements, i, direction="after")
                
                # Add context to table metadata
                enhanced_element = ParsedElement(
                    element_type=element.element_type,
                    content=element.content,
                    metadata={
                        **element.metadata,
                        'context_before': context_before,
                        'context_after': context_after,
                        'has_context': bool(context_before or context_after)
                    },
                    page_num=element.page_num,
                    bbox=element.bbox,
                    parent_id=element.parent_id,
                    element_id=element.element_id
                )
                associated_elements.append(enhanced_element)
            else:
                associated_elements.append(element)
        
        return associated_elements
    
    def _find_table_context(
        self, 
        elements: List[ParsedElement], 
        table_idx: int, 
        direction: str,
        max_distance: int = 2
    ) -> str:
        """Find contextual text near a table"""
        context_parts = []
        
        if direction == "before":
            start_idx = max(0, table_idx - max_distance)
            end_idx = table_idx
            search_elements = elements[start_idx:end_idx]
        else:  # after
            start_idx = table_idx + 1
            end_idx = min(len(elements), table_idx + 1 + max_distance)
            search_elements = elements[start_idx:end_idx]
        
        for element in search_elements:
            if element.element_type in ["paragraph", "heading"]:
                # Look for table references
                content_lower = element.content.lower()
                if any(keyword in content_lower for keyword in 
                       ["table", "chart", "figure", "shows", "data", "results"]):
                    context_parts.append(element.content)
        
        return " ".join(context_parts)
