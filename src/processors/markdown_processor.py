"""Markdown document processor for Hulo documentation."""

import re
import uuid
from pathlib import Path
from typing import List, Dict, Any
from datetime import datetime

import markdown

from ..models.document import Document, DocumentChunk
from ..config.settings import settings


class MarkdownProcessor:
    """Process Markdown documents for the knowledge base."""
    
    def __init__(self):
        self.md = markdown.Markdown(extensions=['meta', 'toc', 'fenced_code'])
    
    def extract_frontmatter(self, content: str) -> tuple[Dict[str, Any], str]:
        """Extract frontmatter from markdown content."""
        frontmatter = {}
        body = content
        
        # Check for frontmatter
        if content.startswith('---'):
            parts = content.split('---', 2)
            if len(parts) >= 3:
                frontmatter_text = parts[1].strip()
                body = parts[2].strip()
                
                # Parse frontmatter
                for line in frontmatter_text.split('\n'):
                    if ':' in line:
                        key, value = line.split(':', 1)
                        key = key.strip()
                        value = value.strip()
                        
                        # Handle list values
                        if value.startswith('[') and value.endswith(']'):
                            value = [item.strip().strip('"\'') for item in value[1:-1].split(',')]
                        # Handle boolean values
                        elif value.lower() in ('true', 'false'):
                            value = value.lower() == 'true'
                        # Handle numeric values
                        elif value.isdigit():
                            value = int(value)
                        elif value.replace('.', '').isdigit():
                            value = float(value)
                        
                        frontmatter[key] = value
        
        return frontmatter, body
    
    def extract_title(self, content: str, frontmatter: Dict[str, Any]) -> str:
        """Extract document title."""
        # First try frontmatter
        if 'title' in frontmatter:
            return str(frontmatter['title'])
        
        # Then try first heading
        lines = content.split('\n')
        for line in lines:
            if line.startswith('#'):
                return line.lstrip('#').strip()
        
        # Fallback to filename
        return "Untitled"
    
    def extract_language(self, file_path: Path) -> str:
        """Extract language from file path."""
        path_str = str(file_path)
        if '/zh/' in path_str or '\\zh\\' in path_str:
            return 'zh'
        return 'en'
    
    def extract_category(self, file_path: Path) -> str:
        """Extract category from file path."""
        parts = file_path.parts
        for i, part in enumerate(parts):
            if part in ['grammar', 'blueprints', 'libs', 'guide', 'toolchain', 'others']:
                return part
        return 'general'
    
    def extract_tags(self, frontmatter: Dict[str, Any]) -> List[str]:
        """Extract tags from frontmatter."""
        tags = frontmatter.get('tag', [])
        if isinstance(tags, str):
            tags = [tags]
        elif not isinstance(tags, list):
            tags = []
        return tags
    
    def clean_content(self, content: str) -> str:
        """Clean markdown content for better processing."""
        # Remove HTML comments
        content = re.sub(r'<!--.*?-->', '', content, flags=re.DOTALL)
        
        # Remove VuePress specific components
        content = re.sub(r'<Catalog\s*/>', '', content)
        content = re.sub(r'<.*?/>', '', content)
        
        # Remove excessive whitespace
        content = re.sub(r'\n\s*\n\s*\n', '\n\n', content)
        
        return content.strip()
    
    def split_into_chunks(self, content: str, chunk_size: int = None, chunk_overlap: int = None) -> List[str]:
        """Split content into chunks for vector storage based on Markdown headers."""
        if chunk_size is None:
            chunk_size = settings.chunk_size
        if chunk_overlap is None:
            chunk_overlap = settings.chunk_overlap
        
        # First, identify all headers and their positions
        lines = content.split('\n')
        header_positions = []
        
        for i, line in enumerate(lines):
            if line.strip().startswith('#'):
                header_level = len(line) - len(line.lstrip('#'))
                header_text = line.strip('#').strip()
                header_positions.append({
                    'index': i,
                    'level': header_level,
                    'text': header_text,
                    'line': line
                })
        
        # If no headers found, fall back to simple chunking
        if not header_positions:
            return self._simple_split_chunks(content, chunk_size, chunk_overlap)
        
        # Split based on headers - simpler and more reliable approach
        chunks = []
        
        for i, header in enumerate(header_positions):
            start_line = header['index']
            
            # Find the end of this section (next header of same or higher level)
            end_line = len(lines)
            for j in range(i + 1, len(header_positions)):
                next_header = header_positions[j]
                if next_header['level'] <= header['level']:
                    end_line = next_header['index']
                    break
            
            # Extract the section content
            section_lines = lines[start_line:end_line]
            section_content = '\n'.join(section_lines)
            
            # If section is too large, split it further
            if len(section_content) > chunk_size * 1.5:
                sub_chunks = self._split_large_section(section_lines, chunk_size, chunk_overlap)
                chunks.extend(sub_chunks)
            else:
                chunks.append(section_content)
        
        return chunks
    
    def _simple_split_chunks(self, content: str, chunk_size: int, chunk_overlap: int) -> List[str]:
        """Fallback simple chunking when no headers are found."""
        chunks = []
        lines = content.split('\n')
        current_chunk = []
        current_size = 0
        
        for line in lines:
            line_size = len(line) + 1
            
            if current_size + line_size > chunk_size and current_chunk:
                chunk_text = '\n'.join(current_chunk)
                chunks.append(chunk_text)
                
                # Overlap
                overlap_lines = []
                overlap_size = 0
                for i in range(len(current_chunk) - 1, -1, -1):
                    if overlap_size + len(current_chunk[i]) + 1 <= chunk_overlap:
                        overlap_lines.insert(0, current_chunk[i])
                        overlap_size += len(current_chunk[i]) + 1
                    else:
                        break
                
                current_chunk = overlap_lines
                current_size = overlap_size
            
            current_chunk.append(line)
            current_size += line_size
        
        if current_chunk:
            chunk_text = '\n'.join(current_chunk)
            chunks.append(chunk_text)
        
        return chunks
    
    def _split_large_section(self, section_lines: List[str], chunk_size: int, chunk_overlap: int) -> List[str]:
        """Split a large section into smaller chunks while preserving structure."""
        chunks = []
        current_chunk = []
        current_size = 0
        
        for line in section_lines:
            line_size = len(line) + 1
            
            # If we encounter a sub-header (h3, h4, h5, h6) and chunk is getting large, break
            if line.strip().startswith('###') and current_chunk and current_size > chunk_size * 0.7:
                chunk_text = '\n'.join(current_chunk)
                chunks.append(chunk_text)
                current_chunk = []
                current_size = 0
            
            if current_size + line_size > chunk_size and current_chunk:
                chunk_text = '\n'.join(current_chunk)
                chunks.append(chunk_text)
                
                # Overlap
                overlap_lines = []
                overlap_size = 0
                for i in range(len(current_chunk) - 1, max(0, len(current_chunk) - 3), -1):
                    if overlap_size + len(current_chunk[i]) + 1 <= chunk_overlap:
                        overlap_lines.insert(0, current_chunk[i])
                        overlap_size += len(current_chunk[i]) + 1
                    else:
                        break
                
                current_chunk = overlap_lines
                current_size = overlap_size
            
            current_chunk.append(line)
            current_size += line_size
        
        if current_chunk:
            chunk_text = '\n'.join(current_chunk)
            chunks.append(chunk_text)
        
        return chunks
    
    def process_file(self, file_path: Path) -> Document:
        """Process a single markdown file."""
        try:
            content = file_path.read_text(encoding='utf-8')
        except UnicodeDecodeError:
            content = file_path.read_text(encoding='latin-1')
        
        # Extract frontmatter and body
        frontmatter, body = self.extract_frontmatter(content)
        
        # Clean content
        body = self.clean_content(body)
        
        # Extract metadata
        title = self.extract_title(body, frontmatter)
        language = self.extract_language(file_path)
        category = self.extract_category(file_path)
        tags = self.extract_tags(frontmatter)
        
        # Create document
        document = Document(
            id=str(uuid.uuid4()),
            title=title,
            content=body,
            language=language,
            category=category,
            tags=tags,
            metadata=frontmatter,
            file_path=str(file_path),
            created_at=datetime.now(),
            updated_at=datetime.now()
        )
        
        return document
    
    def create_chunks(self, document: Document) -> List[DocumentChunk]:
        """Create chunks from a document."""
        chunks = self.split_into_chunks(document.content)
        document_chunks = []
        
        for i, chunk_content in enumerate(chunks):
            # Skip empty chunks
            if not chunk_content.strip():
                continue
                
            chunk = DocumentChunk(
                id=f"{document.id}_chunk_{i}",
                document_id=document.id,
                content=chunk_content,
                language=document.language,
                category=document.category,
                chunk_index=len(document_chunks),  # Use actual chunk index
                start_pos=document.content.find(chunk_content),
                end_pos=document.content.find(chunk_content) + len(chunk_content),
                metadata={
                    "title": document.title,
                    "tags": document.tags,
                    "file_path": document.file_path
                }
            )
            document_chunks.append(chunk)
        
        return document_chunks 