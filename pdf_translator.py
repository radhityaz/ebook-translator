"""
Technical PDF Textbook Translator
Interactive CLI for translating technical PDFs while preserving structure
Handles: math formulas, tables, figures, and technical content
"""

import os
import json
import re
import time
import threading
import sys
from pathlib import Path
from typing import List, Dict, Tuple, Optional
from getpass import getpass

import fitz  # PyMuPDF
import pdfplumber
from openai import OpenAI
from tqdm import tqdm

# ==================== SPINNER UTILITY ====================
class Spinner:
    """Animated spinner for long-running operations"""
    
    def __init__(self, message="Processing"):
        self.spinner_chars = ['⠋', '⠙', '⠹', '⠸', '⠼', '⠴', '⠦', '⠧', '⠇', '⠏']
        self.message = message
        self.running = False
        self.thread = None
    
    def spin(self):
        """Spinner animation loop"""
        idx = 0
        while self.running:
            sys.stdout.write(f'\r{self.spinner_chars[idx]} {self.message}')
            sys.stdout.flush()
            idx = (idx + 1) % len(self.spinner_chars)
            time.sleep(0.1)
        sys.stdout.write('\r' + ' ' * (len(self.message) + 5) + '\r')
        sys.stdout.flush()
    
    def start(self):
        """Start the spinner"""
        self.running = True
        self.thread = threading.Thread(target=self.spin)
        self.thread.start()
    
    def stop(self):
        """Stop the spinner"""
        self.running = False
        if self.thread:
            self.thread.join()


# ==================== CONFIGURATION ====================
class Config:
    """Configuration for PDF Translator"""
    
    def __init__(self):
        self.api_key = ""
        self.base_url = ""
        self.model = ""
        self.source_lang = ""
        self.target_lang = ""
        self.input_pdf = ""
        self.output_pdf = ""
        self.chunk_size = 800000
        self.enable_glossary = True
        
    def save_to_file(self, filepath: str):
        """Save config to JSON file (without API key)"""
        config_data = {
            "base_url": self.base_url,
            "model": self.model,
            "source_lang": self.source_lang,
            "target_lang": self.target_lang,
            "chunk_size": self.chunk_size,
            "enable_glossary": self.enable_glossary
        }
        with open(filepath, 'w') as f:
            json.dump(config_data, f, indent=2)
    
    def load_from_file(self, filepath: str):
        """Load config from JSON file"""
        if os.path.exists(filepath):
            with open(filepath, 'r') as f:
                config_data = json.load(f)
                self.base_url = config_data.get("base_url", "")
                self.model = config_data.get("model", "")
                self.source_lang = config_data.get("source_lang", "")
                self.target_lang = config_data.get("target_lang", "")
                self.chunk_size = config_data.get("chunk_size", 800000)
                self.enable_glossary = config_data.get("enable_glossary", True)
            return True
        return False


# ==================== CONTENT DETECTOR ====================
class ContentDetector:
    """Detect special content types (math, tables, figures)"""
    
    @staticmethod
    def contains_math(text: str) -> bool:
        """Detect if text contains mathematical expressions"""
        # Common math indicators
        math_patterns = [
            r'\$.*?\$',  # LaTeX inline math
            r'\\\(.*?\\\)',  # LaTeX inline math alt
            r'[∫∑∏√±∞≈≠≤≥∂∇]',  # Math symbols
            r'[α-ωΑ-Ω]',  # Greek letters
            r'\b[a-z]\s*=\s*[0-9]',  # Variable equations
            r'[xy]\^[0-9]',  # Powers
            r'\d+\s*[+\-×÷]\s*\d+',  # Arithmetic
        ]
        
        for pattern in math_patterns:
            if re.search(pattern, text):
                return True
        return False
    
    @staticmethod
    def is_reference(text: str) -> bool:
        """Detect if text is a reference/citation"""
        ref_patterns = [
            r'\(\d{4}\)',  # (2020)
            r'et al\.',
            r'[A-Z][a-z]+,\s*[A-Z]\.',  # Author names
            r'vol\.\s*\d+',
            r'pp\.\s*\d+-\d+',
        ]
        
        for pattern in ref_patterns:
            if re.search(pattern, text):
                return True
        return False
    
    @staticmethod
    def is_figure_caption(text: str) -> bool:
        """Detect figure captions"""
        return bool(re.match(r'(Figure|Fig\.|FIGURE)\s+\d+', text, re.IGNORECASE))
    
    @staticmethod
    def is_table_caption(text: str) -> bool:
        """Detect table captions"""
        return bool(re.match(r'(Table|TABLE)\s+\d+', text, re.IGNORECASE))


# ==================== PDF PROCESSOR ====================
class TechnicalPDFProcessor:
    """Advanced PDF processor for technical documents"""
    
    def __init__(self):
        self.content_detector = ContentDetector()
    
    def extract_with_structure(self, pdf_path: str) -> List[Dict]:
        """Extract text with structure detection"""
        pages_data = []
        
        print(f"\n📄 Analyzing PDF structure...")
        
        # Use pdfplumber for better table detection
        with pdfplumber.open(pdf_path) as pdf:
            total_pages = len(pdf.pages)
            
            for page_num in tqdm(range(total_pages), desc="Processing"):
                page = pdf.pages[page_num]
                
                # Extract tables
                tables = page.extract_tables()
                
                # Extract text with positions
                words = page.extract_words()
                
                # Group words into blocks
                blocks = self._group_words_into_blocks(words)
                
                page_data = {
                    "page_num": page_num,
                    "width": page.width,
                    "height": page.height,
                    "blocks": blocks,
                    "tables": tables,
                    "has_tables": len(tables) > 0
                }
                
                pages_data.append(page_data)
        
        return pages_data
    
    def _group_words_into_blocks(self, words: List[Dict]) -> List[Dict]:
        """Group words into logical text blocks"""
        if not words:
            return []
        
        blocks = []
        current_block = {
            "text": "",
            "words": [],
            "bbox": None,
            "type": "text"
        }
        
        # Sort words by vertical then horizontal position
        sorted_words = sorted(words, key=lambda w: (w['top'], w['x0']))
        
        prev_top = None
        line_threshold = 5  # pixels
        
        for word in sorted_words:
            # Check if new line/block
            if prev_top is not None and abs(word['top'] - prev_top) > line_threshold:
                if current_block["text"]:
                    blocks.append(current_block.copy())
                    current_block = {
                        "text": "",
                        "words": [],
                        "bbox": None,
                        "type": "text"
                    }
            
            current_block["text"] += word['text'] + " "
            current_block["words"].append(word)
            prev_top = word['top']
        
        # Add last block
        if current_block["text"]:
            blocks.append(current_block)
        
        # Detect block types
        for block in blocks:
            text = block["text"].strip()
            
            if self.content_detector.contains_math(text):
                block["type"] = "math"
            elif self.content_detector.is_reference(text):
                block["type"] = "reference"
            elif self.content_detector.is_figure_caption(text):
                block["type"] = "figure_caption"
            elif self.content_detector.is_table_caption(text):
                block["type"] = "table_caption"
        
        return blocks
    
    def rebuild_pdf(self, pages_data: List[Dict], output_path: str):
        """Rebuild PDF with translated content"""
        doc = fitz.open()
        
        print(f"\n📝 Building translated PDF...")
        
        for page_data in tqdm(pages_data, desc="Building"):
            page = doc.new_page(
                width=page_data["width"],
                height=page_data["height"]
            )
            
            y_position = 50  # Start position
            
            # Add translated blocks
            for block in page_data["blocks"]:
                if "translated_text" in block and block["translated_text"]:
                    text = block["translated_text"]
                else:
                    text = block["text"]
                
                # Insert text
                try:
                    rect = fitz.Rect(50, y_position, page_data["width"] - 50, y_position + 100)
                    page.insert_textbox(
                        rect,
                        text,
                        fontsize=10,
                        fontname="helv",
                        align=fitz.TEXT_ALIGN_LEFT
                    )
                    y_position += 15
                except:
                    pass
            
            # Add tables (simplified - in real implementation, format properly)
            if page_data.get("has_tables") and "translated_tables" in page_data:
                y_position += 20
                for table in page_data["translated_tables"]:
                    table_text = str(table)
                    rect = fitz.Rect(50, y_position, page_data["width"] - 50, y_position + 100)
                    try:
                        page.insert_textbox(
                            rect,
                            f"[TABLE]\n{table_text}",
                            fontsize=9,
                            fontname="helv"
                        )
                        y_position += 50
                    except:
                        pass
        
        doc.save(output_path)
        doc.close()


# ==================== TEXT CHUNKER ====================
class SmartChunker:
    """Intelligent text chunking for technical documents"""
    
    @staticmethod
    def chunk_pages(pages_data: List[Dict], max_size: int) -> List[Dict]:
        """Chunk pages intelligently"""
        chunks = []
        current_chunk = {
            "text_blocks": [],
            "tables": [],
            "pages": [],
            "size": 0
        }
        
        for page_data in pages_data:
            page_text = ""
            text_blocks = []
            
            # Collect translatable blocks
            for block in page_data["blocks"]:
                if block["type"] in ["text", "table_caption", "figure_caption"]:
                    page_text += block["text"] + "\n"
                    text_blocks.append(block)
            
            estimated_tokens = len(page_text) // 4
            
            if current_chunk["size"] + estimated_tokens > max_size and current_chunk["text_blocks"]:
                chunks.append(current_chunk)
                current_chunk = {
                    "text_blocks": [],
                    "tables": [],
                    "pages": [],
                    "size": 0
                }
            
            current_chunk["text_blocks"].extend(text_blocks)
            current_chunk["pages"].append(page_data)
            current_chunk["size"] += estimated_tokens
            
            # Add tables separately
            if page_data.get("has_tables"):
                current_chunk["tables"].extend(page_data["tables"])
        
        if current_chunk["text_blocks"]:
            chunks.append(current_chunk)
        
        return chunks


# ==================== LLM TRANSLATOR ====================
class TechnicalTranslator:
    """LLM-based translator for technical content"""
    
    def __init__(self, config: Config):
        self.config = config
        self.client = OpenAI(
            api_key=config.api_key,
            base_url=config.base_url
        )
        self.glossary = {}
    
    def build_glossary(self, sample_text: str) -> Dict[str, str]:
        """Build terminology glossary"""
        if not self.config.enable_glossary:
            return {}
        
        print("\n📚 Building technical terminology glossary...")
        estimated_tokens = len(sample_text) // 4
        print(f"   └─ Sample size: ~{estimated_tokens:,} tokens")
        
        prompt = f"""Analyze this technical {self.config.source_lang} text and identify 30-40 key technical terms, formulas, proper nouns, and specialized vocabulary that should be consistently translated to {self.config.target_lang}.

IMPORTANT: 
- For mathematical symbols and formulas, keep them in original form (universal)
- For author names and citations, keep in original form
- Focus on domain-specific technical terms

Return ONLY a JSON object: {{"term1": "translation1", "term2": "translation2", ...}}

Text sample:
{sample_text[:3000]}"""
        
        spinner = Spinner("   └─ Analyzing terminology... ⏳")
        spinner.start()
        
        try:
            start_time = time.time()
            response = self.client.chat.completions.create(
                model=self.config.model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.3
            )
            elapsed = time.time() - start_time
            spinner.stop()
            
            result = response.choices[0].message.content
            start = result.find("{")
            end = result.rfind("}") + 1
            
            if start >= 0 and end > start:
                self.glossary = json.loads(result[start:end])
                print(f"   └─ ✅ Built glossary with {len(self.glossary)} terms ({elapsed:.1f}s)")
        except Exception as e:
            spinner.stop()
            print(f"   └─ ⚠️  Glossary building failed: {e}")
            self.glossary = {}
        
        return self.glossary
    
    def translate_chunk(self, chunk: Dict, chunk_num: int, total_chunks: int) -> Dict:
        """Translate a chunk of content"""
        # Combine text blocks
        text_to_translate = "\n\n".join([
            block["text"] for block in chunk["text_blocks"]
        ])
        
        chunk_tokens = len(text_to_translate) // 4
        estimated_time = chunk_tokens / 1000
        
        print(f"\n🌍 Translating chunk {chunk_num}/{total_chunks}")
        print(f"   └─ Chunk size: ~{chunk_tokens:,} tokens")
        print(f"   └─ Text blocks: {len(chunk['text_blocks'])}")
        print(f"   └─ Tables: {len(chunk['tables'])}")
        print(f"   └─ Estimated time: ~{int(estimated_time // 60)}m {int(estimated_time % 60)}s")
        
        glossary_str = "\n".join([f"- {k}: {v}" for k, v in self.glossary.items()])
        
        prompt = f"""Translate this technical {self.config.source_lang} text to {self.config.target_lang}.

CRITICAL RULES:
1. PRESERVE mathematical formulas, equations, and symbols EXACTLY as-is (they are universal)
2. PRESERVE author names, citations, and references in original form
3. Use consistent technical terminology from this glossary:
{glossary_str}

4. Maintain paragraph structure and formatting
5. Keep technical accuracy - this is educational/scientific content
6. Translate figure/table captions but keep numbering
7. This is chunk {chunk_num}/{total_chunks}

Text to translate:
{text_to_translate}

Provide ONLY the translated text, maintaining the same paragraph structure."""
        
        spinner = Spinner(f"   └─ Translating with {self.config.model}... ⏳")
        spinner.start()
        
        try:
            start_time = time.time()
            response = self.client.chat.completions.create(
                model=self.config.model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.3,
                max_tokens=self.config.chunk_size
            )
            elapsed = time.time() - start_time
            spinner.stop()
            
            translated = response.choices[0].message.content
            output_tokens = len(translated) // 4
            
            print(f"   └─ ✅ Completed in {elapsed:.1f}s (~{output_tokens:,} tokens)")
            
            # Split back into blocks
            translated_parts = translated.split("\n\n")
            for i, block in enumerate(chunk["text_blocks"]):
                if i < len(translated_parts):
                    block["translated_text"] = translated_parts[i]
            
            return chunk
            
        except Exception as e:
            spinner.stop()
            print(f"   └─ ❌ Translation failed: {e}")
            return chunk


# ==================== MAIN TRANSLATOR ====================
class InteractivePDFTranslator:
    """Main interactive translator"""
    
    def __init__(self):
        self.config = Config()
        self.processor = TechnicalPDFProcessor()
        self.chunker = SmartChunker()
        self.translator = None
    
    def interactive_setup(self):
        """Interactive CLI setup"""
        print("\n" + "="*60)
        print("🌐 TECHNICAL PDF TRANSLATOR")
        print("="*60)
        
        # Try to load previous config
        config_file = "translator_config.json"
        if os.path.exists(config_file):
            use_saved = input(f"\n📁 Found saved config. Use it? (Y/n): ").strip().lower()
            if use_saved != 'n':
                self.config.load_from_file(config_file)
                print("✅ Loaded saved configuration")
        
        # Input PDF
        print("\n📂 PDF FILE SELECTION")
        print("-" * 60)
        default_input = self.config.input_pdf or ""
        self.config.input_pdf = input(f"Input PDF path [{default_input}]: ").strip() or default_input
        
        if not os.path.exists(self.config.input_pdf):
            print(f"❌ Error: File not found: {self.config.input_pdf}")
            sys.exit(1)
        
        # Output PDF
        default_output = self.config.output_pdf or self.config.input_pdf.replace(".pdf", "_translated.pdf")
        self.config.output_pdf = input(f"Output PDF path [{default_output}]: ").strip() or default_output
        
        # LLM Configuration
        print("\n🤖 LLM API CONFIGURATION")
        print("-" * 60)
        default_url = self.config.base_url or "https://api.openai.com/v1"
        self.config.base_url = input(f"API Base URL [{default_url}]: ").strip() or default_url
        
        self.config.api_key = getpass("API Key (hidden): ").strip()
        
        if not self.config.api_key:
            print("❌ API Key is required!")
            sys.exit(1)
        
        # Model selection
        print("\n🎯 MODEL SELECTION")
        print("-" * 60)
        print("Examples:")
        print("  - xai/grok-4-fast (2M context, recommended for large books)")
        print("  - openai/gpt-4o")
        print("  - anthropic/claude-3.5-sonnet")
        print("  - google/gemini-1.5-pro")
        
        default_model = self.config.model or "xai/grok-4-fast"
        self.config.model = input(f"Model name [{default_model}]: ").strip() or default_model
        
        # Translation settings
        print("\n🌍 TRANSLATION SETTINGS")
        print("-" * 60)
        default_source = self.config.source_lang or "English"
        self.config.source_lang = input(f"Source language [{default_source}]: ").strip() or default_source
        
        default_target = self.config.target_lang or "Indonesian"
        self.config.target_lang = input(f"Target language [{default_target}]: ").strip() or default_target
        
        # Advanced settings
        print("\n⚙️  ADVANCED SETTINGS")
        print("-" * 60)
        default_chunk = str(self.config.chunk_size)
        chunk_input = input(f"Chunk size in tokens [{default_chunk}]: ").strip()
        self.config.chunk_size = int(chunk_input) if chunk_input else self.config.chunk_size
        
        glossary_input = input(f"Enable terminology glossary? (Y/n): ").strip().lower()
        self.config.enable_glossary = glossary_input != 'n'
        
        # Save config
        save_config = input(f"\n💾 Save this configuration? (Y/n): ").strip().lower()
        if save_config != 'n':
            self.config.save_to_file(config_file)
            print(f"✅ Configuration saved to {config_file}")
        
        # Summary
        print("\n" + "="*60)
        print("📋 CONFIGURATION SUMMARY")
        print("="*60)
        print(f"Input:           {self.config.input_pdf}")
        print(f"Output:          {self.config.output_pdf}")
        print(f"Model:           {self.config.model}")
        print(f"Translation:     {self.config.source_lang} → {self.config.target_lang}")
        print(f"Chunk size:      {self.config.chunk_size:,} tokens")
        print(f"Glossary:        {'Enabled' if self.config.enable_glossary else 'Disabled'}")
        print("="*60)
        
        confirm = input("\n▶️  Start translation? (Y/n): ").strip().lower()
        if confirm == 'n':
            print("❌ Translation cancelled")
            sys.exit(0)
    
    def translate(self):
        """Main translation workflow"""
        start_time = time.time()
        
        # Initialize translator
        self.translator = TechnicalTranslator(self.config)
        
        # Step 1: Extract PDF structure
        pages_data = self.processor.extract_with_structure(self.config.input_pdf)
        print(f"✅ Extracted {len(pages_data)} pages")
        
        # Step 2: Build glossary
        if self.config.enable_glossary:
            sample_text = "\n".join([
                block["text"]
                for page in pages_data[:10]
                for block in page["blocks"]
                if block["type"] in ["text", "table_caption"]
            ])
            self.translator.build_glossary(sample_text)
        
        # Step 3: Chunk content
        print(f"\n✂️  Chunking content...")
        chunks = self.chunker.chunk_pages(pages_data, self.config.chunk_size)
        print(f"📦 Created {len(chunks)} chunks")
        
        # Step 4: Translate chunks
        print(f"\n{'='*60}")
        print("🚀 TRANSLATION PHASE")
        print("="*60)
        
        for i, chunk in enumerate(chunks):
            self.translator.translate_chunk(chunk, i + 1, len(chunks))
            time.sleep(1)  # Rate limiting
        
        # Step 5: Rebuild PDF
        self.processor.rebuild_pdf(pages_data, self.config.output_pdf)
        
        total_time = time.time() - start_time
        
        print(f"\n{'='*60}")
        print("✨ TRANSLATION COMPLETED!")
        print("="*60)
        print(f"Output file:     {self.config.output_pdf}")
        print(f"Total time:      {int(total_time // 60)}m {int(total_time % 60)}s")
        print(f"Pages:           {len(pages_data)}")
        print(f"Chunks:          {len(chunks)}")
        print("="*60 + "\n")


# ==================== MAIN ====================
def main():
    """Main entry point"""
    translator = InteractivePDFTranslator()
    
    try:
        translator.interactive_setup()
        translator.translate()
    except KeyboardInterrupt:
        print("\n\n❌ Translation interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
