# ============================================================
# src/pdf_loader.py
# ============================================================
# PURPOSE:
#   This file handles reading PDF files and extracting their text.
#   We use PyMuPDF (imported as "fitz") because it handles complex
#   legal PDF formatting better than basic PDF readers.
#
# WHY PyMuPDF?
#   - Handles multi-column layouts common in legal documents
#   - Preserves page numbers (useful for citing sources)
#   - Fast and reliable on large PDFs
# ============================================================

import fitz  # PyMuPDF library - imported as "fitz"
import os


def load_pdf(file_path: str) -> list[dict]:
    """
    Read a single PDF file and return a list of pages.
    
    Each page is a dictionary with:
      - 'text': the extracted text from that page
      - 'page_number': which page it came from
      - 'source': the filename (so we can cite it later)
    
    Args:
        file_path: Path to the PDF file on disk
        
    Returns:
        List of page dictionaries
    """
    
    # Check the file actually exists before trying to open it
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"PDF not found: {file_path}")
    
    pages = []  # We'll collect all pages here
    
    # Open the PDF using PyMuPDF
    pdf_document = fitz.open(file_path)
    
    # Get just the filename (not the full path) for display
    file_name = os.path.basename(file_path)
    
    # Loop through every page in the PDF
    for page_number in range(len(pdf_document)):
        
        # Get the current page object
        page = pdf_document[page_number]
        
        # Extract all the text from this page
        page_text = page.get_text()
        
        # Skip pages that are empty or have very little text
        # (some PDFs have blank separator pages)
        if len(page_text.strip()) < 50:
            continue
        
        # Store the page info as a dictionary
        page_info = {
            "text": page_text,
            "page_number": page_number + 1,  # +1 because humans count from 1, not 0
            "source": file_name
        }
        
        pages.append(page_info)
    
    # Close the PDF to free memory
    pdf_document.close()
    
    print(f"✅ Loaded '{file_name}' — {len(pages)} pages extracted")
    
    return pages


def load_multiple_pdfs(folder_path: str) -> list[dict]:
    """
    Load ALL PDF files from a folder.
    
    This is useful when the user uploads multiple legal documents
    at once (e.g., a contract + an NDA + a case law PDF).
    
    Args:
        folder_path: Path to a folder containing PDF files
        
    Returns:
        Combined list of pages from all PDFs
    """
    
    all_pages = []  # Collect pages from all PDFs here
    
    # List all files in the folder
    files_in_folder = os.listdir(folder_path)
    
    # Filter to only keep .pdf files
    pdf_files = [f for f in files_in_folder if f.lower().endswith(".pdf")]
    
    if not pdf_files:
        print("⚠️  No PDF files found in the folder.")
        return []
    
    print(f"📂 Found {len(pdf_files)} PDF file(s) to load...")
    
    # Load each PDF one by one
    for pdf_file in pdf_files:
        full_path = os.path.join(folder_path, pdf_file)
        
        try:
            pages = load_pdf(full_path)
            all_pages.extend(pages)  # Add this PDF's pages to the master list
        except Exception as error:
            print(f"❌ Could not load '{pdf_file}': {error}")
    
    print(f"📄 Total pages loaded: {len(all_pages)}")
    
    return all_pages
