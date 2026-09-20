import os
from fpdf import FPDF

# Paths
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
DOCS_DIR = os.path.join(ROOT, "docs")
MARKDOWN_FILE = os.path.join(DOCS_DIR, "final_report.md")
PDF_OUTPUT = os.path.join(DOCS_DIR, "finalreport.pdf")

def clean_text(text):
    """Clean text to remove problematic characters."""
    # Replace unicode characters
    replacements = {
        '→': '->',
        '←': '<-',
        '✓': '[OK]',
        '✅': '[OK]',
        '⚠️': 'WARNING',
        '✔': 'OK',
        '✘': 'X',
        '📊': '',
        '📅': '',
        '📸': '',
        '🔁': '',
        '—': ' - ',
        '–': '-',
        '×': 'x',
        '•': '-',
        '\u2014': ' - ',
        '\u2022': '-',
        '\u2026': '...',
    }
    
    for old, new in replacements.items():
        text = text.replace(old, new)
    
    # Remove any remaining non-ASCII characters except common ones
    cleaned = ''
    for char in text:
        if ord(char) < 128 or char in ' \n\t':
            cleaned += char
        else:
            cleaned += ' '
    
    return cleaned

def markdown_to_pdf(md_path, pdf_path):
    """Convert markdown file to PDF using FPDF."""
    
    # Read markdown content
    with open(md_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Clean the content
    content = clean_text(content)
    
    # Create PDF
    pdf = FPDF(orientation='P', unit='pt', format='A4')
    pdf.set_auto_page_break(auto=True, margin=50)
    pdf.add_page()
    
    # Set font - use Helvetica (built-in, no deprecation warnings)
    pdf.set_font('Helvetica', '', 10)
    
    # Process line by line
    lines = content.split('\n')
    
    for line in lines:
        # Handle headers
        if line.startswith('# '):
            pdf.set_font('Helvetica', 'B', 16)
            pdf.ln(8)
            pdf.cell(0, 16, line[2:].strip(), ln=True)
            pdf.set_font('Helvetica', '', 10)
            pdf.ln(4)
            continue
        elif line.startswith('## '):
            pdf.set_font('Helvetica', 'B', 13)
            pdf.ln(6)
            pdf.cell(0, 14, line[3:].strip(), ln=True)
            pdf.set_font('Helvetica', '', 10)
            pdf.ln(3)
            continue
        elif line.startswith('### '):
            pdf.set_font('Helvetica', 'B', 11)
            pdf.ln(4)
            pdf.cell(0, 12, line[4:].strip(), ln=True)
            pdf.set_font('Helvetica', '', 10)
            pdf.ln(2)
            continue
        elif line.startswith('**') and line.endswith('**'):
            pdf.set_font('Helvetica', 'B', 10)
            pdf.cell(0, 12, line[2:-2].strip(), ln=True)
            pdf.set_font('Helvetica', '', 10)
            continue
        elif line.startswith('- '):
            # Bullet point - use simple text
            bullet_text = '- ' + line[2:].strip()
            pdf.cell(0, 12, bullet_text, ln=True)
            continue
        elif line.strip() == '':
            pdf.ln(4)
            continue
        elif line.startswith('```'):
            # Code block - skip
            continue
        elif line.startswith('**Purpose**'):
            pdf.set_font('Helvetica', 'B', 10)
            pdf.cell(0, 12, line.strip(), ln=True)
            pdf.set_font('Helvetica', '', 10)
            continue
        else:
            # Regular text
            if line.strip():
                pdf.cell(0, 12, line.strip(), ln=True)
    
    pdf.output(pdf_path)
    print(f'PDF generated at: {pdf_path}')

if __name__ == '__main__':
    if os.path.exists(MARKDOWN_FILE):
        markdown_to_pdf(MARKDOWN_FILE, PDF_OUTPUT)
    else:
        print(f'Error: Markdown file not found at {MARKDOWN_FILE}')
