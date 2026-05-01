from fpdf import FPDF
import markdown
import re
from datetime import datetime
import os

class PharmaReport(FPDF):
    def header(self):
        # Using a standard font that might support more characters or fallback
        self.set_font('helvetica', 'B', 15)
        self.cell(0, 10, 'PHARMA-GUARD AI - Tibbi Analiz Raporu', 0, 1, 'C')
        self.ln(5)

    def footer(self):
        self.set_y(-15)
        self.set_font('helvetica', 'I', 8)
        self.cell(0, 10, f'Sayfa {self.page_no()} | Uretim Tarihi: {datetime.now().strftime("%Y-%m-%d %H:%M")}', 0, 0, 'C')

def create_pdf_report(markdown_text: str, filename: str = "rapor.pdf"):
    pdf = PharmaReport()
    pdf.add_page()
    pdf.set_font("helvetica", size=12)
    
    # Simple markdown to PDF logic
    # Removing markdown symbols
    clean_text = markdown_text.replace("#", "").replace("*", "").replace("`", "")
    
    # Instead of encoding to latin-1, we use fpdf2's ability to handle unicode if possible.
    # We also replace Turkish specific characters with their ASCII equivalents for safety 
    # if a proper unicode font isn't loaded.
    tr_map = str.maketrans("ğĞüÜşŞİıçÇöÖ", "gGuUsSIicCoO")
    
    # Set margins explicitly
    pdf.set_margins(15, 15, 15)
    
    lines = clean_text.split('\n')
    for line in lines:
        clean_line = line.strip()
        if not clean_line:
            pdf.ln(5)
            continue
        
        # Translate Turkish characters and strip out emojis/unsupported chars
        translated_line = clean_line.translate(tr_map)
        safe_line = "".join(c for c in translated_line if ord(c) < 256).strip()
        
        if len(safe_line) > 0:
            # Ensure we are at the start of the line to avoid horizontal space errors
            pdf.set_x(15)
            # Use 0 to indicate full width from current X to right margin
            pdf.multi_cell(0, 8, txt=safe_line, align='L')
            pdf.ln(2)
            
    if not os.path.exists("data"):
        os.makedirs("data", exist_ok=True)
        
    output_path = os.path.join("data", filename)
    pdf.output(output_path)
    return output_path

def save_uploaded_file(uploaded_file):
    try:
        if not os.path.exists("data"):
            os.makedirs("data", exist_ok=True)
        with open(os.path.join("data", uploaded_file.name), "wb") as f:
            f.write(uploaded_file.getbuffer())
        return True
    except Exception as e:
        return False
