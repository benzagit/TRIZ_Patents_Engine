import fitz  # PyMuPDF
import easyocr
import pandas as pd
import os

# Initialize EasyOCR (CPU mode)
reader = easyocr.Reader(['en'], gpu=False)

def extract_text_from_pdf_images(pdf_path):
    full_text_chunks = []
    try:
        # Open the PDF
        doc = fitz.open(pdf_path)
        print(f"Processing {pdf_file}: {len(doc)} pages...")
        
        # Process each page
        for page_num in range(len(doc)):
            page = doc[page_num]
            # Convert page to image (300 DPI)
            pix = page.get_pixmap(matrix=fitz.Matrix(300/72, 300/72))
            img_bytes = pix.tobytes("png")
            # OCR with EasyOCR
            result = reader.readtext(img_bytes, detail=0)
            page_text = " ".join(result)
            if page_text.strip():
                full_text_chunks.append({
                    "pdf": os.path.basename(pdf_path),
                    "page": page_num + 1,
                    "text": page_text
                })
                print(f"  Page {page_num + 1}: Extracted {len(page_text)} characters")
            else:
                print(f"  Page {page_num + 1}: No text detected")
        
        doc.close()
        return full_text_chunks
    
    except Exception as e:
        print(f"Error processing {pdf_path}: {e}")
        return []

# Directory with your PDFs
pdf_folder = r"C:\venv\.venv\serpapi_pdfs"  # Adjust if needed
output_csv = "all_patents_corpus.csv"

# List to store results
all_data = []

# Process each PDF
for pdf_file in os.listdir(pdf_folder):
    if pdf_file.endswith(".pdf"):
        pdf_path = os.path.join(pdf_folder, pdf_file)
        if os.path.exists(pdf_path):
            text_data = extract_text_from_pdf_images(pdf_path)
            all_data.extend(text_data)
        else:
            print(f"File {pdf_path} not found!")

# Save to CSV
if all_data:
    df = pd.DataFrame(all_data)
    df.to_csv(output_csv, index=False)
    print(f"Text extracted and saved to {output_csv}")
else:
    print("No text extracted from any PDF!")

