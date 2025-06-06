# TRIZ Patent Innovation Engine
test
## Overview
This project implements a TRIZ (Theory of Inventive Problem Solving)-based innovation engine to analyze patent documents and generate novel solutions for technical contradictions. It integrates three Python scripts to fetch patent data using the SERP API, extract text via OCR, and deploy a Streamlit application for solution generation, evaluation, and visualization. The system leverages advanced APIs (SERP API, NVIDIA Nemotron 70B) and OCR technology (EasyOCR) to support innovation, with a focus on fields like microprocessor technology.

- **Purpose**: Automate the discovery of patent-based insights and propose innovative solutions using TRIZ principles.
- **Target Audience**: Researchers, engineers, and innovators interested in patent analysis and technical problem-solving.
- **Date**: Created on June 02, 2025, at 02:01 AM +01.

## Project Structure
The project consists of three main scripts, each with a specific role in the workflow:

1. **`serpapi_fetcher.py`**: Fetches patent documents from Google Patents using the SERP API and downloads PDF files.
2. **`ocr_processor.py`**: Extracts text from PDF patent documents using OCR and compiles it into a `.csv` corpus.
3. **`solution_generator.py`**: Deploys a Streamlit application to process the corpus, apply TRIZ methodology, evaluate solutions, and visualize results (including a radar plot).

## Installation

### Prerequisites
- Python 3.8 or higher.
- Internet connection for API calls and downloads.
- Tesseract OCR installed (e.g., `sudo apt install tesseract-ocr` on Ubuntu or download from [here](https://github.com/UB-Mannheim/tesseract/wiki) for Windows).

### Dependencies
Install the required Python packages using the following command:
```bash
pip install -r requirements.txt
```
Create a `requirements.txt` file with the following content:
```
serpapi
requests
beautifulsoup4
fitz==0.0.1.dev2  # PyMuPDF
easyocr
pandas
streamlit
sentence-transformers
faiss-cpu
spacy
plotly
nltk
textblob
numpy
```
- **Additional Setup**: Install the spaCy model: `python -m spacy download en_core_web_sm`.

### API Keys
- **SERP API Key**: Obtain a key from [SerpApi](https://serpapi.com/) and replace the placeholder in `serpapi_fetcher.py` (line 26) with your key.
- **NVIDIA API Key**: Required for `solution_generator.py` (replace the placeholder in line 15 with your NVIDIA API key).

## Usage

### Workflow
Follow these steps to run the project:

1. **Fetch Patent PDFs**:
   - Run `serpapi_fetcher.py` to download patent PDFs:
     ```bash
     python serpapi_fetcher.py
     ```
   - This script searches for patents related to "microprocessors" (configurable via `TOPIC`) and saves PDFs to `C:\venv\.venv\serpapi_pdfs`.
   - Output: PDF files in the specified directory.

2. **Extract Text from PDFs**:
   - Run `ocr_processor.py` to convert PDFs to text and create a corpus:
     ```bash
     python ocr_processor.py
     ```
   - This script processes all `.pdf` files in `C:\venv\.venv\patents` (adjust the path if needed) and outputs `all_patents_corpus.csv`.
   - Output: A `.csv` file with columns `pdf`, `page`, and `text`.

3. **Run the TRIZ Engine**:
   - Launch the Streamlit application with:
     ```bash
     streamlit run solution_generator.py
     ```
   - Use the UI to input queries, generate TRIZ and Patents-based solutions, evaluate them, and export results.
   - Input: `all_patents_corpus.csv` (must be in the same directory).
   - Output: Visualizations (radar plot), evaluation scores, and exportable `.csv`/`Excel` files.

### Detailed Instructions

#### `serpapi_fetcher.py`
- **Purpose**: Retrieves patent documents from Google Patents using SERP API.
- **Functionality**:
  - Searches for patents with a configurable query (default: "microprocessor").
  - Downloads up to 10 PDF files to a specified directory (`SAVE_DIR`).
  - Uses asynchronous requests to handle multiple downloads efficiently.
- **Key Features**:
  - Filters results based on topic-related keywords (e.g., "microprocessor", "cpu").
  - Includes error handling for failed downloads or API issues.
- **Customization**:
  - Modify `TOPIC` (line 26) and `SAVE_DIR` (line 27) to target different patent domains or locations.
  - Adjust `NUM_PATENTS` (line 24) to change the number of downloads.

#### `ocr_processor.py`
- **Purpose**: Extracts text from patent PDFs using OCR for further analysis.
- **Functionality**:
  - Processes all `.pdf` files in a specified folder (`pdf_folder`).
  - Converts each page to an image (300 DPI) and applies EasyOCR to extract text.
  - Saves the extracted text into `all_patents_corpus.csv` with metadata (PDF name, page number).
- **Key Features**:
  - Handles multi-page PDFs and empty pages gracefully.
  - Outputs structured data suitable for machine learning or TRIZ analysis.
- **Customization**:
  - Update `pdf_folder` (line 27) to match the directory containing your PDFs (e.g., output from `serpapi_fetcher.py`).
  - Adjust DPI in `fitz.Matrix(300/72, 300/72)` (line 19) for better or faster OCR.

#### `solution_generator.py`
- **Purpose**: Deploys a web-based interface to analyze patents and generate innovative solutions.
- **Functionality**:
  - Loads the `all_patents_corpus.csv` file to extract problematics.
  - Applies TRIZ principles and patent-inspired methods to propose solutions.
  - Evaluates solutions based on Clarity, Relevance, Novelty, and Feasibility, with total scores for TRIZ, Patents, and overall performance.
  - Visualizes results with a radar plot and progress bars.
- **Key Features**:
  - Interactive UI for query input and solution review.
  - Exports solutions and evaluations to `.csv` or `.xlsx` files.
  - Integrates NVIDIA Nemotron 70B for advanced text generation.
- **Customization**:
  - Replace the NVIDIA API key (line 15) with your own.
  - Adjust the TRIZ matrix (line 70) and parameters (lines 65-69) for domain-specific analysis.
  - Modify `all_patents_corpus.csv` path if stored elsewhere.

## Output
- **`serpapi_fetcher.py`**: PDF files in `SAVE_DIR`.
- **`ocr_processor.py`**: `all_patents_corpus.csv` with extracted text.
- **`solution_generator.py`**:
  - On-screen: Radar plot, evaluation tables, and progress bars.
  - Downloadable: `triz_solutions.csv`, `triz_evaluation.csv`, `triz_solutions.xlsx`, `triz_evaluation.xlsx`, and `short_evaluation.xlsx`.

## Troubleshooting
- **SERP API Errors**: Ensure your API key is valid and has sufficient credits. Check internet connectivity.
- **OCR Issues**: Verify Tesseract installation and adjust DPI if text is unreadable.
- **Streamlit Failures**: Ensure all dependencies are installed and `all_patents_corpus.csv` exists before running.
- **Logs**: Enable debug mode in `serpapi_fetcher.py` or check Streamlit’s debug checkbox for detailed error messages.

## Academic Relevance
- **Innovation**: Applies TRIZ methodology to automate inventive problem-solving, a cornerstone of engineering and design research. This project extends TRIZ by integrating patent data, offering a novel approach to technical contradiction resolution.
- **Data Processing**: Demonstrates the use of OCR and API integration for large-scale text analysis, contributing to digital humanities and data science by enabling scalable patent corpus creation.
- **Visualization**: The radar plot and evaluation metrics provide a quantitative and visual method to assess solution quality, enhancing academic presentations or publications in innovation studies.
- **Interdisciplinary Impact**: Combines computer science (machine learning, UI design) with intellectual property analysis, fostering cross-disciplinary research.

## Future Improvements
- Enhance OCR accuracy with multi-language support (e.g., EasyOCR with additional languages).
- Integrate real-time API updates in the Streamlit app.
- Expand the TRIZ matrix for broader industry applications (e.g., biotechnology, renewable energy).
- Add machine learning models to predict solution feasibility based on historical patent data.

## Contributors
- Developed by : Benakka Zaid & Amcassou Hanane

## License
All rights reserved. 
