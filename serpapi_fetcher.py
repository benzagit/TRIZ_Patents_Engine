import os
import requests
from serpapi import GoogleSearch

# Configuration
NUM_PATENTS = 10  # Number of patents to download
TOPIC = "microprocessors"  # Change this to your desired topic
SEARCH_QUERY = f'microprocessor OR "{TOPIC}"'  # Broadened query to catch variations
SAVE_DIR = r"C:\venv\.venv\serpapi_pdfs"  # Target directory for saving PDFs
API_KEY = "4d390a763dbfaac40be9bfb4193df24d8c9568c66cda999f23e8016e4196e5c8"  # Your SerpAPI key

# Create the save directory if it doesn't exist
if not os.path.exists(SAVE_DIR):
    os.makedirs(SAVE_DIR)

# SerpAPI search parameters
params = {
    "engine": "google_patents",
    "q": SEARCH_QUERY,
    "api_key": API_KEY,
    "num": 50,  # Fetch more results to ensure we have enough to filter
    "hl": "en"
}

# Perform the search
search = GoogleSearch(params)
results = search.get_dict()

# Check if results are empty
if not results.get("organic_results"):
    print("No results found for the query. Try broadening the search query or checking the API key.")
    exit()

# Extract and filter patent results
patents = results.get("organic_results", [])
filtered_patents = []
topic_keywords = [TOPIC.lower(), "microprocessor", "processor", "cpu"]  # Broader keywords for filtering

for patent in patents:
    if len(filtered_patents) >= NUM_PATENTS:
        break
    title = patent.get("title", "").lower()
    snippet = patent.get("snippet", "").lower()

    # Check if title or snippet contains topic-related keywords
    has_keyword = any(keyword in title or keyword in snippet for keyword in topic_keywords)

    if has_keyword:
        filtered_patents.append(patent)

# Download filtered PDFs
for i, patent in enumerate(filtered_patents[:NUM_PATENTS]):
    pdf_url = patent.get("pdf")
    if pdf_url:
        pdf_name = pdf_url.split("/")[-1]
        pdf_path = os.path.join(SAVE_DIR, pdf_name)
        
        print(f"Downloading {pdf_name}...")
        response = requests.get(pdf_url)
        if response.status_code == 200:
            with open(pdf_path, "wb") as f:
                f.write(response.content)
            print(f"Saved to {pdf_path}")
        else:
            print(f"Failed to download {pdf_name}")
    else:
        print(f"No PDF link for patent {i+1}")

# Final message
if len(filtered_patents) == 0:
    print(f"No {TOPIC}-related patents found after filtering. Try adjusting the topic keywords or query.")
else:
    print(f"Finished downloading {len(filtered_patents[:NUM_PATENTS])} {TOPIC}-related patents to {SAVE_DIR}")