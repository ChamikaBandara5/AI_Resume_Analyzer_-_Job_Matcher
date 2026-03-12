

# AI Job Application Assistant

A web application built with Streamlit that helps users optimize job applications. Users can upload their CV, provide a job description or URL, and get AI-powered analysis including ATS score, skill gap detection, CV improvement suggestions, and a tailored cover letter in PDF. Optionally, it can search LinkedIn for matching job postings using SerpAPI.

---

## Features

* Upload CV (PDF, DOCX, TXT) and extract text
* Analyze CV against a job description
* ATS-style scoring and skill gap analysis
* CV improvement suggestions
* Generate professional cover letter (PDF)
* Optional LinkedIn job search via SerpAPI

---

## Technology

* **Frontend/UI:** Streamlit
* **AI:** OpenRouter GPT / OpenAI API
* **PDF Generation:** FPDF2
* **CV Parsing:** PyPDF2, python-docx
* **Web Scraping:** BeautifulSoup
* **LinkedIn Search:** SerpAPI

---

## Installation

1. Clone the repository:

```bash
git clone https://github.com/manojaberathna24/Job-assistent.git

```

2. Create and activate a virtual environment:

```bash
python -m venv venv
# Windows
venv\Scripts\activate
# macOS/Linux
source venv/bin/activate
```

3. Install dependencies:

```bash
pip install -r requirements.txt
```

---

## API Setup

* **OpenRouter API Key** (required for AI analysis)
* **SerpAPI Key** (optional for LinkedIn search)

Enter keys in the app sidebar or set as environment variables:

```bash
export OPENROUTER_API_KEY="sk-or-yourkey"
export SERPAPI_KEY="your-serpapi-key"
```

---

## Running the App

```bash
streamlit run app.py
```

* Open `http://localhost:8501` in your browser
* Upload CV, provide job description, analyze, and download cover letter

---

## Notes

* Use **DejaVuSans.ttf** font in `generate_cover_letter_pdf()` for Unicode support.
* Ensure valid API keys for OpenRouter and SerpAPI.

---
try this->

https://jobassistent.streamlit.app/

