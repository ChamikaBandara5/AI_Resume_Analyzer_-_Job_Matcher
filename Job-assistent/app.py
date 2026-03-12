import os
import io
import json
import textwrap
from typing import Optional, Dict, Any

import streamlit as st
import requests
from fpdf import FPDF
from PyPDF2 import PdfReader
from docx import Document
from bs4 import BeautifulSoup


# -----------------------------
# Utility: Read CV file content
# -----------------------------
def read_pdf(file) -> str:
    try:
        reader = PdfReader(file)
        text = []
        for page in reader.pages:
            text.append(page.extract_text() or "")
        return "\n".join(text).strip()
    except Exception as e:
        return f"ERROR_READING_PDF: {e}"


def read_docx(file) -> str:
    try:
        doc = Document(file)
        text = []
        for para in doc.paragraphs:
            text.append(para.text)
        return "\n".join(text).strip()
    except Exception as e:
        return f"ERROR_READING_DOCX: {e}"


def read_txt(file) -> str:
    try:
        return file.read().decode("utf-8", errors="ignore").strip()
    except Exception as e:
        return f"ERROR_READING_TXT: {e}"


def extract_cv_text(uploaded_file) -> str:
    if uploaded_file is None:
        return ""
    filename = uploaded_file.name.lower()
    if filename.endswith(".pdf"):
        return read_pdf(uploaded_file)
    elif filename.endswith(".docx"):
        return read_docx(uploaded_file)
    elif filename.endswith(".txt"):
        return read_txt(uploaded_file)
    else:
        return ""


# --------------------------------------
# Utility: Fetch job description from URL
# --------------------------------------
def fetch_job_description_from_url(url: str) -> str:
    try:
        headers = {
            "User-Agent": (
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                "AppleWebKit/537.36 (KHTML, like Gecko) "
                "Chrome/123.0 Safari/537.36"
            )
        }
        resp = requests.get(url, headers=headers, timeout=15)
        resp.raise_for_status()
        html = resp.text

        soup = BeautifulSoup(html, "html.parser")

        # Try common containers
        candidate_selectors = [
            "[class*=description]",
            "[class*=job-description]",
            "[class*=description__text]",
            "section[aria-label*=Description]",
            "section[aria-label*=Job]",
            "div[data-test-id*=job-description]",
        ]
        for selector in candidate_selectors:
            el = soup.select_one(selector)
            if el and el.get_text(strip=True):
                return el.get_text(separator="\n", strip=True)

        # Fallback: large body text
        body_text = soup.get_text(separator="\n", strip=True)
        # Heuristic: take middle chunk
        lines = [l for l in body_text.splitlines() if len(l.strip()) > 40]
        if len(lines) > 80:
            lines = lines[20:200]
        return "\n".join(lines).strip()
    except Exception as e:
        return f"ERROR_FETCHING_URL: {e}"


# ----------------------------------------
# Utility: Call OpenRouter Chat Completion
# ----------------------------------------
def call_openrouter_chat(
    messages,
    api_key: str,
    model: str = "openrouter/auto",
    temperature: float = 0.2,
    max_tokens: int = 2000,
) -> str:
    url = "https://openrouter.ai/api/v1/chat/completions"

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
        # Optional but recommended
        "HTTP-Referer": "https://your-app-url.com",  # change if you deploy
        "X-Title": "AI Job Application Assistant",
    }

    payload = {
        "model": model,
        "messages": messages,
        "temperature": temperature,
        "max_tokens": max_tokens,
    }

    resp = requests.post(url, headers=headers, json=payload, timeout=60)
    resp.raise_for_status()
    data = resp.json()
    return data["choices"][0]["message"]["content"]


# ---------------------------------------------------
# Utility: Build a job search query from the CV using LLM
# ---------------------------------------------------
def build_job_search_query_from_cv(cv_text: str, api_key: str, model: str) -> str:
    """
    Ask the LLM to infer the main target role and key skills from the CV,
    and return a short search query string that we can use with Google/SerpAPI.
    """
    system_prompt = """You are an expert career coach and job search assistant.

Given a candidate CV, infer:
- The best target job title (or 1–2 titles)
- 3–6 core skills or technologies

Respond ONLY with a short search query suitable for searching LinkedIn job postings.
Do NOT add explanations or extra text.

Examples:
- "senior software engineer python django backend developer"
- "junior data analyst sql excel power bi entry level"
"""

    user_prompt = f"""Here is the candidate CV:

\"\"\"CV
{cv_text}
\"\"\"

Return just a concise search query that describes the type of roles this candidate should look for.
"""

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]

    try:
        query = call_openrouter_chat(messages, api_key=api_key, model=model, temperature=0.3, max_tokens=64)
        return query.strip()
    except Exception as e:
        st.error(f"Could not build search query from CV: {e}")
        return ""


# ---------------------------------------------------
# Utility: Search LinkedIn jobs using SerpAPI (Google)
# ---------------------------------------------------
def search_linkedin_jobs(
    base_query: str,
    serpapi_key: str,
    min_results: int = 5,
    max_results: int = 7,
) -> list[Dict[str, str]]:
    """
    Use SerpAPI's Google engine to search for LinkedIn job postings
    that match the given query. Returns a list of dicts with title, link, and snippet.

    NOTE:
    - Requires a SerpAPI API key (https://serpapi.com/).
    - We try to return recent and active-looking roles only:
      * Restrict Google search to recent results (past week) when possible.
      * Filter out snippets that clearly mention "no longer accepting applications", "expired", etc.
    """
    if not serpapi_key:
        st.error("Please provide your SerpAPI key in the sidebar to search LinkedIn jobs.")
        return []

    # Force LinkedIn jobs domain
    full_query = f"{base_query} site:linkedin.com/jobs"

    params = {
        "engine": "google",
        "q": full_query,
        "api_key": serpapi_key,
        "num": max_results,
        # Try to bias towards the most recent postings (past week).
        # See SerpAPI docs: this translates to Google's "past week" filter.
        "tbs": "qdr:w",
    }

    try:
        resp = requests.get("https://serpapi.com/search.json", params=params, timeout=30)
        resp.raise_for_status()
        data = resp.json()
    except requests.exceptions.HTTPError as e:
        st.error(f"SerpAPI HTTP error: {e}")
        return []
    except Exception as e:
        st.error(f"Error calling SerpAPI: {e}")
        return []

    results = []
    for item in data.get("organic_results", []):
        link = item.get("link", "") or ""
        if "linkedin.com/jobs" not in link:
            continue
        title = item.get("title", "LinkedIn Job")
        snippet = item.get("snippet", "")

        # Basic text-based filter to avoid clearly closed/expired postings
        combined = f"{title} {snippet}".lower()
        bad_phrases = [
            "no longer accepting applications",
            "no longer accepting applicants",
            "no longer available",
            "job is closed",
            "position is closed",
            "this job has expired",
            "expired job",
        ]
        if any(p in combined for p in bad_phrases):
            continue

        results.append(
            {
                "title": title,
                "link": link,
                "snippet": snippet,
            }
        )
        if len(results) >= max_results:
            break

    # Ensure we have at least min_results if possible
    return results[: max_results] if len(results) >= min_results else results


# -------------------------------------------------------
# Utility: Analyze CV + JD with structured JSON instruction
# -------------------------------------------------------
def analyze_with_llm(
    cv_text: str, job_text: str, api_key: str, model: str
) -> Optional[Dict[str, Any]]:
    system_prompt = """You are an expert ATS (Applicant Tracking System) analyzer and career coach.

Given:
- A candidate CV
- A target job description

You MUST respond in STRICT JSON with this exact top-level structure:

{
  "ats_score": {
    "score": 0-100 (number),
    "reasoning": "short explanation"
  },
  "skill_gap_analysis": {
    "missing_skills": ["skill1", "skill2", ...],
    "nice_to_have_skills": ["skill1", "skill2", ...],
    "summary": "1-3 paragraph description"
  },
  "cv_improvement_suggestions": {
    "high_impact_changes": ["...", "..."],
    "bullet_point_rewrites": ["...", "..."],
    "overall_feedback": "2-4 paragraphs of feedback"
  },
  "cover_letter": {
    "title": "Professional cover letter title",
    "body": "Complete, ready-to-send cover letter of 4-8 paragraphs, using the candidate name if available from CV and addressed to the hiring manager."
  }
}

Rules:
- The JSON MUST be valid and parseable.
- Do NOT wrap the JSON in markdown.
- Do NOT include any commentary outside the JSON.
"""

    user_prompt = f"""Here is the candidate CV:

\"\"\"CV
{cv_text}
\"\"\"

Here is the job description:

\"\"\"JOB_DESCRIPTION
{job_text}
\"\"\"

Now perform the ATS analysis and respond only with the JSON described in the system prompt.
"""

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]

    # Call OpenRouter and handle common HTTP errors (e.g., 401 Unauthorized)
    try:
        raw = call_openrouter_chat(messages, api_key=api_key, model=model)
    except requests.exceptions.HTTPError as e:
        status = e.response.status_code if e.response is not None else "unknown"
        try:
            err_json = e.response.json() if e.response is not None else {}
        except Exception:
            err_json = {}
        detail = err_json.get("error", {}).get("message") or err_json.get("message") or str(e)

        if status == 401:
            st.error(
                "OpenRouter returned **401 Unauthorized**.\n\n"
                "- Make sure you pasted a **valid OpenRouter API key** (starts with `sk-or-`).\n"
                "- Check that the key is **active** and **not revoked** on your OpenRouter dashboard.\n"
                "- If you set `OPENROUTER_API_KEY` in the environment, restart the app / terminal.\n\n"
                f"Server message: {detail}"
            )
        else:
            st.error(f"OpenRouter request failed with HTTP {status}: {detail}")
        return None
    except Exception as e:
        st.error(f"Unexpected error calling OpenRouter: {e}")
        return None

    # Try to locate JSON in the response
    try:
        # Trim any leading / trailing whitespace
        raw_stripped = raw.strip()

        # If model still returned markdown, try to extract {...}
        if raw_stripped.startswith("```"):
            # Remove fences if present
            raw_stripped = raw_stripped.strip("`")
            # Find first { and last }
            start = raw_stripped.find("{")
            end = raw_stripped.rfind("}")
            raw_stripped = raw_stripped[start : end + 1]

        parsed = json.loads(raw_stripped)
        return parsed
    except Exception as e:
        st.error(f"Could not parse AI response as JSON. Raw response:\\n\\n{raw}\\n\\nError: {e}")
        return None


# ---------------------------------
# Utility: Create PDF for download
# ---------------------------------
def generate_cover_letter_pdf(title: str, body: str) -> bytes:
    pdf = FPDF()
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.add_page()
    
    # We use built-in Helvetica. For characters outside latin-1, replace them to avoid FPDF errors.
    def safe_latin1(text):
        return text.encode('latin-1', 'replace').decode('latin-1')

    # Title styling
    pdf.set_font("Helvetica", "B", 18)
    pdf.set_text_color(33, 37, 41)
    pdf.multi_cell(0, 10, safe_latin1(title), 0, "L")
    pdf.ln(4)

    # Body styling
    pdf.set_font("Helvetica", "", 12)
    pdf.set_text_color(55, 55, 55)

    usable_width = pdf.w - 2 * pdf.l_margin
    paragraphs = body.split("\n\n") if body else []
    for paragraph in paragraphs:
        clean_para = safe_latin1(paragraph.strip())
        if not clean_para:
            pdf.ln(4)
            continue
        wrapped = textwrap.wrap(
            clean_para,
            width=95,
            break_long_words=True,
            break_on_hyphens=False,
        )
        if not wrapped:
            pdf.ln(4)
            continue
        for line in wrapped:
            safe_line = line.replace("\t", "    ")
            pdf.multi_cell(usable_width, 7, safe_line, 0, "L")
        pdf.ln(4)

    # Export to bytes
    pdf_bytes = pdf.output(dest="S")
    return pdf_bytes


# -------------------
# Streamlit UI Layout
# -------------------
def main():
    st.set_page_config(
        page_title="AI Job Application Assistant",
        page_icon="🧠",
        layout="wide",
    )

    # Initialize in-memory session storage for API keys (reset when app/server restarts)
    if "openrouter_key" not in st.session_state:
        st.session_state["openrouter_key"] = ""
    if "serpapi_key" not in st.session_state:
        st.session_state["serpapi_key"] = ""

    # Custom CSS for a premium dark UI
    st.markdown(
        """
        <style>
        .main {
            background-color: #0b0f19;
            color: #e2e8f0;
        }
        .block-container {
            padding-top: 2rem;
            padding-bottom: 2rem;
            max-width: 1100px;
        }
        /* Sleek header band */
        .header-band {
            background: linear-gradient(135deg, #1e293b, #0f172a);
            border-radius: 1rem;
            padding: 1.8rem 2rem;
            color: #f8fafc;
            box-shadow: 0 10px 30px rgba(0, 0, 0, 0.5);
            border: 1px solid rgba(255, 255, 255, 0.05);
            margin-bottom: 2rem;
        }
        .title-text {
            font-size: 2.4rem;
            font-weight: 800;
            background: -webkit-linear-gradient(45deg, #38bdf8, #818cf8);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            letter-spacing: -0.5px;
        }
        .subtitle-text {
            font-size: 1.05rem;
            color: #94a3b8;
            margin-top: 0.5rem;
        }
        .hero-links {
            margin-top: 1.2rem;
            display: flex;
            flex-wrap: wrap;
            gap: 0.8rem;
        }
        .hero-label {
            display: inline-flex;
            align-items: center;
            padding: 0.4rem 1rem;
            border-radius: 8px;
            background: rgba(56, 189, 248, 0.1);
            color: #38bdf8;
            font-weight: 600;
            font-size: 0.85rem;
            border: 1px solid rgba(56, 189, 248, 0.2);
        }
        .section-header {
            font-size: 1.3rem;
            font-weight: 700;
            color: #e2e8f0;
            margin-bottom: 0.8rem;
            margin-top: 1.5rem;
            border-bottom: 1px solid #1e293b;
            padding-bottom: 0.5rem;
        }
        .pill {
            display: inline-block;
            padding: 0.3rem 0.8rem;
            border-radius: 6px;
            background: rgba(16, 185, 129, 0.15);
            color: #10b981;
            font-size: 0.8rem;
            font-weight: 600;
            margin-right: 0.6rem;
            border: 1px solid rgba(16, 185, 129, 0.3);
        }
        .pill-soft {
            display: inline-block;
            padding: 0.3rem 0.8rem;
            border-radius: 6px;
            background: rgba(239, 68, 68, 0.15);
            color: #ef4444;
            font-size: 0.8rem;
            font-weight: 500;
            margin-right: 0.5rem;
            border: 1px solid rgba(239, 68, 68, 0.3);
        }
        .ats-score-badge {
            font-weight: 800;
            font-size: 2rem;
            color: #38bdf8;
        }
        .score-box {
            background: #1e293b;
            border-radius: 12px;
            padding: 1.5rem;
            border: 1px solid #334155;
            text-align: center;
            box-shadow: 0 4px 15px rgba(0,0,0,0.2);
        }
        .job-card {
            background: #1e293b;
            border-radius: 12px;
            padding: 1.2rem;
            border: 1px solid #334155;
            margin-bottom: 1rem;
            transition: transform 0.2s ease;
        }
        .job-card:hover {
            transform: translateY(-2px);
            border-color: #475569;
        }
        .job-title {
            font-weight: 700;
            color: #f1f5f9;
            font-size: 1.1rem;
            margin-bottom: 0.4rem;
        }
        .job-link {
            font-size: 0.9rem;
            color: #60a5fa;
            text-decoration: none;
        }
        .job-link:hover {
            text-decoration: underline;
        }
        .job-tag {
            display: inline-block;
            padding: 0.2rem 0.6rem;
            border-radius: 4px;
            background: #334155;
            color: #cbd5e1;
            font-size: 0.75rem;
            margin-right: 0.4rem;
            margin-top: 0.4rem;
        }
        .stButton>button[kind="primary"] {
            background: #38bdf8;
            color: #0f172a;
            border-radius: 8px;
            padding: 0.6rem 2rem;
            border: none;
            font-weight: 700;
            font-size: 1rem;
            transition: all 0.2s ease;
        }
        .stButton>button[kind="primary"]:hover {
            background: #7dd3fc;
            box-shadow: 0 0 15px rgba(56, 189, 248, 0.4);
        }
        /* Fix text elements inside containers */
        .stMarkdown p {
            color: #cbd5e1;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )

    # -------------
    # Sidebar
    # -------------
    with st.sidebar:
        st.image(
            "https://images.pexels.com/photos/3183150/pexels-photo-3183150.jpeg",
            use_column_width=True,
        )
        st.markdown("### AI Job Application Assistant")

        st.markdown(
            """
            Use this assistant to analyze your CV against a target job,
            get an ATS-style score, identify skill gaps, and generate
            a tailored cover letter as PDF.
            """
        )

        st.markdown("---")

        # API 1: OpenRouter (for all AI analysis and text generation)
        st.markdown("#### OpenRouter Settings (AI analysis)")
        api_key_input = st.text_input(
            "OpenRouter API Key",
            type="password",
            value=st.session_state["openrouter_key"],
            placeholder="sk-or-...",
            help="Get a free key from openrouter.ai and paste it here. Stored only in this session.",
        )
        # Save key only in Streamlit session memory (not on disk). Closing the app resets it.
        st.session_state["openrouter_key"] = api_key_input.strip()
        api_key = st.session_state["openrouter_key"]

        model = st.selectbox(
            "AI Model",
            options=[
                "google/gemini-2.5-flash:free",
                "meta-llama/llama-3.1-8b-instruct:free",
                "mistralai/mistral-nemo:free",
                "openrouter/auto",
                "openai/gpt-4o",
                "anthropic/claude-3.5-sonnet",
            ],
            index=0,
            help="Select a ':free' model if you haven't added credits to your OpenRouter account.",
        )

        st.markdown("---")

        # API 2: SerpAPI (for real LinkedIn job search through Google)
        st.markdown("#### SerpAPI Settings (LinkedIn job search)")
        serpapi_input = st.text_input(
            "SerpAPI API Key",
            type="password",
            value=st.session_state["serpapi_key"],
            placeholder="Your SerpAPI key",
            help="Used to search LinkedIn jobs via Google. Stored only in this session. Get one at serpapi.com.",
        )
        st.session_state["serpapi_key"] = serpapi_input.strip()
        serpapi_key = st.session_state["serpapi_key"]

        st.markdown("---")
        st.markdown("#### Upload Your CV")
        cv_file = st.file_uploader(
            "Supported: PDF, DOCX, TXT",
            type=["pdf", "docx", "txt"],
        )

        st.markdown("---")
        st.caption("Tip: For best results, use a well-formatted CV and a detailed job description.")

    # -------------
    # Main Header
    # -------------
    col_left, col_right = st.columns([2, 1], gap="large")

    with col_left:
        st.markdown(
            """
            <div class="header-band">
                <div class="title-text">AI Application Assistant & Job Matcher</div>
                <div class="subtitle-text">
                    Elevate your career with AI. Upload your CV, analyze skill gaps against top job descriptions, and generate tailored cover letters instantly.
                </div>
                <div class="hero-links">
                     <span class="hero-label">🎯 High-Precision ATS Scoring</span>
                     <span class="hero-label">🔍 Deep Skill Gap Insights</span>
                     <span class="hero-label">✍️ AI Cover Letter Generator</span>
                     <span class="hero-label">🚀 Live Job Discovery</span>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with col_right:
        st.image(
            "https://images.pexels.com/photos/5474295/pexels-photo-5474295.jpeg",
            caption="Powered by Advanced NLP & GPT Models",
            use_column_width=True,
        )

    st.markdown("---")

    # -------------
    # Job description input
    # -------------
    st.markdown('<div class="section-header">1. Job Description</div>', unsafe_allow_html=True)
    jd_tab1, jd_tab2 = st.tabs(["Paste job description", "Paste job URL"])

    job_description_text = ""

    with jd_tab1:
        job_description_text = st.text_area(
            "Job description text",
            height=220,
            placeholder="Paste the full job description here (recommended)...",
        )

    with jd_tab2:
        job_url = st.text_input(
            "Job posting URL (LinkedIn, company site, etc.)",
            placeholder="https://www.linkedin.com/jobs/view/...",
        )
        if job_url:
            if st.button("Fetch job description from URL"):
                with st.spinner("Fetching and extracting job description from URL..."):
                    fetched = fetch_job_description_from_url(job_url)
                    if fetched.startswith("ERROR_FETCHING_URL"):
                        st.error(f"Could not fetch job description: {fetched}")
                    else:
                        job_description_text = fetched
                        st.success("Job description fetched successfully! You can fine-tune it in the text tab.")
                        # Show fetched text for editing
                        st.text_area(
                            "Fetched job description (editable)",
                            value=job_description_text,
                            height=220,
                        )

    st.markdown("---")

    # -------------
    # CV Preview
    # -------------
    st.markdown('<div class="section-header">2. CV Content Preview</div>', unsafe_allow_html=True)
    cv_text = extract_cv_text(cv_file) if cv_file else ""
    if cv_text:
        with st.expander("Show extracted CV text"):
            preview = cv_text[:5000]
            if len(cv_text) > 5000:
                preview += "\n\n...[truncated]"
            st.text(preview)
    else:
        st.info("Upload your CV from the sidebar to enable a complete analysis.")

    st.markdown("---")

    # -------------
    # Main navigation: ATS analysis vs LinkedIn jobs
    # -------------
    tab_analysis, tab_jobs = st.tabs(
        ["ATS Analysis & Cover Letter", "LinkedIn Job Matches"]
    )

    # --- Tab 1: ATS Analysis & Cover Letter ---
    with tab_analysis:
        st.markdown('<div class="section-header">3. AI Analysis & Cover Letter</div>', unsafe_allow_html=True)
        col_analyze_left, col_analyze_right = st.columns([1, 3])

        with col_analyze_left:
            analyze_clicked = st.button("🚀 Analyze & Generate", type="primary")

        if analyze_clicked:
            if not api_key:
                st.error("Please provide your OpenRouter API key in the sidebar.")
                return
            if not cv_text:
                st.error("Please upload your CV in the sidebar.")
                return
            if not job_description_text.strip():
                st.error("Please paste a job description or fetch it from a URL.")
                return

            with st.spinner("Calling AI and generating your ATS analysis and cover letter..."):
                result = analyze_with_llm(cv_text, job_description_text, api_key=api_key, model=model)

            if result is None:
                return

            ats = result.get("ats_score", {})
            skill_gaps = result.get("skill_gap_analysis", {})
            cv_improve = result.get("cv_improvement_suggestions", {})
            cover_letter = result.get("cover_letter", {})

            # -------------
            # ATS Score
            # -------------
            st.markdown('<div id="ats-score"></div>', unsafe_allow_html=True)
            st.markdown("#### ATS Score")
            col_score, col_reason = st.columns([1, 3])

            with col_score:
                st.markdown('<div class="score-box">', unsafe_allow_html=True)
                score_value = ats.get("score", 0)
                try:
                    score_value = float(score_value)
                except Exception:
                    score_value = 0
                st.markdown(
                    f'<span class="ats-score-badge">{score_value:.1f}/100</span>',
                    unsafe_allow_html=True,
                )
                if score_value >= 80:
                    st.success("Excellent match for this job.")
                elif score_value >= 60:
                    st.info("Good match with room for improvement.")
                else:
                    st.warning("Below average match. Consider improving your CV and skills alignment.")
                st.markdown("</div>", unsafe_allow_html=True)

            with col_reason:
                st.write(ats.get("reasoning", ""))

            st.markdown("---")

            # -------------
            # Skill Gap Analysis
            # -------------
            st.markdown('<div id="skill-gap"></div>', unsafe_allow_html=True)
            st.markdown("#### Skill Gap Analysis")
            col_gap_left, col_gap_right = st.columns(2)

            with col_gap_left:
                missing = skill_gaps.get("missing_skills", []) or []
                st.markdown("**Missing / Weak Skills**")
                if missing:
                    for s in missing:
                        st.markdown(f"- {s}")
                else:
                    st.write("No major missing skills detected.")

            with col_gap_right:
                nice_to_have = skill_gaps.get("nice_to_have_skills", []) or []
                st.markdown("**Nice-to-have Skills**")
                if nice_to_have:
                    for s in nice_to_have:
                        st.markdown(f"- {s}")
                else:
                    st.write("No additional nice-to-have skills identified.")

            st.markdown("**Summary**")
            st.write(skill_gaps.get("summary", ""))

            st.markdown("---")

            # -------------
            # CV Improvement Suggestions
            # -------------
            st.markdown('<div id="cv-improvements"></div>', unsafe_allow_html=True)
            st.markdown("#### CV Improvement Suggestions")
            col_cv_left, col_cv_right = st.columns(2)

            with col_cv_left:
                hi_changes = cv_improve.get("high_impact_changes", []) or []
                st.markdown("**High-impact Changes**")
                for item in hi_changes:
                    st.markdown(f"- {item}")

            with col_cv_right:
                rewrites = cv_improve.get("bullet_point_rewrites", []) or []
                st.markdown("**Example Bullet Point Rewrites**")
                for item in rewrites:
                    st.markdown(f"- {item}")

            st.markdown("**Overall Feedback**")
            st.write(cv_improve.get("overall_feedback", ""))

            st.markdown("---")

            # -------------
            # Cover Letter
            # -------------
            st.markdown('<div id="cover-letter"></div>', unsafe_allow_html=True)
            st.markdown("#### Tailored Cover Letter")
            cl_title = cover_letter.get("title", "Cover Letter")
            cl_body = cover_letter.get("body", "")

            st.subheader(cl_title)
            st.write(cl_body)

            pdf_bytes = generate_cover_letter_pdf(cl_title, cl_body)

            st.download_button(
                label="📄 Download Cover Letter as PDF",
                data=pdf_bytes,
                file_name="cover_letter.pdf",
                mime="application/pdf",
            )

            st.success("Analysis complete! You can now download your cover letter as a PDF.")

    # --- Tab 2: LinkedIn Jobs ---
    with tab_jobs:
        st.markdown('<div id="linkedin-jobs"></div>', unsafe_allow_html=True)
        st.markdown('<div class="section-header">4. Find Matching LinkedIn Job Vacancies</div>', unsafe_allow_html=True)
        st.caption("Use your CV to automatically search for 5–7 relevant LinkedIn job postings.")

        find_jobs_clicked = st.button("🔍 Find LinkedIn Jobs that Match My CV")

        if find_jobs_clicked:
            if not cv_text:
                st.error("Please upload your CV in the sidebar first.")
                return

            if not api_key:
                st.error("To build a smart search query from your CV, please enter your OpenRouter API key in the sidebar.")
                return

            with st.spinner("Analyzing your CV to build a job search query..."):
                search_query = build_job_search_query_from_cv(cv_text, api_key=api_key, model=model)

            if not search_query:
                st.error("Could not build a search query from your CV.")
                return

            st.info(f"Search query derived from your CV: **{search_query}**")

            with st.spinner("Searching LinkedIn job postings via SerpAPI..."):
                jobs = search_linkedin_jobs(search_query, serpapi_key=serpapi_key)

            if not jobs:
                st.warning("No LinkedIn job postings were found for this query. Try adjusting your CV or query.")
                return

            st.markdown("#### Matching LinkedIn Job Vacancies")
            st.caption("These are real LinkedIn job links discovered via Google (SerpAPI).")

            for i, job in enumerate(jobs, start=1):
                with st.container():
                    st.markdown('<div class="job-card">', unsafe_allow_html=True)
                    st.markdown(f"<div class='job-title'>{i}. {job['title']}</div>", unsafe_allow_html=True)
                    if job.get("snippet"):
                        st.write(job["snippet"])
                    st.markdown(
                        f"<a href='{job['link']}' class='job-link' target='_blank'>Open on LinkedIn &raquo;</a>",
                        unsafe_allow_html=True,
                    )
                    st.markdown("</div>", unsafe_allow_html=True)


if __name__ == "__main__":
    main()


