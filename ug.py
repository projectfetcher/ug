import re
import csv
import json
import time
import base64
import hashlib
import logging
import argparse
import os
from datetime import datetime
from dateutil.relativedelta import relativedelta
from urllib.parse import unquote
from typing import Optional

import requests
import pandas as pd
from bs4 import BeautifulSoup

# Optional NLP deps — imported lazily so scrape-only mode works without them
try:
    import language_tool_python
    from sentence_transformers import SentenceTransformer, util
    NLP_AVAILABLE = True
except ImportError:
    NLP_AVAILABLE = False

# ─────────────────────────────────────────────────────────────
#  CONFIG  —  edit these before running
# ─────────────────────────────────────────────────────────────

# ── Scraper ───────────────────────────────────────────────────
BASE_URL    = os.getenv("UG_BASE_URL", "")
LIST_PATH   = "/jobs/newest-jobs"
PAGE_SIZE   = 20
DELAY       = 1.5
MAX_RETRIES = 3

# ── Mistral ───────────────────────────────────────────────────
MISTRAL_API_KEY = os.getenv("MISTRAL_API_KEY", "YOUR_MISTRAL_API_KEY_HERE")
MISTRAL_URL     = "https://api.mistral.ai/v1/chat/completions"
MISTRAL_MODEL   = "mistral-small-latest"

# ── WordPress ─────────────────────────────────────────────────
WP_SITE_URL   = os.getenv("WP_SITE_URL",   "")
WP_USERNAME   = os.getenv("WP_USERNAME",   "")
WP_APP_PASSWORD = os.getenv("WP_APP_PASSWORD", "")

WP_BASE       = f"{WP_SITE_URL}/wp-json/wp/v2"
WP_URL        = f"{WP_BASE}/job-listings"
WP_MEDIA_URL  = f"{WP_BASE}/media"
WP_COMPANY_URL = f"{WP_BASE}/companies"          # adjust if CPT slug differs

# ── Tracker file ─────────────────────────────────────────────
PROCESSED_IDS_FILE = "processed_jobs.csv"

# ── Job type normalisation map ────────────────────────────────
JOB_TYPE_MAPPING = {
    "full time":  "full-time",
    "full-time":  "full-time",
    "part time":  "part-time",
    "part-time":  "part-time",
    "contract":   "contract",
    "contractor": "contract",
    "internship": "internship",
    "volunteer":  "volunteer",
    "temporary":  "temporary",
    "casual":     "casual",
    "freelance":  "freelance",
}

# ─────────────────────────────────────────────────────────────
#  LOGGING
# ─────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-7s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("ug_scraper")

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/124.0.0.0 Safari/537.36"
    ),
    "Accept-Language": "en-US,en;q=0.9",
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
}

# ─────────────────────────────────────────────────────────────
#  COLUMN DEFINITIONS
# ─────────────────────────────────────────────────────────────
COLUMNS = [
    "job_title", "job_type", "qualifications", "experience",
    "location", "field", "date_posted", "deadline",
    "job_description", "application", "company_url", "company_name",
    "company_logo", "industry", "founded", "company_type",
    "website", "address", "company_details",
    "job_url", "estimated_deadline", "salary_range",
]

COLUMN_LABELS = {
    "job_title":          "Job Title",
    "job_type":           "Job Type",
    "qualifications":     "Qualifications",
    "experience":         "Experience",
    "location":           "Location",
    "field":              "Field",
    "date_posted":        "Date Posted",
    "deadline":           "Deadline",
    "job_description":    "Job Description",
    "application":        "Application",
    "company_url":        "Company URL",
    "company_name":       "Company Name",
    "company_logo":       "Company Logo",
    "industry":           "Industry",
    "founded":            "Founded",
    "company_type":       "Company Type",
    "website":            "Website",
    "address":            "Address",
    "company_details":    "Company Details",
    "job_url":            "Job URL",
    "estimated_deadline": "Estimated Deadline",
    "salary_range":       "Salary Range",
}

GENERIC_PATTERNS = [
    r"^fresh jobs? at ",
    r"^latest (jobs?|recruitment|openings?) at ",
    r"^(job )?(openings?|opportunities|vacancies|positions?) at ",
    r"^open roles? at ",
    r"^careers? at ",
    r"^several jobs? at ",
    r"^multiple (positions?|jobs?) at ",
    r"^new recruitment at ",
]
_GENERIC_RE = re.compile("|".join(GENERIC_PATTERNS), re.I)

# ─────────────────────────────────────────────────────────────
#  NLP TOOLS  (grammar + similarity)
# ─────────────────────────────────────────────────────────────
_grammar_tool    = None
_similarity_model = None


def _get_grammar_tool():
    global _grammar_tool
    if _grammar_tool is None and NLP_AVAILABLE:
        try:
            _grammar_tool = language_tool_python.LanguageTool(
                "en-US", remote_server="https://api.languagetool.org"
            )
        except Exception as e:
            log.warning(f"LanguageTool init failed: {e}")
    return _grammar_tool


def _get_similarity_model():
    global _similarity_model
    if _similarity_model is None and NLP_AVAILABLE:
        try:
            _similarity_model = SentenceTransformer("all-MiniLM-L6-v2", device="cpu")
        except Exception as e:
            log.warning(f"SentenceTransformer init failed: {e}")
    return _similarity_model


def grammar_correct(text: str) -> str:
    tool = _get_grammar_tool()
    if not tool:
        return text
    try:
        return language_tool_python.utils.correct(text, tool.check(text))
    except Exception:
        return text


def similarity_score(a: str, b: str) -> float:
    model = _get_similarity_model()
    if not model:
        return 0.75  # neutral fallback when model unavailable
    try:
        emb = model.encode([a, b], convert_to_tensor=True)
        return float(util.pytorch_cos_sim(emb[0], emb[1]))
    except Exception:
        return 0.0


# ─────────────────────────────────────────────────────────────
#  MOJIBAKE / SANITISE HELPERS
# ─────────────────────────────────────────────────────────────
_MOJIBAKE = [
    ("Â", ""), ("â€™", "\u2019"), ("â€œ", "\u201c"),
    ("â€\x9d", "\u201d"), ("â€", "\u201d"),
    ("â€\x93", "\u2013"), ("â€\x94", "\u2014"),
    ("â€¢", "\u2022"), ("â„¢", "™"),
    ("\u00a0", " "), ("\u200b", ""), ("\ufeff", ""),
]


def _fix_mojibake(text: str) -> str:
    for pattern, replacement in _MOJIBAKE:
        text = text.replace(pattern, replacement)
    text = re.sub(r"[\x00-\x08\x0B\x0C\x0E-\x1F\x7F]", "", text)
    return text


def sanitize_text(text, is_url=False, is_email=False) -> str:
    if not isinstance(text, str):
        text = str(text) if pd.notna(text) else ""
    text = text.strip()
    if text in ("nan", "None", "NaN", "", "N/A", "n/a", "NA", "na"):
        return ""
    text = _fix_mojibake(text)
    if is_url or is_email:
        return re.sub(r"[ \t\r\n\f\v]+", " ", text).strip()
    text = re.sub(r"#+\s*", "", text)
    text = re.sub(r"\*\*", "", text)
    text = re.sub(
        r"[^\x20-\x7E\n\u00C0-\u017F\u2013\u2014\u2018-\u201D\u2022]", "", text
    )
    text = re.sub(r"[ \t]+", " ", text)
    return re.sub(r"\n{3,}", "\n\n", text).strip()


def clean_text(value: str) -> str:
    if not value:
        return ""
    value = _fix_mojibake(value)
    value = re.sub(r"[\x00-\x08\x0B\x0C\x0E-\x1F\x7F]", "", value)
    value = re.sub(r"\s+", " ", value).strip()
    value = re.sub(r"^N/A$", "", value, flags=re.I)
    return value


def clean_output(text: str) -> str:
    text = _fix_mojibake(text)
    for pat in [
        r"\[/?INST\]", r"</?s>",
        r"(?i)(rewritten?|rephrased?|output|paraphrase[d]?)[:\s]+",
        r"\*\*", r"###", r"---",
    ]:
        text = re.sub(pat, "", text)
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return grammar_correct(text.strip())


# ─────────────────────────────────────────────────────────────
#  FIELD CLEANERS
# ─────────────────────────────────────────────────────────────

def clean_job_type(raw: str) -> str:
    if not raw:
        return ""
    mapping = {
        "FULL_TIME": "Full Time", "FULLTIME": "Full Time", "FULL-TIME": "Full Time",
        "PART_TIME": "Part Time", "PARTTIME": "Part Time", "PART-TIME": "Part Time",
        "CONTRACT":  "Contract",  "CONTRACTOR": "Contract",
        "INTERNSHIP": "Internship", "VOLUNTEER": "Volunteer",
        "TEMPORARY": "Temporary", "CASUAL": "Casual", "FREELANCE": "Freelance",
    }
    key = raw.strip().upper().replace(" ", "_")
    return mapping.get(key, raw.strip().replace("_", " ").title())


def clean_experience(raw: str) -> str:
    if not raw:
        return ""
    raw = raw.strip()
    if re.search(r"\byears?\b", raw, re.I):
        return raw
    m = re.match(r"^(\d+)\s*(?:months?)?$", raw, re.I)
    if m:
        months = int(m.group(1))
        years  = months / 12
        if years == int(years):
            return f"{int(years)} year{'s' if years != 1 else ''}"
        return f"{years:.1f} years"
    return raw


def clean_location(raw: str) -> str:
    if not raw:
        return ""
    parts = re.split(r"[|\n,/]", raw)
    seen, result = set(), []
    for p in parts:
        p = p.strip()
        if p and p.lower() not in seen and p.lower() not in ("uganda", ""):
            seen.add(p.lower())
            result.append(p)
    return result[0] if result else raw.strip()


def clean_field(raw: str) -> str:
    if not raw:
        return ""
    raw = re.sub(r"\s*jobs?\s+in\s+\w+", "", raw, flags=re.I).strip()
    raw = re.sub(r",\s*", ", ", raw).strip(", ")
    return raw


def clean_deadline(raw: str) -> str:
    if not raw:
        return ""
    raw = raw.strip()
    m = re.match(r"(\d{4})-(\d{2})-(\d{2})", raw)
    if m:
        return f"{m.group(3)}-{m.group(2)}-{m.group(1)}"
    if re.match(r"\d{2}-\d{2}-\d{4}", raw):
        return raw
    for fmt in ("%A %B %d %Y", "%B %d %Y", "%d %B %Y"):
        try:
            dt = datetime.strptime(re.sub(r",", "", raw), fmt)
            return dt.strftime("%d-%m-%Y")
        except ValueError:
            pass
    return raw


def clean_date_posted(raw: str) -> str:
    return clean_deadline(raw)


def clean_application(raw: str) -> str:
    if not raw:
        return ""
    raw = raw.strip()
    if "application-email" in raw:
        m = re.search(r"application-email\]=([^&]+)", raw)
        if m:
            return unquote(m.group(1)).strip()
    if re.match(r"^[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z]{2,}$", raw):
        return raw
    return raw


def add_three_months(date_str: str) -> str:
    cleaned = clean_deadline(date_str)
    try:
        dt  = datetime.strptime(cleaned, "%d-%m-%Y")
        new = dt + relativedelta(months=3)
        return new.strftime("%d-%m-%Y")
    except Exception:
        return (datetime.today() + relativedelta(months=3)).strftime("%d-%m-%Y")


def normalise_job_type(raw: str) -> str:
    return JOB_TYPE_MAPPING.get(raw.lower().strip(), "full-time")


# ─────────────────────────────────────────────────────────────
#  TITLE HELPERS
# ─────────────────────────────────────────────────────────────

def extract_raw_title(soup: BeautifulSoup) -> str:
    t = soup.find("div", itemprop="title")
    if t:
        val = t.get_text(strip=True)
        if val:
            return val
    span = soup.find("span", class_="jsjobs-main-page-title")
    if span:
        val = span.get_text(strip=True)
        if val:
            return val
    body = soup.find("div", class_="jsjobs_description_data")
    if body:
        m = re.search(r"Vacancy title:\s*\n?([^\n\[]+)", body.get_text())
        if m:
            return m.group(1).strip()
    og = soup.find("meta", property="og:title")
    if og and og.get("content"):
        return re.sub(r"\s*[\|\-–]\s*.*$", "", og["content"]).strip()
    t = soup.find("title")
    if t:
        return re.sub(r"\s*[\|\-–]\s*.*$", "", t.get_text()).strip()
    return ""


def clean_title(raw: str) -> Optional[str]:
    if not raw:
        return None
    title = raw.strip()
    if _GENERIC_RE.match(title):
        return None
    m = re.match(r"^(.+?)\s+job\s+at\s+.+$", title, re.I)
    if m:
        title = m.group(1).strip()
    else:
        m = re.match(r"^(.+?)\s+at\s+.+$", title, re.I)
        if m:
            title = m.group(1).strip()
    title = re.sub(r"[\s,\-–]+$", "", title).strip()
    if len(title) > 150:
        title = title[:147] + "…"
    return title or None


# ─────────────────────────────────────────────────────────────
#  SIDE-PANEL FIELD EXTRACTOR
# ─────────────────────────────────────────────────────────────

def extract_info_field(soup: BeautifulSoup, label: str) -> str:
    for wrapper in soup.find_all("div", class_="js_job_data_wrapper"):
        title_el = wrapper.find("span", class_="js_job_data_title")
        if not title_el:
            continue
        if label.lower() in title_el.get_text(strip=True).lower():
            val_el = wrapper.find("span", class_="js_job_data_value")
            return val_el.get_text(strip=True) if val_el else ""
    return ""


# ─────────────────────────────────────────────────────────────
#  DESCRIPTION EXTRACTOR
# ─────────────────────────────────────────────────────────────

def extract_description(soup: BeautifulSoup) -> str:
    container = (
        soup.find("div", class_="jsjobs_description_data") or
        soup.find("div", itemprop="description")
    )
    if not container:
        return ""

    lines = []

    def walk(el):
        tag = el.name if el.name else None
        if tag in ("script", "style", "noscript"):
            return
        if tag in ("h1", "h2", "h3", "h4", "h5", "h6"):
            text = el.get_text(" ", strip=True)
            if text:
                lines.append("")
                lines.append(text.upper())
                lines.append("")
        elif tag == "li":
            text = el.get_text(" ", strip=True)
            if text:
                lines.append(f"  • {text}")
        elif tag == "p":
            text = el.get_text(" ", strip=True)
            if text:
                lines.append(text)
                lines.append("")
        elif tag == "br":
            lines.append("")
        elif tag in ("ul", "ol", "div", "section", "article", "aside"):
            for child in el.children:
                walk(child)
        elif tag is None:
            text = str(el).strip()
            if text:
                lines.append(text)
        else:
            for child in el.children:
                walk(child)

    for child in container.children:
        walk(child)

    result, blank_count = [], 0
    for line in lines:
        if line == "":
            blank_count += 1
            if blank_count <= 1:
                result.append("")
        else:
            blank_count = 0
            result.append(line)

    return "\n".join(result).strip()


# ─────────────────────────────────────────────────────────────
#  VERBOSE PRINTER
# ─────────────────────────────────────────────────────────────

def print_job(job: dict, index: int, total: int):
    width = 78
    print(f"\n{'═' * width}")
    print(f"  JOB {index} / {total}  ─  {job.get('job_url', '')}")
    print(f"{'═' * width}")

    for col in COLUMNS:
        label = COLUMN_LABELS.get(col, col)
        value = job.get(col, "") or ""

        if col == "job_description":
            print(f"\n  {'─' * (width - 2)}")
            print(f"  {'JOB DESCRIPTION':^{width - 2}}")
            print(f"  {'─' * (width - 2)}")
            if value:
                for line in value.split("\n"):
                    while len(line) > width - 4:
                        print(f"  {line[:width - 4]}")
                        line = "  " + line[width - 4:]
                    print(f"  {line}")
            else:
                print("  (no description)")
            print(f"  {'─' * (width - 2)}\n")

        elif col == "company_details":
            print(f"\n  {'─' * (width - 2)}")
            print(f"  {'COMPANY DETAILS':^{width - 2}}")
            print(f"  {'─' * (width - 2)}")
            if value:
                for line in value.split("\n"):
                    while len(line) > width - 4:
                        print(f"  {line[:width - 4]}")
                        line = "  " + line[width - 4:]
                    print(f"  {line}")
            else:
                print("  (not available)")
            print(f"  {'─' * (width - 2)}\n")

        else:
            display   = str(value) if value else "(not found)"
            prefix    = f"  {label:<22}: "
            available = width - len(prefix)
            indent    = " " * len(prefix)
            is_url    = display.startswith("http") or display.startswith("//")
            if is_url:
                print(f"{prefix}")
                print(f"{indent}{display}")
            elif len(display) <= available:
                print(f"{prefix}{display}")
            else:
                print(f"{prefix}{display[:available]}")
                remaining = display[available:]
                while remaining:
                    print(f"{indent}{remaining[:available]}")
                    remaining = remaining[available:]

    print(f"{'═' * width}\n")


# ─────────────────────────────────────────────────────────────
#  HTTP HELPER
# ─────────────────────────────────────────────────────────────
session = requests.Session()
session.headers.update(HEADERS)


def fetch(url: str, retries: int = MAX_RETRIES) -> Optional[BeautifulSoup]:
    for attempt in range(1, retries + 1):
        try:
            r = session.get(url, timeout=20)
            r.raise_for_status()
            return BeautifulSoup(r.text, "lxml")
        except Exception as e:
            log.warning(f"Attempt {attempt}/{retries} failed for {url}: {e}")
            if attempt < retries:
                time.sleep(DELAY * attempt)
    return None


# ─────────────────────────────────────────────────────────────
#  COMPANY PAGE SCRAPER
# ─────────────────────────────────────────────────────────────

def scrape_company_page(url: str) -> dict:
    out = {"company_details": "", "founded": "", "company_type": "", "address": ""}
    if not url:
        return out
    soup = fetch(url)
    if not soup:
        return out
    time.sleep(DELAY)

    about = (
        soup.find("div", class_="jsjobs-full-width-data") or
        soup.find("div", attrs={"itemprop": "description"})
    )
    if about:
        out["company_details"] = clean_text(about.get_text(" ", strip=True))

    for li in soup.find_all("li"):
        lbl_el = li.find("span", class_="comp-info-title")
        val_el = li.find("span", class_="comp-info-desc")
        if not lbl_el or not val_el:
            continue
        lbl = lbl_el.get_text(strip=True).lower()
        val = val_el.get_text(strip=True)
        if "founded" in lbl:
            out["founded"] = val
        elif "type" in lbl:
            out["company_type"] = val
        elif "address" in lbl or "location" in lbl:
            out["address"] = val

    return out


# ─────────────────────────────────────────────────────────────
#  SINGLE JOB DETAIL SCRAPER
# ─────────────────────────────────────────────────────────────

def scrape_job(url: str) -> Optional[dict]:
    soup = fetch(url)
    if not soup:
        return None
    time.sleep(DELAY)

    raw_title = extract_raw_title(soup)
    job_title = clean_title(raw_title)
    if not job_title:
        return None

    def sd(itemprop: str) -> str:
        el = soup.find(attrs={"itemprop": itemprop})
        return clean_text(el.get_text(" ", strip=True)) if el else ""

    raw_type   = sd("employmentType") or extract_info_field(soup, "Job Type")
    job_type   = clean_job_type(raw_type)
    qual_el    = soup.find(attrs={"itemprop": "credentialCategory"})
    qualifications = clean_text(qual_el.get_text(strip=True)) if qual_el else ""
    exp_el     = soup.find(attrs={"itemprop": "monthsOfExperience"})
    raw_exp    = clean_text(exp_el.get_text(strip=True)) if exp_el else ""
    experience = clean_experience(raw_exp)

    raw_loc  = extract_info_field(soup, "Duty Station") or sd("addressLocality")
    location = clean_location(raw_loc)
    raw_field = extract_info_field(soup, "Job Category") or sd("occupationalCategory")
    field     = clean_field(raw_field)

    raw_posted = re.sub(r"T\d{2}:\d{2}:\d{2}.*", "",
                        sd("datePosted") or extract_info_field(soup, "Posted")).strip()
    date_posted  = clean_date_posted(raw_posted)
    raw_deadline = re.sub(r"T\d{2}:\d{2}:\d{2}.*", "",
                          sd("validThrough") or extract_info_field(soup, "Deadline of this Job")).strip()
    deadline     = clean_deadline(raw_deadline)
    est_deadline = add_three_months(date_posted) if date_posted else ""

    sal_val  = soup.find("div", itemprop="value")
    sal_unit = soup.find("div", itemprop="unitText")
    salary   = ""
    if sal_val and sal_val.get_text(strip=True):
        cur_el = soup.find("div", itemprop="currency")
        cur    = cur_el.get_text(strip=True) if cur_el else ""
        salary = f"{cur} {sal_val.get_text(strip=True)} / {sal_unit.get_text(strip=True) if sal_unit else 'month'}".strip()
    if not salary:
        og = soup.find("meta", property="og:description")
        if og:
            m = re.search(r"Base Salary:\s*([^\n,\]]+)", og.get("content", ""))
            if m:
                val = m.group(1).strip()
                if val.lower() not in ("not disclosed", "n/a", ""):
                    salary = val

    job_description = extract_description(soup)

    application = ""
    for a in soup.find_all("a", href=True):
        txt  = a.get_text(strip=True).lower()
        href = a["href"]
        if "apply" in txt or "click here" in txt:
            if "application-email" in href:
                application = clean_application(href)
                break
            if href.startswith("http") and BASE_URL not in href:
                application = href
                break
    if not application:
        m = re.search(r"[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z]{2,}", job_description)
        if m:
            application = m.group(0)
    if not application:
        for a in soup.find_all("a", href=True):
            if "application-email" in a["href"]:
                application = clean_application(a["href"])
                break

    comp_anchor      = soup.find("a", class_="js_job_company_anchor")
    company_name     = ""
    company_page_url = ""
    if comp_anchor:
        company_name = clean_text(comp_anchor.get_text(strip=True))
        href = comp_anchor.get("href", "")
        if href:
            company_page_url = href if href.startswith("http") else BASE_URL + href
    if not company_name:
        org_name = soup.find("div", itemprop="name")
        if org_name:
            company_name = clean_text(org_name.get_text(strip=True))

    logo_el = soup.find("img", class_="js_jobs_company_logo")
    company_logo = ""
    if logo_el:
        raw_src = (
            logo_el.get("src") or
            logo_el.get("data-src") or
            logo_el.get("data-lazy-src") or ""
        ).strip()
        if raw_src.startswith("//"):
            company_logo = "https:" + raw_src
        elif raw_src.startswith("/"):
            company_logo = BASE_URL + raw_src
        elif raw_src.startswith("http"):
            company_logo = raw_src
        if any(x in company_logo for x in ("blank.gif", "placeholder", "no-image")):
            company_logo = ""

    website_el      = soup.find("div", itemprop="url")
    company_website = clean_text(website_el.get_text(strip=True)) if website_el else ""
    industry        = clean_text(sd("industry"))

    company_extra = {}
    if company_page_url:
        log.info(f"    → Fetching company page: {company_page_url}")
        company_extra = scrape_company_page(company_page_url)

    return {
        "job_title":          job_title,
        "job_type":           job_type,
        "qualifications":     qualifications,
        "experience":         experience,
        "location":           location,
        "field":              field,
        "date_posted":        date_posted,
        "deadline":           deadline,
        "job_description":    job_description,
        "application":        application,
        "company_url":        company_page_url,
        "company_name":       company_name,
        "company_logo":       company_logo,
        "industry":           industry,
        "founded":            company_extra.get("founded", ""),
        "company_type":       company_extra.get("company_type", ""),
        "website":            company_website,
        "address":            company_extra.get("address", ""),
        "company_details":    company_extra.get("company_details", ""),
        "job_url":            url,
        "estimated_deadline": est_deadline,
        "salary_range":       salary,
    }


# ─────────────────────────────────────────────────────────────
#  LISTING PAGE  —  collect job URLs
# ─────────────────────────────────────────────────────────────

def collect_job_urls(max_pages: Optional[int] = None) -> list:
    seen, urls, page = set(), [], 0
    while True:
        if max_pages and page >= max_pages:
            break
        offset = page * PAGE_SIZE
        url    = BASE_URL + LIST_PATH + (f"?start={offset}" if offset else "")
        log.info(f"📄 Listing page {page + 1}: {url}")
        soup = fetch(url)
        if not soup:
            log.warning(f"Failed to fetch listing page {page + 1}, stopping.")
            break
        found = 0
        for a in soup.find_all("a", href=True):
            href = a["href"]
            if "job-detail" not in href:
                continue
            full = href if href.startswith("http") else BASE_URL + href
            full = re.sub(r"/nav-\d+.*$", "", full)
            if full not in seen:
                seen.add(full)
                urls.append(full)
                found += 1
        log.info(f"   Found {found} new URLs  (total: {len(urls)})")
        if found == 0:
            log.info("   No new jobs — end of listings.")
            break
        page += 1
        time.sleep(DELAY)
    return urls


# ─────────────────────────────────────────────────────────────
#  SAVE CSV / JSON
# ─────────────────────────────────────────────────────────────

def save_csv(jobs: list, filename: str):
    with open(filename, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=COLUMNS, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(jobs)
    log.info(f"💾 Saved {len(jobs)} jobs → {filename}")


def save_json(jobs: list, filename: str):
    with open(filename, "w", encoding="utf-8") as f:
        json.dump(jobs, f, ensure_ascii=False, indent=2)
    log.info(f"💾 Saved {len(jobs)} jobs → {filename}")


# ═══════════════════════════════════════════════════════════════
#  MISTRAL PARAPHRASING
# ═══════════════════════════════════════════════════════════════

def mistral_generate(prompt: str, max_tokens: int = 400, temperature: float = 0.7) -> str:
    try:
        response = requests.post(
            MISTRAL_URL,
            headers={
                "Authorization": f"Bearer {MISTRAL_API_KEY}",
                "Content-Type":  "application/json",
            },
            json={
                "model":       MISTRAL_MODEL,
                "messages":    [{"role": "user", "content": prompt}],
                "max_tokens":  max_tokens,
                "temperature": temperature,
            },
            timeout=30,
        )
        response.raise_for_status()
        return response.json()["choices"][0]["message"]["content"].strip()
    except Exception as e:
        log.error(f"Mistral API error: {e}")
        return ""


def paraphrase_title(title: str) -> str:
    clean = sanitize_text(title)
    if not clean:
        return title

    print(f"\n ┌─ TITLE PARAPHRASE {'─'*45}")
    print(f" │ Original : \"{clean}\"")
    print(f" │ {'─'*60}")

    best_result, best_sim = None, 0.0

    for attempt in range(4):
        temp = round(0.68 + attempt * 0.06, 2)
        print(f" │ Attempt {attempt+1} (temp={temp}):")

        prompt = (
            f"Rewrite this job title professionally using different words. "
            f"Output ONLY the rewritten title, nothing else. "
            f"Keep it between 4 and 12 words.\n\nJob title: {clean}"
        )
        raw    = mistral_generate(prompt, max_tokens=50, temperature=temp)
        result = clean_output(raw).split("\n")[0].strip().strip('"').strip("'")

        wc     = len(result.split()) if result else 0
        sim    = similarity_score(clean, result) if result else 0.0
        is_dup = result.lower().strip() == clean.lower().strip()

        print(f" │    Output  : \"{result}\"")
        print(f" │    Words   : {wc} | Similarity: {sim:.3f} | Duplicate: {'Yes ⚠️' if is_dup else 'No'}")

        valid = bool(result) and 4 <= wc <= 14 and sim >= 0.55 and not is_dup
        if not valid:
            reasons = []
            if not result:  reasons.append("empty output")
            if wc < 4:      reasons.append(f"too short ({wc} words, min=4)")
            if wc > 14:     reasons.append(f"too long ({wc} words, max=14)")
            if sim < 0.55:  reasons.append(f"sim={sim:.3f} < 0.55")
            if is_dup:      reasons.append("identical to original")
            print(f" │    → ❌ REJECTED — {', '.join(reasons)}")
        else:
            if sim > best_sim:
                best_sim, best_result = sim, result
                print(f" │    → ✅ ACCEPTED — new best (sim={sim:.3f})")
            else:
                print(f" │    → ✅ VALID but not better (best sim={best_sim:.3f})")

        print(f" │ {'─'*60}")
        time.sleep(1)

    if best_result:
        print(f" │ 🏆 FINAL: \"{best_result}\" (sim={best_sim:.3f})")
        print(f" └{'─'*65}")
        return best_result
    else:
        print(f" │ ⚠️  No valid paraphrase → keeping original: \"{clean}\"")
        print(f" └{'─'*65}")
        return clean


def paraphrase_description(text: str) -> str:
    clean = sanitize_text(text)
    if not clean:
        return text

    paragraphs    = [p.strip() for p in clean.split("\n") if p.strip()]
    rewritten     = []
    success_count = 0

    print(f"\n ┌─ DESCRIPTION PARAPHRASE ({len(paragraphs)} paragraphs) {'─'*25}")

    for i, para in enumerate(paragraphs):
        orig_wc = len(para.split())
        print(f"\n │ ┌─ Paragraph {i+1}/{len(paragraphs)} {'─'*50}")
        print(f" │ │ ORIGINAL ({orig_wc} words):")
        orig_line = []
        for w in para.split():
            orig_line.append(w)
            if len(" ".join(orig_line)) >= 100:
                print(f" │ │    {' '.join(orig_line)}")
                orig_line = []
        if orig_line:
            print(f" │ │    {' '.join(orig_line)}")
        print(f" │ │ {'─'*60}")

        prompt = (
            f"Rewrite this job description paragraph professionally. "
            f"Keep ALL facts, requirements, and responsibilities. "
            f"Use different sentence structure and vocabulary. "
            f"Output ONLY the rewritten paragraph — no labels, no explanation.\n\n"
            f"Original:\n{para}"
        )

        best_result, best_sim, accepted_text = None, 0.0, None

        for attempt in range(3):
            temp   = round(0.65 + attempt * 0.08, 2)
            print(f" │ │ Attempt {attempt+1}/3 (temp={temp}):")
            raw    = mistral_generate(prompt, max_tokens=500, temperature=temp)
            result = clean_output(raw).strip()
            rw     = len(result.split()) if result else 0
            sim    = similarity_score(para, result) if result and rw >= 5 else 0.0

            if result:
                print(f" │ │    Paraphrased ({rw} words, sim={sim:.3f}):")
                line = []
                for w in result.split():
                    line.append(w)
                    if len(" ".join(line)) >= 100:
                        print(f" │ │       {' '.join(line)}")
                        line = []
                if line:
                    print(f" │ │       {' '.join(line)}")
            else:
                print(f" │ │    Paraphrased : (no output from model)")

            valid = bool(result) and rw >= 8 and sim >= 0.48
            if not valid:
                reasons = []
                if not result:  reasons.append("empty output")
                if rw < 8:      reasons.append(f"too short ({rw} words, min=8)")
                if sim < 0.48:  reasons.append(f"sim={sim:.3f} < 0.48")
                print(f" │ │    → ❌ REJECTED — {', '.join(reasons)}")
                if result and sim > best_sim:
                    best_sim, best_result = sim, result
                    print(f" │ │       (stored as best fallback, sim={sim:.3f})")
            else:
                print(f" │ │    → ✅ ACCEPTED on attempt {attempt+1}")
                rewritten.append(result)
                success_count += 1
                accepted_text = result
                break
            print(f" │ │ {'─'*60}")
            time.sleep(1)

        if accepted_text is None:
            print(f" │ │ {'─'*60}")
            if best_result and best_sim >= 0.40:
                print(f" │ │ 🔁 FALLBACK — Using best attempt (sim={best_sim:.3f})")
                rewritten.append(best_result)
                success_count += 1
            else:
                print(f" │ │ ⚠️  KEPT ORIGINAL (best sim={best_sim:.3f}, threshold=0.40)")
                rewritten.append(para)
        print(f" │ └{'─'*62}")

    print(f"\n │ SUMMARY: {success_count}/{len(paragraphs)} paragraphs paraphrased")
    print(f" └{'─'*80}\n")
    return "\n\n".join(rewritten)


def paraphrase_company(text: str) -> str:
    clean = sanitize_text(text)
    if not clean:
        return text

    print(f"\n ┌─ COMPANY PARAPHRASE {'─'*43}")
    orig_wc = len(clean.split())
    print(f" │ Original ({orig_wc} words):")
    line = []
    for w in clean.split():
        line.append(w)
        if len(" ".join(line)) >= 100:
            print(f" │    {' '.join(line)}")
            line = []
    if line:
        print(f" │    {' '.join(line)}")
    print(f" │ {'─'*60}")

    prompt = (
        f"Rewrite this company description professionally. "
        f"Preserve all facts. Use different wording. "
        f"Output ONLY the rewritten description.\n\nOriginal:\n{clean}"
    )
    raw    = mistral_generate(prompt, max_tokens=600, temperature=0.68)
    result = clean_output(raw)
    rw     = len(result.split()) if result else 0
    sim    = similarity_score(clean, result) if result and rw >= 10 else 0.0

    if result and rw >= 10:
        print(f" │ Paraphrased ({rw} words, sim={sim:.3f}):")
        line = []
        for w in result.split():
            line.append(w)
            if len(" ".join(line)) >= 100:
                print(f" │    {' '.join(line)}")
                line = []
        if line:
            print(f" │    {' '.join(line)}")
        print(f" │ → ✅ ACCEPTED")
        print(f" └{'─'*65}")
        time.sleep(1)
        return result
    else:
        reasons = []
        if not result:  reasons.append("empty output")
        if rw < 10:     reasons.append(f"too short ({rw} words, min=10)")
        print(f" │ → ❌ REJECTED — {', '.join(reasons)} — keeping original")
        print(f" └{'─'*65}")
        time.sleep(1)
        return clean


def paraphrase_tagline(text: str) -> str:
    clean = sanitize_text(text[:300])
    if not clean:
        return text

    print(f"\n ┌─ TAGLINE PARAPHRASE {'─'*43}")
    print(f" │ Original : \"{clean}\"")
    print(f" │ {'─'*60}")

    prompt = (
        f"Rewrite this company tagline as a crisp, professional phrase. "
        f"Output ONLY the rewritten tagline (5–12 words). No explanation.\n\n"
        f"Original: {clean}"
    )
    raw    = mistral_generate(prompt, max_tokens=35, temperature=0.75)
    result = clean_output(raw).split("\n")[0].strip().strip('"').strip("'")
    wc     = len(result.split()) if result else 0
    sim    = similarity_score(clean, result) if result else 0.0

    print(f" │ Paraphrased : \"{result}\"")
    print(f" │ Words: {wc} | Similarity: {sim:.3f}")

    if result and 3 <= wc <= 15:
        print(f" │ → ✅ ACCEPTED")
        print(f" └{'─'*65}")
        time.sleep(1)
        return result
    else:
        reasons = []
        if not result:  reasons.append("empty output")
        if wc < 3:      reasons.append(f"too short ({wc} words, min=3)")
        if wc > 15:     reasons.append(f"too long ({wc} words, max=15)")
        print(f" │ → ❌ REJECTED — {', '.join(reasons)} — keeping original")
        print(f" └{'─'*65}")
        time.sleep(1)
        return clean


# ═══════════════════════════════════════════════════════════════
#  DUPLICATE TRACKER
# ═══════════════════════════════════════════════════════════════

def _init_tracker():
    if not os.path.exists(PROCESSED_IDS_FILE):
        pd.DataFrame(columns=[
            "Job ID", "Job URL", "Job Title", "Company Name",
            "Status", "Timestamp", "Sheet Row",
        ]).to_csv(PROCESSED_IDS_FILE, index=False)


def load_processed_ids() -> tuple:
    _init_tracker()
    df = pd.read_csv(PROCESSED_IDS_FILE)
    return (
        set(df["Job ID"].fillna("").astype(str)),
        set(df.get("Job URL", pd.Series()).fillna("").astype(str)),
    )


def _upsert_row(job_id: str, updates: dict):
    _init_tracker()
    df   = pd.read_csv(PROCESSED_IDS_FILE)
    mask = df["Job ID"].astype(str) == str(job_id)
    if mask.any():
        for col, val in updates.items():
            if col in df.columns:
                df.loc[mask, col] = val
        df.loc[mask, "Timestamp"] = datetime.now().isoformat()
    else:
        row = {"Job ID": job_id, "Timestamp": datetime.now().isoformat()}
        row.update(updates)
        df = pd.concat([df, pd.DataFrame([row])], ignore_index=True)
    df.to_csv(PROCESSED_IDS_FILE, index=False)


def make_job_id(job: dict, idx: int) -> str:
    src = sanitize_text(str(job.get("job_url", "")), is_url=True)
    if src:
        return hashlib.md5(src.encode()).hexdigest()[:16]
    seed = f"{job.get('job_title','')}{job.get('company_name','')}{idx}"
    return hashlib.md5(seed.encode()).hexdigest()[:16]


def mark_read(job_id, job_url, title, company, sheet_row):
    _upsert_row(job_id, {
        "Job URL": job_url, "Job Title": title,
        "Company Name": company, "Status": "read", "Sheet Row": sheet_row,
    })


def mark_posted(job_id, wp_id, wp_url):
    _upsert_row(job_id, {"Status": f"posted|wp_id={wp_id}|{wp_url}"})


def mark_failed(job_id, reason):
    _upsert_row(job_id, {"Status": f"failed|{reason}"})


def print_tracker_summary():
    if not os.path.exists(PROCESSED_IDS_FILE):
        return
    df     = pd.read_csv(PROCESSED_IDS_FILE)
    print(f"\n{'═'*55}")
    print(f" TRACKER SUMMARY ({len(df)} total records)")
    print(f"{'═'*55}")
    counts = df["Status"].str.split("|").str[0].value_counts()
    icons  = {"read": "🔵", "paraphrased": "🟡", "posted": "✅", "failed": "❌"}
    for status, count in counts.items():
        print(f" {icons.get(status,'⚪')} {status:<15} {count}")
    print(f"{'═'*55}\n")


# ═══════════════════════════════════════════════════════════════
#  WORDPRESS HELPERS
# ═══════════════════════════════════════════════════════════════

def wp_headers() -> dict:
    token = base64.b64encode(f"{WP_USERNAME}:{WP_APP_PASSWORD}".encode()).decode()
    return {"Authorization": f"Basic {token}", "Content-Type": "application/json"}


def upload_logo(logo_url: str) -> Optional[int]:
    logo_url = sanitize_text(logo_url, is_url=True)
    if not logo_url or not logo_url.startswith("http"):
        return None
    ext = logo_url.lower().rsplit(".", 1)[-1]
    if ext not in ("png", "jpg", "jpeg", "webp"):
        return None
    try:
        img = requests.get(logo_url, timeout=10)
        img.raise_for_status()
        h = wp_headers()
        h["Content-Disposition"] = f"attachment; filename={logo_url.split('/')[-1]}"
        h["Content-Type"]        = img.headers.get("content-type", "image/jpeg")
        r = requests.post(
            WP_MEDIA_URL, headers=h, data=img.content,
            auth=(WP_USERNAME, WP_APP_PASSWORD), timeout=15, verify=False,
        )
        r.raise_for_status()
        return r.json().get("id")
    except Exception as e:
        log.error(f"Logo upload error: {e}")
        return None


def get_or_create_term(taxonomy_url: str, name: str) -> Optional[int]:
    if not name or not name.strip():
        return None
    slug = re.sub(r"[^a-z0-9-]", "-", name.lower().strip())
    try:
        r     = requests.get(f"{taxonomy_url}?slug={slug}", headers=wp_headers(), timeout=10, verify=False)
        terms = r.json()
        if isinstance(terms, list) and terms:
            return terms[0]["id"]
    except Exception:
        pass
    try:
        r = requests.post(
            taxonomy_url, json={"name": name, "slug": slug},
            headers=wp_headers(), auth=(WP_USERNAME, WP_APP_PASSWORD),
            timeout=10, verify=False,
        )
        return r.json().get("id")
    except Exception as e:
        log.error(f"Term create error '{name}': {e}")
        return None


def save_company(job: dict) -> tuple:
    """Create or find a company CPT post in WordPress."""
    name = sanitize_text(job.get("company_name", ""))
    if not name or name in ("Unknown Company", "nan"):
        return None, None
    slug = re.sub(r"[^a-z0-9-]", "-", name.lower())
    try:
        r     = requests.get(f"{WP_COMPANY_URL}?slug={slug}", headers=wp_headers(), timeout=10, verify=False)
        posts = r.json()
        if isinstance(posts, list) and posts:
            log.info(f"⏭ Company exists: {name}")
            return posts[0]["id"], posts[0].get("link")
    except Exception:
        pass

    attachment_id = upload_logo(job.get("company_logo", ""))
    raw           = job.get("company_details", "")
    details       = paraphrase_company(raw) if raw else ""
    tagline       = paraphrase_tagline(raw[:300]) if raw else ""

    payload = {
        "title":          name,
        "content":        details,
        "status":         "publish",
        "featured_media": attachment_id or 0,
        "meta": {
            "_company_name":     name,
            "_company_logo":     str(attachment_id) if attachment_id else "",
            "_company_industry": sanitize_text(job.get("industry", "")),
            "_company_website":  sanitize_text(job.get("website", ""), is_url=True),
            "_company_tagline":  tagline,
        },
    }
    try:
        r = requests.post(
            WP_COMPANY_URL, json=payload, headers=wp_headers(),
            auth=(WP_USERNAME, WP_APP_PASSWORD), timeout=15, verify=False,
        )
        r.raise_for_status()
        post = r.json()
        log.info(f"✅ Company posted: {name} → ID {post.get('id')}")
        return post.get("id"), post.get("link")
    except Exception as e:
        log.error(f"Company post error '{name}': {e}")
        return None, None


def post_job_to_wp(job: dict, title: str, description: str) -> tuple:
    """Post a single job to WordPress WP Job Manager."""
    h = wp_headers()

    # Pre-seed common job types
    for jt_label in ["Full Time", "Part Time", "Contract",
                     "Temporary", "Freelance", "Internship", "Volunteer"]:
        get_or_create_term(f"{WP_BASE}/job_listing_type", jt_label)

    location    = sanitize_text(job.get("location", "Uganda"))
    raw_type    = sanitize_text(job.get("job_type", "Full Time"))
    job_type_s  = normalise_job_type(raw_type)
    company     = sanitize_text(job.get("company_name", ""))
    application = sanitize_text(job.get("application", ""), is_url=True)
    deadline    = sanitize_text(job.get("deadline", ""))
    logo_url    = sanitize_text(job.get("company_logo", ""), is_url=True)
    co_website  = sanitize_text(job.get("website", ""), is_url=True)
    qualif      = sanitize_text(job.get("qualifications", ""))
    experience  = sanitize_text(job.get("experience", ""))
    industry    = sanitize_text(job.get("industry", ""))
    co_address  = sanitize_text(job.get("address", ""))
    job_field   = sanitize_text(job.get("field", ""))
    job_url     = sanitize_text(job.get("job_url", ""), is_url=True)
    co_founded  = sanitize_text(job.get("founded", ""))
    co_type     = sanitize_text(job.get("company_type", ""))
    salary      = sanitize_text(job.get("salary_range", ""))
    if not deadline:
        deadline = sanitize_text(job.get("estimated_deadline", ""))

    is_email = bool(re.match(r"^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$", application))
    is_url_v = bool(re.match(r"^https?://[^\s]+$", application))
    if not (is_email or is_url_v):
        application = ""

    slug = re.sub(r"[^a-z0-9-]", "-", title.lower())[:80]
    try:
        r     = requests.get(f"{WP_URL}?slug={slug}", headers=h, timeout=10, verify=False)
        posts = r.json()
        if isinstance(posts, list) and posts:
            log.info(f"⏭ Job already on WP: {title}")
            return posts[0]["id"], posts[0].get("link")
    except Exception:
        pass

    attachment_id    = upload_logo(logo_url)
    region_term_id   = get_or_create_term(f"{WP_BASE}/job_listing_region", location)
    job_type_term_id = get_or_create_term(
        f"{WP_BASE}/job_listing_type", job_type_s.replace("-", " ").title()
    )

    payload = {
        "title":          title,
        "content":        description,
        "status":         "publish",
        "featured_media": attachment_id or 0,
        "meta": {
            "_job_title":          title,
            "_job_location":       location,
            "_job_type":           job_type_s,
            "_job_description":    description,
            "_application":        application,
            "_job_expires":        deadline,
            "_company_name":       company,
            "_company_website":    co_website,
            "_company_logo":       str(attachment_id) if attachment_id else "",
            "_company_industry":   industry,
            "_company_address":    co_address,
            "_company_founded":    co_founded,
            "_company_type":       co_type,
            "_job_qualifications": qualif,
            "_job_experiences":    experience,
            "_job_field":          job_field,
            "_job_source_url":     job_url,
            "_job_salary":         salary,
        },
    }
    if region_term_id:
        payload["job_listing_region"] = [region_term_id]
    if job_type_term_id:
        payload["job_listing_type"] = [job_type_term_id]

    for attempt in range(3):
        try:
            r = requests.post(
                WP_URL, json=payload, headers=h,
                auth=(WP_USERNAME, WP_APP_PASSWORD), timeout=20, verify=False,
            )
            r.raise_for_status()
            post = r.json()
            log.info(f"✅ Job posted: '{title}' → WP ID {post.get('id')}")
            return post.get("id"), post.get("link")
        except Exception as e:
            log.error(f"Job post attempt {attempt+1} failed: {e}")
            if attempt < 2:
                time.sleep(2 ** attempt)
    return None, None


# ═══════════════════════════════════════════════════════════════
#  MAIN
# ═══════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="Scrape GreatUgandaJobs → Paraphrase → Post to WordPress"
    )
    parser.add_argument("--pages",        type=int, default=None,
                        help="Max listing pages (default: all)")
    parser.add_argument("--output",       default="jobs",
                        help="Output filename without extension (default: jobs)")
    parser.add_argument("--json",         action="store_true",
                        help="Also save JSON output")
    parser.add_argument("--no-paraphrase", action="store_true",
                        help="Skip Mistral paraphrasing (post raw text)")
    parser.add_argument("--no-post",      action="store_true",
                        help="Skip WordPress posting (scrape + save only)")
    args = parser.parse_args()

    SEP = "═" * 78
    print(f"\n{SEP}")
    print(f"{'  GreatUgandaJobs Scraper + Paraphraser + WP Poster':^78}")
    print(f"{SEP}\n")

    if not args.no_post and MISTRAL_API_KEY == "YOUR_MISTRAL_API_KEY_HERE":
        log.warning("⚠️  MISTRAL_API_KEY not set — paraphrasing will fail.")
    if not args.no_post and WP_SITE_URL == "https://yoursite.com":
        log.warning("⚠️  WP_SITE_URL not configured — WordPress posting will fail.")

    # ── Step 1: collect URLs ──────────────────────────────────
    job_urls = collect_job_urls(max_pages=args.pages)
    total    = len(job_urls)
    print(f"\n{'─' * 78}")
    print(f"  Total unique job URLs found: {total}")
    print(f"{'─' * 78}\n")

    # Load already-processed IDs to skip duplicates
    processed_ids, processed_urls = load_processed_ids()

    # ── Step 2: scrape, paraphrase, post ─────────────────────
    jobs   = []
    errors = 0
    posted = 0
    skipped_dup = 0

    for i, url in enumerate(job_urls, 1):
        log.info(f"[{i}/{total}] Scraping: {url}")

        # Skip already-posted URLs
        if url in processed_urls:
            log.info(f"  ⏭ Already processed, skipping: {url}")
            skipped_dup += 1
            continue

        try:
            job = scrape_job(url)
            if not job:
                print(f"\n  ⚠  [{i}/{total}] Skipped (no usable title): {url}\n")
                continue

            jobs.append(job)
            print_job(job, i, total)

            job_id = make_job_id(job, i)
            mark_read(job_id, url, job["job_title"], job["company_name"], i)

            if not args.no_post:
                # ── Paraphrase ────────────────────────────────
                if args.no_paraphrase:
                    final_title = job["job_title"]
                    final_desc  = job["job_description"]
                else:
                    log.info(f"  ✏  Paraphrasing title…")
                    final_title = paraphrase_title(job["job_title"])
                    log.info(f"  ✏  Paraphrasing description…")
                    final_desc  = paraphrase_description(job["job_description"])

                # ── Company ──────────────────────────────────
                save_company(job)

                # ── Post job ──────────────────────────────────
                log.info(f"  🚀 Posting to WordPress: \"{final_title}\"")
                wp_id, wp_link = post_job_to_wp(job, final_title, final_desc)
                if wp_id:
                    mark_posted(job_id, wp_id, wp_link or "")
                    posted += 1
                else:
                    mark_failed(job_id, "wp_post_failed")
                    errors += 1

        except Exception as e:
            errors += 1
            log.error(f"  ✗ ERROR [{i}/{total}]: {e}")

        time.sleep(DELAY)

    # ── Step 3: save CSV / JSON ───────────────────────────────
    csv_file = args.output + ".csv"
    save_csv(jobs, csv_file)
    if args.json:
        save_json(jobs, args.output + ".json")

    # ── Summary ───────────────────────────────────────────────
    print_tracker_summary()
    print(f"\n{'═' * 78}")
    print(f"  SUMMARY")
    print(f"{'─' * 78}")
    print(f"  Total URLs found     : {total}")
    print(f"  Jobs scraped         : {len(jobs)}")
    print(f"  Skipped (duplicates) : {skipped_dup}")
    if not args.no_post:
        print(f"  Posted to WordPress  : {posted}")
    print(f"  Errors               : {errors}")
    print(f"  CSV output           : {csv_file}")
    if args.json:
        print(f"  JSON output          : {args.output}.json")
    print(f"{'═' * 78}\n")


if __name__ == "__main__":
    main()
