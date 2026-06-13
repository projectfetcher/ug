"""
GreatUgandaJobs.com Scraper
Scrapes all jobs from greatugandajobs.com and saves to CSV / JSON.

Usage:
    python scraper.py                      # scrape all pages → jobs.csv
    python scraper.py --pages 5            # limit to 5 listing pages
    python scraper.py --output my_jobs     # custom output filename (no ext)
    python scraper.py --json               # also save jobs.json

Requirements:
    pip install requests beautifulsoup4 lxml
"""

import re
import csv
import json
import time
import logging
import argparse
from datetime import datetime, date
from dateutil.relativedelta import relativedelta
from urllib.parse import urlencode, urlparse, parse_qs, unquote
from typing import Optional
import requests
from bs4 import BeautifulSoup

# ─────────────────────────────────────────────────────────────
#  CONFIG
# ─────────────────────────────────────────────────────────────
BASE_URL   = "https://www.greatugandajobs.com"
LIST_PATH  = "/jobs/newest-jobs"
PAGE_SIZE  = 20          # jobs per listing page
DELAY      = 1.5         # seconds between requests (be polite)
MAX_RETRIES = 3

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-7s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("guj_scraper")

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/124.0.0.0 Safari/537.36"
    ),
    "Accept-Language": "en-US,en;q=0.9",
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
}

COLUMNS = [
    "job_title", "job_type", "qualifications", "experience",
    "location", "field", "date_posted", "deadline",
    "job_description", "application", "company_url", "company_name",
    "company_logo", "industry", "founded", "company_type",
    "website", "address", "company_details",
    "job_url", "estimated_deadline", "salary_range",
]

# Generic titles to skip (collection listings, not real jobs)
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
#  CLEANING HELPERS
# ─────────────────────────────────────────────────────────────

def clean_text(value: str) -> str:
    """Strip mojibake, excess whitespace, N/A placeholders."""
    if not value:
        return ""
    value = (
        value.replace("Â", "")
             .replace("â€™", "\u2019").replace("â€œ", "\u201c")
             .replace("â€\x9d", "\u201d").replace("â€", "\u201d")
             .replace("â€\x93", "\u2013").replace("â€\x94", "\u2014")
             .replace("â€¢", "\u2022")
    )
    value = re.sub(r"[\x00-\x08\x0B\x0C\x0E-\x1F\x7F]", "", value)
    value = re.sub(r"\s+", " ", value).strip()
    value = re.sub(r"^N/A$", "", value, flags=re.I)
    return value


def clean_job_type(raw: str) -> str:
    """
    FULL_TIME  → Full Time
    PART_TIME  → Part Time
    CONTRACTOR → Contractor
    """
    if not raw:
        return ""
    mapping = {
        "FULL_TIME": "Full Time",
        "FULLTIME": "Full Time",
        "FULL-TIME": "Full Time",
        "PART_TIME": "Part Time",
        "PARTTIME": "Part Time",
        "PART-TIME": "Part Time",
        "CONTRACT": "Contract",
        "CONTRACTOR": "Contract",
        "INTERNSHIP": "Internship",
        "VOLUNTEER": "Volunteer",
        "TEMPORARY": "Temporary",
        "CASUAL": "Casual",
        "FREELANCE": "Freelance",
    }
    key = raw.strip().upper().replace(" ", "_")
    if key in mapping:
        return mapping[key]
    # Title-case fallback
    return raw.strip().replace("_", " ").title()


def clean_experience(raw: str) -> str:
    """
    '96' or '96 months' → '8 years'
    '5 years'           → '5 years'
    '2-3 years'         → '2-3 years'
    """
    if not raw:
        return ""
    raw = raw.strip()
    # Already has "year" in it
    if re.search(r"\byears?\b", raw, re.I):
        return raw
    # Pure number or "N months"
    m = re.match(r"^(\d+)\s*(?:months?)?$", raw, re.I)
    if m:
        months = int(m.group(1))
        years  = months / 12
        if years == int(years):
            return f"{int(years)} year{'s' if years != 1 else ''}"
        else:
            return f"{years:.1f} years"
    return raw


def clean_location(raw: str) -> str:
    """
    'Kampala | Kampala'   → 'Kampala'
    'Kampala | Uganda'    → 'Kampala'
    'Kampala\nKampala'    → 'Kampala'
    """
    if not raw:
        return ""
    # Split on pipe, comma, newline → take first unique meaningful part
    parts = re.split(r"[|\n,/]", raw)
    seen, result = set(), []
    for p in parts:
        p = p.strip()
        if p and p.lower() not in seen and p.lower() not in ("uganda", ""):
            seen.add(p.lower())
            result.append(p)
    return result[0] if result else raw.strip()


def clean_field(raw: str) -> str:
    """
    'Administrative jobs in Uganda' → 'Administrative'
    'Management,Sales & Retail'     → 'Management, Sales & Retail'
    """
    if not raw:
        return ""
    # Remove trailing "jobs in Uganda" / "jobs in Kenya" etc.
    raw = re.sub(r"\s*jobs?\s+in\s+\w+", "", raw, flags=re.I).strip()
    # Normalize commas
    raw = re.sub(r",\s*", ", ", raw).strip(", ")
    return raw


def clean_deadline(raw: str) -> str:
    """
    'Thursday, June 25 2026'       → '25-06-2026'
    '2026-06-20T17:00:00+00:00'    → '20-06-2026'
    '20-06-2026'                   → '20-06-2026'
    '12-06-2026'                   → '12-06-2026'
    """
    if not raw:
        return ""
    raw = raw.strip()
    # ISO datetime
    m = re.match(r"(\d{4})-(\d{2})-(\d{2})", raw)
    if m:
        return f"{m.group(3)}-{m.group(2)}-{m.group(1)}"
    # Already DD-MM-YYYY
    if re.match(r"\d{2}-\d{2}-\d{4}", raw):
        return raw
    # Human-readable: "Thursday, June 25 2026"
    for fmt in ("%A, %B %d %Y", "%B %d %Y", "%d %B %Y"):
        try:
            dt = datetime.strptime(re.sub(r",", "", raw), fmt.replace(",", ""))
            return dt.strftime("%d-%m-%Y")
        except ValueError:
            pass
    return raw


def clean_date_posted(raw: str) -> str:
    """Same as deadline cleaning."""
    return clean_deadline(raw)


def clean_application(raw: str) -> str:
    """
    Extract email from query-string application URLs:
    /company-application-form?...form[application-email]=jobs@x.com&...
    OR return raw URL / email as-is.
    """
    if not raw:
        return ""
    raw = raw.strip()

    # Internal application form URL → extract email
    if "application-email" in raw:
        m = re.search(r"application-email\]=([^&]+)", raw)
        if m:
            return unquote(m.group(1)).strip()

    # Already an email
    if re.match(r"^[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z]{2,}$", raw):
        return raw

    # URL containing "apply" or external apply link — keep as-is
    return raw


def clean_description(raw: str) -> str:
    """Normalise whitespace in the full description text."""
    if not raw:
        return ""
    # Collapse multiple blank lines to single
    raw = re.sub(r"\n{3,}", "\n\n", raw)
    raw = re.sub(r"[ \t]+", " ", raw)
    return raw.strip()


def add_three_months(date_str: str) -> str:
    """Given a date string (any format), return date + 3 months as DD-MM-YYYY."""
    cleaned = clean_deadline(date_str)
    try:
        dt = datetime.strptime(cleaned, "%d-%m-%Y")
        new = dt + relativedelta(months=3)
        return new.strftime("%d-%m-%Y")
    except Exception:
        # Fallback: today + 3 months
        new = datetime.today() + relativedelta(months=3)
        return new.strftime("%d-%m-%Y")


# ─────────────────────────────────────────────────────────────
#  TITLE HELPERS
# ─────────────────────────────────────────────────────────────

def extract_raw_title(soup: BeautifulSoup) -> str:
    # 1. Structured data (most reliable)
    t = soup.find("div", itemprop="title")
    if t:
        val = t.get_text(strip=True)
        if val:
            return val

    # 2. jsjobs page-title span
    span = soup.find("span", class_="jsjobs-main-page-title")
    if span:
        val = span.get_text(strip=True)
        if val:
            return val

    # 3. Vacancy title in description body
    body = soup.find("div", class_="jsjobs_description_data")
    if body:
        m = re.search(r"Vacancy title:\s*\n?([^\n\[]+)", body.get_text())
        if m:
            return m.group(1).strip()

    # 4. og:title
    og = soup.find("meta", property="og:title")
    if og and og.get("content"):
        return re.sub(r"\s*[\|\-–]\s*.*$", "", og["content"]).strip()

    # 5. <title> tag
    t = soup.find("title")
    if t:
        return re.sub(r"\s*[\|\-–]\s*.*$", "", t.get_text()).strip()

    return ""


def clean_title(raw: str) -> Optional[str]:
    if not raw:
        return None
    title = raw.strip()

    if _GENERIC_RE.match(title):
        log.debug(f"Skipping generic title: {title}")
        return None

    # Strip " job at Company Name"
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
    """
    Find  <span class="js_job_data_title">Label:</span>
           <span class="js_job_data_value">Value</span>
    inside any  div.js_job_data_wrapper
    """
    for wrapper in soup.find_all("div", class_="js_job_data_wrapper"):
        title_el = wrapper.find("span", class_="js_job_data_title")
        if not title_el:
            continue
        if label.lower() in title_el.get_text(strip=True).lower():
            val_el = wrapper.find("span", class_="js_job_data_value")
            return val_el.get_text(strip=True) if val_el else ""
    return ""


# ─────────────────────────────────────────────────────────────
#  JOB DESCRIPTION EXTRACTOR  (complete, well-formatted)
# ─────────────────────────────────────────────────────────────

def extract_description(soup: BeautifulSoup) -> str:
    """
    Pull the full job description preserving logical line breaks:
    - headings get a blank line before/after
    - list items get a bullet prefix
    """
    container = soup.find("div", class_="jsjobs_description_data")
    if not container:
        # Fallback: the itemprop description div
        container = soup.find("div", itemprop="description")
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

        elif tag in ("br",):
            lines.append("")

        elif tag in ("ul", "ol"):
            for child in el.children:
                walk(child)

        elif tag in ("div", "section", "article", "aside"):
            for child in el.children:
                walk(child)

        elif tag is None:
            # NavigableString
            text = str(el).strip()
            if text:
                lines.append(text)

        else:
            for child in el.children:
                walk(child)

    for child in container.children:
        walk(child)

    # Collapse > 2 consecutive blank lines
    result = []
    blank_count = 0
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
#  COMPANY PAGE SCRAPER
# ─────────────────────────────────────────────────────────────

def scrape_company_page(url: str) -> dict:
    """Fetch extra company details from the employer profile page."""
    out = {
        "company_details": "", "founded": "",
        "company_type": "", "address": "",
    }
    if not url:
        return out
    soup = fetch(url)
    if not soup:
        return out
    time.sleep(DELAY)

    # Company about text
    about = (
        soup.find("div", class_="jsjobs-full-width-data") or
        soup.find("div", attrs={"itemprop": "description"})
    )
    if about:
        out["company_details"] = clean_text(about.get_text(" ", strip=True))

    # Meta info rows: Founded, Type, Address
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

    # ── Title ──────────────────────────────────────────────────
    raw_title = extract_raw_title(soup)
    job_title = clean_title(raw_title)
    if not job_title:
        log.debug(f"No usable title at {url}")
        return None
    log.info(f"  Title: {job_title}")

    # ── Structured data helpers ────────────────────────────────
    def sd(itemprop: str) -> str:
        el = soup.find(attrs={"itemprop": itemprop})
        return clean_text(el.get_text(" ", strip=True)) if el else ""

    # ── Job type ───────────────────────────────────────────────
    raw_type = sd("employmentType") or extract_info_field(soup, "Job Type")
    job_type = clean_job_type(raw_type)

    # ── Qualifications ─────────────────────────────────────────
    qual_el = soup.find(attrs={"itemprop": "credentialCategory"})
    qualifications = clean_text(qual_el.get_text(strip=True)) if qual_el else ""

    # ── Experience ─────────────────────────────────────────────
    exp_el = soup.find(attrs={"itemprop": "monthsOfExperience"})
    raw_exp = clean_text(exp_el.get_text(strip=True)) if exp_el else ""
    experience = clean_experience(raw_exp)

    # ── Location ───────────────────────────────────────────────
    raw_loc = extract_info_field(soup, "Duty Station") or sd("addressLocality")
    location = clean_location(raw_loc)

    # ── Field / Category ───────────────────────────────────────
    raw_field = extract_info_field(soup, "Job Category") or sd("occupationalCategory")
    field = clean_field(raw_field)

    # ── Dates ──────────────────────────────────────────────────
    raw_posted = sd("datePosted") or extract_info_field(soup, "Posted")
    # Strip time component
    raw_posted = re.sub(r"T\d{2}:\d{2}:\d{2}.*", "", raw_posted).strip()
    date_posted = clean_date_posted(raw_posted)

    raw_deadline = sd("validThrough") or extract_info_field(soup, "Deadline of this Job")
    raw_deadline = re.sub(r"T\d{2}:\d{2}:\d{2}.*", "", raw_deadline).strip()
    deadline = clean_deadline(raw_deadline)

    est_deadline = add_three_months(date_posted) if date_posted else ""

    # ── Salary ─────────────────────────────────────────────────
    sal_val = soup.find("div", itemprop="value")
    sal_unit = soup.find("div", itemprop="unitText")
    salary = ""
    if sal_val and sal_val.get_text(strip=True):
        cur_el = soup.find("div", itemprop="currency")
        cur = cur_el.get_text(strip=True) if cur_el else ""
        salary = f"{cur} {sal_val.get_text(strip=True)} / {sal_unit.get_text(strip=True) if sal_unit else 'month'}".strip()
    # Fallback: og:description
    if not salary:
        og = soup.find("meta", property="og:description")
        if og:
            m = re.search(r"Base Salary:\s*([^\n,\]]+)", og.get("content", ""))
            if m:
                val = m.group(1).strip()
                if val.lower() not in ("not disclosed", "n/a", ""):
                    salary = val

    # ── Job description (complete) ─────────────────────────────
    job_description = extract_description(soup)

    # ── Application ────────────────────────────────────────────
    # Priority: external "Apply Now" link → internal form email → any email in description
    application = ""

    # Check jsjobs apply button area
    for a in soup.find_all("a", href=True):
        txt = a.get_text(strip=True).lower()
        href = a["href"]
        if "apply" in txt or "click here" in txt:
            # Internal company form → extract email
            if "application-email" in href:
                application = clean_application(href)
                break
            # External HTTP link
            if href.startswith("http") and BASE_URL not in href:
                application = href
                break

    # Fallback: email anywhere in description
    if not application:
        m = re.search(r"[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z]{2,}", job_description)
        if m:
            application = m.group(0)

    # Fallback: any form href on page with application-email param
    if not application:
        for a in soup.find_all("a", href=True):
            if "application-email" in a["href"]:
                application = clean_application(a["href"])
                break

    # ── Company info ───────────────────────────────────────────
    comp_anchor = soup.find("a", class_="js_job_company_anchor")
    company_name = ""
    company_page_url = ""
    if comp_anchor:
        company_name = clean_text(comp_anchor.get_text(strip=True))
        href = comp_anchor.get("href", "")
        if href:
            company_page_url = href if href.startswith("http") else BASE_URL + href

    # Structured data fallback for name
    if not company_name:
        org_name = soup.find("div", itemprop="name")
        if org_name:
            company_name = clean_text(org_name.get_text(strip=True))

    logo_el = soup.find("img", class_="js_jobs_company_logo")
    company_logo = logo_el["src"] if logo_el else ""
    if company_logo and not company_logo.startswith("http"):
        company_logo = "https:" + company_logo

    website_el = soup.find("div", itemprop="url")
    company_website = clean_text(website_el.get_text(strip=True)) if website_el else ""

    industry = clean_text(sd("industry"))

    # ── Extra company details from profile page ────────────────
    company_extra = {}
    if company_page_url:
        log.info(f"  Fetching company page: {company_page_url}")
        company_extra = scrape_company_page(company_page_url)

    return {
        "job_title":        job_title,
        "job_type":         job_type,
        "qualifications":   qualifications,
        "experience":       experience,
        "location":         location,
        "field":            field,
        "date_posted":      date_posted,
        "deadline":         deadline,
        "job_description":  job_description,
        "application":      application,
        "company_url":      company_page_url,
        "company_name":     company_name,
        "company_logo":     company_logo,
        "industry":         industry,
        "founded":          company_extra.get("founded", ""),
        "company_type":     company_extra.get("company_type", ""),
        "website":          company_website,
        "address":          company_extra.get("address", ""),
        "company_details":  company_extra.get("company_details", ""),
        "job_url":          url,
        "estimated_deadline": est_deadline,
        "salary_range":     salary,
    }


# ─────────────────────────────────────────────────────────────
#  LISTING PAGE SCRAPER  (collects job URLs)
# ─────────────────────────────────────────────────────────────

def collect_job_urls(max_pages: Optional[int] = None) -> list:
    """Iterate listing pages and collect unique job-detail URLs."""
    seen = set()
    urls = []
    page = 0

    while True:
        if max_pages and page >= max_pages:
            break

        offset = page * PAGE_SIZE
        url = BASE_URL + LIST_PATH + (f"?start={offset}" if offset else "")
        log.info(f"Listing page {page + 1}: {url}")

        soup = fetch(url)
        if not soup:
            log.warning(f"Failed to fetch listing page {page + 1}, stopping.")
            break

        found_this_page = 0

        # Primary selectors for job links
        for a in soup.find_all("a", href=True):
            href = a["href"]
            if "job-detail" not in href:
                continue
            full = href if href.startswith("http") else BASE_URL + href
            # Strip nav suffix (/nav-19 etc.)
            full = re.sub(r"/nav-\d+.*$", "", full)
            if full not in seen:
                seen.add(full)
                urls.append(full)
                found_this_page += 1

        log.info(f"  Found {found_this_page} new URLs (total so far: {len(urls)})")

        if found_this_page == 0:
            log.info("No new jobs on this page — reached end of listings.")
            break

        page += 1
        time.sleep(DELAY)

    return urls


# ─────────────────────────────────────────────────────────────
#  SAVE RESULTS
# ─────────────────────────────────────────────────────────────

def save_csv(jobs: list, filename: str):
    with open(filename, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=COLUMNS, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(jobs)
    log.info(f"Saved {len(jobs)} jobs → {filename}")


def save_json(jobs: list, filename: str):
    with open(filename, "w", encoding="utf-8") as f:
        json.dump(jobs, f, ensure_ascii=False, indent=2)
    log.info(f"Saved {len(jobs)} jobs → {filename}")


# ─────────────────────────────────────────────────────────────
#  MAIN
# ─────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Scrape GreatUgandaJobs.com")
    parser.add_argument("--pages", type=int, default=None,
                        help="Max listing pages to scrape (default: all)")
    parser.add_argument("--output", default="jobs",
                        help="Output filename without extension (default: jobs)")
    parser.add_argument("--json", action="store_true",
                        help="Also save a JSON file")
    args = parser.parse_args()

    log.info("=" * 55)
    log.info("  GreatUgandaJobs Scraper  starting…")
    log.info("=" * 55)

    # Install dateutil if missing
    try:
        from dateutil.relativedelta import relativedelta
    except ImportError:
        import subprocess, sys
        subprocess.check_call([sys.executable, "-m", "pip", "install",
                               "python-dateutil", "-q"])

    # Step 1: collect URLs
    job_urls = collect_job_urls(max_pages=args.pages)
    log.info(f"\nTotal unique job URLs collected: {len(job_urls)}\n")

    # Step 2: scrape each job
    jobs = []
    errors = 0
    for i, url in enumerate(job_urls, 1):
        log.info(f"[{i}/{len(job_urls)}] {url}")
        try:
            job = scrape_job(url)
            if job:
                jobs.append(job)
            else:
                log.info("  Skipped (no usable title)")
        except Exception as e:
            errors += 1
            log.error(f"  ERROR: {e}")

    # Step 3: save
    csv_file  = args.output + ".csv"
    json_file = args.output + ".json"
    save_csv(jobs, csv_file)
    if args.json:
        save_json(jobs, json_file)

    log.info("=" * 55)
    log.info(f"  Done. Scraped: {len(jobs)} | Errors: {errors}")
    log.info("=" * 55)


if __name__ == "__main__":
    main()
