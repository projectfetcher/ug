"""
GreatUgandaJobs.com Scraper  (ug.py)
Scrapes all jobs from greatugandajobs.com and saves to CSV / JSON.
Full verbose output prints every scraped field to the terminal.

Usage:
    python ug.py                      # scrape all pages → jobs.csv
    python ug.py --pages 5            # limit to 5 listing pages
    python ug.py --output my_jobs     # custom output filename (no ext)
    python ug.py --json               # also save jobs.json

Requirements:
    pip install requests beautifulsoup4 lxml python-dateutil
"""

import re
import csv
import json
import time
import logging
import argparse
from datetime import datetime
from dateutil.relativedelta import relativedelta
from urllib.parse import unquote
from typing import Optional

import requests
from bs4 import BeautifulSoup

# ─────────────────────────────────────────────────────────────
#  CONFIG
# ─────────────────────────────────────────────────────────────
BASE_URL    = "https://www.greatugandajobs.com"
LIST_PATH   = "/jobs/newest-jobs"
PAGE_SIZE   = 20
DELAY       = 1.5
MAX_RETRIES = 3

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
#  VERBOSE PRINTER
# ─────────────────────────────────────────────────────────────

def print_job(job: dict, index: int, total: int):
    """Print every field of a scraped job in a readable box."""
    width = 78
    border = "═" * width

    print(f"\n{'═' * width}")
    print(f"  JOB {index} / {total}  ─  {job.get('job_url', '')}")
    print(f"{'═' * width}")

    for col in COLUMNS:
        label = COLUMN_LABELS.get(col, col)
        value = job.get(col, "") or ""

        # Multiline fields: indent continuation lines
        if col == "job_description":
            print(f"\n  {'─' * (width - 2)}")
            print(f"  {'JOB DESCRIPTION':^{width - 2}}")
            print(f"  {'─' * (width - 2)}")
            if value:
                for line in value.split("\n"):
                    # Wrap long lines at (width-4) chars
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
            display = str(value) if value else "(not found)"
            # Wrap long single-line values
            prefix = f"  {label:<22}: "
            available = width - len(prefix)
            if len(display) <= available:
                print(f"{prefix}{display}")
            else:
                # First chunk
                print(f"{prefix}{display[:available]}")
                remaining = display[available:]
                indent = " " * (len(prefix))
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
#  CLEANING HELPERS
# ─────────────────────────────────────────────────────────────

def clean_text(value: str) -> str:
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
    if not raw:
        return ""
    mapping = {
        "FULL_TIME":   "Full Time",
        "FULLTIME":    "Full Time",
        "FULL-TIME":   "Full Time",
        "PART_TIME":   "Part Time",
        "PARTTIME":    "Part Time",
        "PART-TIME":   "Part Time",
        "CONTRACT":    "Contract",
        "CONTRACTOR":  "Contract",
        "INTERNSHIP":  "Internship",
        "VOLUNTEER":   "Volunteer",
        "TEMPORARY":   "Temporary",
        "CASUAL":      "Casual",
        "FREELANCE":   "Freelance",
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
#  DESCRIPTION EXTRACTOR  (complete, well-formatted)
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

    # Collapse consecutive blank lines
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

    # Job Type
    raw_type = sd("employmentType") or extract_info_field(soup, "Job Type")
    job_type = clean_job_type(raw_type)

    # Qualifications
    qual_el = soup.find(attrs={"itemprop": "credentialCategory"})
    qualifications = clean_text(qual_el.get_text(strip=True)) if qual_el else ""

    # Experience
    exp_el  = soup.find(attrs={"itemprop": "monthsOfExperience"})
    raw_exp = clean_text(exp_el.get_text(strip=True)) if exp_el else ""
    experience = clean_experience(raw_exp)

    # Location
    raw_loc  = extract_info_field(soup, "Duty Station") or sd("addressLocality")
    location = clean_location(raw_loc)

    # Field
    raw_field = extract_info_field(soup, "Job Category") or sd("occupationalCategory")
    field     = clean_field(raw_field)

    # Dates
    raw_posted = re.sub(r"T\d{2}:\d{2}:\d{2}.*", "",
                        sd("datePosted") or extract_info_field(soup, "Posted")).strip()
    date_posted = clean_date_posted(raw_posted)

    raw_deadline = re.sub(r"T\d{2}:\d{2}:\d{2}.*", "",
                          sd("validThrough") or extract_info_field(soup, "Deadline of this Job")).strip()
    deadline     = clean_deadline(raw_deadline)
    est_deadline = add_three_months(date_posted) if date_posted else ""

    # Salary
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

    # Full description
    job_description = extract_description(soup)

    # Application
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

    # Company
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

    logo_el      = soup.find("img", class_="js_jobs_company_logo")
    company_logo = logo_el["src"] if logo_el else ""
    if company_logo and not company_logo.startswith("http"):
        company_logo = "https:" + company_logo

    website_el      = soup.find("div", itemprop="url")
    company_website = clean_text(website_el.get_text(strip=True)) if website_el else ""
    industry        = clean_text(sd("industry"))

    # Company profile page
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
#  SAVE
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


# ─────────────────────────────────────────────────────────────
#  MAIN
# ─────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Scrape GreatUgandaJobs.com — full verbose")
    parser.add_argument("--pages",  type=int, default=None,
                        help="Max listing pages (default: all)")
    parser.add_argument("--output", default="jobs",
                        help="Output filename without extension (default: jobs)")
    parser.add_argument("--json",   action="store_true",
                        help="Also save JSON output")
    args = parser.parse_args()

    SEP = "═" * 78
    print(f"\n{SEP}")
    print(f"{'  GreatUgandaJobs Scraper  (ug.py)':^78}")
    print(f"{SEP}\n")

    # ── Step 1: collect URLs ──────────────────────────────────
    job_urls = collect_job_urls(max_pages=args.pages)
    total    = len(job_urls)
    print(f"\n{'─' * 78}")
    print(f"  Total unique job URLs found: {total}")
    print(f"{'─' * 78}\n")

    # ── Step 2: scrape & print each job ──────────────────────
    jobs   = []
    errors = 0

    for i, url in enumerate(job_urls, 1):
        log.info(f"[{i}/{total}] Scraping: {url}")
        try:
            job = scrape_job(url)
            if job:
                jobs.append(job)
                # ── FULL VERBOSE PRINT ──────────────────────
                print_job(job, i, total)
            else:
                print(f"\n  ⚠  [{i}/{total}] Skipped (no usable title): {url}\n")
        except Exception as e:
            errors += 1
            log.error(f"  ✗ ERROR [{i}/{total}]: {e}")

    # ── Step 3: save ─────────────────────────────────────────
    csv_file = args.output + ".csv"
    save_csv(jobs, csv_file)
    if args.json:
        save_json(jobs, args.output + ".json")

    # ── Summary ───────────────────────────────────────────────
    print(f"\n{'═' * 78}")
    print(f"  SUMMARY")
    print(f"{'─' * 78}")
    print(f"  Total URLs found   : {total}")
    print(f"  Jobs scraped       : {len(jobs)}")
    print(f"  Skipped / errors   : {errors}")
    print(f"  Output file        : {csv_file}")
    if args.json:
        print(f"  JSON file          : {args.output}.json")
    print(f"{'═' * 78}\n")


if __name__ == "__main__":
    main()
