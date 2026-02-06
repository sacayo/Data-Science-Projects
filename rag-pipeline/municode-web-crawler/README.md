# Municode Web Crawler

> Selenium-based scraper that automatically downloads county ordinance PDFs from Municode Library — running in Google Colab with Google Drive integration.

This is **Step 0** in the RAG pipeline. It automates the data collection process: navigating Municode's municipal code library, filtering for county-level governments, and downloading their ordinance PDFs for processing by the [Data Engineering Pipeline](../data-engineering/README.md).

---

## How It Works

```
┌──────────────────┐     ┌──────────────────┐     ┌──────────────────┐     ┌──────────────────┐
│  Municode        │     │  County Filter   │     │  PDF Download    │     │  S3 Upload       │
│  Library         │     │                  │     │                  │     │                  │
│  municode.com/ga │────▶│  Identifies      │────▶│  Clicks download │────▶│  Google Drive    │
│                  │     │  county-level    │     │  buttons + modal │     │  → aws s3 sync   │
│  State listing   │     │  municipalities  │     │  confirmations   │     │  → S3 bucket     │
│  page            │     │  (not cities)    │     │                  │     │                  │
└──────────────────┘     └──────────────────┘     └──────────────────┘     └──────────────────┘
```

1. **Navigate** to a state listing page on Municode (e.g., `/ga`)
2. **Filter** for county-level municipalities (matching "county", "parish", "borough")
3. **Download** ordinance PDFs by clicking download buttons and confirming modals
4. **Save** to Google Drive, then sync to S3

---

## Quick Start

1. Upload `municode_crawler.ipynb` to Google Drive
2. Open with Google Colab
3. Edit `DOWNLOAD_DIR` to your preferred folder:
   ```python
   DOWNLOAD_DIR = "/content/drive/MyDrive/municode_downloads"
   ```
4. Run all cells (`Runtime → Run all`)

The crawler will install Chrome/Selenium, mount Drive, and start downloading.

---

## Configuration

### Target Different States

```python
# Change the URL in the main code
driver.get("https://library.municode.com/ga")  # Georgia (default)
driver.get("https://library.municode.com/ca")  # California
driver.get("https://library.municode.com/fl")  # Florida
driver.get("https://library.municode.com/tx")  # Texas
```

### County Detection

The crawler identifies counties using name patterns:

```python
def is_county_level(name: str, url: str) -> bool:
    county_terms = [" county", " parish", " borough"]
    return any(t in name.lower() for t in county_terms)
```

### Download Timeout

Default: 15 minutes per batch. Adjust in the waiting loop:

```python
if time.time() - start > 900:  # seconds
    break
```

---

## Transferring to S3

After downloading, upload to S3 for the data engineering pipeline:

```bash
# From Colab or local machine (update state code: ga, fl, ca, tx)
aws s3 sync /path/to/downloads s3://your-bucket/input/pdfs/ga/ \
    --exclude "*.txt" \
    --exclude "*.crdownload"
```

---

## Performance

| Metric | Georgia (baseline) |
|--------|-------------------|
| Counties | ~159 |
| Download time | 2-4 hours |
| Total size | ~5-10GB |
| Success rate | ~95% |

Failed URLs are logged to `failed_urls.txt` for manual retry.

---

## Known Limitations

- **Colab runtime**: 12-hour session limit
- **Drive storage**: 15GB on free accounts
- **Sequential downloads**: Single-threaded, not parallel
- **No resume**: Must restart if interrupted (failed URLs saved for retry)
- **HTML dependency**: If Municode changes their page structure, CSS selectors need updating

---

## Project Structure

```
municode-web-crawler/
├── municode_crawler.ipynb   # Main Colab notebook
└── README.md
```

## Next Step

After collecting PDFs and uploading to S3, run the [Data Engineering Pipeline](../data-engineering/README.md) to extract and chunk text.
