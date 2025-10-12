# Building a Real Estate Scraper with Scrapy and Flask

This guide walks through the complete process of creating a web scraper for sreality.cz (a Czech real estate site) using Scrapy, storing the data in SQLite, and presenting it through a Flask web application.

## Table of Contents
- [Project Overview](#project-overview)
- [Prerequisites](#prerequisites)
- [Project Structure](#project-structure)
- [Part 1: Setting Up the Scrapy Project](#part-1-setting-up-the-scrapy-project)
- [Part 2: Defining the Spider](#part-2-defining-the-spider)
- [Part 3: Creating the Data Pipeline](#part-3-creating-the-data-pipeline)
- [Part 4: Building the Flask Web App](#part-4-building-the-flask-web-app)
- [Part 5: Running the Complete Application](#part-5-running-the-complete-application)

## Project Overview

This project consists of two main components:
1. A **Scrapy scraper** that extracts apartment listings from sreality.cz
2. A **Flask web application** that displays the scraped data with filtering and analytics

The data flow is:
- The Scrapy spider crawls sreality.cz and extracts apartment data
- Data is processed and stored in a SQLite database
- The Flask app reads from the database and displays the data in a user-friendly interface
- Analytics functionality shows price per square meter statistics

## Prerequisites

```
scrapy
flask
sqlalchemy
jinja2
```

Install these dependencies with:

```bash
pip install scrapy flask sqlalchemy jinja2
```

### Understanding Each Component

#### Core Libraries

- **Scrapy**: A powerful web crawling and scraping framework that makes it easy to extract data from websites. It handles HTTP requests, follows links, and provides a structured way to process and store the scraped data.

- **Flask**: A lightweight web framework that allows us to build the user interface for viewing the scraped data. Flask handles HTTP routes, template rendering, and serves the web application.

- **SQLAlchemy**: A SQL toolkit and Object-Relational Mapping (ORM) library that provides a way to interact with databases using Python objects. In this project, it's used to query the SQLite database from the Flask application.

- **Jinja2**: A modern and designer-friendly templating engine for Python. Flask uses Jinja2 to render HTML templates with dynamic content. It allows embedding Python-like expressions in HTML templates.

#### Project Components

- **sreality/sreality/items.py**: Defines the data structure for the items being scraped. This creates a clear contract for what data is expected and helps maintain consistency.

- **sreality/sreality/spiders/apartments.py**: Contains the spider code that performs the actual web scraping. It defines how to extract data from the API and transform it into structured items.

- **sreality/sreality/pipelines.py**: Processes items after they've been scraped. In this project, it calculates additional data (price per m²) and stores items in a SQLite database.

- **sreality/sreality/settings.py**: Contains configuration settings for the Scrapy project, such as user agent, download delays, and which pipelines to use.

- **flask_app/app.py**: The Flask web application that provides a user interface to view and analyze the scraped data. It connects to the SQLite database and can trigger the Scrapy spider to fetch new data.

- **flask_app/templates/**: Contains HTML templates for rendering the web pages. These templates use Jinja2 syntax to display dynamic data from the database.

#### Data Flow Between Components

1. The **Scrapy Spider** (`apartments.py`) makes requests to sreality.cz's API and extracts structured data
2. Data flows through the **Scrapy Pipeline** (`pipelines.py`), which enriches it and stores it in the **SQLite database**
3. The **Flask application** (`app.py`) reads from the database using **SQLAlchemy** and passes data to **Jinja2 templates**
4. The **HTML templates** render this data into a user-friendly web interface
5. When users request more data, the Flask app can trigger the Scrapy spider to fetch additional pages

## Project Structure

```
lecture_11/
│
├── requirements.txt           # Project dependencies
├── sreality/                  # Scrapy project
│   ├── apartments.db          # SQLite database
│   ├── scrapy.cfg             # Scrapy configuration
│   └── sreality/              # Scrapy app code
│       ├── __init__.py
│       ├── items.py           # Data models
│       ├── middlewares.py     # Request/response middleware
│       ├── pipelines.py       # Data processing pipeline
│       ├── settings.py        # Scrapy settings
│       └── spiders/           # Web crawlers
│           ├── __init__.py
│           └── apartments.py  # Main spider for sreality.cz
│
└── flask_app/                 # Flask web application
    ├── app.py                 # Flask application code
    └── templates/             # HTML templates
        ├── index.html         # Main listings page
        └── analytics.html     # Analytics dashboard
```

## Part 1: Setting Up the Scrapy Project

### 1.1 Creating a New Scrapy Project

Start by creating a new Scrapy project:

```bash
scrapy startproject sreality
cd sreality
```

This creates the basic structure for your Scrapy project.

### 1.2 Defining Data Models

Edit `sreality/sreality/items.py` to define the data structure for apartment listings:

```python
import scrapy

class SrealityItem(scrapy.Item):
    # define the fields for your item here like:
    # name = scrapy.Field()
    pass

class ApartmentItem(scrapy.Item):
    listing_id = scrapy.Field()
    url = scrapy.Field()
    title = scrapy.Field()
    price_num = scrapy.Field()
    location = scrapy.Field()
    rooms = scrapy.Field()
    area_m2 = scrapy.Field()
    image_url = scrapy.Field()
    scraped_at = scrapy.Field()
    price_per_m2 = scrapy.Field()
```

Each field represents a piece of data we'll extract from the apartment listings.

## Part 2: Defining the Spider

### 2.1 Creating the Spider

Create a new spider by generating `sreality/sreality/spiders/apartments.py`:

```python
# sreality/sreality/spiders/apartments.py

import json
import time
import re
import scrapy
from sreality.items import ApartmentItem


class ApartmentsSpider(scrapy.Spider):
    name = "apartments"
    allowed_domains = ["www.sreality.cz"]

    def __init__(self, page=1, *args, **kwargs):
        super().__init__(*args, **kwargs)
        tms = int(time.time() * 1000)
        self.start_urls = [
            (
                "https://www.sreality.cz/api/cs/v2/estates"
                f"?category_main_cb=1"
                f"&category_type_cb=1"
                f"&per_page=50"
                f"&page={int(page)}"
                f"&tms={tms}"
            )
        ]

    def parse(self, response):
        data = json.loads(response.text)
        estates = data.get("_embedded", {}).get("estates", [])
        for e in estates:
            item = ApartmentItem()

            lid = str(e.get("hash_id"))
            item["listing_id"] = lid

            # Parse title & fallback
            title = e.get("name") or e.get("locality") or ""
            item["title"] = title

            # Build front-end detail URL:
            # rooms_slug = e.g. "3+1" or "2+kk"
            m_slug = re.search(r"bytu\s+([\d\+kkmxMEONet]+)", title)
            rooms_slug = m_slug.group(1) if m_slug else ""
            # slugify locality
            loc = e.get("locality", "") or ""
            loc_slug = loc.lower().replace(" ", "-")
            item["url"] = (
                f"https://www.sreality.cz/detail/"
                f"prodej/byt/{rooms_slug}/{loc_slug}/{lid}"
            )

            # Price
            item["price_num"] = e.get("price_czk", {}).get("value_raw")

            # Extract rooms (float) from title e.g. "3+1"
            m_rooms = re.search(r"bytu\s+([\d\.]+)", title)
            item["rooms"] = float(m_rooms.group(1)) if m_rooms else None

            # Extract area in m2: number before "m²"
            m_area = re.search(r"(\d+)\s?m", title)
            item["area_m2"] = float(m_area.group(1)) if m_area else None

            # Location
            item["location"] = loc

            # First image
            imgs = e.get("_links", {}).get("images") or []
            item["image_url"] = imgs[0].get("href") if imgs else None

            yield item

        # Automatic pagination through the v2 API
        paging = data.get("paging", {})
        current = paging.get("page", 1)
        total = paging.get("pagesCount", 1)
        if current < total:
            nxt = current + 1
            tms = int(time.time() * 1000)
            base = response.url.split("?")[0]
            qs = (
                "category_main_cb=1"
                "&category_type_cb=1"
                f"&per_page=50&page={nxt}&tms={tms}"
            )
            yield scrapy.Request(f"{base}?{qs}", callback=self.parse)
```

### 2.2 Understanding the Spider in Detail

This spider uses an API-based approach rather than traditional HTML scraping, making it more efficient and reliable.

#### API-Based Scraping Approach

The spider uses sreality.cz's JSON API directly instead of scraping HTML. This approach offers several advantages:
- The API returns structured data in JSON format, which is easier to parse
- It avoids issues with JavaScript rendering and complex HTML structures
- It's less likely to break when the website's UI changes

#### Spider Initialization and URL Construction

```python
def __init__(self, page=1, *args, **kwargs):
    super().__init__(*args, **kwargs)
    tms = int(time.time() * 1000)
    self.start_urls = [
        (
            "https://www.sreality.cz/api/cs/v2/estates"
            f"?category_main_cb=1"      # Selects apartments
            f"&category_type_cb=1"      # Selects "for sale" listings
            f"&per_page=50"             # 50 results per page
            f"&page={int(page)}"        # Which page to fetch
            f"&tms={tms}"               # Timestamp to avoid caching
        )
    ]
```

The URL parameters are carefully constructed:
- `category_main_cb=1` - Specifies apartments (as opposed to houses)
- `category_type_cb=1` - Specifies "for sale" (as opposed to "for rent")
- `per_page=50` - Requests 50 listings per page
- `page={int(page)}` - Allows scraping a specific page number passed as a parameter
- `tms={tms}` - Adds timestamp (in milliseconds) to prevent caching issues

#### Parsing the API Response

```python
def parse(self, response):
    data = json.loads(response.text)    # Convert JSON string to Python dictionary
    estates = data.get("_embedded", {}).get("estates", [])  # Get the array of properties
```

The parsing process:
1. The `response.text` contains a JSON string from the API
2. `json.loads()` converts it to a Python dictionary for easier access
3. The actual property listings are nested in `data["_embedded"]["estates"]`
4. Using `.get()` with default values ensures no errors if the structure changes or values are missing

#### Extracting Data for Each Property

For each property in the results, we create a new item and populate it with data:

#### Basic Property Identification

```python
lid = str(e.get("hash_id"))  # Get the unique listing ID
item["listing_id"] = lid
```

Every property has a unique hash ID that identifies it in the system.

#### Title Extraction with Fallback

```python
title = e.get("name") or e.get("locality") or ""
item["title"] = title
```

This line uses Python's short-circuit evaluation to:
1. Try to get the property name first
2. If that's missing or empty, fall back to the locality
3. If both are missing, use an empty string

#### Building the Front-End URL

```python
# Extract the room configuration (e.g., "3+1" or "2+kk") from the title
m_slug = re.search(r"bytu\s+([\d\+kkmxMEONet]+)", title)
rooms_slug = m_slug.group(1) if m_slug else ""

# Convert location to URL-friendly format
loc = e.get("locality", "") or ""
loc_slug = loc.lower().replace(" ", "-")

# Construct full URL
item["url"] = (
    f"https://www.sreality.cz/detail/"
    f"prodej/byt/{rooms_slug}/{loc_slug}/{lid}"
)
```

URL construction uses:
1. Regular expression with a capturing group to extract room information from the title
   - The pattern `r"bytu\s+([\d\+kkmxMEONet]+)"` looks for room descriptions like "2+kk" or "3+1"
   - The capturing group `()` extracts just that part
2. Simple string transformation to create a URL-friendly location slug
   - Convert to lowercase and replace spaces with hyphens
3. Template string to build the complete URL for the listing's web page

#### Extracting Price

```python
item["price_num"] = e.get("price_czk", {}).get("value_raw")
```

This navigates through nested dictionaries to get the raw price value:
1. Access the `price_czk` dictionary within the listing
2. Extract the `value_raw` field which contains the numeric price
3. Using `.get()` with nested dictionaries prevents errors if fields are missing

#### Extracting Room Count

```python
m_rooms = re.search(r"bytu\s+([\d\.]+)", title)
item["rooms"] = float(m_rooms.group(1)) if m_rooms else None
```

This regex extracts the room count from titles like "Prodej bytu 3+1":
1. Find a pattern like "bytu 3" in the title
2. Extract the number part with a capturing group
3. Convert it to a float for numerical analysis
4. Store None if the pattern isn't found

#### Extracting Area in Square Meters

```python
m_area = re.search(r"(\d+)\s?m", title)
item["area_m2"] = float(m_area.group(1)) if m_area else None
```

This extracts the apartment size:
1. Looks for a number followed by "m" (e.g., "75 m²")
2. Captures the numeric part with parentheses in the regex
3. Converts to float for numerical calculations
4. Defaults to None if not found

#### Location and Image URL

```python
item["location"] = loc  # Use the location we extracted earlier

# Get the first image from the listing
imgs = e.get("_links", {}).get("images") or []
item["image_url"] = imgs[0].get("href") if imgs else None
```

For images:
1. Navigate to the `_links` → `images` array in the JSON structure
2. Get the first image if available
3. Extract its `href` attribute which contains the URL
4. Default to None if no images are available

#### Handling Pagination

```python
# Get pagination info
paging = data.get("paging", {})
current = paging.get("page", 1)
total = paging.get("pagesCount", 1)

# If there are more pages, go to the next one
if current < total:
    nxt = current + 1
    tms = int(time.time() * 1000)
    base = response.url.split("?")[0]
    qs = (
        "category_main_cb=1"
        "&category_type_cb=1"
        f"&per_page=50&page={nxt}&tms={tms}"
    )
    yield scrapy.Request(f"{base}?{qs}", callback=self.parse)
```

The pagination mechanism:
1. Gets current page number and total pages from API response
2. If there are more pages, constructs a URL for the next page
3. Generates a fresh timestamp to avoid caching
4. Yields a new request to that URL, which will trigger this `parse` method again
5. This creates a recursive cycle that walks through all available pages

#### Tips for Understanding API Structure

To discover similar elements in other APIs:

1. **Inspect Network Traffic**: Use your browser's DevTools Network tab when browsing sreality.cz to see API calls
2. **Study JSON Structure**: Examine the raw JSON response to understand data organization
3. **Look for Patterns**: Note how data is nested (e.g., `_embedded.estates[]`)
4. **Test Queries**: Try modifying URL parameters to see how they affect results

The advantage of this approach is that you can directly access structured data without having to parse complex HTML, making the scraper more robust and efficient.

## Part 3: Creating the Data Pipeline

### 3.1 Implementing a SQLite Pipeline

Edit `sreality/sreality/pipelines.py` to create a pipeline that stores the data in SQLite:

```python
# pipelines.py

from itemadapter import ItemAdapter
import sqlite3
from datetime import datetime


class SrealityPipeline:
    def process_item(self, item, spider):
        return item


class SQLitePipeline:
    def open_spider(self, spider):
        self.conn = sqlite3.connect("apartments.db")
        self.cur = self.conn.cursor()
        # note the new price_per_m2 column
        self.cur.execute(
            """
        CREATE TABLE IF NOT EXISTS apartments (
            id            INTEGER PRIMARY KEY,
            listing_id    TEXT    UNIQUE,
            url           TEXT,
            title         TEXT,
            price_num     INTEGER,
            location      TEXT,
            rooms         REAL,
            area_m2       REAL,
            image_url     TEXT,
            price_per_m2  REAL,
            scraped_at    TEXT
        )
        """
        )

    def close_spider(self, spider):
        self.conn.commit()
        self.cur.close()
        self.conn.close()

    def process_item(self, item, spider):
        # compute price_per_m2 if possible
        price = item.get("price_num")
        area = item.get("area_m2")
        if price is not None and area:
            item["price_per_m2"] = float(price) / area
        else:
            item["price_per_m2"] = None

        # timestamp
        item["scraped_at"] = datetime.utcnow().isoformat()

        # insert into SQLite, now including price_per_m2
        self.cur.execute(
            """
          INSERT OR REPLACE INTO apartments
            (listing_id, url, title,
             price_num, location, rooms,
             area_m2, image_url, price_per_m2,
             scraped_at)
          VALUES (?,?,?,?,?,?,?,?,?,?)
        """,
            (
                item["listing_id"],
                item.get("url"),
                item.get("title"),
                item.get("price_num"),
                item.get("location"),
                item.get("rooms"),
                item.get("area_m2"),
                item.get("image_url"),
                item.get("price_per_m2"),
                item.get("scraped_at"),
            ),
        )
        return item
```

### 3.2 Enabling the Pipeline

Edit `sreality/sreality/settings.py` to enable the SQLite pipeline:

```python
# Add this to the bottom of settings.py
ITEM_PIPELINES = {
    'sreality.pipelines.SQLitePipeline': 300,
}
```

## Part 4: Building the Flask Web App

### 4.1 Creating the Flask Application

Create `flask_app/app.py`:

```python
# flask_app/app.py

import os
import sys
import time
import subprocess

from flask import Flask, render_template, jsonify, request
from sqlalchemy import create_engine, text

app = Flask(__name__)

# ─── locate your Scrapy project one level up ─────────────────────────────────
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
SCRAPY_DIR = os.path.join(BASE_DIR, "sreality")
# ──────────────────────────────────────────────────────────────────────────────

# point SQLAlchemy at the SQLite DB that Scrapy's pipeline writes to
DB_PATH = os.path.join(SCRAPY_DIR, "apartments.db")
DB_URL = f"sqlite:///{DB_PATH}"
engine = create_engine(DB_URL, connect_args={"check_same_thread": False})


def run_spider_for_page(page: int):
    """Spawn a separate Scrapy process to crawl exactly page N."""
    subprocess.run(
        ["scrapy", "crawl", "apartments", "-a", f"page={page}"],
        cwd=SCRAPY_DIR,
        check=True,
    )


@app.route("/")
def index():
    # seed page 1 if table empty
    with engine.connect() as conn:
        cnt = conn.execute(text("SELECT COUNT(*) FROM apartments")).scalar()
    if cnt == 0:
        run_spider_for_page(1)

    # show latest 50 (page 1)
    with engine.connect() as conn:
        rows = (
            conn.execute(
                text(
                    """
            SELECT listing_id, url, title,
                   price_num, location, rooms, area_m2, image_url, price_per_m2
              FROM apartments
             ORDER BY scraped_at DESC
             LIMIT 50
        """
                )
            )
            .mappings()
            .all()
        )
    listings = [dict(r) for r in rows]
    return render_template("index.html", listings=listings)


@app.route("/api/load_page")
def load_page():
    page = int(request.args.get("page", 1))

    # run Scrapy to fetch & insert page N
    run_spider_for_page(page)

    # now grab that same batch of 50 from SQLite by ascending scraped_at
    offset = 50 * (page - 1)
    with engine.connect() as conn:
        rows = (
            conn.execute(
                text(
                    """
            SELECT listing_id, url, title,
                   price_num, location, rooms, area_m2, image_url, price_per_m2
              FROM apartments
             ORDER BY scraped_at ASC
             LIMIT 50 OFFSET :offset
        """
                ),
                {"offset": offset},
            )
            .mappings()
            .all()
        )

    # return all 50 (or fewer if you're off the end)
    return jsonify([dict(r) for r in rows])


@app.route("/analytics")
def analytics():
    # pull area and price_per_m2
    with engine.connect() as conn:
        rows = (
            conn.execute(
                text(
                    """
            SELECT listing_id, url, title, image_url, price_num,
                   location, rooms, area_m2, price_per_m2
              FROM apartments
             WHERE area_m2 > 0
               AND price_per_m2 IS NOT NULL
        """
                )
            )
            .mappings()
            .all()
        )

    records = [dict(r) for r in rows]
    return render_template("analytics.html", records=records)


@app.route("/api/analytics_data")
def analytics_data():
    # pull the same columns as /analytics, but send as JSON
    with engine.connect() as conn:
        rows = (
            conn.execute(
                text(
                    """
            SELECT listing_id, url, title, image_url, price_num, 
                   location, rooms, area_m2, price_per_m2
              FROM apartments
             WHERE area_m2 > 0
               AND price_per_m2 >= 1
        """
                )
            )
            .mappings()
            .all()
        )
    records = [dict(r) for r in rows]
    return jsonify(records=records)


if __name__ == "__main__":
    app.run(debug=True)
```

### 4.2 Creating HTML Templates

Create `flask_app/templates/index.html` for displaying apartment listings.

Create `flask_app/templates/analytics.html` for displaying price analytics.

## Part 5: Running the Complete Application

### 5.1 Running the Scrapy Spider Alone

To run the Scrapy spider and collect data:

```bash
cd sreality
scrapy crawl apartments
```

This will store the data in `apartments.db`.

### 5.2 Running the Flask App

To run the Flask web application:

```bash
cd flask_app
python app.py
```

The app will:
1. Check if there's data in the database
2. Run the spider to collect data if needed
3. Serve the web interface on http://127.0.0.1:5000/

### 5.3 Using the Application

- The main page (`/`) shows apartment listings with pagination
- The analytics page (`/analytics`) shows price statistics
- Use the "Load More" button to trigger scraping of additional pages

## Key Features Explained

### Scrapy Features

1. **API-based Scraping**: Instead of parsing HTML, this scraper directly uses sreality.cz's JSON API
2. **Regular Expression Parsing**: Extracts structured data from text fields
3. **Dynamic Parameters**: Uses timestamps and pagination
4. **Flexible Page Selection**: Allows scraping specific pages through parameters

### Data Processing

1. **SQLite Database**: Stores data persistently
2. **Data Enrichment**: Calculates price per square meter
3. **Unique Constraints**: Prevents duplicate listings
4. **Timestamp Tracking**: Records when data was scraped

### Flask Integration

1. **On-Demand Scraping**: Runs the spider when needed
2. **Pagination UI**: Loads more listings as the user scrolls
3. **Analytics Dashboard**: Visualizes price data
4. **Asynchronous Loading**: Fetches data without page reload

### Key Technical Concepts

1. **Subprocess Management**: Flask app spawns Scrapy processes
2. **Database Connection Sharing**: Both Scrapy and Flask use the same database
3. **Stateless API Design**: Clear separation between data collection and presentation
4. **Data Visualization**: Uses JavaScript to create interactive charts
