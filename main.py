import logging
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.common.exceptions import TimeoutException, NoSuchElementException
from webdriver_manager.chrome import ChromeDriverManager
import time
import os

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

url = (
    "https://www.chunkbase.com/apps/seed-map"
    "#seed=-1015452521318102722&platform=bedrock_1_21_90"
    "&dimension=overworld&x=-36&z=-1380&zoom=0.442"
)

# -----------------------
# Chrome options (safer)
# -----------------------
options = webdriver.ChromeOptions()
options.add_argument("--start-maximized")
options.add_argument("--disable-infobars")
options.add_argument("--disable-extensions")
options.add_argument("--disable-notifications")
# set a common user agent to reduce automation detection:
options.add_argument(
    "--user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
    "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/141.0.0.0 Safari/537.36"
)
# NOTE: do NOT disable GPU or images here (map needs WebGL & tiles)
# options.add_argument("--headless=new")  # uncomment to run headless after verifying UI

driver = webdriver.Chrome(service=Service(ChromeDriverManager().install()), options=options)
wait = WebDriverWait(driver, 40)

try:
    logging.info("Opening Chunkbase Seed Map...")
    driver.get(url)

    # Early page-source snapshot for debugging
    print("\n=== First 2000 characters of page_source (early) ===")
    print(driver.page_source[:2000])
    print("====================================================\n")

    # Wait for page ready
    wait.until(lambda d: d.execute_script("return document.readyState") == "complete")
    logging.info("document.readyState == complete")

    # Wait for map canvas to appear (mapbox or generic canvas)
    try:
        canvas = wait.until(
            EC.presence_of_element_located(
                (By.CSS_SELECTOR, "canvas.mapboxgl-canvas, canvas")
            )
        )
        logging.info("Map canvas found.")
    except TimeoutException:
        logging.warning("Map canvas not found within timeout. Proceeding to try seed input anyway.")

    # Try to find seed input (robust fallbacks)
    try:
        seed_box = wait.until(EC.presence_of_element_located((By.ID, "seed")))
    except TimeoutException:
        logging.warning("seed not found by ID. Trying fallback selectors...")
        # try multiple fallbacks
        seed_box = None
        fallbacks = [
            (By.CSS_SELECTOR, "input[placeholder*='Seed']"),
            (By.CSS_SELECTOR, "input[type='text']"),
            (By.XPATH, "//input[contains(@id,'seed') or contains(@name,'seed')]"),
        ]
        for by, sel in fallbacks:
            try:
                seed_box = wait.until(EC.presence_of_element_located((by, sel)))
                logging.info(f"Found seed box using fallback: {by} {sel}")
                break
            except Exception:
                continue
        if not seed_box:
            raise NoSuchElementException("Seed input not found by any selector.")

    # Change seed
    new_seed = "8273935"
    seed_box.clear()
    seed_box.send_keys(new_seed)
    seed_box.send_keys(Keys.RETURN)
    logging.info(f"Seed input changed to {new_seed}")

    # Wait until URL fragment updates OR the map canvas shows new content
    wait.until(lambda d: new_seed in d.current_url or "8273935" in d.page_source)
    logging.info("URL/page shows new seed (or page source updated).")

    # small delay for tiles to load
    time.sleep(2)

    # Save screenshot for inspection
    out = "chunkbase_result.png"
    driver.save_screenshot(out)
    logging.info(f"Saved screenshot -> {out}")

    # Try to extract visible text (useful summary)
    body_text = driver.find_element(By.TAG_NAME, "body").text
    snippet = "\n".join(body_text.splitlines()[:40])  # first 40 lines
    print("\n=== Visible text snippet ===")
    print(snippet)
    print("===========================\n")

    logging.info("Done. Inspect screenshot and text snippet above.")

except TimeoutException as e:
    logging.error("Timeout while waiting for elements: %s", e)
    print("Page source snapshot (2000 chars):\n", driver.page_source[:2000])
except Exception as e:
    logging.exception("Unexpected error: %s", e)
finally:
    # keep the browser open for a few more seconds if you want to inspect visually
    # comment out the next two lines if you want it to close immediately
    logging.info("Pausing 5s before cleanup so you can inspect the browser window.")
    time.sleep(5)
    logging.info("Closing browser.")
    driver.quit()
