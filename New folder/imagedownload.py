import os
import json
import requests
from bs4 import BeautifulSoup

# Load JSON File
json_path = "ffhq-dataset-v1-processed.json"
save_folder = "ffhq_images"
os.makedirs(save_folder, exist_ok=True)

# Read JSON Data
with open(json_path, "r") as f:
    data = json.load(f)

# Extract URLs from JSON
image_urls = [data[key]["url"] for key in data if "url" in data[key]]

# Function to Scrape Direct Image URL from Flickr
def scrape_flickr_image(flickr_url):
    try:
        headers = {"User-Agent": "Mozilla/5.0"}
        response = requests.get(flickr_url, headers=headers)

        if response.status_code == 200:
            soup = BeautifulSoup(response.text, "html.parser")

            # Find the highest resolution image (Original or Large size)
            img_tag = soup.find("meta", {"property": "og:image"})
            if img_tag:
                return img_tag["content"]  # Direct image URL

        return None
    except Exception as e:
        print(f"⚠️ Error scraping {flickr_url}: {e}")
        return None

# Download Images
for i, flickr_url in enumerate(image_urls[:5000]):  # Adjust limit as needed
    direct_img_url = scrape_flickr_image(flickr_url)
    if direct_img_url:
        try:
            img_data = requests.get(direct_img_url).content
            img_path = os.path.join(save_folder, f"image_{i+1}.jpg")
            with open(img_path, "wb") as img_file:
                img_file.write(img_data)
            print(f"✅ Downloaded: {img_path}")
        except Exception as e:
            print(f"❌ Failed to download {direct_img_url}: {e}")
    else:
        print(f"❌ No direct image found for {flickr_url}")

print("✅ Finished downloading available images!")
