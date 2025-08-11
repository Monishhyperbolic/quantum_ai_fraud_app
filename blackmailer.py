import requests
import time
import random
import os
from bs4 import BeautifulSoup
from urllib.parse import urljoin, urlparse

# --- Configuration ---
TARGET_DOMAINS = [
    'nobroker.in' 
]

HEADERS = {
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
}
POLICY_KEYWORDS = ['privacy', 'legal', 'policy', 'data']
OUTPUT_DIR = 'privacy_policies'
if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)

# --- NEW: Known URLs Dictionary ---
# This is the most reliable way to get the correct page for major sites.
KNOWN_POLICY_URLS = {
    'google.com': 'https://policies.google.com/privacy',
    'wikipedia.org': 'https://foundation.wikimedia.org/wiki/Privacy_policy',
    'reddit.com': 'https://www.reddit.com/policies/privacy-policy'
    # Add other known URLs for your top 50 list here
}

# --- Main Scraper Logic ---

def find_privacy_policy_url(domain, session):
    """
    Tries to find the privacy policy URL, checking a known list first.
    """
    # 1. Check our manual list first for the highest accuracy.
    if domain in KNOWN_POLICY_URLS:
        print(f"[+] Found known policy URL for {domain}.")
        return KNOWN_POLICY_URLS[domain]

    # 2. If not in the list, fall back to searching the homepage.
    print(f"[*] No known URL for {domain}. Searching homepage...")
    base_url = f"https://www.{domain}"
    try:
        response = session.get(base_url, headers=HEADERS, timeout=15)
        response.raise_for_status()
        soup = BeautifulSoup(response.text, 'html.parser')
        links = soup.find_all('a', href=True)
        
        for link in links:
            link_text = link.get_text().lower()
            link_href = link['href'].lower()
            
            # Refined check to avoid irrelevant links
            if any(keyword in link_text for keyword in POLICY_KEYWORDS) or \
               any(keyword in link_href for keyword in POLICY_KEYWORDS):
                
                # Ignore links related to accounts, sign-in, etc.
                if 'sign' in link_href or 'account' in link_href or 'login' in link_href:
                    continue

                policy_url = urljoin(base_url, link['href'])
                if urlparse(policy_url).netloc == urlparse(base_url).netloc:
                    print(f"[+] Found potential policy link: {policy_url}")
                    return policy_url
                    
    except requests.exceptions.RequestException as e:
        print(f"[!] Error fetching homepage for {domain}: {e}")
        return None
    
    print(f"[-] Could not find a policy link on the homepage for {domain}.")
    return None

def scrape_policy_text(url, session):
    # This function remains the same as before.
    try:
        print(f"[*] Scraping content from {url}...")
        response = session.get(url, headers=HEADERS, timeout=15)
        response.raise_for_status()
        soup = BeautifulSoup(response.text, 'html.parser')
        main_content = soup.find('main') or soup.find('article') or soup.find('body')
        if main_content:
            return main_content.get_text(separator='\n', strip=True)
        return "Could not extract main content."
    except requests.exceptions.RequestException as e:
        print(f"[!] Error scraping policy page {url}: {e}")
        return None

def save_policy(domain, text):
    # This function remains the same as before.
    filename = os.path.join(OUTPUT_DIR, f"{domain}_privacy_policy.txt")
    try:
        with open(filename, 'w', encoding='utf-8') as f:
            f.write(text)
        print(f"[+] Successfully saved policy for {domain} to {filename}")
    except IOError as e:
        print(f"[!] Error saving file for {domain}: {e}")

# The main execution block also remains the same.
if __name__ == "__main__":
    with requests.Session() as session:
        for domain in TARGET_DOMAINS:
            policy_url = find_privacy_policy_url(domain, session)
            
            if policy_url:
                time.sleep(random.uniform(3, 7))
                policy_text = scrape_policy_text(policy_url, session)
                if policy_text:
                    save_policy(domain, policy_text)
            
            print("---")
            time.sleep(random.uniform(5, 10))

    print("\n[✔] Scraping complete.")