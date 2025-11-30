import requests
from bs4 import BeautifulSoup
from sentence_transformers import SentenceTransformer, util
from urllib.parse import urljoin

# Load a pre-trained sentence transformer model
model = SentenceTransformer('all-MiniLM-L6-v2')

# Reference related link texts (anchor points)
reference_texts = ["Privacy Policy", "Terms of Service", "Terms and Conditions", "FAQ", "Data Policy", "Legal", "Security"]

# Embed reference texts
reference_embeddings = model.encode(reference_texts, convert_to_tensor=True)

# Set to track visited URLs (to avoid duplicates)
visited_urls = set()

def is_related_link(link_text):
    # Encode the link text using the pre-trained model
    link_embedding = model.encode(link_text, convert_to_tensor=True)

    # Compute cosine similarity between link text and reference texts
    cosine_scores = util.pytorch_cos_sim(link_embedding, reference_embeddings)

    # Return True if the highest similarity score exceeds a certain threshold (e.g., 0.7)
    max_score = cosine_scores.max().item()
    return max_score > 0.8, max_score

def scrape_page(url):
    try:
        # Send a GET request to the URL
        response = requests.get(url)
        
        if response.status_code == 200:
            soup = BeautifulSoup(response.content, 'html.parser')
            
            # Extract visible text from the main page
            main_text = ' '.join([elem.get_text(separator=' ', strip=True) for elem in soup.find_all(['p', 'h1', 'h2', 'h3', 'li', 'span', 'div'])])

            return main_text, soup  # Return both text and soup for further link scraping
        else:
            print(f"Failed to retrieve the page. Status code: {response.status_code}")
            return "", None
    except Exception as e:
        print(f"An error occurred: {e}")
        return "", None


def scrape_related_pages(url, main_page_soup):
    related_pages_text = ""
    links = main_page_soup.find_all('a', href=True)

    for link in links:
        link_text = link.get_text().strip()
        related_url = urljoin(url, link['href'])  # Construct full URL

        # Check if the URL has already been visited to avoid duplicates
        if related_url not in visited_urls:
            # Check if the link is related to the topic (privacy, terms, FAQ, etc.)
            is_related, score = is_related_link(link_text)

            if is_related:
                print(f"Related link found: '{link_text}' (Score: {score})")
                
                # Mark this URL as visited
                visited_urls.add(related_url)

                # Scrape the related page
                related_page_text, _ = scrape_page(related_url)

                # Add the related page's text to the combined text
                if related_page_text:
                    related_pages_text += f"\n\n--- Scraped from {related_url} ---\n\n"
                    related_pages_text += related_page_text

    return related_pages_text


def scrape_website_and_related(url):
    # Step 1: Scrape the main page
    main_page_text, main_page_soup = scrape_page(url)

    # Step 2: Scrape related pages if main page soup is available
    if main_page_soup:
        related_pages_text = scrape_related_pages(url, main_page_soup)

        # Combine the main page text and related pages text
        combined_text = main_page_text + related_pages_text

        # Save the combined text to a file
        with open('scraped_text_with_dynamic_related_links.txt', 'w', encoding='utf-8') as file:
            file.write(combined_text)

        print("Website and related pages scraped successfully. Text saved to scraped_text_with_dynamic_related_links.txt.")
    else:
        print("Failed to scrape the main page.")

# Example usage
url = 'https://policies.google.com/terms?hl=t'  # Replace with the actual URL of the website to scrape
scrape_website_and_related(url)