from fastapi import FastAPI, HTTPException, Depends
from pydantic import BaseModel
from typing import List, Dict, Union
from sqlalchemy import create_engine, Column, String, Integer, JSON
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
import os
import google.generativeai as genai
import requests
from bs4 import BeautifulSoup
import asyncio
from dotenv import load_dotenv
import requests
from bs4 import BeautifulSoup
from sentence_transformers import SentenceTransformer, util
from urllib.parse import urljoin
from urllib.parse import urlparse
load_dotenv()

# Set up CockroachDB connection
DATABASE_URL = os.getenv("DATABASE_URL")

engine = create_engine(DATABASE_URL, connect_args={"sslmode": "verify-full"})
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()

genai.configure(api_key=os.getenv("GENERATIVEAI_API_KEY"))

# Generative AI model setup
generation_config = {
    "temperature": 0,
    "top_p": 0.95,
    "top_k": 40,
    "max_output_tokens": 8192,
    "response_mime_type": "text/plain",
}

model_gemini = genai.GenerativeModel(
  model_name="gemini-1.5-flash-8b",
  generation_config=generation_config,
  system_instruction="You are tasked with analyzing the terms of service (ToS) or privacy policy of a website. Your goal is to parse the content and identify sections that may contain invalid or problematic clauses. Please review the content carefully and categorize the following 5-7 common types of invalid or inappropriate terms of service and also give explanation on each type:\n\n\tTypes of Invalid Terms of Service:\n\n\t\t1.\tUnfair Limitation of Liability: Look for clauses that attempt to absolve the company of all legal responsibility, especially in cases where it may be unreasonable (e.g., “We are not responsible for any damages under any circumstances”).\n        2.\tArbitrary Changes to Terms: Identify terms that allow the website to change the ToS at any time without notifying the user or obtaining their consent (e.g., “We reserve the right to change these terms at any time without notice”).\n        3.\tExcessive Data Collection and Sharing: Pay attention to clauses that allow for unlimited data collection or sharing of user data with third parties without proper transparency or consent (e.g., “We may share your data with third parties for any reason”).\n        4.\tUnilateral Content Ownership: Check for any statements where the company claims ownership over any content the user creates or uploads, regardless of the user’s rights (e.g., “We own all content you upload to this site”).\n        5.\tMandatory Arbitration and Waiver of Class Actions: Highlight clauses that force users into arbitration and deny them the right to pursue legal action or join class-action lawsuits (e.g., “You agree to mandatory arbitration and waive the right to join class actions”).\n        6.\tUnreasonable Termination of Services: Look for provisions where the service reserves the right to terminate the user’s account or access without providing any justification (e.g., “We may terminate your account at any time without cause”).\n        7.\tAutomatic Renewals Without Clear Notice: Flag any terms that bind users to automatic renewal of services without providing clear notice or requiring explicit consent from the user (e.g., “Your subscription will renew automatically unless you cancel”).\n\n\tScoring Scale:\n\n\t\t\t1: Bad (Highly problematic and unfair to users)\n        \t2: Medium (Potentially problematic but not severe)\n        \t3: Moderate (Reasonable and user-friendly)\n            4: Good (Highly user-friendly and protective of user rights)\n            5: Excellent (Best practices and exemplary terms of service)\n\n\tOutput Format:\n        {summary: \"<your summary about privacy policy and ToS>\",topic: {\n        Unfair Limitation of Liability: {explanation: \"<your explanation>\", score: <1-5>},\n        Arbitrary Changes to Terms: {explanation: \"<your explanation>\", score: <1-5>},\n        Excessive Data Collection and Sharing: {explanation: \"<your explanation>\", score: <1-5>},\n        Unilateral Content Ownership: {explanation: \"<your explanation>\", score: <1-5>},\n        Mandatory Arbitration and Waiver of Class Actions: {explanation: \"<your explanation>\", score: <1-5>},\n        Unreasonable Termination of Services: {explanation: \"<your explanation>\", score: <1-5>},\n        Automatic Renewals Without Clear Notice: {explanation: \"<your explanation>\", score: <1-5>}\n        }}\n\tProvide the output in the following JSON format, and each type is assigned with a score between 1 and 3 based on the scale and the explanation about that score\n    Your Output:\n        •\tParse the terms of service and categorize any sections that match the types of invalid ToS mentioned above.\n\t    •\tProvide a score for each type based on how well or poorly the website adheres to best practices.\n\t    •\tIf a type does not exist in the terms, assign a score of 5 (Excellent).\n\t    •\tSummarize the overall terms of service, identifying key issues or confirming that they are mostly compliant and user-friendly.",
)

# Load a pre-trained sentence transformer model
model = SentenceTransformer('all-MiniLM-L6-v2')

# Reference related link texts (anchor points)
reference_texts = ["Privacy Policy","terms-of-use-and-privacy", "Terms of Service", "Terms and Conditions", "FAQ", "Data Policy", "Legal", "Security"]

# Embed reference texts
reference_embeddings = model.encode(reference_texts, convert_to_tensor=True)

# Set to track visited URLs (to avoid duplicates)


def is_related_link(link_text):
    # Encode the link text using the pre-trained model
    link_embedding = model.encode(link_text, convert_to_tensor=True)

    # Compute cosine similarity between link text and reference texts
    cosine_scores = util.pytorch_cos_sim(link_embedding, reference_embeddings)

    # Return True if the highest similarity score exceeds a certain threshold (e.g., 0.7)
    max_score = cosine_scores.max().item()
    return max_score > 0.8, max_score

def scrape_page(url,visited_urls):
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


def scrape_related_pages(url, main_page_soup, visited_urls):
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
                related_page_text, _ = scrape_page(related_url,visited_urls)

                # Add the related page's text to the combined text
                if related_page_text:
                    related_pages_text += f"\n\n--- Scraped from {related_url} ---\n\n"
                    related_pages_text += related_page_text

    return related_pages_text


def scrape_website_and_related(url,visited_urls):
    # Step 1: Scrape the main page
    main_page_text, main_page_soup = scrape_page(url,visited_urls)

    # Step 2: Scrape related pages if main page soup is available
    if main_page_soup:
        related_pages_text = scrape_related_pages(url, main_page_soup,visited_urls)

        # Combine the main page text and related pages text
        combined_text = main_page_text + related_pages_text
        
        print("Website and related pages scraped successfully. Text saved to scraped_text_with_dynamic_related_links.txt.")
        return combined_text
    else:
        print("Failed to scrape the main page.")


app = FastAPI()

# SQLAlchemy Database Model
class WebsiteData(Base):
    __tablename__ = "websitedata"
    
    id = Column(Integer, primary_key=True, index=True)
    web_url = Column(String, unique=True, index=True)
    summary = Column(String)
    topic = Column(JSON)

# Create the tables in CockroachDB
Base.metadata.create_all(bind=engine)

# Pydantic models
class MainAPIRequest(BaseModel):
    web_url: str
    terms_urls: List[str]

    def is_valid(self):
        return bool(self.web_url and isinstance(self.terms_urls, list))

class MainAPIResponse(BaseModel):
    summary: str
    topic: Dict[str, Dict[str, Union[str, int]]]

class WebScrapperRequest(BaseModel):
    web_url: str
    terms_urls: List[str]

    def is_valid(self):
        return bool(self.web_url and isinstance(self.terms_urls, list))

class WebScrapperResponse(BaseModel):
    web_url: str
    all_content: str

class GenerativeAIResponse(BaseModel):
    summary: str
    topic: Dict[str, Dict[str, Union[str, int]]]

# Chat API request and response model
class ChatMessage(BaseModel):
    role: str
    msg: str

class ChatAPIRequest(BaseModel):
    web_url: str
    conversation: List[ChatMessage]

    def is_valid(self):
        return bool(self.web_url and isinstance(self.conversation, list))

class ChatAPIResponse(BaseModel):
    conversation: List[ChatMessage]

# Dependency to get the database session
def get_db():
    db = SessionLocal()
    try:
        return db
    finally:
        db.close()


# Mock Generative AI API function
async def generative_ai_api(data: str):
    text = str(model_gemini.generate_content(data).text)
    text = text[text.find("{"):text.rfind("}")+1]  # Extract the JSON-like content
    text = eval(text)
    return GenerativeAIResponse(**text)



def get_main_url(url):
    parsed_url = urlparse(url)
    # Rebuild the URL using only the scheme (http/https) and the netloc (domain)
    main_url = f"{parsed_url.scheme}://{parsed_url.netloc}"
    return main_url

# Function to store in the CockroachDB database
def store_in_database(db, web_url: str, summary: str, topic: Dict[str, int]):
    # convert web_url to only main url like https://www.google.com/webhp?hl=th&sa=X&ved=0ahUKEwin08Kj2YSJAxXRSGcHHcpLEBEQPAgI
    # to https://www.google.com
    web_url = get_main_url(web_url)

    db_data = WebsiteData(web_url=web_url, summary=summary, topic=topic)
    db.add(db_data)
    db.commit()

# Function to check the database for existing data
def check_database(db, web_url: str):
    web_url = get_main_url(web_url)
    return db.query(WebsiteData).filter(WebsiteData.web_url == web_url).first()

# Web Scrapper API
# Web Scrapper API - modified to scrape all terms_urls
async def web_scrapper_api(data: WebScrapperRequest):
    visited_urls = set()
    if not data.is_valid():
        raise HTTPException(status_code=400, detail="Invalid input format")

    all_combined_content = ""
    
    # Iterate over each URL in the terms_urls
    for url in data.terms_urls:
        print(f"Scraping {url}...")
        page_content = scrape_website_and_related(url,visited_urls)
        
        # If content is scraped successfully, add it to the combined content
        if page_content:
            all_combined_content += f"\n\n--- Scraped from {url} ---\n\n"
            all_combined_content += page_content

    if not all_combined_content:
        raise HTTPException(status_code=404, detail="Failed to scrape content from provided URLs")

    return WebScrapperResponse(web_url=data.web_url, all_content=all_combined_content)

# Main API endpoint
@app.post("/main-api", response_model=MainAPIResponse)
async def main_api(request_data: MainAPIRequest, db: SessionLocal = Depends(get_db)):
    if not request_data.is_valid():
        raise HTTPException(status_code=400, detail="Invalid input format")

    # Step 1: Check if the data already exists in the database
    existing_data = check_database(db, request_data.web_url)
    if existing_data:
        print("EXISTED")
        return MainAPIResponse(summary=existing_data.summary, topic=existing_data.topic)
    
    # Step 2: Call Web Scrapper API
    scrapper_data = await web_scrapper_api(WebScrapperRequest(
        web_url=request_data.web_url,
        terms_urls=request_data.terms_urls
    ))

    # Step 3: Call Generative AI API
    ai_data = await generative_ai_api(scrapper_data.all_content)

    # Step 4: Store the result in the database
    store_in_database(db, request_data.web_url, ai_data.summary, ai_data.topic)

    # Step 5: Return the final response
    return MainAPIResponse(summary=ai_data.summary, topic=ai_data.topic)

@app.post("/chat-api", response_model=ChatAPIResponse)
async def chat_api(request_data: ChatAPIRequest, db: SessionLocal = Depends(get_db)):
    print(request_data)
    if not request_data.is_valid():
        raise HTTPException(status_code=400, detail="Invalid input format")

    # Step 1: Fetch the web data from the database
    web_data = fetch_web_data_from_db(db, request_data.web_url)
    if not web_data:
        raise HTTPException(status_code=404, detail="Website data not found")

    # Step 2: Generate a response for the conversation
    ai_response = await generate_chat_response(web_data.summary, request_data.conversation)

    # Step 3: Append the bot's response to the conversation
    request_data.conversation.append(ChatMessage(role="bot", msg=ai_response))

    # Step 4: Return the updated conversation
    return ChatAPIResponse(conversation=request_data.conversation)


# Fetch web data from the database
def fetch_web_data_from_db(db, web_url: str):
    web_url = get_main_url(web_url)
    return db.query(WebsiteData).filter(WebsiteData.web_url == web_url).first()

# Mock chat conversation generator with Gemini
async def generate_chat_response(data: str, conversation: List[ChatMessage]):
    # Prepare the chat conversation prompt for Gemini AI
    chat_prompt = "Here is the privacy policy json data you need to read and understand this to answer the following question (DONT output this json it's for you to understand only):\n\n" + data + "\n\n"
    for message in conversation:
        chat_prompt += f"{message.role.capitalize()}: {message.msg}\n"

    chat_prompt += "Please output in natural language and remember you are assistant\n"
    chat_prompt += "Bot: "

    # Generate AI response from Gemini
    ai_response = model_gemini.generate_content(chat_prompt).text
    
    return ai_response.strip()

# Chat API endpoint



if __name__ == '__main__':
    #import uvicorn
    #uvicorn.run(app, host="0.0.0.0", port=8000)
    # mock = MainAPIRequest(web_url="https://www.google.com", terms_urls=["https://policies.google.com/privacy?hl=en-US", "https://policies.google.com/terms?hl=en-US"])
    # my_db = get_db()
    # print(asyncio.run(main_api(mock,my_db)))

    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
