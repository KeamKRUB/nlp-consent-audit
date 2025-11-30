from fastapi import FastAPI, HTTPException, Depends
from pydantic import BaseModel
from typing import List, Dict, Union
from sqlalchemy import create_engine, Column, String, Integer, JSON, UniqueConstraint
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
import os
import google.generativeai as genai
import requests
from bs4 import BeautifulSoup
import asyncio
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer, util
from urllib.parse import urljoin, urlparse

load_dotenv()

# Set up CockroachDB connection
DATABASE_URL = os.getenv("DATABASE_URL")
engine = create_engine(DATABASE_URL, connect_args={"sslmode": "verify-full"})
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()

genai.configure(api_key=os.getenv("GENERATIVEAI_API_KEY"))

# Generative AI model setup
generation_config = {
    "temperature": 1,
    "top_p": 0.95,
    "top_k": 40,
    "max_output_tokens": 8192,
    "response_mime_type": "text/plain",
}

model_gemini = genai.GenerativeModel(
        model_name="gemini-1.5-flash-8b",
        generation_config=generation_config,
    )

# Load a pre-trained sentence transformer model
model = SentenceTransformer('all-MiniLM-L6-v2')

reference_texts = ["Privacy Policy","terms-of-use-and-privacy", "Terms of Service", "Terms and Conditions", "FAQ", "Data Policy", "Legal", "Security"]
reference_embeddings = model.encode(reference_texts, convert_to_tensor=True)

def is_related_link(link_text):
    link_embedding = model.encode(link_text, convert_to_tensor=True)
    cosine_scores = util.pytorch_cos_sim(link_embedding, reference_embeddings)
    max_score = cosine_scores.max().item()
    return max_score > 0.8, max_score

def scrape_page(url, visited_urls):
    try:
        response = requests.get(url)
        if response.status_code == 200:
            soup = BeautifulSoup(response.content, 'html.parser')
            main_text = ' '.join([elem.get_text(separator=' ', strip=True) for elem in soup.find_all(['p', 'h1', 'h2', 'h3', 'li', 'span', 'div'])])
            return main_text, soup
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
        related_url = urljoin(url, link['href'])

        if related_url not in visited_urls:
            is_related, score = is_related_link(link_text)
            if is_related:
                visited_urls.add(related_url)
                related_page_text, _ = scrape_page(related_url, visited_urls)
                if related_page_text:
                    related_pages_text += f"\n\n--- Scraped from {related_url} ---\n\n{related_page_text}"
    return related_pages_text

def scrape_website_and_related(url, visited_urls):
    main_page_text, main_page_soup = scrape_page(url, visited_urls)
    if main_page_soup:
        related_pages_text = scrape_related_pages(url, main_page_soup, visited_urls)
        combined_text = main_page_text + related_pages_text
        return combined_text
    else:
        print("Failed to scrape the main page.")
        return ""

app = FastAPI()

# SQLAlchemy Database Model
class WebsiteData(Base):
    __tablename__ = "websitedata"
    
    id = Column(Integer, primary_key=True, index=True)
    web_url = Column(String, index=True)
    summary = Column(String)
    topic = Column(JSON)
    language = Column(String, index=True)  # Language field
    
    __table_args__ = (
        UniqueConstraint('web_url', 'language', name='unique_web_url_language'),  # Unique by web_url and language
    )

# Create the tables in CockroachDB
Base.metadata.create_all(bind=engine)

# Pydantic models
class MainAPIRequest(BaseModel):
    web_url: str
    terms_urls: List[str]
    language: str = "en"  # Default to English

    def is_valid(self):
        return bool(self.web_url and isinstance(self.terms_urls, list))

class MainAPIResponse(BaseModel):
    summary: str
    topic: Dict[str, Dict[str, Union[str, int]]]
    language: str = "en"  # Include language in response

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

# Dependency to get the database session
def get_db():
    db = SessionLocal()
    try:
        return db
    finally:
        db.close()

# Mock Generative AI API function
async def generative_ai_api(data: str, language: str = "en"):
    # read prompt.txt
    with open("prompt.txt", "r") as file:
        prompt = file.read()
    if language != "en":
        prompt += f" ***Please ensure that the analysis and responses are in {language} language.***"
    prompt += "\nAnd please make sure not to forget to provide explanation for each type"
    
    text = str(model_gemini.generate_content(data+"\n"+prompt).text)
    text = text[text.find("{"):text.rfind("}")+1]  # Extract the JSON-like content
    text = eval(text)
    return GenerativeAIResponse(**text)

def get_main_url(url):
    parsed_url = urlparse(url)
    return f"{parsed_url.scheme}://{parsed_url.netloc}"

def store_in_database(db, web_url: str, summary: str, topic: Dict[str, int], language: str = "en"):
    web_url = get_main_url(web_url)
    db_data = WebsiteData(web_url=web_url, summary=summary, topic=topic, language=language)
    db.add(db_data)
    db.commit()

def check_database(db, web_url: str, language: str = "en"):
    web_url = get_main_url(web_url)
    return db.query(WebsiteData).filter(WebsiteData.web_url == web_url, WebsiteData.language == language).first()

# Web Scrapper API
async def web_scrapper_api(data: WebScrapperRequest):
    visited_urls = set()
    if not data.is_valid():
        raise HTTPException(status_code=400, detail="Invalid input format")

    all_combined_content = ""
    for url in data.terms_urls:
        page_content = scrape_website_and_related(url, visited_urls)
        if page_content:
            all_combined_content += f"\n\n--- Scraped from {url} ---\n\n{page_content}"

    if not all_combined_content:
        raise HTTPException(status_code=404, detail="Failed to scrape content from provided URLs")

    return WebScrapperResponse(web_url=data.web_url, all_content=all_combined_content)

# New translation function using a new Gemini agent for translation
async def translate_data(summary: str, topic: Dict[str, Dict[str, Union[str, int]]], target_language: str):
    prompt = f"Translate the following structure into {target_language} while keeping the format and structure exactly the same. Ensure the translation is accurate and the meaning is preserved.\n\n"
    prompt += f"{{summary: \"{summary}\", topic: {str(topic)}}}"
    prompt += f"\nIn topic dont change the keys keep the key english only translate the values"
    # Use a new Gemini agent for translation
    translation_config = {
        "temperature": 1,
        "top_p": 0.95,
        "top_k": 40,
        "max_output_tokens": 8192,
        "response_mime_type": "text/plain",
    }

    translation_agent = genai.GenerativeModel(
        model_name="gemini-1.5-flash-8b",
        generation_config=translation_config,
    )

    # Generate translation using the translation agent
    translation_response = str(translation_agent.generate_content(prompt).text)
    
    # Extract the translated JSON portion from the response
    translated_json = translation_response[translation_response.find("{"):translation_response.rfind("}") + 1]
    
    # Convert the translated string back into a Python dictionary and return it
    translated_data = eval(translated_json)
    
    return GenerativeAIResponse(**translated_data)


# Main API endpoint
@app.post("/main-api", response_model=MainAPIResponse)
async def main_api(request_data: MainAPIRequest, db: SessionLocal = Depends(get_db)):
    if not request_data.is_valid():
        raise HTTPException(status_code=400, detail="Invalid input format")

    # Step 1: Check if the data already exists in the requested language
    existing_data = check_database(db, request_data.web_url, request_data.language)
    if existing_data:
        return MainAPIResponse(summary=existing_data.summary, topic=existing_data.topic, language=request_data.language)

    # Step 2: If data doesn't exist in the requested language, check if it exists in any language
    any_language_data = db.query(WebsiteData).filter(WebsiteData.web_url == get_main_url(request_data.web_url)).first()
    if any_language_data:
        # Step 3: Translate the existing data into the requested target language using the new translation function
        translated_data = await translate_data(any_language_data.summary, any_language_data.topic, request_data.language)

        # Step 4: Store the translated result in the database
        store_in_database(db, request_data.web_url, translated_data.summary, translated_data.topic, request_data.language)

        # Step 5: Return the translated result
        return MainAPIResponse(summary=translated_data.summary, topic=translated_data.topic, language=request_data.language)

    # Step 6: If no data exists at all, follow the normal scraping and processing flow
    scrapper_data = await web_scrapper_api(WebScrapperRequest(
        web_url=request_data.web_url,
        terms_urls=request_data.terms_urls
    ))

    ai_data = await generative_ai_api(scrapper_data.all_content, request_data.language)

    store_in_database(db, request_data.web_url, ai_data.summary, ai_data.topic, request_data.language)

    return MainAPIResponse(summary=ai_data.summary, topic=ai_data.topic, language=request_data.language)

# same api with GET
@app.get("/main-api", response_model=MainAPIResponse)
async def main_api_get(web_url: str, terms_urls: List[str], language: str = "en", db: SessionLocal = Depends(get_db)):
    request_data = MainAPIRequest(web_url=web_url, terms_urls=terms_urls, language=language)
    return await main_api(request_data, db)

# Chat API

class ChatMessage(BaseModel):
    role: str
    msg: str

class ChatAPIRequest(BaseModel):
    web_url: str
    conversation: List[ChatMessage]
    language: str = "en"

    def is_valid(self):
        return bool(self.web_url and isinstance(self.conversation, list))

class ChatAPIResponse(BaseModel):
    conversation: List[ChatMessage]


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
    
    # output is sentence after "Bot: "
    return ai_response[ai_response.find("Bot: ")+5:]


@app.post("/chat-api", response_model=ChatAPIResponse)
async def chat_api(request_data: ChatAPIRequest, db: SessionLocal = Depends(get_db)):
    print(request_data)
    if not request_data.is_valid():
        raise HTTPException(status_code=400, detail="Invalid input format")

    # Step 1: Fetch the web data from the database
    web_data = check_database(db, request_data.web_url, request_data.language)
    if not web_data:
        raise HTTPException(status_code=404, detail="Website data not found")

    # Step 2: Generate a response for the conversation
    ai_response = await generate_chat_response(web_data.summary, request_data.conversation)

    # Step 3: Append the bot's response to the conversation
    request_data.conversation.append(ChatMessage(role="bot", msg=ai_response))

    # Step 4: Return the updated conversation
    return ChatAPIResponse(conversation=request_data.conversation)

@app.get("/chat-api", response_model=ChatAPIResponse)
async def chat_api_get(web_url: str, conversation: List[ChatMessage], language: str = "en", db: SessionLocal = Depends(get_db)):
    request_data = ChatAPIRequest(web_url=web_url, conversation=conversation, language=language)
    return await chat_api(request_data, db)


if __name__ == '__main__':
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)