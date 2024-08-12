import requests
from bs4 import BeautifulSoup
from urllib.parse import urljoin, urlparse
import time
from sentence_transformers import SentenceTransformer
import chromadb

# Initialize the sentence transformer model for embeddings
model = SentenceTransformer('paraphrase-MiniLM-L6-v2')

# Initialize ChromaDB client 
client = chromadb.PersistentClient(path="chroma_db")

def get_absolute_url(base_url, link):
    """Convert relative URLs to absolute URLs."""
    return urljoin(base_url, link)

def is_valid_url(url, base_url):
    """Check if the URL is valid, not a file, and starts with the base URL."""
    parsed = urlparse(url)
    return (
        parsed.scheme in ["http", "https"] and
        url.startswith(base_url) and
        not url.endswith(('.pdf', '.doc', '.xls', '.png', '.jpg', '.gif'))
    )

def store_in_chromadb(url, content, collection_name):
    """Store the URL and its content in ChromaDB."""
    # Convert content to embedding
    embedding = model.encode(content).tolist()  # Convert embedding to list

    # Create a document with URL, content, and embedding
    doc = {'content': content, 'embedding': embedding, 'metadata': {'url': url}}

    # Generate a unique ID for the document (you can use a hash or a UUID library)
    doc_id = hash(url)  # Using hash as an example

    # Store the document in ChromaDB, providing the document ID and collection name
    client.get_or_create_collection(name=collection_name).add(
        documents=[doc['content']], 
        embeddings=[doc['embedding']],
        metadatas=[doc['metadata']],
        ids=[str(doc_id)]  # Provide the ID for the document
    )

def crawl(start_url, max_depth=2):
    """Crawl the website starting from start_url up to max_depth levels."""
    visited = set()  # Set to track visited URLs
    to_visit = [(start_url, 0)]  # Queue for URLs to visit
    urls = set()  # Set to store unique URLs

    # Parse the base domain for collection naming
    base_domain = urlparse(start_url).netloc.replace('.', '_')

    while to_visit:
        url, depth = to_visit.pop(0)

        # Skip if URL is already visited or depth limit reached
        if url in visited or depth > max_depth:
            continue

        print(f"Crawling: {url} (depth {depth})")
        visited.add(url)
        urls.add(url)  # Add to unique URL set
        print(url)  # Print URL as it is added

        try:
            reqs = requests.get(url, timeout=5)
            soup = BeautifulSoup(reqs.text, 'html.parser')

            # Get the content of the page in paragraphs
            page_content = "\n".join(paragraph.get_text() for paragraph in soup.find_all('p'))

            # Print the content of the page
            print("Page Content:")
            print(page_content)
            print("\n" + "-"*40 + "\n")

            # Store the URL and content in ChromaDB with collection name as the base domain
            store_in_chromadb(url, page_content, base_domain)

            # Find all 'a' tags and extract the href attributes
            for link in soup.find_all('a'):
                href = link.get('href')
                if href:
                    full_url = get_absolute_url(url, href)
                    if is_valid_url(full_url, start_url):
                        if full_url not in visited:
                            to_visit.append((full_url, depth + 1))
        except requests.RequestException as e:
            print(f"Request failed: {e}")

        # Be respectful and avoid hammering the server
        time.sleep(1)

    print(f"Total number of unique URLs found: {len(urls)}")

# Start crawling from the specified URL
crawl('https://nnrg.edu.in/')