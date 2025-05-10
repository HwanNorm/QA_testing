import os
import json
import time
import re
import sys
import shutil
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError

# Try importing the required packages
try:
    from langchain_chroma import Chroma
    from langchain.prompts import ChatPromptTemplate
    from langchain_ollama.llms import OllamaLLM
    import langchain_text_splitters as text_splitters
    from langchain_core.documents import Document
    from langchain_ollama import OllamaEmbeddings
except ImportError as e:
    missing_package = str(e).split("'")[1]
    print(f"Error: Missing required package: {missing_package}")
    print("\nPlease install all required packages with:")
    print(
        "pip install langchain langchain_community langchain_chroma langchain_ollama chromadb google-api-python-client")
    sys.exit(1)

CHROMA_PATH = "chroma"
CURRENT_VIDEO_ID = None


def get_embedding_function():
    """Get the embedding function for the vector database"""
    try:
        return OllamaEmbeddings(model="mxbai-embed-large")
    except Exception as e:
        print(f"Error setting up Ollama embeddings: {str(e)}")
        print("Make sure Ollama is installed and running on your system.")
        print("Visit https://ollama.ai/ for installation instructions.")
        sys.exit(1)


def extract_video_id(url):
    """Extract YouTube video ID from URL."""
    patterns = [
        r'(?:v=|\/)([0-9A-Za-z_-]{11}).*',
        r'(?:embed\/)([0-9A-Za-z_-]{11})',
        r'(?:watch\?v=)([0-9A-Za-z_-]{11})'
    ]

    for pattern in patterns:
        match = re.search(pattern, url)
        if match:
            return match.group(1)

    # If it's already a video ID (11 characters)
    if len(url) == 11:
        return url

    return None


def get_comments(video_id, api_key):
    """Fetch comments and replies from YouTube."""
    # Create a YouTube API client
    youtube = build('youtube', 'v3', developerKey=api_key)

    # Call the API to get the comments
    comments = []
    next_page_token = None
    total_comments = 0

    print("Fetching comments from YouTube API...")

    try:
        while True:
            # Request comments
            request = youtube.commentThreads().list(
                part='snippet,replies',
                videoId=video_id,
                pageToken=next_page_token,
                maxResults=100,  # Maximum allowed by API
                textFormat='plainText'
            )
            response = request.execute()

            # Extract top-level comments and replies
            items_count = len(response.get('items', []))
            if items_count == 0:
                print("No comments found or all comments processed.")
                break

            print(f"Processing batch of {items_count} comment threads...")

            for item in response.get('items', []):
                # Top-level comment
                top_level_comment = item['snippet']['topLevelComment']['snippet']
                comment = top_level_comment['textDisplay']
                author = top_level_comment['authorDisplayName']

                # Add top-level comment
                comments.append({
                    'author': author,
                    'text': comment,
                    'is_reply': False
                })
                total_comments += 1

                # Replies (if any)
                if 'replies' in item:
                    for reply in item['replies']['comments']:
                        reply_author = reply['snippet']['authorDisplayName']
                        reply_comment = reply['snippet']['textDisplay']

                        # Add reply
                        comments.append({
                            'author': reply_author,
                            'text': reply_comment,
                            'is_reply': True,
                            'replied_to': author
                        })
                        total_comments += 1

            # Print progress
            print(f"Fetched {total_comments} comments so far...")

            # Check for more comments (pagination)
            next_page_token = response.get('nextPageToken')
            if not next_page_token:
                break  # No more pages, exit the loop

            # Add a small delay to avoid hitting API rate limits
            time.sleep(0.5)

    except Exception as e:
        print(f"Error fetching comments: {str(e)}")

    print(f"Completed fetching {total_comments} comments.")
    return comments


def get_video_title(video_id, api_key):
    """Get the title of a YouTube video."""
    try:
        youtube = build('youtube', 'v3', developerKey=api_key)
        response = youtube.videos().list(
            part='snippet',
            id=video_id
        ).execute()

        if not response['items']:
            return "Unknown Video"

        return response['items'][0]['snippet']['title']
    except Exception as e:
        print(f"Error fetching video title: {str(e)}")
        return "Unknown Video"


def save_comments_to_chroma(comments, video_id, video_title):
    """Populate comments into Chroma database."""
    global CURRENT_VIDEO_ID

    # Check if we already have a database for this video
    if CURRENT_VIDEO_ID == video_id and os.path.exists(CHROMA_PATH):
        print(f"Using existing Chroma database for video ID: {video_id}")
        return len(comments)

    # If video ID changed or no database exists, rebuild it
    if os.path.exists(CHROMA_PATH):
        print(f"Video ID changed from {CURRENT_VIDEO_ID} to {video_id}. Removing existing Chroma database.")
        shutil.rmtree(CHROMA_PATH)

    # Create the database
    print(f"Creating new Chroma vector database for video ID: {video_id}")
    db = Chroma(persist_directory=CHROMA_PATH,
                embedding_function=get_embedding_function())

    # Create Document objects for each comment
    documents = []

    # Add video title as a document
    title_doc = Document(
        page_content=f"VIDEO TITLE: {video_title}",
        metadata={"type": "title", "video_id": video_id}
    )
    documents.append(title_doc)

    # Process each comment
    for idx, comment in enumerate(comments, start=1):
        # Format the comment text
        if comment.get('is_reply', False):
            content = f"REPLY - Author: {comment['author']}\nComment: {comment['text']}"
            metadata = {
                "type": "reply",
                "author": comment['author'],
                "replied_to": comment.get('replied_to', 'Unknown')
            }
        else:
            content = f"Author: {comment['author']}\nComment: {comment['text']}"
            metadata = {
                "type": "comment",
                "author": comment['author']
            }

        document = Document(page_content=content, metadata=metadata)
        documents.append(document)

    # Split the documents to ensure they're not too large
    text_splitter = text_splitters.CharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=50
    )
    split_documents = text_splitter.split_documents(documents)

    # Add documents to Chroma in batches
    batch_size = 100
    for i in range(0, len(split_documents), batch_size):
        batch = split_documents[i:i + batch_size]
        db.add_documents(batch)
        print(f"Added batch of {len(batch)} documents to Chroma (total {i + len(batch)})")

    # Update the current video ID
    CURRENT_VIDEO_ID = video_id

    # Save metadata
    with open(os.path.join(CHROMA_PATH, "video_metadata.json"), "w", encoding="utf-8") as f:
        json.dump({
            "video_id": video_id,
            "title": video_title,
            "comment_count": len(comments)
        }, f, ensure_ascii=False, indent=2)

    print(f"Successfully added {len(split_documents)} documents to Chroma database.")
    return len(comments)


def calculate_optimal_k(total_comments):
    """Calculate the optimal k value based on total comment count."""
    if total_comments < 100:
        return total_comments
    elif total_comments < 500:
        return min(100, total_comments)
    elif total_comments < 2000:
        return min(200, total_comments)
    else:
        return min(300, total_comments)


def answer_question(question, k=None):
    """
    Answer a question based on the YouTube comments.

    Args:
        question: The user's question about the video comments
        k: Number of relevant comments to retrieve for context

    Returns:
        Answer generated by the LLM
    """
    # Check if database exists
    if not os.path.exists(CHROMA_PATH):
        return "Error: No comment database found! Please analyze a YouTube video first."

    # Start timing
    start_time = time.time()

    # Load the Chroma vector store
    print("Connecting to vector database...")
    db = Chroma(persist_directory=CHROMA_PATH,
                embedding_function=get_embedding_function())

    # Get the total number of documents in the database
    doc_count = len(db.get()['ids'])

    # Calculate optimal k if not specified
    if k is None:
        k = calculate_optimal_k(doc_count)
        print(f"Auto-calculated optimal k value: {k} (based on {doc_count} total documents)")

    # Get video metadata if available
    video_title = "Unknown Video"
    try:
        with open(os.path.join(CHROMA_PATH, "video_metadata.json"), "r") as f:
            metadata = json.load(f)
            video_title = metadata.get("title", "Unknown Video")
    except:
        pass

    # Define focused QA prompt template
    PROMPT_TEMPLATE = """
    You are a YouTube comment analyst answering questions about video comments.

    VIDEO: {video_title}

    QUESTION: {question}

    COMMENTS:
    {context}

    Answer the question using ONLY the information in these comments. Your response should:

    1. Start with a direct answer to the question
    2. Group similar opinions or information together
    3. Include specific quotes from commenters as evidence when helpful
    4. Stay focused on addressing exactly what was asked

    If the comments don't contain information to answer the question, simply state that.
    DO NOT invent or assume information not present in the comments.
    """

    print(f"Retrieving {k} most relevant comments for: {question}")

    # Retrieve relevant documents
    results = db.similarity_search_with_score(question, k=k)

    # Sort by relevance score (lower is better)
    sorted_results = sorted(results, key=lambda x: x[1])

    # Build context string from retrieved documents
    context_parts = []
    for i, (doc, score) in enumerate(sorted_results[:min(k, len(sorted_results))]):
        context_parts.append(f"[{i + 1}] {doc.page_content}")

    context_text = "\n\n".join(context_parts)

    # Format prompt with context
    prompt_template = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)
    prompt = prompt_template.format(
        question=question,
        context=context_text,
        video_title=video_title
    )

    # Generate answer with Ollama
    print("Generating answer...")
    model = OllamaLLM(model="llama3.2")

    generation_start = time.time()
    response_text = model.invoke(prompt)
    generation_time = time.time() - generation_start

    total_time = time.time() - start_time
    print(f"Answer generated in {generation_time:.2f} seconds")
    print(f"Total processing time: {total_time:.2f} seconds")

    return response_text


def analyze_youtube_video(youtube_url, api_key="AIzaSyDj7I12G6kpxEt4esWYXh2XwVAOXu7mbz0"):
    """
    Analyze YouTube comments from a URL.

    Args:
        youtube_url: URL or video ID of the YouTube video
        api_key: YouTube API key

    Returns:
        Number of comments processed
    """
    # Extract video ID
    video_id = extract_video_id(youtube_url)
    if not video_id:
        print("Invalid YouTube URL or video ID")
        return 0

    print(f"Analyzing video ID: {video_id}")

    # Get video title
    video_title = get_video_title(video_id, api_key)
    print(f"Video title: {video_title}")

    # Get comments
    comments = get_comments(video_id, api_key)

    # Save to Chroma
    comment_count = save_comments_to_chroma(comments, video_id, video_title)

    print(f"Analysis complete! Processed {comment_count} comments.")
    return comment_count