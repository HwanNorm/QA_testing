import os
import shutil
import json
import time
import re
import sys
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
CONFIG_FILE = "config.json"


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

    return None


def get_video_metadata(youtube, video_id):
    """Get the title and description of a video."""
    try:
        # Get video details
        response = youtube.videos().list(
            part='snippet',
            id=video_id
        ).execute()

        if not response['items']:
            return "Unknown Title", "No description available"

        video_data = response['items'][0]['snippet']
        title = video_data.get('title', "Unknown Title")
        description = video_data.get('description', "No description available")

        return title, description
    except HttpError as e:
        print(f"Error fetching video metadata: {e.resp.status} {e.content}")
        return "Unknown Title", "No description available"


def get_video_comments(youtube, video_id, max_comments=None):
    """Get comments for a specific YouTube video with simplified metadata."""
    comments = []
    next_page_token = None
    comment_count = 0

    print(f"Fetching comments for video: {video_id}")

    try:
        while True:
            # Get top-level comments (threads)
            response = youtube.commentThreads().list(
                part='snippet,replies',
                videoId=video_id,
                maxResults=100,  # Maximum allowed by API
                pageToken=next_page_token,
                textFormat='plainText'
            ).execute()

            # Process each comment thread
            for item in response['items']:
                # Get top-level comment
                top_level_comment = item['snippet']['topLevelComment']['snippet']

                # Simplified comment data - only essential fields
                comment_data = {
                    'author': top_level_comment['authorDisplayName'],
                    'text': top_level_comment['textDisplay'],
                    'is_reply': False
                }

                comments.append(comment_data)
                comment_count += 1

                # Check if we need to get replies
                if item['snippet'].get('totalReplyCount', 0) > 0:
                    # Get replies for this comment
                    replies = get_comment_replies(youtube, item['id'])
                    comments.extend(replies)
                    comment_count += len(replies)

                # Check if we've reached the maximum comments limit
                if max_comments and comment_count >= max_comments:
                    print(f"Reached maximum comment limit: {max_comments}")
                    return comments[:max_comments]

            # Check if there are more comments
            next_page_token = response.get('nextPageToken')
            if not next_page_token:
                break

            # Add a small delay to avoid hitting API rate limits
            time.sleep(0.5)

    except HttpError as e:
        print(f"An HTTP error occurred: {e.resp.status} {e.content}")

    print(f"Total comments fetched: {comment_count}")
    return comments


def get_comment_replies(youtube, thread_id):
    """Get replies to a comment thread with simplified metadata."""
    replies_list = []

    try:
        # Get replies to the comment thread
        response = youtube.comments().list(
            part='snippet',
            parentId=thread_id,
            maxResults=100,
            textFormat='plainText'
        ).execute()

        # Process each reply
        for item in response.get('items', []):
            snippet = item['snippet']

            # Simplified reply data - only essential fields
            reply_data = {
                'author': snippet['authorDisplayName'],
                'text': snippet['textDisplay'],
                'is_reply': True
            }

            replies_list.append(reply_data)

    except HttpError as e:
        print(f"An HTTP error occurred when fetching replies: {e.resp.status} {e.content}")

    return replies_list


def save_comments(comments, video_id, video_title, video_description):
    """Save comments to JSON file with proper UTF-8 encoding."""
    if not comments:
        print("No comments to save")
        return None, None

    # Create output directory
    os.makedirs('comments', exist_ok=True)

    # Prepare filenames
    json_filename = os.path.join('comments', f"{video_id}_comments.json")
    metadata_filename = os.path.join('comments', f"{video_id}_metadata.json")

    # Save as JSON with proper encoding
    with open(json_filename, 'w', encoding='utf-8') as file:
        json.dump(comments, file, ensure_ascii=False, indent=2)

    # Save video metadata separately
    with open(metadata_filename, 'w', encoding='utf-8') as file:
        json.dump({
            'video_id': video_id,
            'title': video_title,
            'description': video_description
        }, file, ensure_ascii=False, indent=2)

    print(f"Comments saved to JSON: {json_filename}")
    print(f"Video metadata saved to: {metadata_filename}")

    return json_filename, metadata_filename


def load_comments_to_db(json_file, metadata_file=None):
    """Load comments from JSON file into Chroma DB with simplified processing"""
    print("Loading comments into vector database...")
    start_time = time.time()

    try:
        # Clear existing database if it exists
        if os.path.exists(CHROMA_PATH):
            print("Removing existing database...")
            shutil.rmtree(CHROMA_PATH)

        # Read the JSON file
        with open(json_file, 'r', encoding='utf-8') as f:
            comments = json.load(f)

        # Load video metadata if available
        video_title = "Unknown Title"
        video_description = "No description available"

        if metadata_file and os.path.exists(metadata_file):
            try:
                with open(metadata_file, 'r', encoding='utf-8') as f:
                    metadata = json.load(f)
                    video_title = metadata.get('title', "Unknown Title")
                    video_description = metadata.get('description', "No description available")

                print(f"Loaded video metadata: {video_title}")
            except Exception as e:
                print(f"Error loading metadata: {str(e)}")

        # Store video metadata in a separate document
        metadata_doc = Document(
            page_content=f"VIDEO TITLE: {video_title}",
            metadata={
                "title": video_title
            }
        )

        print(f"Processing {len(comments)} comments...")

        # Combine all the comments into documents
        documents = [metadata_doc]  # Start with the metadata document

        # Add progress tracking
        total_comments = len(comments)
        for i, comment in enumerate(comments):
            if i % 100 == 0:
                print(f"Processing comment {i}/{total_comments}...")

            # Simplified formatting - less metadata in content
            content = f"Author: {comment['author']}\nComment: {comment['text']}"
            if comment.get('is_reply', False):
                content = f"REPLY: {content}"

            # Create a document with minimal metadata
            doc = Document(
                page_content=content,
                metadata={
                    "author": comment['author'],
                    "video_title": video_title
                }
            )
            documents.append(doc)

        # Split the documents to ensure they're not too large
        print("Splitting documents into chunks...")
        # Increased chunk size for fewer chunks and faster processing
        text_splitter = text_splitters.CharacterTextSplitter(
            chunk_size=1500,  # Increased from 1000
            chunk_overlap=50  # Reduced from 100
        )
        split_documents = text_splitter.split_documents(documents)

        print(f"Generating embeddings for {len(split_documents)} chunks...")
        embedding_start_time = time.time()

        # Create and populate the Chroma database with higher batch size
        db = Chroma.from_documents(
            documents=split_documents,
            embedding=get_embedding_function(),
            persist_directory=CHROMA_PATH
        )

        embedding_end_time = time.time()
        embedding_duration = embedding_end_time - embedding_start_time
        print(
            f"Embedding generation completed in {embedding_duration:.2f} seconds ({embedding_duration / 60:.2f} minutes)")

        # Save the video metadata to a file that will be used during queries
        with open(os.path.join(CHROMA_PATH, "video_metadata.json"), "w", encoding="utf-8") as f:
            json.dump({
                "title": video_title,
                "description": video_description
            }, f, ensure_ascii=False, indent=2)

        print("Database persisted to disk")
        print(f"Successfully loaded {len(split_documents)} comment chunks into the database")

        total_duration = time.time() - start_time
        print(f"Total database loading completed in {total_duration:.2f} seconds ({total_duration / 60:.2f} minutes)")

        return True, video_title, video_description
    except Exception as e:
        print(f"Error loading comments to database: {str(e)}")
        return False, None, None


def load_api_key():
    """Load the API key from the config file"""
    if not os.path.exists(CONFIG_FILE):
        return None

    try:
        with open(CONFIG_FILE, 'r') as f:
            config = json.load(f)
        return config.get("youtube_api_key")
    except Exception as e:
        print(f"Error loading API key: {str(e)}")
        return None


def save_api_key(api_key):
    """Save the API key to a config file"""
    config = {"youtube_api_key": api_key}
    try:
        with open(CONFIG_FILE, 'w') as f:
            json.dump(config, f)
        print(f"API key saved successfully to {CONFIG_FILE}")
        return True
    except Exception as e:
        print(f"Error saving API key: {str(e)}")
        return False


def main():
    print("=" * 80)
    print("YouTube Comment Loader (Simplified)")
    print("=" * 80)

    # Get API key
    api_key = load_api_key()
    if not api_key:
        api_key = input("Enter your YouTube API Key: ")
        if api_key:
            save_key = input("Save this API key for future use? (Y/n): ").lower()
            if save_key != "n":
                save_api_key(api_key)
    else:
        use_saved = input(f"Use saved API key? (Y/n): ").lower()
        if use_saved == "n":
            api_key = input("Enter your YouTube API Key: ")
            if api_key:
                save_key = input("Save this API key for future use? (Y/n): ").lower()
                if save_key != "n":
                    save_api_key(api_key)

    if not api_key:
        print("No API key provided. Exiting.")
        return

    # Initialize YouTube API client
    youtube = build('youtube', 'v3', developerKey=api_key)

    # Get video URL from user
    url = input("Enter YouTube video URL: ")
    video_id = extract_video_id(url)

    if not video_id:
        print("Invalid YouTube URL or video ID.")
        return

    # Get max comments (optional)
    max_comments_input = input("Enter maximum number of comments to fetch (leave blank for all): ")
    max_comments = int(max_comments_input) if max_comments_input.strip() else None

    # Get video metadata
    print("Fetching video title and description...")
    video_title, video_description = get_video_metadata(youtube, video_id)
    print(f"Video Title: {video_title}")
    print(f"Video ID: {video_id}")

    # Extract comments from YouTube
    comments = get_video_comments(youtube, video_id, max_comments)

    if not comments:
        print("No comments found or unable to fetch comments.")
        return

    # Save comments to files
    json_file, metadata_file = save_comments(comments, video_id, video_title, video_description)

    if not json_file:
        print("Failed to save comments.")
        return

    # Load comments into vector database
    success, title, description = load_comments_to_db(json_file, metadata_file)

    if success:
        print("\nComments successfully loaded and embedded!")
        print(f"You can now run qa_test.py to ask questions about the video comments")
    else:
        print("\nFailed to load comments into the database.")


if __name__ == "__main__":
    main()