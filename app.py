from flask import Flask, request, jsonify, render_template, send_from_directory
import os
import time
from youtube_qa import analyze_youtube_video, answer_question, extract_video_id, CURRENT_VIDEO_ID

app = Flask(__name__, static_folder='static')


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/api/analyze', methods=['POST'])
def analyze():
    data = request.json
    youtube_url = data.get('youtube_url')
    api_key = data.get('api_key', 'AIzaSyDj7I12G6kpxEt4esWYXh2XwVAOXu7mbz0')

    if not youtube_url:
        return jsonify({'error': 'No YouTube URL provided'}), 400

    try:
        # Extract video ID
        video_id = extract_video_id(youtube_url)
        if not video_id:
            return jsonify({'error': 'Invalid YouTube URL or video ID'}), 400

        # Run the analysis
        start_time = time.time()
        comment_count = analyze_youtube_video(youtube_url, api_key)
        processing_time = time.time() - start_time

        if comment_count == 0:
            return jsonify({'error': 'No comments found or unable to analyze video'}), 400

        # Read the metadata file if it exists
        metadata = {}
        metadata_path = os.path.join("chroma", "video_metadata.json")
        if os.path.exists(metadata_path):
            try:
                with open(metadata_path, 'r', encoding='utf-8') as f:
                    import json
                    metadata = json.load(f)
            except Exception as e:
                print(f"Error reading metadata: {str(e)}")

        return jsonify({
            'success': True,
            'video_id': video_id,
            'comment_count': comment_count,
            'processing_time': f"{processing_time:.2f} seconds",
            'video_title': metadata.get('title', 'Unknown Video')
        })
    except Exception as e:
        print(f"Error: {str(e)}")
        return jsonify({'error': str(e)}), 500


@app.route('/api/ask', methods=['POST'])
def ask_question_api():
    data = request.json
    question = data.get('question')
    k = data.get('k')  # Optional parameter for number of comments

    if not question:
        return jsonify({'error': 'No question provided'}), 400

    try:
        # Convert k to integer if provided
        if k is not None:
            try:
                k = int(k)
            except ValueError:
                return jsonify({'error': 'Parameter k must be an integer'}), 400

        # Process the question
        start_time = time.time()
        answer = answer_question(question, k=k)
        processing_time = time.time() - start_time

        return jsonify({
            'answer': answer,
            'processing_time': f"{processing_time:.2f} seconds"
        })
    except Exception as e:
        print(f"Error in Q&A: {str(e)}")
        return jsonify({'error': str(e)}), 500


@app.route('/api/status', methods=['GET'])
def get_status():
    try:
        status = {
            'database_exists': os.path.exists("chroma"),
            'current_video_id': CURRENT_VIDEO_ID
        }

        # Add metadata from JSON file if it exists
        metadata_path = os.path.join("chroma", "video_metadata.json")
        if os.path.exists(metadata_path):
            try:
                with open(metadata_path, 'r', encoding='utf-8') as f:
                    import json
                    metadata = json.load(f)
                status['metadata'] = metadata
            except Exception as e:
                status['metadata_error'] = str(e)

        return jsonify(status)
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/static/<path:path>')
def serve_static(path):
    return send_from_directory('static', path)


if __name__ == '__main__':
    app.run(debug=True, port=5000)