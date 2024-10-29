from flask import Flask, request, jsonify, send_file
from GP_TTS.EdgeTTSModel import synthesize_speech
from GP_STT.WhisperSTTModel import pipe
import librosa
import os

app = Flask(__name__)


@app.route('/')
def home():
    return "Welcome to LinkAI."


@app.route('/stt', methods=['POST'])
def convert_voice_to_text():
    audio_file = request.files['audio']
    audio_data, _ = librosa.load(audio_file, sr=16000)
    text = pipe(audio_data)['text'].strip()
    return jsonify({'text': text})


@app.route('/tts', methods=['POST'])
async def convert_text_to_voice():
    data = request.get_json()
    text = data.get('text')

    audio_path = await synthesize_speech(text=text)

    if os.path.exists(audio_path):
        print(audio_path)
        return send_file(audio_path, as_attachment=True, download_name="generated_audio.mp3")
    else:
        return jsonify({'error': 'Failed to generate audio'}), 500


if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5003)
