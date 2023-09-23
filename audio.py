from pytube import YouTube
from pyannote.audio import Pipeline
from pydub import AudioSegment
import os

with open("./keys.json", 'r') as j:
    keys = json.loads(j.read())
    HUGGING_FACE_AUTH_TOKEN = keys["HUGGING_FACE_AUTH_TOKEN"]

def download_audio(link):
    yt = YouTube(link)
    audio_stream = yt.streams.filter(only_audio=True).first()
    audio_stream.download()
    return audio_stream.default_filename

def classify_voices(audio_file):
    pipeline = Pipeline.from_pretrained("pyannote/speaker-diarization",
                                        use_auth_token=HUGGING_FACE_AUTH_TOKEN)

    diarization = pipeline(audio_file)

    voices = {}
    for turn, _, speaker in diarization.itertracks(yield_label=True):
        print(f"start={turn.start:.1f}s stop={turn.end:.1f}s speaker_{speaker}")
        if speaker not in voices:
            voices[speaker] = [(turn.start, turn.end)]
        else:
            if voices[speaker][-1][1] == turn.start:
                voices[speaker][-1] = (voices[speaker][-1][0], turn.end)
            else:
                voices[speaker].append((turn.start, turn.end))

    return voices

def extract_tracks(audio_file, voice_intervals):
    audio = AudioSegment.from_file(audio_file, format="mp3")
    for speaker, intervals in voice_intervals.items():
        combined = AudioSegment.empty()  # Start with an empty segment
        for start, end in intervals:
            segment = audio[start * 1000:end * 1000]  # Convert from seconds to milliseconds
            combined += segment
        combined.export(f"{speaker}.mp3", format="mp3")

# Example Usage
if __name__ == "__main__":
    #youtube_link = 'YOUR_YOUTUBE_LINK_HERE'
    #downloaded_audio = download_audio(youtube_link)
    downloaded_audio = "angel.mp3"
    voice_intervals = classify_voices(downloaded_audio)
    extract_tracks(downloaded_audio, voice_intervals)
    #os.remove(downloaded_audio)  # Remove the downloaded file
