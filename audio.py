from pytube import YouTube
from pyannote.audio import Pipeline
from pydub import AudioSegment
import json
import os

with open("./keys.json", 'r') as j:
    keys = json.loads(j.read())
    HUGGING_FACE_AUTH_TOKEN = keys["HUGGING_FACE_AUTH_TOKEN"]

LINKS = []
TIMESTAMPS = []


def download_audio(link, output_folder='.'):
    """
    Downloads a YouTube video as an MP3.

    Parameters:
    - link (str): YouTube video URL.
    - output_folder (str, optional): Folder where the MP3 will be saved. Defaults to the current folder.

    Returns:
    - str: Path to the downloaded MP3.
    """
    yt = YouTube(link)
    audio_stream = yt.streams.filter(only_audio=True, file_extension='mp4').first()
    download_path = audio_stream.download(output_path=output_folder)
    
    # Convert the downloaded mp4 audio to mp3
    audio = AudioSegment.from_file(download_path, format="mp4")
    mp3_path = download_path.replace(".mp4", ".mp3")
    audio.export(mp3_path, format="mp3")

    # Optionally, if you want to remove the original .mp4 file after conversion:
    # import os
    # os.remove(download_path)

    return mp3_path


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
    print(voices)
    return voices

def extract_speaker_segments(audio_file, voice_intervals):
    audio = AudioSegment.from_file(audio_file, format="mp3")
    for speaker, intervals in voice_intervals.items():
        combined = AudioSegment.empty()  # Start with an empty segment
        for (start, end) in intervals:
            segment_start = int(start * 1000)  # Convert from seconds to milliseconds
            segment_end = int(end * 1000)  # Convert from seconds to milliseconds
            segment = audio[segment_start:segment_end]
            combined += segment
        combined.export(f"{speaker}.mp3", format="mp3")


def create_aggregated_track(youtube_links, speaker_timestamps):
    assert len(youtube_links) == len(speaker_timestamps), "Ensure each URL has a corresponding list of timestamps."

    final_combined = AudioSegment.empty()
    for i, link in enumerate(youtube_links):
        downloaded_audio = download_audio(link)
        extracted_segment = extract_speaker_segments(downloaded_audio, speaker_timestamps[i])
        final_combined += extracted_segment
        os.remove(downloaded_audio)

    final_combined.export("final_speaker_track.mp3", format="mp3")

# Example Usage
if __name__ == "__main__":
    #youtube_link = 'YOUR_YOUTUBE_LINK_HERE'
    #downloaded_audio = download_audio(youtube_link)
    downloaded_audio = "angel.mp3"
    voice_intervals = classify_voices(downloaded_audio)
    extract_speaker_segments(downloaded_audio, voice_intervals)
    create_aggregated_track(LINKS, TIMESTAMPS)
    #os.remove(downloaded_audio)  # Remove the downloaded file
