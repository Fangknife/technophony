import numpy as np
import os
import json
import subprocess
from subprocess import run
import librosa
import librosa.display
import matplotlib.pyplot as plt
import whisper
from vosk import Model, KaldiRecognizer
from videogrep import parse_transcript
from moviepy.editor import *
from moviepy.video.tools.subtitles import SubtitlesClip
from moviepy.config import change_settings
change_settings({"IMAGEMAGICK_BINARY": r"C:\\Program Files\\ImageMagick-7.1.1-Q16-HDRI\\magick.exe"})
from PIL import Image

###-------------------------------------------------------------------------------------------------------------
folderDir = "./audioFiles"    # folder with .wav audio files, also where things will be saved.
###-------------------------------------------------------------------------------------------------------------

def saveSpec(folderDir, file):
    path = folderDir + file
    y, sr = librosa.load(path)
    D = librosa.stft(y)
    S_db = librosa.amplitude_to_db(np.abs(D), ref=np.max)
    fig, ax = plt.subplots()
    img = librosa.display.specshow(S_db, x_axis='time', y_axis='log', ax=ax)
    ax.set(title=file)
    fig.colorbar(img, ax=ax, format="%+2.f dB")
    plt.savefig(folderDir + os.path.splitext(os.path.basename(folderDir + file))[0] + "_spectro.png")
    plt.close()

### vvv batch export spectrograms .pngs
for file in os.listdir(folderDir):
    if (file.endswith(".wav")):
        saveSpec(folderDir, file)

###-------------------------------------------------------------------------------------------------------------

def transcribeWhisper(file, outputFile):
    thisFile = folderDir + file
    audio = whisper.load_audio(thisFile)
    audio = whisper.pad_or_trim(audio)
    result = model.transcribe(thisFile, language="en", no_speech_threshold=True)
    with open(outputFile, "a", encoding="utf-8") as f:
        print(file[:-4] + ":" + result["text"], file=f)

### vvv batch transcribe to same .txt file -- whisper
outputFile = folderDir + + "transcript_whisper.txt"
model = whisper.load_model("base")
for file in os.listdir(folderDir):
    if (file.endswith(".wav")):
        transcribeWhisper(file, outputFile)

###-------------------------------------------------------------------------------------------------------------

def transcribeVosk(file, outputFile):
    with subprocess.Popen(["ffmpeg", "-loglevel", "quiet", "-i",
                                folderDir + file,
                                "-ar", str(SAMPLE_RATE) , "-ac", "1", "-f", "s16le", "-"],
                                stdout=subprocess.PIPE) as process:
            tmpScript = ""
            while True:
                data = process.stdout.read(4000)
                if len(data) == 0:
                    break
                if rec.AcceptWaveform(data):
                    thisScript = rec.Result()[14:-3]
                    if thisScript != "":
                        tmpScript += thisScript + ". "
            with open(outputFile, "a", encoding="utf-8") as f:
                print(file + ":" + tmpScript, file=f)

### vvv batch transcribe to same .txt -- vosk
outputFile = folderDir +"transcript_vosk.txt"
SAMPLE_RATE = 16000
model = Model(lang="en-us")
rec = KaldiRecognizer(model, SAMPLE_RATE)
for file in os.listdir(folderDir):
    if (file.endswith(".wav")):
        transcribeVosk(file, outputFile)


###-------------------------------------------------------------------------------------------------------------

def captionVosk(folderDir, file):
    ### vvv make transcript .json - via videogrep
    fileBaseName = os.path.splitext(os.path.basename(folderDir + file))[0]
    args = [
        "videogrep",
        "--input", folderDir + file,
        "--transcribe"
    ]
    run(args)

    if os.path.isfile(folderDir + fileBaseName + ".json"):
        os.rename(folderDir + fileBaseName + ".json", fileBaseName + ".json")
        transcript = parse_transcript(file) #  file.json
        subs = []
        for sentence in transcript:
            for word in sentence["words"]:
                subs.append(((word["start"], word["end"]), word["word"]))

    ### vvv create a .mp4 with subtitles on black bg - via moviepy
        im = Image.new('RGB', (500, 300), (0, 0, 0))
        im.save('tmpImg.jpg', quality=95)
        video = ImageClip("tmpImg.jpg")
        video = video.set_duration(subs[len(subs)-1][0][1])
        audioclip = AudioFileClip(folderDir + file)
        video = video.set_audio(audioclip) 
        video.write_videofile("tmpImg.mp4", fps=15)
        video = VideoFileClip("tmpImg.mp4")

        generator = lambda txt: TextClip(txt, font='Arial', fontsize=50, color='white')
        subtitles = SubtitlesClip(subs, generator)
        result = CompositeVideoClip([video, subtitles.set_pos(('center','bottom'))])
        result.write_videofile(folderDir + fileBaseName + "_capt.mp4", fps=video.fps, temp_audiofile="tmpAudio.m4a", 
                            remove_temp=True, codec="libx264", audio_codec="aac")
        os.remove("tmpImg.jpg")
        os.remove("tmpImg.mp4")
        os.rename( fileBaseName + ".json", folderDir + fileBaseName + ".json")

### vvv save a .mp4 with timed captions on bg
### will only wsave a video if vosk detected words
for file in os.listdir(folderDir):
    if (file.endswith(".wav")):
        captionVosk(folderDir, file)

###-------------------------------------------------------------------------------------------------------------

