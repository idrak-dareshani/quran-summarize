import time
import whisper

model = whisper.load_model("medium")

audio_file = "input_audio.mp3"

start_time = time.time()

# Transcribe the audio file
result = model.transcribe(audio_file, language="en")

end_time = time.time()

# Print the transcription
# print(result["text"])

elapsed = end_time - start_time
minutes = int(elapsed // 60)
seconds = int(elapsed % 60)
print(f"Time taken: {minutes} min {seconds} sec")

# Save the transcription to a text file
with open("transcription.txt", "w", encoding="utf-8") as f:
    f.write(result["text"])