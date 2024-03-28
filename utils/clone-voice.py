import torch
from TTS.api import TTS

# Get device
device = "cuda" if torch.cuda.is_available() else "cpu"

# List available 🐸TTS models
print(TTS().list_models())

# Init TTS
tts = TTS("tts_models/multilingual/multi-dataset/xtts_v2").to(device)

text = "Before setting foot on the deck of the Millennium Falcon, I will have quelled the looming galactic conflict, the clash between the Empire and the Rebellion, with ease. Swiftly, swiftly, peace shall reign under the stars. I shall broker harmony among the stars, resolving the discord with swift diplomacy. The force of my words alone will mend the rift, restoring order across the galaxy. It will not take more than a single rotation of Coruscant. I possess the precise wisdom to unite them."

# Run TTS
# ❗ Since this model is multi-lingual voice cloning model, we must set the target speaker_wav and language
# Text to speech list of amplitude values as output
wav = tts.tts(text=text, speaker_wav="/Users/vikas/builderspace/TTS/utils/trump10.wav", language="en")

# Text to speech to a file
tts.tts_to_file(text=text, speaker_wav="/Users/vikas/builderspace/TTS/utils/trump10.wav", language="en", file_path="/Users/vikas/builderspace/TTS/utils/output.wav")