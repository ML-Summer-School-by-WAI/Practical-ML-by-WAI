
import soundfile
import numpy as np
from io import BytesIO
def audio_array_to_buffer(audio_array : np.array,sample_rate: int) -> BytesIO:
    buffer = BytesIO()
    soundfile.write("test.wav",audio_array, sample_rate,format="WAV",subtype="PCM_16")
    soundfile.write(buffer,audio_array, sample_rate,format="WAV",subtype="PCM_16")
    buffer.seek(0)
    return buffer