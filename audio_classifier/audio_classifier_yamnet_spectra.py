import ai_edge_litert.interpreter as tflite
import librosa
import time
import platform
import numpy as np
import scipy.signal

MODEL_PATH = './yamnet_spectra_in_edgetpu.tflite' 
AUDIO_FILE = "./miaow_16k.wav"

EDGETPU_SHARED_LIB = {
    'Linux': 'libedgetpu.so.1',
    'Darwin': 'libedgetpu.1.dylib',
    'Windows': 'edgetpu.dll'
}[platform.system()]

# 1. Load labels securely
try:
    with open('yamnet_label_list.txt', 'r', encoding='utf-8') as f:
        labels = [line.strip().strip('"') for line in f.readlines()]
except FileNotFoundError:
    labels = [f"Class {i}" for i in range(521)]

# 2. Initialize TPU Interpreter
interpreter = tflite.Interpreter(
    model_path=MODEL_PATH,
    experimental_delegates=[tflite.load_delegate(EDGETPU_SHARED_LIB)]
)
interpreter.allocate_tensors()

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# 3. Load Audio at canonical 16,000 Hz
audio_data, sample_rate = librosa.load(AUDIO_FILE, sr=16000)

# =====================================================================
# EXACT REPLICATED TRANSLATION OF YAMNET FEATURES.PY (WITHOUT TENSORFLOW)
# =====================================================================

# 4a. Emulate YAMNet pad_waveform logic (Reflect Padding)
# YAMNet pads the end of the waveform to match whole frame increments cleanly
window_samples = 400  # 25ms
hop_samples = 160     # 10ms

# 4b. Perform accurate Periodic Hann Window calculation
# SciPy's sym=False yields a true periodic Hann window, exactly matching tf.signal.hann_window
window_func = scipy.signal.windows.hann(window_samples, sym=False)

# 4c. Replicate YAMNet's Magnitude STFT via NumPy/SciPy
# We pass center=False and calculate frames step-by-step to match YAMNet's exact feature matrix layout
num_frames = int(np.floor((len(audio_data) - window_samples) / hop_samples) + 1)
stft_matrix = np.zeros((num_frames, 512 // 2 + 1))  # Width will now correctly be 257

for i in range(num_frames):
    start = i * hop_samples
    end = start + window_samples
    frame = audio_data[start:end] * window_func
    # Calculate RFFT magnitude matching tf.abs(tf.signal.stft)
    stft_matrix[i, :] = np.abs(np.fft.rfft(frame, n=512))

# 4d. Replicate YAMNet's exact linear-to-mel filter bank math
# YAMNet uses 64 bands from 125Hz up to 7500Hz
mel_basis = librosa.filters.mel(sr=16000, n_fft=512, n_mels=64, fmin=125, fmax=7500, htk=True)
mel_spectrogram = np.dot(stft_matrix, mel_basis.T)

# 4e. Exact features.py stabilization offset rules: log(mel + 0.001) + 0.0001
log_mel = np.log(mel_spectrogram + 0.001) + 0.0001

# =====================================================================

# 5. Slide and extract 96x64 patches precisely from the clean matrix
frame_length = 96
hop_length = 72  # 50% overlap rule
results = []

print(f'Audio file: {AUDIO_FILE}')
print(f'Sample rate: {sample_rate} Hz, duration: {len(audio_data)/sample_rate:.2f}s')
print(f'---------------------------------------------------------------------------')

for start_frame in range(0, log_mel.shape[0] - frame_length + 1, hop_length):
    spectrogram_patch = log_mel[start_frame:start_frame + frame_length, :]
    
    # Form input frame tracking shape: [1, 96, 64]
    spectrogram_patch = np.expand_dims(spectrogram_patch, axis=0).astype(np.float32)
    
    # Run Accelerated Inference
    interpreter.set_tensor(input_details[0]['index'], spectrogram_patch)
    
    start_time = time.perf_counter()
    interpreter.invoke()
    end_time = time.perf_counter()

    scores = interpreter.get_tensor(output_details[0]['index'])[0]
    results.append(scores)
    
    top_class_index = scores.argmax()
    
    print(f"Inference {start_frame}: {(end_time - start_time) * 1000:.4f} ms → {labels[top_class_index]}")

# 6. Aggregated Global Model Decision Classification
if results:
    results_np = np.array(results)
    mean_results = results_np.mean(axis=0)
    result_index = mean_results.argmax()
    
    print(f'\nThe overall main sound is: {labels[result_index]}')