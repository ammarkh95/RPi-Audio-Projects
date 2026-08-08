import ai_edge_litert.interpreter as tflite
import zipfile, time, platform
import librosa
import numpy as np

USE_TPU = False
MODEL_PATH = './yamnet_edge.tflite' if USE_TPU else './yamnet_cpu.tflite'
INPUT_SIZE = 15600
AUDIO_FILE = './miaow_16k_left_record.wav'

EDGETPU_SHARED_LIB = {
    'Linux': 'libedgetpu.so.1',
    'Darwin': 'libedgetpu.1.dylib',
    'Windows': 'edgetpu.dll',
}[platform.system()]

# --- labels (zip-packed in CPU model, sidecar for compiled model) ---
try:
    with zipfile.ZipFile(MODEL_PATH) as z:
        labels = [l.decode('utf-8').strip() for l in z.open('yamnet_label_list.txt').readlines()]
except zipfile.BadZipFile:
    with open('yamnet_label_list.txt') as f:
        labels = [l.strip() for l in f]

# --- interpreter ---
delegates = [tflite.load_delegate(EDGETPU_SHARED_LIB)] if USE_TPU else []
interpreter = tflite.Interpreter(model_path=MODEL_PATH, experimental_delegates=delegates)
interpreter.allocate_tensors()  # before get_*_details

input_details = interpreter.get_input_details()[0]
output_details = interpreter.get_output_details()[0]
in_dtype, out_dtype = input_details['dtype'], output_details['dtype']
in_quant, out_quant = input_details['quantization'], output_details['quantization']

# --- audio ---
audio_data, sample_rate = librosa.load(AUDIO_FILE, sr=None)
print(f'Audio file: {AUDIO_FILE}')
print(f'Sample rate: {sample_rate} Hz, duration: {len(audio_data)/sample_rate:.2f}s')
print(f'---------------------------------------------------------------------------')

remainder = len(audio_data) % INPUT_SIZE
if remainder:
    audio_data = np.pad(audio_data, (0, INPUT_SIZE - remainder))

frames = librosa.util.frame(audio_data, frame_length=INPUT_SIZE, hop_length=INPUT_SIZE).T

def quantize(x, dtype, quant):
    if dtype in (np.int8, np.uint8):
        scale, zp = quant
        return (x / scale + zp).astype(dtype)
    return x.astype(np.float32)

def dequantize(x, dtype, quant):
    if dtype in (np.int8, np.uint8):
        scale, zp = quant
        return (x.astype(np.float32) - zp) * scale
    return x

# --- inference ---
results = []
for i, frame in enumerate(frames):
    x = quantize(frame, in_dtype, in_quant).reshape(input_details['shape'])
    interpreter.set_tensor(input_details['index'], x)

    t0 = time.perf_counter()
    interpreter.invoke()
    t1 = time.perf_counter()

    raw = interpreter.get_tensor(output_details['index'])
    scores = dequantize(raw, out_dtype, out_quant)
    results.append(scores)

    # Extract the top label and its corresponding confidence score
    top_index = scores.argmax()
    label = labels[top_index]
    confidence = np.max(scores) # Grabs the highest probability in the array
    
    print(f"Inference {i}: {(t1 - t0) * 1000:.4f} ms → {label} (Confidence: {confidence:.4f})")

mean_scores = np.mean(results, axis=0)
overall_top_index = mean_scores.argmax()
print(f'\nThe main sound is: {labels[overall_top_index]} (Mean Confidence: {np.max(mean_scores):.4f})')