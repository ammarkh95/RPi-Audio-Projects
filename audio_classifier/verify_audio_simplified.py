import os
import subprocess
import urllib.request
import csv
import numpy as np
import soundfile as sf
import scipy.signal
from ai_edge_litert.interpreter import Interpreter

# Force Matplotlib to use a headless backend
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# ==========================================
# CONFIGURATION
# ==========================================
MODEL_PATH = 'yamnet_cpu.tflite'
ORIGINAL_AUDIO_PATH = 'speech_whistling2.wav' 
RECORDED_AUDIO_PATH = 'speech_whistling2_right_record.wav'     
CLASS_MAP_PATH = 'yamnet_class_map.csv'
CLASS_MAP_URL = 'https://raw.githubusercontent.com/tensorflow/models/master/research/audioset/yamnet/yamnet_class_map.csv'

TARGET_SAMPLE_RATE = 16000
YAMNET_FRAME_SAMPLES = 15600 
FRAME_DURATION = 0.96 # YAMNet processes 0.96 seconds of audio at a time

# ==========================================
# AUDIO PROCESSING FUNCTIONS
# ==========================================
def record_audio_from_mic(duration=3):
    print(f"🎙️ Recording {duration} seconds from DIY Coupler...")
    record_command = (
        f"arecord -D stereo_mics -c 2 -r 48000 -f S24_LE -d {duration + 1} -t wav | "
        f"sox -t wav - -c 1 -r {TARGET_SAMPLE_RATE} -b 16 {RECORDED_AUDIO_PATH} trim 0.5 {duration}"
    )
    subprocess.run(record_command, shell=True, check=True, stderr=subprocess.DEVNULL)
    print("✅ Recording complete and formatted to 16kHz Mono.")

def load_and_format_audio(file_path):
    wav_data, sample_rate = sf.read(file_path)
    if len(wav_data.shape) > 1:
        wav_data = np.mean(wav_data, axis=1)
        
    if sample_rate != TARGET_SAMPLE_RATE:
        num_samples = int(len(wav_data) * float(TARGET_SAMPLE_RATE) / sample_rate)
        wav_data = scipy.signal.resample(wav_data, num_samples)
        
    remainder = len(wav_data) % YAMNET_FRAME_SAMPLES
    if remainder != 0:
        padding = YAMNET_FRAME_SAMPLES - remainder
        wav_data = np.pad(wav_data, (0, padding), 'constant')
        
    return wav_data.astype(np.float32)

def get_yamnet_scores(audio_data, interpreter):
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    num_frames = len(audio_data) // YAMNET_FRAME_SAMPLES
    all_scores = []

    for i in range(num_frames):
        start_idx = i * YAMNET_FRAME_SAMPLES
        end_idx = start_idx + YAMNET_FRAME_SAMPLES
        frame = audio_data[start_idx:end_idx]

        interpreter.set_tensor(input_details[0]['index'], frame)
        interpreter.invoke()

        frame_scores = interpreter.get_tensor(output_details[0]['index'])[0]
        all_scores.append(frame_scores)
    
    return np.array(all_scores)

def calculate_mse(original_scores_2d, recorded_scores_2d):
    return np.mean((original_scores_2d - recorded_scores_2d) ** 2)

# ==========================================
# VISUALIZATION FUNCTIONS
# ==========================================
def load_class_names():
    if not os.path.exists(CLASS_MAP_PATH):
        print("📥 Downloading YAMNet class dictionary...")
        urllib.request.urlretrieve(CLASS_MAP_URL, CLASS_MAP_PATH)
    
    class_names = []
    with open(CLASS_MAP_PATH, 'r') as csvfile:
        reader = csv.reader(csvfile)
        next(reader) 
        for row in reader:
            class_names.append(row[2]) 
    return class_names

def plot_snapshots(orig_scores_2d, rec_scores_2d, class_names, top_n=3):
    """Generates a PowerPoint-optimized grid of bar charts (3 columns)."""
    num_frames = orig_scores_2d.shape[0]
    
    # 1. Force exactly 3 columns (unless the audio is shorter than 3 frames)
    ncols = min(3, num_frames)
    nrows = (num_frames + ncols - 1) // ncols
    
    # Scale figure height based on rows, keeping the 16-inch widescreen width
    fig_height = max(5.0, 4.5 * nrows)
    fig, axes = plt.subplots(nrows, ncols, figsize=(16, fig_height))
    
    if num_frames == 1:
        axes = [axes]
    else:
        axes = axes.flatten()
        
    for i in range(num_frames):
        ax = axes[i]
        
        orig_frame = orig_scores_2d[i]
        rec_frame = rec_scores_2d[i]
        
        # Grab top N classes for this specific frame
        top_orig = list(np.argsort(orig_frame)[::-1][:top_n])
        top_rec = list(np.argsort(rec_frame)[::-1][:top_n])
        
        combined_indices = list(set(top_orig + top_rec))
        combined_indices.sort(key=lambda x: orig_frame[x], reverse=True)
        
        labels = [class_names[j] for j in combined_indices]
        orig_vals = [orig_frame[j] for j in combined_indices]
        rec_vals = [rec_frame[j] for j in combined_indices]
        
        # Truncate very long class names so they don't overlap horizontally under the bars
        short_labels = [lbl[:15] + ".." if len(lbl) > 15 else lbl for lbl in labels]
        
        x = np.arange(len(labels))
        width = 0.35
        
        rects1 = ax.bar(x - width/2, orig_vals, width, label='Original', color='#2ca02c')
        rects2 = ax.bar(x + width/2, rec_vals, width, label='Recorded', color='#d62728')
        
        start_time = i * FRAME_DURATION
        end_time = (i + 1) * FRAME_DURATION
        
        # 2. Text formatting
        ax.set_title(f'Frame {i+1} ({start_time:.2f}s - {end_time:.2f}s)', fontsize=14, fontweight='bold')
        ax.set_ylabel('Confidence', fontsize=12)
        ax.set_xticks(x)
        ax.set_xticklabels(short_labels, rotation=22, ha="right", fontsize=11)
        
        # 3. Increase Y-limit to 1.25 so 1.0 scores have headroom for their text labels
        ax.set_ylim([0, 1.25])
        ax.grid(axis='y', linestyle='--', alpha=0.5)
        
        # 4. Shrink font size on top of the bars to prevent overlapping numbers
        ax.bar_label(rects1, fmt='%.2f', padding=2, fontsize=9)
        ax.bar_label(rects2, fmt='%.2f', padding=2, fontsize=9)
        
        # Place legend only on the first chart
        if i == 0:
            ax.legend(loc='upper right', fontsize=11)
            
    # Hide any unused subplots (if you have exactly 6 frames, this won't trigger)
    for j in range(num_frames, len(axes)):
        fig.delaxes(axes[j])
        
    fig.suptitle('YAMNet Classification Snapshots (0.96s Frames)', fontsize=22, fontweight='bold', y=0.97)
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    
    filename = 'classification_snapshots_ppt.png'
    plt.savefig(filename, dpi=300)
    print(f"📊 PPT-optimized snapshot grid successfully saved as '{filename}'")
# ==========================================
# MAIN EXECUTION
# ==========================================
if __name__ == "__main__":
    # record_audio_from_mic(duration=3)

    print("🧠 Loading YAMNet TFLite Model...")
    interpreter = Interpreter(model_path=MODEL_PATH)
    interpreter.allocate_tensors()

    print("🔄 Formatting waveforms (Full Duration)...")
    original_waveform = load_and_format_audio(ORIGINAL_AUDIO_PATH)
    recorded_waveform = load_and_format_audio(RECORDED_AUDIO_PATH)

    print("🔍 Generating semantic fingerprints frame-by-frame...")
    original_scores_2d = get_yamnet_scores(original_waveform, interpreter)
    recorded_scores_2d = get_yamnet_scores(recorded_waveform, interpreter)

    original_scores_2d = original_scores_2d[: len(recorded_scores_2d)] 

    # print("🧮 Calculating Mean Squared Error...")
    # mse = calculate_mse(original_scores_2d, recorded_scores_2d)
    
    # print("\n" + "="*45)
    # print("🎯 TEST BENCH RESULTS")
    # print("="*45)
    # print(f"Mean Squared Error (MSE): {mse:.4f}")
    # if mse < 0.05:
    #     print("✅ PASS: The hearing aid preserved the overall audio signature.")
    # else:
    #     print("❌ FAIL: The audio was significantly altered or noisy overall.")
    # print("="*45)

    print("🎨 Generating performance snapshot graph...")
    class_names = load_class_names()
    plot_snapshots(original_scores_2d, recorded_scores_2d, class_names, top_n=3)