# Pico 2 Hearing Aid DSP (3-Band WDRC)

A real-time, zero-latency digital signal processing (DSP) hearing aid pipeline built for the Raspberry Pi Pico 2. This project implements a 3-Band Wide Dynamic Range Compressor (WDRC) utilizing the Pico's hardware Floating-Point Unit (FPU), Direct Memory Access (DMA), and Programmable I/O (PIO) to deliver continuous audio.

## Features
* **Zero Latency I/O:** Uses DMA ping-pong buffering to read from the ADC and write to the I2S DAC simultaneously while the CPU processes 128-sample blocks.
* **Optimized 3-Band WDRC:** Independent downward expansion (noise gating) and compression for Low, Mid, and High frequency bands.
* **Phase-Corrected Crossover:** Inverts the mid-band polarity upon recombination to prevent phase cancellation notches (Linkwitz-Riley style).
* **Rational Soft Clipper:** Gracefully rounds off loud transients before they hit the digital ceiling, preventing harsh integer overflow distortion.
* **Denormal Flushing:** Eliminates FPU stalls during absolute silence.

---

## Project Architecture

### Hardware Mapping
This project is configured for a MAX9814 analog microphone and a standard I2S DAC.

| Peripheral | Component | Pin | Notes |
| :--- | :--- | :--- | :--- |
| **ADC Input** | MAX9814 Mic Out | `26` | 12-bit ADC at 44.1 kHz |
| **I2S MCLK** | DAC Master Clock | `2` | Driven by PIO |
| **I2S BCK** | DAC Bit Clock | `3` | Driven by PIO |
| **I2S LRCK** | DAC Word/LR Clock | `4` | Driven by PIO |
| **I2S DOUT** | DAC Data In | `5` | Driven by PIO |

### DSP Signal Flow
The audio pipeline executes block-by-block with the following floating-point signal path:
1. **Input Cleanup:** Raw ADC samples undergo adaptive DC bias removal and a 100Hz biquad high-pass filter to eliminate structural rumble.
2. **Crossover Split:** The signal is routed through cascaded IIR biquad filters to split it into three bands:
   * **Low Band:** < 500 Hz
   * **Mid Band:** 500 Hz – 3000 Hz
   * **High Band:** > 3000 Hz
3. **Dynamics Processing:** Each band passes through an envelope follower triggering a soft downward expander (to remove analog mic hiss) and a WDRC compressor (to control loudness based on specific ratios and thresholds).
4. **Recombination & Limiting:** The bands are summed together (with the mid-band inverted for phase alignment), scaled by the master volume, and passed through a soft-knee limiter.
5. **Output:** The resulting mono signal is duplicated to stereo left/right, bit-shifted to 32-bit I2S framing, and loaded into the TX DMA buffer.

---

## Static Tuning Configuration

The DSP settings can be modified easily without touching the complex WDRC math. Adjust the `MY_TUNING` struct at the bottom of the source code to change the EQ, master volume, and noise reduction behavior. 

```c
static const HearingAidUI MY_TUNING = {
    .bass_slider = 1.2f,     // >1.0 boosts bass, <1.0 cuts it
    .middle_slider = 1.4f,   // Adjusts the 500Hz-3kHz vocal band
    .treble_slider = 1.1f,   // Controls high-frequency detail
    .volume_slider = 1.0f,   // Master output multiplier
    .noise_reduction = 0.3f, // 0.0 (off) to 1.0 (max suppression)
    .clarity_mode = false    // True = dynamic vocal boost, False = natural dynamics
};