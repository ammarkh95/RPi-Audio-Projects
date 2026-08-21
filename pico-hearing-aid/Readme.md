# Pico 2 Hearing Aid DSP (3-Band WDRC)

A real-time, zero-latency digital signal processing (DSP) hearing aid pipeline built for the Raspberry Pi Pico 2. This project implements a 3-Band Wide Dynamic Range Compressor (WDRC) utilizing the Pico's hardware Floating-Point Unit (FPU), Direct Memory Access (DMA), and Programmable I/O (PIO) to deliver continuous audio.

## Features
* **Zero Latency I/O:** Uses DMA ping-pong buffering to read from the ADC and write to the I2S DAC simultaneously while the CPU processes 128-sample blocks.
* **Smart WDRC Decoupling:** User EQ slider adjustments fold directly into the compressor makeup gains and inversely scale the thresholds, guaranteeing that tone adjustments never accidentally drive the signal into clipping.
* **4th-Order Linkwitz-Riley (LR4) Crossover:** Cascaded biquad filters ensure all three bands sum perfectly in-phase with a flat magnitude response. The low band utilizes a bespoke alignment path to perfectly time-match the phase delay of the higher bands.
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
1. **Input Cleanup:** Raw ADC samples undergo adaptive DC bias removal and a ~100 Hz biquad high-pass filter to eliminate structural rumble.
2. **LR4 Crossover Split:** The signal is routed through 4th-order Linkwitz-Riley filters (cascaded Butterworth biquads) to split it into three bands:
   * **Low Band:** < 500 Hz (Passed through an all-pass proxy to time-align with the 3 kHz split).
   * **Mid Band:** 500 Hz – 3000 Hz
   * **High Band:** > 3000 Hz
3. **Dynamics Processing:** Each band passes through an envelope follower triggering a soft downward expander (to fade out analog mic hiss) and a WDRC compressor (to control loudness based on distinct ratios and thresholds).
4. **Recombination & Limiting:** The bands are summed completely in-phase, scaled by the master volume, and passed through a soft-knee limiter to protect the DAC.
5. **Output:** The resulting mono signal is duplicated to stereo left/right, bit-shifted to 32-bit I2S framing, and loaded into the TX DMA buffer.

---

## Static Tuning Configuration

The DSP settings can be modified intuitively using normalized floating-point sliders. Adjust the static variables under the `USER TUNING CONTROLS` section of the source code to change the EQ, master volume, and noise reduction behavior. 

**Slider Range:** `-5.0` (Less/Cut) to `+5.0` (More/Boost), where `0.0` is neutral.

```c
// Frequency sliders
static float user_bass   = -2.0f;  // Low-band loudness / warmth
static float user_middle =  2.0f;  // Speech body / voice presence
static float user_treble =  3.0f;  // Consonant clarity / brightness

// Overall loudness slider
static float user_volume =  1.0f;  // Global output gain

// Noise control slider
// Higher = aggressive analog noise reduction, but weaker quiet/far sounds.
static float user_noise_reduction = 5.0f; 

// Sound character slider
// -5 (comfort/smoother) to +5 (clarity/brighter speech).
// Tilts the overall frequency response and compression thresholds.
static float user_clarity = 4.0f;