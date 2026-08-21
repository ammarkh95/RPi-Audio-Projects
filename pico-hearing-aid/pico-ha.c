#include <stdio.h>
#include <math.h>
#include "pico/stdlib.h"
#include "hardware/pio.h"
#include "hardware/dma.h"
#include "hardware/adc.h"
#include "hardware/irq.h"
#include "hardware/sync.h"
#include "i2s.pio.h"

#define SAMPLE_RATE     44100
#define BUF_LEN         256          // I2S buffer (L+R interleaved, so 128 frames)
#define ADC_BUF_LEN     128          // frames (mono ADC samples)
#define ADC_PIN         26           // MAX9814 OUT

// ---- I2S pins ----
const uint8_t MCLK_PIN     = 2;
const uint8_t OUT_BCK_PIN  = 3;
const uint8_t OUT_LRCK_PIN = 4;
const uint8_t DOUT_PIN     = 5;

// ---- PIO State machines for I2S MCLK, CLK and LRCLK ----
uint8_t smMCLK;
uint8_t smCLK;

// ---- I2S DMA ping-pong ----
static int i2s_dma_a, i2s_dma_b;
uint32_t i2sBufA[BUF_LEN] __attribute__((aligned(16)));
uint32_t i2sBufB[BUF_LEN] __attribute__((aligned(16)));

// ---- ADC DMA ping-pong ----
static int adc_dma_a, adc_dma_b;
uint16_t adcBufA[ADC_BUF_LEN];
uint16_t adcBufB[ADC_BUF_LEN];

// Which ADC buffer just got filled
volatile bool adc_buffer_ready = false;
volatile bool adc_select = false;     // false=A just filled, true=B just filled

// Which I2S buffer needs refilling
volatile bool i2s_need_fill = false;
volatile bool i2s_select   = false;    // false=fill A, true=fill B

// ============================================================
// --- HEARING AID DSP PIPELINE: OPTIMIZED 3-BAND WDRC ---
// ============================================================
//
// Pipeline:
// ADC
//   -> adaptive DC/bias removal
//   -> DC blocker
//   -> input rumble high-pass
//   -> cascaded 3-band split
//   -> independent expansion/compression per band
//   -> recombine
//   -> fast soft limiter
//   -> stereo I2S output
//
// Designed for Fs = 44100 Hz and 128-sample processing blocks.
// Adds no extra block latency.

#define MB_NUM_BANDS 3

// Enable tiny-value flushing in filters.
// This avoids possible slow floating-point behavior during silence.
#define DSP_FLUSH_DENORMALS 1

// Precomputed envelope coefficients.
// Avoid calling expf() inside the real-time sample loop.
#define MB_ATTACK_COEFF       0.00565348f   // 4 ms at 44100 Hz
#define MB_RELEASE_COEFF      0.00018895f   // 120 ms at 44100 Hz

// Extra smoothing on computed gain to prevent zipper noise.
#define MB_GAIN_SMOOTH_COEFF  0.01500000f

typedef struct {
    // Biquad coefficients.
    // a0 is assumed to be 1.0.
    float b0, b1, b2;
    float a1, a2;

    // Delay memory for Direct Form I.
    float x1, x2;
    float y1, y2;
} MB_Biquad;

typedef struct {
    // Envelope and smoothed gain are persistent per band.
    float env;
    float gain_smooth;

    // Downward expander settings for quiet analog noise.
    float noise_floor;
    float min_gain;

    // WDRC compressor settings for loudness control.
    float threshold;
    float ratio;
    float makeup_gain;
} MB_BandDynamics;

// Tracks the real ADC midpoint, which may drift away from exactly 2048.
static float adc_bias_est = 2048.0f;
static float dc_x1 = 0.0f;
static float dc_y1 = 0.0f;

// ------------------------------------------------------------
// Input cleanup filter
// ------------------------------------------------------------

// Approx 100 Hz high-pass, Fs = 44.1 kHz.
static MB_Biquad input_hpf = {
    .b0 =  0.98995425f,
    .b1 = -1.97990850f,
    .b2 =  0.98995425f,
    .a1 = -1.97980793f,
    .a2 =  0.98000908f,
    .x1 = 0.0f, .x2 = 0.0f,
    .y1 = 0.0f, .y2 = 0.0f
};

// ------------------------------------------------------------
// LR4 3-band analysis split
// ------------------------------------------------------------
//
// LR4 = two identical 2nd-order Butterworth filters in series.
// Default recombination should be same polarity.
//
// Split:
//   low  = LR4 low-pass 500 Hz
//   rest = LR4 high-pass 500 Hz
//   mid  = LR4 low-pass 3000 Hz applied to rest
//   high = LR4 high-pass 3000 Hz applied to rest

static MB_Biquad split_lp_500_a = {
    .b0 =  0.00120740f,
    .b1 =  0.00241480f,
    .b2 =  0.00120740f,
    .a1 = -1.89931218f,
    .a2 =  0.90414178f,
    .x1 = 0.0f, .x2 = 0.0f,
    .y1 = 0.0f, .y2 = 0.0f
};

static MB_Biquad split_lp_500_b = {
    .b0 =  0.00120740f,
    .b1 =  0.00241480f,
    .b2 =  0.00120740f,
    .a1 = -1.89931218f,
    .a2 =  0.90414178f,
    .x1 = 0.0f, .x2 = 0.0f,
    .y1 = 0.0f, .y2 = 0.0f
};

static MB_Biquad split_hp_500_a = {
    .b0 =  0.95172821f,
    .b1 = -1.90345642f,
    .b2 =  0.95172821f,
    .a1 = -1.90112625f,
    .a2 =  0.90578659f,
    .x1 = 0.0f, .x2 = 0.0f,
    .y1 = 0.0f, .y2 = 0.0f
};

static MB_Biquad split_hp_500_b = {
    .b0 =  0.95172821f,
    .b1 = -1.90345642f,
    .b2 =  0.95172821f,
    .a1 = -1.90112625f,
    .a2 =  0.90578659f,
    .x1 = 0.0f, .x2 = 0.0f,
    .y1 = 0.0f, .y2 = 0.0f
};

static MB_Biquad split_lp_3000_a = {
    .b0 =  0.03357181f,
    .b1 =  0.06714362f,
    .b2 =  0.03357181f,
    .a1 = -1.41898265f,
    .a2 =  0.56108439f,
    .x1 = 0.0f, .x2 = 0.0f,
    .y1 = 0.0f, .y2 = 0.0f
};

static MB_Biquad split_lp_3000_b = {
    .b0 =  0.03357181f,
    .b1 =  0.06714362f,
    .b2 =  0.03357181f,
    .a1 = -1.41898265f,
    .a2 =  0.56108439f,
    .x1 = 0.0f, .x2 = 0.0f,
    .y1 = 0.0f, .y2 = 0.0f
};

static MB_Biquad split_hp_3000_a = {
    .b0 =  0.74501676f,
    .b1 = -1.49003352f,
    .b2 =  0.74501676f,
    .a1 = -1.41898265f,
    .a2 =  0.56108439f,
    .x1 = 0.0f, .x2 = 0.0f,
    .y1 = 0.0f, .y2 = 0.0f
};

static MB_Biquad split_hp_3000_b = {
    .b0 =  0.74501676f,
    .b1 = -1.49003352f,
    .b2 =  0.74501676f,
    .a1 = -1.41898265f,
    .a2 =  0.56108439f,
    .x1 = 0.0f, .x2 = 0.0f,
    .y1 = 0.0f, .y2 = 0.0f
};

// ------------------------------------------------------------
// Low-band phase alignment through the 3000 Hz LR4 split
// ------------------------------------------------------------
//
// In a cascaded 3-way LR4 split, mid/high pass through the 3000 Hz
// crossover. Low does not. This alignment path passes low through
// an equivalent LP3000+HP3000 summed LR4 path, approximating an
// all-pass delay/phase match without changing low-band magnitude.
//
// These must use separate states from the real mid/high split.

static MB_Biquad align_lp_3000_a = {
    .b0 =  0.03357181f,
    .b1 =  0.06714362f,
    .b2 =  0.03357181f,
    .a1 = -1.41898265f,
    .a2 =  0.56108439f,
    .x1 = 0.0f, .x2 = 0.0f,
    .y1 = 0.0f, .y2 = 0.0f
};

static MB_Biquad align_lp_3000_b = {
    .b0 =  0.03357181f,
    .b1 =  0.06714362f,
    .b2 =  0.03357181f,
    .a1 = -1.41898265f,
    .a2 =  0.56108439f,
    .x1 = 0.0f, .x2 = 0.0f,
    .y1 = 0.0f, .y2 = 0.0f
};

static MB_Biquad align_hp_3000_a = {
    .b0 =  0.74501676f,
    .b1 = -1.49003352f,
    .b2 =  0.74501676f,
    .a1 = -1.41898265f,
    .a2 =  0.56108439f,
    .x1 = 0.0f, .x2 = 0.0f,
    .y1 = 0.0f, .y2 = 0.0f
};

static MB_Biquad align_hp_3000_b = {
    .b0 =  0.74501676f,
    .b1 = -1.49003352f,
    .b2 =  0.74501676f,
    .a1 = -1.41898265f,
    .a2 =  0.56108439f,
    .x1 = 0.0f, .x2 = 0.0f,
    .y1 = 0.0f, .y2 = 0.0f
};
// ------------------------------------------------------------
// Per-band dynamics
// ------------------------------------------------------------

static const MB_BandDynamics base_dyn[MB_NUM_BANDS] = {
    // Low band
    {
        .env = 0.0f,
        .gain_smooth = 1.0f,
        .noise_floor = 10.0f,
        .min_gain = 0.25f,
        .threshold = 220.0f,
        .ratio = 2.5f,
        .makeup_gain = 0.80f
    },

    // Mid band
    {
        .env = 0.0f,
        .gain_smooth = 1.0f,
        .noise_floor = 6.0f,
        .min_gain = 0.45f,
        .threshold = 220.0f,
        .ratio = 2.2f,
        .makeup_gain = 2.40f
    },

    // High band
    {
        .env = 0.0f,
        .gain_smooth = 1.0f,
        .noise_floor = 5.0f,
        .min_gain = 0.35f,
        .threshold = 140.0f,
        .ratio = 1.8f,
        .makeup_gain = 2.60f
    }
};

static MB_BandDynamics mb_dyn[MB_NUM_BANDS] = {0};

// ============================================================
// --- USER TUNING CONTROLS: STATIC PROOF OF CONCEPT ---
// ============================================================
//
// Slider range:
//   -5 = less
//    0 = neutral
//   +5 = more
//
// These are static for now. Later they can be updated from BLE,
// buttons, UART commands, saved presets, or a phone app.

#define USER_SLIDER_MIN  (-5.0f)
#define USER_SLIDER_MAX  ( 5.0f)

// Frequency sliders.
static float user_bass   = 0.0f;   // low-band loudness / warmth
static float user_middle = 0.0f;   // speech body / voice presence
static float user_treble = 0.0f;   // consonant clarity / brightness

// Overall loudness slider.
static float user_volume = 1.0f;

// Noise control slider.
// Higher = more noise reduction, but weaker far/quiet sounds.
static float user_noise_reduction = 5.0f;

// Sound character slider.
// -5 = comfort/smoother
//  0 = balanced
// +5 = clarity/brighter speech
static float user_clarity = 0.0f;

typedef struct {
    float low_mix;
    float mid_mix;
    float high_mix;
    float output_gain;

    float low_noise_floor;
    float mid_noise_floor;
    float high_noise_floor;

    float low_min_gain;
    float mid_min_gain;
    float high_min_gain;

} UserTuningRuntime;

static UserTuningRuntime user_rt = {
    .low_mix = 0.85f,
    .mid_mix = 1.00f,
    .high_mix = 0.95f,
    .output_gain = 1.00f,

    .low_noise_floor = 10.0f,
    .mid_noise_floor = 6.0f,
    .high_noise_floor = 5.0f,

    .low_min_gain = 0.25f,
    .mid_min_gain = 0.45f,
    .high_min_gain = 0.35f,
};

// ------------------------------------------------------------
// Utility functions
// ------------------------------------------------------------

static inline float clampf_fast(float x, float lo, float hi) {
    if (x < lo) return lo;
    if (x > hi) return hi;
    return x;
}

static inline float biquad_process(float x, MB_Biquad *s) {
    float y = s->b0 * x
            + s->b1 * s->x1
            + s->b2 * s->x2
            - s->a1 * s->y1
            - s->a2 * s->y2;

    s->x2 = s->x1;
    s->x1 = x;
    s->y2 = s->y1;
    s->y1 = y;

#if DSP_FLUSH_DENORMALS
    if (fabsf(y) < 1.0e-12f) {
        y = 0.0f;
    }
#endif

    return y;
}

static inline float slider_norm(float v) {
    // Convert -5..+5 to -1..+1.
    v = clampf_fast(v, USER_SLIDER_MIN, USER_SLIDER_MAX);
    return v / 5.0f;
}

static inline float db_to_linear(float db) {
    // 20*log10(gain) convention for audio amplitude.
    return powf(10.0f, db / 20.0f);
}

static inline float slider_to_gain_db(float slider, float max_db) {
    // Example:
    // slider = +5 and max_db = 6 gives +6 dB.
    // slider = -5 and max_db = 6 gives -6 dB.
    return slider_norm(slider) * max_db;
}

static inline float align_low_through_3000_split(float low) {
    float low_lp = biquad_process(low, &align_lp_3000_a);
    low_lp = biquad_process(low_lp, &align_lp_3000_b);

    float low_hp = biquad_process(low, &align_hp_3000_a);
    low_hp = biquad_process(low_hp, &align_hp_3000_b);

    return low_lp + low_hp;
}

void init_adc_frontend(void) {
    adc_init();
    adc_gpio_init(ADC_PIN);
    adc_select_input(0);
}

// ------------------------------------------------------------
// DC and input cleanup
// ------------------------------------------------------------

static inline float remove_adc_bias_and_dc(float raw_adc) {
    // Very slow tracker: follows DC/bias drift but not normal speech.
    const float bias_alpha = 0.000045f;
    adc_bias_est += bias_alpha * (raw_adc - adc_bias_est);

    float x = raw_adc - adc_bias_est;

    // First-order DC blocker around 20 Hz.
    const float dc_r = 0.99715f;
    float y = x - dc_x1 + dc_r * dc_y1;

    dc_x1 = x;
    dc_y1 = y;

    return y;
}

static void calibrate_adc_bias(void) {
    const int n = 512;
    uint32_t sum = 0;

    adc_select_input(0);

    for (int i = 0; i < n; i++) {
        sum += adc_read() & 0x0FFFu;
        sleep_us(25);
    }

    adc_bias_est = (float)sum / (float)n;
    dc_x1 = 0.0f;
    dc_y1 = 0.0f;
}

// ------------------------------------------------------------
// Per-band expander + WDRC compressor
// ------------------------------------------------------------

static inline float process_band_dynamics(float x, MB_BandDynamics *d) {
    float ax = fabsf(x);

    // Envelope follower: fast attack, slow release.
    if (ax > d->env) {
        d->env += MB_ATTACK_COEFF * (ax - d->env);
    } else {
        d->env += MB_RELEASE_COEFF * (ax - d->env);
    }

    // Soft expander reduces low-level mic noise without hard muting.
    float expander_gain = 1.0f;

    if (d->env < d->noise_floor) {
        float r = d->env / d->noise_floor;
        r = clampf_fast(r, 0.0f, 1.0f);
        expander_gain = d->min_gain + (1.0f - d->min_gain) * r;
    }

    // WDRC: apply makeup gain, then reduce gain above the threshold.
    float compressor_gain = d->makeup_gain;

    if (d->env > d->threshold) {
        float compressed_env = d->threshold + (d->env - d->threshold) / d->ratio;
        compressor_gain *= compressed_env / d->env;
    }

    float target_gain = expander_gain * compressor_gain;

    // Smooth gain changes to reduce pumping and zipper noise.
    d->gain_smooth += MB_GAIN_SMOOTH_COEFF * (target_gain - d->gain_smooth);

    return x * d->gain_smooth;
}

// ------------------------------------------------------------
// Fast soft limiter
// ------------------------------------------------------------

static inline float soft_limiter(float x) {
    const float limit = 1900.0f; // Start limiting gracefully BEFORE the 2047 ceiling
    const float max_out = 2047.0f;
    
    x = clampf_fast(x, -4096.0f, 4096.0f);
    float ax = fabsf(x);

    if (ax <= limit) return x;

    // Smooth curve between 1600 and max_out
    float excess = ax - limit;
    float y = limit + excess / (1.0f + excess / (max_out - limit)); 

    return (x < 0.0f) ? -y : y;
}

// ============================================================
// Process ADC block -> I2S block
// ============================================================

void process_block(uint16_t *adc_in, uint32_t *i2s_out) {
    for (int i = 0; i < ADC_BUF_LEN; i++) {
        // Raw 12-bit ADC sample.
        float x = (float)(adc_in[i] & 0x0FFFu);
        // Remove microphone bias/DC and low-frequency rumble.
        x = remove_adc_bias_and_dc(x);
        x = biquad_process(x, &input_hpf);

        // LR4 split: each branch uses two identical Butterworth biquads.
        float low = biquad_process(x, &split_lp_500_a);
        low = biquad_process(low, &split_lp_500_b);

        // Align low band with the 3000 Hz split phase/delay used by mid/high.
        low = align_low_through_3000_split(low);

        float rest = biquad_process(x, &split_hp_500_a);
        rest = biquad_process(rest, &split_hp_500_b);

        float mid = biquad_process(rest, &split_lp_3000_a);
        mid = biquad_process(mid, &split_lp_3000_b);

        float high = biquad_process(rest, &split_hp_3000_a);
        high = biquad_process(high, &split_hp_3000_b);

        // Compress/expand each band independently.
        low  = process_band_dynamics(low,  &mb_dyn[0]);
        mid  = process_band_dynamics(mid,  &mb_dyn[1]);
        high = process_band_dynamics(high, &mb_dyn[2]);

        // Recombine bands using user-adjustable tone controls.
        float y = user_rt.low_mix  * low
                + user_rt.mid_mix  * mid
                + user_rt.high_mix * high;

        // Apply user volume after band recombination.
        y *= user_rt.output_gain;

        // Protect the DAC/output path.
        y = soft_limiter(y);
        y = clampf_fast(y, -2048.0f, 2047.0f);

        // Duplicate mono signal to left and right I2S channels.
        int32_t sample32 = ((int32_t)y) << 20;

        i2s_out[2 * i + 0] = (uint32_t)sample32;
        i2s_out[2 * i + 1] = (uint32_t)sample32;

    }
}

static void update_user_tuning(void) {
    float bass_n    = slider_norm(user_bass);
    float middle_n  = slider_norm(user_middle);
    float treble_n  = slider_norm(user_treble);
    float volume_n  = slider_norm(user_volume);
    float noise_n   = slider_norm(user_noise_reduction);
    float clarity_n = slider_norm(user_clarity);

    // Preserve live envelope state if tuning changes while audio is running.
    float old_env[MB_NUM_BANDS];
    float old_gain_smooth[MB_NUM_BANDS];

    for (int i = 0; i < MB_NUM_BANDS; i++) {
        old_env[i] = mb_dyn[i].env;
        old_gain_smooth[i] = mb_dyn[i].gain_smooth;

        mb_dyn[i] = base_dyn[i];

        mb_dyn[i].env = old_env[i];
        mb_dyn[i].gain_smooth = old_gain_smooth[i];
    }

    // ------------------------------------------------------------
    // Convert tone sliders to per-band gain in dB.
    // ------------------------------------------------------------
    //
    // These gains are folded into compressor makeup gain, not applied
    // after compression. Threshold scales inversely to preserve WDRC behavior.
    float bass_db   = bass_n   * 4.0f;
    float middle_db = middle_n * 5.0f;
    float treble_db = treble_n * 5.0f;

    // Clarity/comfort tilt.
    middle_db += clarity_n * 1.5f;
    treble_db += clarity_n * 2.5f;
    bass_db   -= clarity_n * 1.0f;

    float band_gain[MB_NUM_BANDS];

    band_gain[0] = db_to_linear(bass_db);
    band_gain[1] = db_to_linear(middle_db);
    band_gain[2] = db_to_linear(treble_db);

    for (int i = 0; i < MB_NUM_BANDS; i++) {
        mb_dyn[i].makeup_gain = base_dyn[i].makeup_gain * band_gain[i];

        // Inverse threshold scaling:
        // if user boosts a band, compression starts earlier;
        // if user cuts a band, compression starts later.
        mb_dyn[i].threshold = base_dyn[i].threshold / band_gain[i];

        mb_dyn[i].threshold = clampf_fast(mb_dyn[i].threshold, 60.0f, 600.0f);
    }

    // ------------------------------------------------------------
    // Noise reduction slider.
    // ------------------------------------------------------------
    //
    // Positive = stronger noise suppression.
    // Negative = more open for quiet/far speech.
    float nr = noise_n;

    mb_dyn[0].noise_floor = 10.0f + nr * 8.0f;
    mb_dyn[1].noise_floor =  6.0f + nr * 6.0f;
    mb_dyn[2].noise_floor =  5.0f + nr * 5.0f;

    mb_dyn[0].noise_floor = clampf_fast(mb_dyn[0].noise_floor, 2.0f, 24.0f);
    mb_dyn[1].noise_floor = clampf_fast(mb_dyn[1].noise_floor, 1.5f, 18.0f);
    mb_dyn[2].noise_floor = clampf_fast(mb_dyn[2].noise_floor, 1.0f, 16.0f);

    mb_dyn[0].min_gain = clampf_fast(0.25f - nr * 0.10f, 0.08f, 0.60f);
    mb_dyn[1].min_gain = clampf_fast(0.45f - nr * 0.18f, 0.15f, 0.75f);
    mb_dyn[2].min_gain = clampf_fast(0.35f - nr * 0.15f, 0.10f, 0.65f);

    // ------------------------------------------------------------
    // Volume slider remains global.
    // ------------------------------------------------------------
    //
    // Keep this moderate. Large output gain can still drive the limiter.
    float volume_db = volume_n * 6.0f;
    user_rt.output_gain = db_to_linear(volume_db);

    // Recombination trims are no longer user EQ sliders.
    user_rt.low_mix  = 0.85f;
    user_rt.mid_mix  = 1.00f;
    user_rt.high_mix = 0.95f;
}


// ============================================================
// DMA IRQ Handler (IRQ1, leaves USB on IRQ0 alone)
// ============================================================

void __isr dma_irq_handler() {
    // --- ADC channels ---
    if (dma_hw->intr & (1u << adc_dma_a)) {
        dma_hw->intr = 1u << adc_dma_a;
        adc_select = false;             // A just filled
        adc_buffer_ready = true;
        // re-arm A to fire after B completes (chain handles that)
        dma_channel_set_write_addr(adc_dma_a, adcBufA, false);
    }
    if (dma_hw->intr & (1u << adc_dma_b)) {
        dma_hw->intr = 1u << adc_dma_b;
        adc_select = true;              // B just filled
        adc_buffer_ready = true;
        dma_channel_set_write_addr(adc_dma_b, adcBufB, false);
    }

    // --- I2S channels ---
    if (dma_hw->intr & (1u << i2s_dma_a)) {
        dma_hw->intr = 1u << i2s_dma_a;
        i2s_select = false;             // need to refill A next
        i2s_need_fill = true;
        dma_channel_set_read_addr(i2s_dma_a, i2sBufA, false);
    }
    if (dma_hw->intr & (1u << i2s_dma_b)) {
        dma_hw->intr = 1u << i2s_dma_b;
        i2s_select = true;              // need to refill B next
        i2s_need_fill = true;
        dma_channel_set_read_addr(i2s_dma_b, i2sBufB, false);
    }
}

// ============================================================
// ADC + DMA Ping-Pong Setup
// ============================================================

void init_adc_dma() {
    adc_fifo_setup(true, true, 1, false, false);
    adc_set_clkdiv((48000000.0f / SAMPLE_RATE) - 1.0f);

    adc_dma_a = dma_claim_unused_channel(true);
    adc_dma_b = dma_claim_unused_channel(true);

    dma_channel_config cA = dma_channel_get_default_config(adc_dma_a);
    channel_config_set_transfer_data_size(&cA, DMA_SIZE_16);
    channel_config_set_read_increment(&cA, false);
    channel_config_set_write_increment(&cA, true);
    channel_config_set_dreq(&cA, DREQ_ADC);
    channel_config_set_chain_to(&cA, adc_dma_b);

    dma_channel_config cB = dma_channel_get_default_config(adc_dma_b);
    channel_config_set_transfer_data_size(&cB, DMA_SIZE_16);
    channel_config_set_read_increment(&cB, false);
    channel_config_set_write_increment(&cB, true);
    channel_config_set_dreq(&cB, DREQ_ADC);
    channel_config_set_chain_to(&cB, adc_dma_a);

    dma_channel_configure(adc_dma_a, &cA, adcBufA, &adc_hw->fifo, ADC_BUF_LEN, false);
    dma_channel_configure(adc_dma_b, &cB, adcBufB, &adc_hw->fifo, ADC_BUF_LEN, false);

    dma_channel_set_irq1_enabled(adc_dma_a, true);
    dma_channel_set_irq1_enabled(adc_dma_b, true);
}


// ============================================================
// I2S PIO Setup (your original code, unchanged)
// ============================================================

void init_i2s() {
    smMCLK = pio_claim_unused_sm(pio0, true);
    uint offsetMCLK = pio_add_program(pio0, &i2s_master_clock_program);
    pio_sm_config cMCLK = i2s_master_clock_program_get_default_config(offsetMCLK);
    sm_config_set_sideset_pins(&cMCLK, MCLK_PIN);
    sm_config_set_clkdiv_int_frac(&cMCLK, 3, 82);
    pio_gpio_init(pio0, MCLK_PIN);
    pio_sm_set_consecutive_pindirs(pio0, smMCLK, MCLK_PIN, 1, true);
    pio_sm_init(pio0, smMCLK, offsetMCLK, &cMCLK);

    smCLK = pio_claim_unused_sm(pio0, true);
    uint offsetCLK = pio_add_program(pio0, &i2s_master_output_program);
    pio_gpio_init(pio0, OUT_BCK_PIN);
    pio_gpio_init(pio0, OUT_LRCK_PIN);
    pio_gpio_init(pio0, DOUT_PIN);
    pio_sm_config cCLK = i2s_master_output_program_get_default_config(offsetCLK);
    sm_config_set_sideset_pins(&cCLK, OUT_BCK_PIN);
    sm_config_set_out_pins(&cCLK, DOUT_PIN, 1);
    sm_config_set_out_shift(&cCLK, false, true, 32);
    sm_config_set_fifo_join(&cCLK, PIO_FIFO_JOIN_TX);
    sm_config_set_clkdiv_int_frac(&cCLK, 26, 147);
    pio_sm_init(pio0, smCLK, offsetCLK, &cCLK);

    pio_sm_set_consecutive_pindirs(pio0, smCLK, OUT_BCK_PIN, 2, true);
    pio_sm_set_consecutive_pindirs(pio0, smCLK, DOUT_PIN, 1, true);
    pio_sm_set_pins(pio0, smCLK, 0);
    pio_sm_clear_fifos(pio0, smCLK);

    pio_sm_exec(pio0, smCLK, pio_encode_jmp(offsetCLK + i2s_master_output_offset_entry_point));

    pio_enable_sm_mask_in_sync(pio0, (1u << smMCLK) | (1u << smCLK));
}

// ============================================================
// I2S DMA Ping-Pong Setup
// ============================================================

void init_i2s_dma() {
    i2s_dma_a = dma_claim_unused_channel(true);
    i2s_dma_b = dma_claim_unused_channel(true);

    dma_channel_config cA = dma_channel_get_default_config(i2s_dma_a);
    channel_config_set_transfer_data_size(&cA, DMA_SIZE_32);
    channel_config_set_read_increment(&cA, true);
    channel_config_set_write_increment(&cA, false);
    channel_config_set_dreq(&cA, pio_get_dreq(pio0, smCLK, true));
    channel_config_set_chain_to(&cA, i2s_dma_b);

    dma_channel_config cB = dma_channel_get_default_config(i2s_dma_b);
    channel_config_set_transfer_data_size(&cB, DMA_SIZE_32);
    channel_config_set_read_increment(&cB, true);
    channel_config_set_write_increment(&cB, false);
    channel_config_set_dreq(&cB, pio_get_dreq(pio0, smCLK, true));
    channel_config_set_chain_to(&cB, i2s_dma_a);

    dma_channel_configure(i2s_dma_a, &cA, &pio0->txf[smCLK], i2sBufA, BUF_LEN, false);
    dma_channel_configure(i2s_dma_b, &cB, &pio0->txf[smCLK], i2sBufB, BUF_LEN, false);

    dma_channel_set_irq1_enabled(i2s_dma_a, true);
    dma_channel_set_irq1_enabled(i2s_dma_b, true);

    irq_set_exclusive_handler(DMA_IRQ_1, dma_irq_handler);
    irq_set_enabled(DMA_IRQ_1, true);
}

// ============================================================
// Main
// ============================================================

int main() {
    stdio_init_all();

    // Pre-fill I2S buffers with silence.
    for (int i = 0; i < BUF_LEN; i++) {
        i2sBufA[i] = 0;
        i2sBufB[i] = 0;
    }

    init_i2s();

    // ADC hardware only. Do this before adc_read() calibration.
    init_adc_frontend();

    // Calibrate before FIFO/DMA setup to avoid FIFO/DMA state corruption.
    calibrate_adc_bias();

    // Now configure DMA/FIFO paths.
    init_adc_dma();
    init_i2s_dma();

    // Apply static user tuning before audio starts.
    update_user_tuning();

    // Start everything.
    adc_run(true);
    dma_channel_start(adc_dma_a);
    dma_channel_start(i2s_dma_a);

    while (true) {
        if (adc_buffer_ready && i2s_need_fill) {
            bool adc_sel;
            bool i2s_sel;

            // Capture buffer flags atomically, then process with interrupts enabled.
            uint32_t save = save_and_disable_interrupts();

            adc_sel = adc_select;
            i2s_sel = i2s_select;

            adc_buffer_ready = false;
            i2s_need_fill = false;

            restore_interrupts(save);

            uint16_t *src = adc_sel ? adcBufB : adcBufA;
            uint32_t *dst = i2s_sel ? i2sBufB : i2sBufA;

            process_block(src, dst);
        }

        tight_loop_contents();
    }
}
