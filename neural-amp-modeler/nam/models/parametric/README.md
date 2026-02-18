# Parametric Amp Profiling — Tester Guide

## What Is This?

This is a new way to capture guitar amplifiers with NAM that gives you **tunable controls after capture**. Instead of getting a frozen snapshot of one amp setting, you get a digital amp with 8 working knobs:

- **Preamp Gain** — Drive/saturation amount
- **Bass** — Low frequency EQ
- **Mid** — Midrange EQ
- **Treble** — High frequency EQ
- **Sag** — Power amp compression (how much the amp "breathes")
- **Presence** — High-frequency clarity/cut
- **Depth** — Low-frequency body/thump
- **Master Volume** — Output level

Think of it like a Kemper profile — you capture your amp once, and the model learns a tunable version of it.

### How It Works (Simple Version)

The model is built as a chain of four stages that mirror a real amp:

```
Guitar In -> [Preamp] -> [Tone Stack] -> [Power Amp] -> [Output] -> Speaker Out
```

Each stage uses real DSP building blocks (filters, nonlinearities) that are "learned" from your amp's captured behavior. Because the internal structure matches a real amp, the knobs behave in a physically plausible way even though they were learned from a single capture.

---

## What You Need

- Python 3.9+ with the NAM training environment set up
- An audio interface (for reamping)
- Your guitar amplifier
- A DAW (Reaper, Logic, Ableton, etc.)

---

## Step-by-Step Testing Instructions

### Step 1: Generate the Profiling Signal

Open a terminal in the `neural-amp-modeler` directory and run:

```
python -m nam.signals --output profiling_signal.wav
```

This creates a ~2.5 minute WAV file (48kHz, 24-bit, mono) containing a carefully designed test signal with multiple phases:

1. **Calibration blips** — for latency alignment
2. **Rising noise** — sweeps from silence to full volume
3. **Pulsating noise** — bursts at different levels
4. **Swept sines** — frequency sweeps at increasing drive
5. **Musical content** — simulated guitar playing dynamics

You only need to generate this file once. The same file is used for all captures.

### Step 2: Set Up Your Amp

1. Set your amp to the tone you want to capture — this becomes the "home base" sound. All knobs at noon/5 is a good starting point.
2. **Do not change any knobs during the capture.** The signal itself tests the amp across its full range.
3. Connect your audio interface output to the amp's input (or use a reamp box).
4. Mic the amp cabinet as you normally would, or use the amp's direct out.

### Step 3: Reamp the Signal

1. Import `profiling_signal.wav` into your DAW on a track.
2. Route that track's output to your audio interface output going to the amp.
3. Record the amp's output (mic or direct) on a second track.
4. Make sure both tracks start at the exact same point — alignment matters.
5. Export the recorded output as a WAV file (48kHz, 24-bit, mono). Call it something like `my_amp_output.wav`.

**Important:** The input and output files must be the same length and sample rate.

### Step 4: Train the Model

```
python -c "
from nam.train.core import train

train(
    input_path='profiling_signal.wav',
    output_path='my_amp_output.wav',
    train_path='./output',
    model_type='ParametricAmp',
    architecture='parametric',
    epochs=100,
    lr=0.004,
    lr_decay=0.007,
    batch_size=16,
)
"
```

Or if you prefer a script, save the above as `train_parametric.py` and run it.

Training should take roughly 15-30 minutes on a GPU, longer on CPU.

The trained model will be saved in the `./output` directory as a `.nam` file.

### Step 5: Test the Model

You can load and run the model in Python to verify it works:

```python
from nam.models.parametric import ParametricAmpInference
import numpy as np

# Load the trained model
inference = ParametricAmpInference.from_nam_file("./output/model.nam")

# Print available knobs
for i, name in enumerate(inference.knob_names):
    print(f"  Knob {i}: {name}")

# Create a test signal (1 second of 220Hz sine)
sr = 48000
t = np.linspace(0, 1, sr)
test_input = (0.5 * np.sin(2 * np.pi * 220 * t)).astype(np.float32)

# Process at default settings (all knobs at 0.5)
output_default = inference.process(test_input)

# Try cranking the gain
inference.set_knob(0, 0.9)  # preamp_gain to 90%
output_high_gain = inference.process(test_input)

# Try scooping the mids
inference.set_knobs(np.array([0.5, 0.7, 0.2, 0.7, 0.5, 0.5, 0.5, 0.5]))
output_scooped = inference.process(test_input)

print(f"Default RMS:    {np.sqrt(np.mean(output_default**2)):.4f}")
print(f"High gain RMS:  {np.sqrt(np.mean(output_high_gain**2)):.4f}")
print(f"Scooped RMS:    {np.sqrt(np.mean(output_scooped**2)):.4f}")
```

---

## What to Look For When Testing

### Does It Sound Like the Amp?

At default knob settings (all at 0.5), the model should reproduce the captured amp tone. Compare the model output against the original recording — it should be close.

### Do the Knobs Do Something?

Each knob should produce an audible change:

| Knob | What to listen for |
|------|--------------------|
| Preamp Gain | More gain = more distortion/saturation, less = cleaner |
| Bass | More = boomier low end, less = tighter |
| Mid | More = honky/full, less = scooped |
| Treble | More = brighter/harsher, less = darker/warmer |
| Sag | More = spongier/compressed feel, less = tighter response |
| Presence | More = more bite on pick attack, less = smoother |
| Depth | More = fuller low-end body, less = thinner |
| Master Volume | Louder/quieter (should not add distortion) |

### Edge Cases to Try

- All knobs at 0 (minimum everything)
- All knobs at 1 (maximum everything)
- Extreme gain (0.0 and 1.0) with moderate EQ
- Rapid knob changes while audio is playing (should not click or glitch)
- Very quiet input signals
- Very loud input signals

---

## Reporting Issues

When reporting a problem, please include:

1. **What amp** you captured (make/model, settings used)
2. **What you expected** vs **what you heard**
3. **Which knob(s)** were involved and at what values
4. **Training output** — any errors or warnings from the training run
5. **The .nam file** if possible (so we can reproduce)

Common things that might not work perfectly yet:

- **Knob range feels too narrow** — the knobs might not produce dramatic enough changes. This is expected in early iterations; the knob-to-parameter mapping is learned from physical priors and may need tuning.
- **Some knobs seem to do nothing** — if the model hasn't learned a meaningful mapping for a particular knob, it may have minimal effect. Note which ones.
- **Audio artifacts at extreme settings** — pushing all knobs to min/max simultaneously may produce unrealistic sounds. Stay within reasonable ranges for initial testing.
- **Training doesn't converge** — if loss doesn't decrease, note the final loss value and number of epochs.

---

## Knob Value Reference

All knobs use a 0.0 to 1.0 range internally. When displayed to users in a plugin, these map to 0-10:

| Internal | Display | Meaning |
|----------|---------|---------|
| 0.0 | 0 | Minimum |
| 0.5 | 5 | Nominal (matches captured tone) |
| 1.0 | 10 | Maximum |

---

## File Reference

| File | What it is |
|------|-----------|
| `profiling_signal.wav` | The test signal you play through your amp |
| `my_amp_output.wav` | Your amp's recorded response to the test signal |
| `./output/model.nam` | The trained parametric model |
