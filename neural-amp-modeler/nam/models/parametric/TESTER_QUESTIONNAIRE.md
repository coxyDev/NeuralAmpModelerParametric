# Parametric Amp Profiling — Tester Feedback Questionnaire

**Tester Name:** _______________
**Date:** _______________
**Build/Version:** _______________

---

## Section 1: Your Setup

**1.1** What amplifier did you capture?
- Make: _______________
- Model: _______________
- Year (if known): _______________

**1.2** What were the amp's knob settings during capture?
| Amp Knob | Setting (clock position or 0-10) |
|----------|----------------------------------|
| Gain/Drive | |
| Bass | |
| Mid | |
| Treble | |
| Presence | |
| Master/Volume | |
| Other: _________ | |

**1.3** Was the amp captured with:
- [ ] Microphone on cabinet (which mic? _______________)
- [ ] Direct out / load box (which? _______________)
- [ ] Other: _______________

**1.4** What audio interface did you use? _______________

**1.5** What DAW did you use for reamping? _______________

**1.6** What sample rate and bit depth was your recording? _______________

---

## Section 2: Signal & Capture Process

**2.1** Did the profiling signal (`profiling_signal.wav`) play back and record without issues?
- [ ] Yes, no problems
- [ ] Minor issues (describe below)
- [ ] Major issues / could not complete

Comments: _______________

**2.2** Was the signal level appropriate for your amp?
- [ ] Too quiet — amp wasn't driven enough even at loud phases
- [ ] About right — clean phases were clean, loud phases pushed the amp
- [ ] Too hot — clipping the input of the amp even on quiet phases

**2.3** Did the total signal length feel:
- [ ] Too short — would have liked more capture time
- [ ] About right
- [ ] Too long — could achieve the same with less

**2.4** Were there any alignment or latency issues when setting up in your DAW?
- [ ] No issues
- [ ] Yes (describe): _______________

**2.5** How long did the full capture process take you (from opening the DAW to having the output WAV exported)?
- [ ] Under 10 minutes
- [ ] 10-20 minutes
- [ ] 20-30 minutes
- [ ] Over 30 minutes

---

## Section 3: Training

**3.1** What hardware did you train on?
- [ ] NVIDIA GPU (which? _______________)
- [ ] Apple Silicon (M1/M2/M3/M4? _______________)
- [ ] CPU only

**3.2** How long did training take? _______________ minutes

**3.3** Did training complete without errors?
- [ ] Yes
- [ ] No — error message: _______________

**3.4** If you monitored the training loss, did it:
- [ ] Decrease steadily and converge
- [ ] Decrease then plateau early
- [ ] Fluctuate / not converge
- [ ] I didn't monitor it

**3.5** Final loss value (if available): _______________

---

## Section 4: Sound Quality at Default Settings

*Set all knobs to 0.5 (nominal). Compare the model's output to your original amp recording.*

**4.1** How close does the model sound to your amp at default knob settings?
- [ ] 1 — Not recognisable as the same amp
- [ ] 2 — Same general character but clearly different
- [ ] 3 — Close but with noticeable differences
- [ ] 4 — Very close, minor differences on careful listening
- [ ] 5 — Indistinguishable or near-indistinguishable

**4.2** If there are differences at default settings, what do you notice?
- [ ] Overall tone/EQ is off
- [ ] Gain/distortion level is wrong
- [ ] Dynamics/touch sensitivity feel different
- [ ] Strange artifacts (buzzing, aliasing, clicking)
- [ ] Low end is different
- [ ] High end is different
- [ ] Compression/sag feel is different
- [ ] Other: _______________

**4.3** Describe the differences in your own words:

_______________

---

## Section 5: Knob Behaviour

*For each knob, sweep it from 0.0 to 1.0 while playing audio and rate the result.*

### Preamp Gain (Knob 0)

**5.1** Does turning up preamp gain add distortion/saturation?
- [ ] Yes, clearly
- [ ] Somewhat — subtle effect
- [ ] No noticeable change
- [ ] Yes but it sounds wrong (describe): _______________

**5.2** Does turning down preamp gain clean up the tone?
- [ ] Yes, clearly
- [ ] Somewhat
- [ ] No noticeable change

**5.3** Rate the preamp gain knob's range (how much does it change the sound?):
- [ ] 1 — Does almost nothing
- [ ] 2 — Too subtle
- [ ] 3 — About right
- [ ] 4 — Too extreme
- [ ] 5 — Completely breaks the sound at extremes

### Bass (Knob 1)

**5.4** Does the bass knob audibly change the low frequencies?
- [ ] Yes, clearly
- [ ] Somewhat
- [ ] No noticeable change

**5.5** Rate the bass knob's range:
- [ ] 1 — Does almost nothing
- [ ] 2 — Too subtle
- [ ] 3 — About right
- [ ] 4 — Too extreme
- [ ] 5 — Completely breaks the sound

### Mid (Knob 2)

**5.6** Does the mid knob audibly change the midrange?
- [ ] Yes, clearly
- [ ] Somewhat
- [ ] No noticeable change

**5.7** Rate the mid knob's range:
- [ ] 1 — Does almost nothing
- [ ] 2 — Too subtle
- [ ] 3 — About right
- [ ] 4 — Too extreme
- [ ] 5 — Completely breaks the sound

### Treble (Knob 3)

**5.8** Does the treble knob audibly change the high frequencies?
- [ ] Yes, clearly
- [ ] Somewhat
- [ ] No noticeable change

**5.9** Rate the treble knob's range:
- [ ] 1 — Does almost nothing
- [ ] 2 — Too subtle
- [ ] 3 — About right
- [ ] 4 — Too extreme
- [ ] 5 — Completely breaks the sound

### Sag (Knob 4)

**5.10** Does turning up sag make the amp feel more compressed/spongy?
- [ ] Yes, clearly
- [ ] Somewhat
- [ ] No noticeable change
- [ ] Yes but it sounds wrong (describe): _______________

**5.11** Rate the sag knob's range:
- [ ] 1 — Does almost nothing
- [ ] 2 — Too subtle
- [ ] 3 — About right
- [ ] 4 — Too extreme
- [ ] 5 — Completely breaks the sound

### Presence (Knob 5)

**5.12** Does the presence knob affect the upper-mid bite and clarity?
- [ ] Yes, clearly
- [ ] Somewhat
- [ ] No noticeable change

**5.13** Rate the presence knob's range:
- [ ] 1 — Does almost nothing
- [ ] 2 — Too subtle
- [ ] 3 — About right
- [ ] 4 — Too extreme
- [ ] 5 — Completely breaks the sound

### Depth (Knob 6)

**5.14** Does the depth knob affect the low-end body and fullness?
- [ ] Yes, clearly
- [ ] Somewhat
- [ ] No noticeable change

**5.15** Rate the depth knob's range:
- [ ] 1 — Does almost nothing
- [ ] 2 — Too subtle
- [ ] 3 — About right
- [ ] 4 — Too extreme
- [ ] 5 — Completely breaks the sound

### Master Volume (Knob 7)

**5.16** Does master volume change the level without altering the tone character?
- [ ] Yes — clean volume change
- [ ] Changes volume but also changes tone
- [ ] No noticeable change

**5.17** Rate the master volume knob's range:
- [ ] 1 — Does almost nothing
- [ ] 2 — Too subtle
- [ ] 3 — About right
- [ ] 4 — Too extreme
- [ ] 5 — Completely breaks the sound

---

## Section 6: Knob Interactions

**6.1** When you adjust multiple EQ knobs together (bass + mid + treble), do the results sound musically useful?
- [ ] Yes — sounds like a real tone stack
- [ ] Somewhat — works but doesn't feel interactive like a real amp
- [ ] No — knobs seem independent with no interaction
- [ ] No — strange artefacts when combining

**6.2** Does adjusting preamp gain interact naturally with the EQ? (On a real amp, more gain changes how the EQ sounds.)
- [ ] Yes, feels interactive
- [ ] No, they seem completely independent
- [ ] Didn't test this

**6.3** Did you find any knob combinations that produce obvious artefacts or broken sounds?

Settings that caused issues: _______________

What happened: _______________

---

## Section 7: Dynamics & Feel

**7.1** Does the model respond to pick dynamics (playing soft vs hard)?
- [ ] Yes — cleans up when playing soft, distorts when digging in
- [ ] Somewhat
- [ ] No — same distortion level regardless of playing dynamics

**7.2** How does the dynamic response compare to your real amp?
- [ ] 1 — Nothing like my amp
- [ ] 2 — Same ballpark but clearly different
- [ ] 3 — Reasonably close
- [ ] 4 — Very close
- [ ] 5 — Identical feel

**7.3** Does the sag/compression feel right for palm mutes and chugging?
- [ ] Yes
- [ ] Somewhat
- [ ] No
- [ ] Not applicable (clean amp capture)

---

## Section 8: Artefacts & Stability

**8.1** Did you hear any of the following?
- [ ] Clicking or popping
- [ ] Buzzing or digital noise
- [ ] Aliasing (metallic / unnatural harmonics)
- [ ] DC offset (waveform drifts away from centre)
- [ ] Volume spikes or unexpected loud noises
- [ ] Silence where there should be sound
- [ ] None of the above

**8.2** Did rapid knob changes cause any glitches?
- [ ] No — smooth transitions
- [ ] Minor glitches (clicks)
- [ ] Major glitches (loud pops, silence, distortion)
- [ ] Didn't test this

**8.3** Did the model remain stable over a long playing session (10+ minutes)?
- [ ] Yes
- [ ] No — degraded over time (describe): _______________
- [ ] Didn't test this long

---

## Section 9: Comparison to Standard NAM

*If you've used standard (non-parametric) NAM captures before:*

**9.1** How does the parametric model's sound quality compare to a standard NAM capture of the same amp?
- [ ] Worse
- [ ] About the same
- [ ] Better
- [ ] Haven't done a standard capture to compare

**9.2** Is the parametric model useful enough to justify the more complex capture process?
- [ ] Yes — the tunable knobs are worth it
- [ ] Maybe — knobs need to work better first
- [ ] No — I'd rather have a more accurate snapshot

---

## Section 10: Overall

**10.1** Overall, how would you rate this first iteration?
- [ ] 1 — Not usable
- [ ] 2 — Interesting concept but needs significant work
- [ ] 3 — Promising, some things work well
- [ ] 4 — Good, would use with improvements
- [ ] 5 — Excellent for a first version

**10.2** Which knob was the MOST useful/effective?

_______________

**10.3** Which knob was the LEAST useful/effective?

_______________

**10.4** What single improvement would make the biggest difference?

_______________

**10.5** Any other comments, observations, or suggestions?

_______________

---

## Attachments Checklist

Please include the following with your questionnaire if possible:

- [ ] The `.nam` model file
- [ ] A short audio clip: real amp vs model at default settings
- [ ] A short audio clip demonstrating a knob that works well
- [ ] A short audio clip demonstrating a knob that doesn't work well
- [ ] Screenshot of training loss curve (if available)
- [ ] Any error messages or console output from training

**Thank you for testing!**
