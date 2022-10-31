## A Unified One-Shot Prosody and Speaker Conversion System with Self-Supervised Discrete Speech Units

 - We present a unified system to realize one-shot voice conversion (VC) on the pitch, rhythm, and speaker attributes. Existing works generally ignore the correlation between prosody and language content, leading to the degradation of naturalness in converted speech. Additionally, the lack of proper language features prevents these systems from accurately preserving language content after conversion. To address these issues, we devise a cascaded modular system leveraging self-supervised discrete speech units as language representation. These discrete units provide duration information essential for rhythm modeling. Our system first extracts utterance-level prosody and speaker representations from the raw waveform. Given the prosody representation, a prosody predictor estimates pitch, energy, and duration for each discrete unit in the utterance. A synthesizer further reconstructs speech based on the predicted prosody, speaker representation, and discrete units. Experiments show that our system outperforms previous approaches in naturalness, intelligibility, speaker transferability, and prosody transferability.
 - Submitted to ICASSP 2023
 - [Code](https://github.com/b04901014/UUVC)
 - [Paper](...)

### Speaker Trasnfer Samples
|Source Speech|Target Speech|AutoVC|SRDVC|Ours (VCTK)|Ours (LibriTTS+VCTK+ESD)|

### Prosody (pitch-energy + rhythm) Transfer Samples
|Source Speech|Target Speech|SRDVC|Ours (VCTK)|Ours (LibriTTS+VCTK+ESD)|

### Rhythm Transfer Samples
|Source Speech|Target Speech|SRDVC|Ours (VCTK)|Ours (LibriTTS+VCTK+ESD)|

### Observations
 - TODO
