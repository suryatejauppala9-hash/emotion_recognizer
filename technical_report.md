# Multimodal Emotion Recognition - Technical Report

**Dataset:** RAVDESS (1,440 clips, 24 actors, 8 emotions)
**Stack:** TensorFlow/Keras, Librosa, Whisper

---

## What I built

Three models: audio-only CNN, text-only RNN, and a combined fusion model. The audio listens to how something is said, the text reads what was said, and the fusion model uses both.

---

## Audio Branch (CNN)

Load audio with librosa, trim/pad to 3 seconds, convert to a 128-bin Mel-Spectrogram, normalize. I also augmented with time stretching, pitch shifting, and random noise to double the dataset size.

The CNN has 3 blocks of paired Conv layers + BatchNorm + MaxPool + Dropout, then a dual GlobalAvgPool + GlobalMaxPool concatenation, then two Dense layers. The 128-dim output of the second Dense is the bottleneck feature vector.

Mel-spectrograms work better than regular spectrograms here because human hearing is logarithmic and mel-scale mirrors that. The dual pooling trick (avg + max together) consistently beats either one alone.

---

## Text Branch (Bi-GRU RNN)

Whisper transcribes the clips, then I tokenize (vocab=5000), pad to length 40, and feed into 3 stacked Bidirectional GRU layers followed by two Dense layers. The 64-dim output is the bottleneck.

The text model is weak on this dataset. Every actor says the same scripted sentences, so the words carry almost zero emotional signal. The audio branch does most of the work.

---

## Fusion (Early Fusion)

Take the 128-dim audio vector and 64-dim text vector, concatenate into 192-dim, pass through Dense(256) -> Dense(128) -> Dense(8, softmax).

We transferred weights from the pretrained unimodal models, then trained in two phases: first freeze the branches and only train the fusion head, then unfreeze everything and fine-tune at a lower learning rate. This stops the new head from trashing the pretrained features on the first pass.

I also tried late fusion (averaging softmax outputs, weighted 0.7/0.3 audio/text, and max-confidence) as a comparison.

---

## Architecture

```
Audio (.wav) -> Mel-Spectrogram -> CNN -> 128-dim -+
                                                    +--> Concat -> Dense(256) -> Dense(8)
Audio (.wav) -> Whisper -> Tokenize -> BiGRU -> 64-dim -+
```

---

## Results

| Model | Accuracy |
|---|---|
| Audio CNN | ~0.62-0.67 |
| Text RNN | ~0.18-0.25 |
| Early Fusion | ~0.65-0.70 |
| Late Fusion (weighted) | close to early fusion |

Audio does the heavy lifting. Text barely helps on its own but adds a small boost in fusion. Early fusion edges out audio-only, weighted late fusion (0.7/0.3) also works and is simpler.

---

## What gets confused

Calm vs neutral, fearful vs sad, happy vs surprised. Angry and disgust are usually fine since they sound distinctive. This is a dataset problem more than a model problem.

---

## Tools

`librosa`, `openai-whisper`, `TensorFlow/Keras`, `scikit-learn`, `matplotlib`, `seaborn`
