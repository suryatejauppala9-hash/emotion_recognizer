# Multimodal Emotion Recognizer

This project is a speech and text multimodal emotion recognition pipeline trained on the Ryerson Audio-Visual Database of Emotional Speech and Song (RAVDESS) dataset.

It uses a dual-input architecture combining:
1. An Audio CNN branch processing Mel-Spectrograms (extracted using Librosa with data augmentation and per-sample normalization).
2. A Text Bidirectional GRU branch processing transcriptions generated via OpenAI's Whisper model.

Bottleneck features from both branches are combined and classified into one of eight emotional states (neutral, calm, happy, sad, angry, fearful, disgust, surprised) using early fusion (two-phase training with branch freezing) or late fusion (averaging, weighted averaging, or max confidence).
