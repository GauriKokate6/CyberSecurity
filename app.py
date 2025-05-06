import streamlit as st
import os
from tortoise.models.classifier import AudioMiniEncoderWithClassifierHead
import io
import librosa
import plotly.express as px
import torch
import torchaudio
import numpy as np
import soundfile as sf
import hashlib
import json
from datetime import datetime
import base64

HASH_DB_PATH = "audio_hash_registry.json"

def load_hash_registry():
    if not os.path.exists(HASH_DB_PATH):
        return []
    with open(HASH_DB_PATH, "r") as f:
        return json.load(f)

def save_hash_to_registry(audio_hash, result_prob, threat_level, file_name, audio_bytes):
    registry = load_hash_registry()
    entry = {
        "hash": audio_hash,
        "timestamp": datetime.now().isoformat(),
        "probability": result_prob,
        "threat_level": threat_level,
        "file_name": file_name,
        "audio_b64": base64.b64encode(audio_bytes).decode("utf-8")
    }
    registry.append(entry)
    with open(HASH_DB_PATH, "w") as f:
        json.dump(registry, f, indent=4)

def check_hash_in_registry(audio_hash):
    registry = load_hash_registry()
    for entry in registry:
        if entry["hash"] == audio_hash:
            return entry
    return None

def hash_similarity_score(hash1, hash2):
    bin1 = bin(int(hash1, 16))[2:].zfill(256)
    bin2 = bin(int(hash2, 16))[2:].zfill(256)
    return 1 - sum(c1 != c2 for c1, c2 in zip(bin1, bin2)) / 256

def get_mel_spectrogram(audio, sample_rate=22000, n_mels=1, f_max=8000):
    mel_spec_transform = torchaudio.transforms.MelSpectrogram(
        sample_rate=sample_rate,
        n_mels=n_mels,
        f_max=f_max
    )
    return mel_spec_transform(audio)

def generate_audio_hash(file_bytes):
    hash_obj = hashlib.sha256()
    hash_obj.update(file_bytes)
    return hash_obj.hexdigest()

def load_audio(audiopath, sampling_rate=22000):
    if isinstance(audiopath, str):
        if audiopath.endswith('.mp3'):
            audio, lsr = librosa.load(audiopath, sr=sampling_rate)
            audio = torch.FloatTensor(audio)
        else:
            raise ValueError(f"Unsupported audio format: {audiopath[-4:]}")
    elif isinstance(audiopath, io.BytesIO):
        audio_data, samplerate = sf.read(audiopath)
        if len(audio_data.shape) > 1:
            audio_data = np.mean(audio_data, axis=1)
        audio = torch.FloatTensor(audio_data)
        lsr = samplerate
    else:
        raise ValueError("Invalid audio path or file type.")

    if lsr != sampling_rate:
        audio = torchaudio.functional.resample(audio, lsr, sampling_rate)

    audio.clip_(-1, 1)
    return audio.unsqueeze(0)

def classify_audio_clip(clip):
    spec = get_mel_spectrogram(clip)
    if len(spec.shape) == 4:
        spec = spec.squeeze(0)
    if spec.ndim == 2:
        spec = spec.unsqueeze(0)
    elif spec.ndim != 3:
        raise ValueError(f"Unexpected spectrogram shape: {spec.shape}")

    classifier = AudioMiniEncoderWithClassifierHead(
        2,
        spec_dim=1,
        embedding_dim=512,
        depth=5,
        downsample_factor=4,
        resnet_blocks=2,
        attn_blocks=4,
        num_attn_heads=4,
        base_channels=32,
        dropout=0,
        kernel_size=5,
        distribute_zero_label=False
    )

    state_dict = torch.load('classifier.pth', map_location=torch.device('cpu'))
    classifier.load_state_dict(state_dict)

    results = torch.nn.functional.softmax(classifier(spec), dim=-1)
    return results[0][0]

def calculate_threat_level(ai_prob, duration, sample_rate):
    threat_score = 0
    if ai_prob > 0.9:
        threat_score += 3
    elif ai_prob > 0.75:
        threat_score += 2
    elif ai_prob > 0.5:
        threat_score += 1
    if duration < 1.5 or duration > 20:
        threat_score += 1

    if threat_score <= 1:
        return "Low"
    elif threat_score == 2:
        return "Moderate"
    elif threat_score == 3:
        return "High"
    else:
        return "Critical"

st.set_page_config(layout="wide")

def main():
    st.title("AI-Generated Voice Detection")
    uploaded_file = st.file_uploader("Upload an audio file", type=["mp3"])

    if uploaded_file is not None:
        uploaded_bytes = uploaded_file.read()
        audio_hash = generate_audio_hash(uploaded_bytes)
        uploaded_file.seek(0)

        existing_entry = check_hash_in_registry(audio_hash)

        if existing_entry:
            st.warning("⚠ This audio hash already exists in the registry.")
            st.info(f"First seen on: {existing_entry['timestamp']}")
            st.info(f"Previous classification: {existing_entry['probability']*100:.2f}% AI, Threat Level: {existing_entry['threat_level']}")
        else:
            st.success("✅ This is a new audio hash (not seen before).")

        if st.button("Analyze Audio"):
            col1, col2, col3, col4 = st.columns(4)

            with col1:
                st.info("Your results are below")
                audio_clip = load_audio(uploaded_file)
                result = classify_audio_clip(audio_clip).item()

                duration = len(audio_clip.squeeze()) / 22000
                threat_level = calculate_threat_level(result, duration, 22000)
                save_hash_to_registry(audio_hash, result, threat_level, uploaded_file.name, uploaded_bytes)

                st.info(f"Result Probability: {result}")
                st.success(f"The uploaded audio is {result * 100:.2f}% likely to be AI Generated.")

                st.subheader("Threat Assessment")
                st.success("Threat Level: " + threat_level)

            with col2:
                st.info("Your uploaded audio is below")
                st.audio(uploaded_file)
                fig = px.line()
                fig.add_scatter(x=list(range(len(audio_clip.squeeze()))), y=audio_clip.squeeze())
                fig.update_layout(title="Waveform Plot", xaxis_title="Time", yaxis_title="Amplitude")
                st.plotly_chart(fig, use_container_width=True)

            with col3:
                st.warning("Disclaimer: These classifications are not always accurate.")

            with col4:
                st.info("🧾 Forensic Audio Hash")
                st.code(audio_hash, language="bash")

            registry = load_hash_registry()
            others = [e for e in registry if e["hash"] != audio_hash and "audio_b64" in e]

            if others:
                similarities = [
                    (
                        entry["hash"],
                        hash_similarity_score(audio_hash, entry["hash"]),
                        entry.get("file_name", "Unknown"),
                        entry.get("audio_b64")
                    )
                    for entry in others
                ]

                similarities.sort(key=lambda x: -x[1])
                best_match = similarities[0]

                st.info(f"Most similar match: {best_match[0][:16]}... ({best_match[1]*100:.2f}% similarity)")
                st.caption(f"Similar to file: {best_match[2]}")

                st.audio(base64.b64decode(best_match[3]), format="audio/mp3")

            else:
                st.info("No similar hashes found in the registry.")

            st.download_button("📥 Download Hash", audio_hash, file_name="audio_hash.txt")

if __name__ == "_main_":
    main()