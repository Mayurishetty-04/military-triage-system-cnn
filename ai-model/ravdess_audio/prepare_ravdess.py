import os
import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np

RAW_DIR = "raw"
OUT_DIR = "dataset"

TRIAGE_MAP = {
    "01": "green",   # neutral
    "02": "green",   # calm
    "03": "green",   # happy
    "04": "yellow",  # sad
    "05": "red",     # angry
    "06": "red",     # fearful
    "07": "yellow",  # disgust
    "08": "yellow",  # surprised
}

TRIAGE_CLASSES = ["green", "yellow", "red", "black"]

# Create output folders if not exist
for label in TRIAGE_CLASSES:
    os.makedirs(os.path.join(OUT_DIR, label), exist_ok=True)

def audio_to_spectrogram(audio_path, out_path):
    y, sr = librosa.load(audio_path, sr=None)
    # optional: trim silence
    y, _ = librosa.effects.trim(y)

    # create mel-spectrogram
    S = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128)
    S_dB = librosa.power_to_db(S, ref=np.max)

    plt.figure(figsize=(2, 2))
    plt.axis("off")
    librosa.display.specshow(S_dB, sr=sr, x_axis=None, y_axis=None)
    plt.tight_layout(pad=0)
    plt.savefig(out_path, bbox_inches="tight", pad_inches=0)
    plt.close()


def process_ravdess():
    count = {"green": 0, "yellow": 0, "red": 0}

    for root, dirs, files in os.walk(RAW_DIR):
        for file in files:
            if not file.lower().endswith(".wav"):
                continue

            parts = file.split("-")
            if len(parts) < 3:
                continue

            emotion_code = parts[2]
            triage_label = TRIAGE_MAP.get(emotion_code)
            if triage_label is None:
                continue

            src_path = os.path.join(root, file)
            out_filename = f"{triage_label}_{count[triage_label]}.png"
            out_path = os.path.join(OUT_DIR, triage_label, out_filename)

            print(f"Processing {src_path} -> {out_path}")
            audio_to_spectrogram(src_path, out_path)
            count[triage_label] += 1

    print("Processed counts:", count)


def create_black_class(num_samples=200, duration=2.0, sr=22050):
    """Create silent audio spectrograms for BLACK class."""
    for i in range(num_samples):
        # pure silence
        y = np.zeros(int(duration * sr), dtype=np.float32)
        # convert to spectrogram and save
        out_path = os.path.join(OUT_DIR, "black", f"black_{i}.png")

        S = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128)
        S_dB = librosa.power_to_db(S, ref=np.max)

        plt.figure(figsize=(2, 2))
        plt.axis("off")
        librosa.display.specshow(S_dB, sr=sr, x_axis=None, y_axis=None)
        plt.tight_layout(pad=0)
        plt.savefig(out_path, bbox_inches="tight", pad_inches=0)
        plt.close()

    print(f"Created {num_samples} BLACK samples")


if __name__ == "__main__":
    process_ravdess()
    create_black_class()
