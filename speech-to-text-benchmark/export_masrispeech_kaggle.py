"""
Script to export MasriSpeech dataset from HuggingFace format to benchmark-compatible format
This should be run in Kaggle after loading and processing the dataset
"""
import os
import csv
from pathlib import Path


def export_masrispeech_to_csv(dataset, output_folder, audio_folder, split="validation"):
    """
    Export MasriSpeech dataset to CSV format for benchmarking

    Args:
        dataset: HuggingFace dataset with 'audio', 'transcription', and optionally 'translation' columns
        output_folder: Folder to save metadata.csv
        audio_folder: Folder where audio files will be saved
        split: Which split to export (train or validation)
    """
    import soundfile
    import scipy.signal as signal

    # Create output folders
    os.makedirs(output_folder, exist_ok=True)
    os.makedirs(audio_folder, exist_ok=True)

    metadata_path = os.path.join(output_folder, "metadata.csv")

    # Determine if dataset has translation column
    has_translation = "translation" in dataset[split].features

    # Write metadata CSV
    fieldnames = ["audio_path", "transcription"]
    if has_translation:
        fieldnames.append("translation")

    with open(metadata_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        for idx, example in enumerate(dataset[split]):
            # Save audio file
            audio_data = example["audio"]
            audio_array = audio_data["array"]
            sample_rate = audio_data["sampling_rate"]

            # Resample to 16kHz if needed
            if sample_rate != 16000:
                num_samples = int(len(audio_array) * 16000 / sample_rate)
                audio_array = signal.resample(audio_array, num_samples)
                sample_rate = 16000

            # Convert to int16
            audio_int16 = (audio_array * 32767).astype("int16")

            # Save as FLAC
            audio_filename = f"masrispeech_{split}_{idx:05d}.flac"
            audio_path = os.path.join(audio_folder, audio_filename)
            soundfile.write(audio_path, audio_int16, samplerate=16000)

            # Write metadata row
            row = {
                "audio_path": audio_path,
                "transcription": example["transcription"]
            }

            if has_translation:
                # Handle None/null translations
                translation = example.get("translation")
                if translation is None or (isinstance(translation, float) and pd.isna(translation)):
                    continue  # Skip rows without translation
                row["translation"] = translation

            writer.writerow(row)

            if (idx + 1) % 100 == 0:
                print(f"Processed {idx + 1} examples...")

    print(f"Export complete! Metadata saved to {metadata_path}")
    print(f"Audio files saved to {audio_folder}")

