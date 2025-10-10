"""
Script to export MasriSpeech dataset from HuggingFace format to benchmark-compatible format
This should be run in Kaggle after loading and processing the dataset with librosa backend

Usage:
    # Make sure to set audio backend BEFORE loading dataset:
    import os
    os.environ['HF_DATASETS_AUDIO_BACKEND'] = 'librosa'

    # Then load and export:
    from export_masrispeech_kaggle import export_masrispeech_to_csv

    metadata_path, count = export_masrispeech_to_csv(
        dataset=dataset,
        output_folder="./masrispeech_benchmark",
        audio_folder="./masrispeech_benchmark/audio",
        split="train",
        max_samples=100
    )
"""
import os
import csv
import numpy as np
from pathlib import Path


def export_masrispeech_to_csv(dataset, output_folder, audio_folder, split="validation", max_samples=None):
    """
    Export MasriSpeech dataset to CSV format for benchmarking

    Args:
        dataset: HuggingFace DatasetDict with 'audio', 'transcription', and optionally 'translation' columns
        output_folder: Folder to save metadata.csv
        audio_folder: Folder where audio files will be saved
        split: Which split to export (train or validation)
        max_samples: Maximum number of samples to export (None for all)

    Returns:
        tuple: (metadata_path, exported_count)
    """
    import soundfile
    import scipy.signal as signal
    import pandas as pd
    from tqdm.auto import tqdm

    # Create output folders
    os.makedirs(output_folder, exist_ok=True)
    os.makedirs(audio_folder, exist_ok=True)

    print(f"\nExporting MasriSpeech {split} split")
    print(f"  Output folder: {output_folder}")
    print(f"  Audio folder: {audio_folder}")

    # Get split data
    split_data = dataset[split]

    # Determine how many samples to export
    total_samples = len(split_data)
    num_samples = min(max_samples, total_samples) if max_samples else total_samples

    print(f"  Total samples in split: {total_samples}")
    print(f"  Exporting: {num_samples}")

    metadata_path = os.path.join(output_folder, "metadata.csv")

    # Check if dataset has translation column
    has_translation = "translation" in split_data.features
    print(f"  Has translations: {has_translation}")

    # CSV fieldnames
    fieldnames = ["audio_path", "transcription"]
    if has_translation:
        fieldnames.append("translation")

    exported_count = 0
    skipped_count = 0

    with open(metadata_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        for idx in tqdm(range(num_samples), desc=f"Exporting {split}"):
            try:
                example = split_data[idx]

                # Get audio data (should be decoded by librosa backend)
                audio_data = example["audio"]

                # Verify we have decoded audio
                if "array" not in audio_data or "sampling_rate" not in audio_data:
                    print(f"\n⚠️  Warning: Audio at index {idx} is not decoded!")
                    print(f"   Audio keys: {audio_data.keys()}")
                    skipped_count += 1
                    continue

                audio_array = audio_data["array"]
                sample_rate = audio_data["sampling_rate"]

                # Ensure numpy array
                if not isinstance(audio_array, np.ndarray):
                    audio_array = np.array(audio_array)

                # Resample to 16kHz if needed
                if sample_rate != 16000:
                    num_samples_new = int(len(audio_array) * 16000 / sample_rate)
                    audio_array = signal.resample(audio_array, num_samples_new)
                    sample_rate = 16000

                # Convert to int16
                if audio_array.dtype in [np.float32, np.float64]:
                    # Audio is in float format (-1.0 to 1.0), convert to int16
                    audio_int16 = (audio_array * 32767).astype(np.int16)
                elif audio_array.dtype == np.int16:
                    audio_int16 = audio_array
                else:
                    # Other formats, try to convert
                    audio_int16 = audio_array.astype(np.int16)

                # Save as FLAC
                audio_filename = f"masrispeech_{split}_{idx:05d}.flac"
                audio_path = os.path.join(audio_folder, audio_filename)
                soundfile.write(audio_path, audio_int16, samplerate=16000)

                # Prepare metadata row
                row = {
                    "audio_path": audio_path,
                    "transcription": example["transcription"]
                }

                if has_translation:
                    # Handle None/null translations
                    translation = example.get("translation")
                    if translation is None:
                        row["translation"] = ""  # Empty string for missing translations
                    elif isinstance(translation, float) and pd.isna(translation):
                        row["translation"] = ""
                    else:
                        row["translation"] = translation

                writer.writerow(row)
                exported_count += 1

            except Exception as e:
                print(f"\n❌ Error processing index {idx}: {e}")
                skipped_count += 1
                continue

    # Summary
    print(f"\n{'='*60}")
    print(f"Export Summary:")
    print(f"  ✓ Exported: {exported_count} samples")
    if skipped_count > 0:
        print(f"  ⚠️  Skipped: {skipped_count} samples")
    print(f"  Metadata: {metadata_path}")
    print(f"  Audio files: {audio_folder}")
    print(f"{'='*60}")

    return metadata_path, exported_count

