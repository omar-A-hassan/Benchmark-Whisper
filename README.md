# Whisper Arabic Benchmark

This is a modified version of the [Picovoice speech-to-text-benchmark](https://github.com/Picovoice/speech-to-text-benchmark) framework, extended to support:

1. **Arabic language** support for speech recognition
2. **Translation tasks** (Arabic → English) using Whisper
3. **MasriSpeech dataset** benchmarking
4. **BLEU metric** for translation quality evaluation

## Overview

This benchmark framework allows you to evaluate Whisper's performance on:
- **Transcription**: Arabic speech → Arabic text (WER metric)
- **Translation**: Arabic speech → English text (BLEU metric)

The modifications support the MasriSpeech-Full dataset from HuggingFace.

## Installation

### Requirements

```bash
pip install torch whisper soundfile datasets scipy editdistance numpy inflect
```

### Optional Dependencies

For using cloud-based engines (not needed for Whisper):
```bash
pip install boto3 azure-cognitiveservices-speech google-cloud-speech ibm-watson pvleopard pvcheetah
```

## Dataset Preparation

### In Kaggle

1. Load the MasriSpeech dataset and add translations:

```python
from datasets import load_dataset, Audio
import pandas as pd

# Load dataset
dataset = load_dataset("NightPrince/MasriSpeech-Full")

# Cast audio to not decode during operations
dataset["validation"] = dataset["validation"].cast_column("audio", Audio(decode=False))

# Load translations CSV
csv_path = "/kaggle/input/translated/Translate - 1 (3).csv"
df = pd.read_csv(csv_path, encoding="utf-8-sig")
translations = df.iloc[:, 1].tolist()

# Add translation column
# (See talsem.ipynb for complete code)
```

2. Export to CSV format:

```python
import os
import csv
import soundfile
import scipy.signal as signal

output_folder = "./masrispeech_benchmark"
audio_folder = os.path.join(output_folder, "audio")
os.makedirs(output_folder, exist_ok=True)
os.makedirs(audio_folder, exist_ok=True)

metadata_path = os.path.join(output_folder, "metadata.csv")

with open(metadata_path, "w", encoding="utf-8", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=["audio_path", "transcription", "translation"])
    writer.writeheader()

    for idx, example in enumerate(dataset["validation"]):
        # Get audio
        audio_data = example["audio"]
        audio_array = audio_data["array"]
        sample_rate = audio_data["sampling_rate"]

        # Resample to 16kHz if needed
        if sample_rate != 16000:
            num_samples = int(len(audio_array) * 16000 / sample_rate)
            audio_array = signal.resample(audio_array, num_samples)

        # Convert to int16 and save
        audio_int16 = (audio_array * 32767).astype("int16")
        audio_filename = f"masrispeech_val_{idx:05d}.flac"
        audio_path = os.path.join(audio_folder, audio_filename)
        soundfile.write(audio_path, audio_int16, samplerate=16000)

        # Write metadata
        translation = example.get("translation")
        if translation is None or (isinstance(translation, float) and pd.isna(translation)):
            translation = ""  # Empty for missing translations

        writer.writerow({
            "audio_path": audio_path,
            "transcription": example["transcription"],
            "translation": translation
        })

        if (idx + 1) % 100 == 0:
            print(f"Processed {idx + 1} examples...")

print(f"Export complete! Files saved to {output_folder}")
```

3. Download the `masrispeech_benchmark` folder

## Usage

### Benchmark Arabic Transcription (WER)

Evaluate Whisper's Arabic speech recognition:

```bash
python3 benchmark.py \
    --engine WHISPER_LARGE_V3 \
    --dataset MASRI_SPEECH \
    --dataset-folder /path/to/masrispeech_benchmark \
    --language AR \
    --num-examples 100 \
    --num-workers 4
```

**Available Whisper transcription engines:**
- `WHISPER_TINY`
- `WHISPER_BASE`
- `WHISPER_SMALL`
- `WHISPER_MEDIUM`
- `WHISPER_LARGE`
- `WHISPER_LARGE_V2`
- `WHISPER_LARGE_V3`

### Benchmark Arabic→English Translation (BLEU)

Evaluate Whisper's translation capability:

```bash
python3 benchmark.py \
    --engine WHISPER_LARGE_V3_TRANSLATE \
    --dataset MASRI_SPEECH_TRANSLATE \
    --dataset-folder /path/to/masrispeech_benchmark \
    --language AR \
    --num-examples 100 \
    --num-workers 4
```

**Available Whisper translation engines:**
- `WHISPER_TINY_TRANSLATE`
- `WHISPER_BASE_TRANSLATE`
- `WHISPER_SMALL_TRANSLATE`
- `WHISPER_MEDIUM_TRANSLATE`
- `WHISPER_LARGE_TRANSLATE`
- `WHISPER_LARGE_V2_TRANSLATE`
- `WHISPER_LARGE_V3_TRANSLATE`

### Parameters

- `--engine`: Which engine to benchmark
- `--dataset`: Dataset type (`MASRI_SPEECH` for transcription, `MASRI_SPEECH_TRANSLATE` for translation)
- `--dataset-folder`: Path to the folder containing `metadata.csv` and audio files
- `--language`: Source language (use `AR` for Arabic)
- `--num-examples`: Number of examples to evaluate (optional, defaults to all)
- `--num-workers`: Number of parallel workers

## Results

Results are saved to `results/{language}/{dataset}/{engine}.log`

### Transcription Results (WER)

Example output for transcription:
```
WER: 15.23
RTF: 0.45
```

- **WER**: Word Error Rate (lower is better)
- **RTF**: Real-time Factor (processing time / audio duration)

### Translation Results (BLEU)

Example output for translation:
```
BLEU: 28.45
RTF: 0.52
```

- **BLEU**: BLEU score, 0-100 (higher is better)
  - Measures translation quality using n-gram overlap with reference
- **RTF**: Real-time Factor

## What Was Modified

### 1. Language Support (`languages.py`)
- Added `AR` (Arabic) language enum
- Added Arabic language code mapping (`ar-SA`)

### 2. Text Normalization (`normalizer.py`)
- Added `ArabicNormalizer` class
- Handles Arabic diacritics removal
- Normalizes Alef variants (أ، إ، آ → ا)
- Normalizes Teh Marbuta (ة → ه)
- Removes tatweel (elongation character)

### 3. Dataset Support (`dataset.py`)
- Added `MASRI_SPEECH` dataset class
  - Loads audio and Arabic transcriptions from CSV
  - Used for transcription benchmarking (WER)
- Added `MASRI_SPEECH_TRANSLATE` dataset class
  - Loads audio and English translations from CSV
  - Used for translation benchmarking (BLEU)

### 4. Whisper Engines (`engine.py`)
- Added Arabic language code to Whisper language mapping
- Modified `Whisper` base class to support `task` parameter
  - `task="transcribe"`: Standard transcription
  - `task="translate"`: Translation to English
- Added translation engine variants:
  - `WhisperTinyTranslate`
  - `WhisperBaseTranslate`
  - `WhisperSmallTranslate`
  - `WhisperMediumTranslate`
  - `WhisperLargeTranslate`
  - `WhisperLargeV2Translate`
  - `WhisperLargeV3Translate`

### 5. Metrics (`metric.py`)
- Added `BLEU` metric enum
- Implemented `BLEUScore` class
  - Calculates 1-4 gram precision
  - Includes brevity penalty
  - Returns score in error format (100 - BLEU) for consistency

### 6. Benchmark Logic (`benchmark.py`)
- Updated to use BLEU metric for translation datasets
- Automatically selects appropriate metric based on dataset type

## Dataset Format

The benchmark expects a folder with:

```
masrispeech_benchmark/
├── metadata.csv
└── audio/
    ├── masrispeech_val_00000.flac
    ├── masrispeech_val_00001.flac
    └── ...
```

**metadata.csv format:**
```csv
audio_path,transcription,translation
audio/masrispeech_val_00000.flac,على إنها عار في الوقت...,It was a shame that she...
audio/masrispeech_val_00001.flac,فأكيد ربنا عوضهم خير...,Surely God compensated them...
```

- `audio_path`: Path to audio file (can be relative or absolute)
- `transcription`: Arabic transcription (required for MASRI_SPEECH)
- `translation`: English translation (required for MASRI_SPEECH_TRANSLATE)

## Example: Complete Benchmark Run

```bash
# 1. Benchmark transcription with Whisper Large-v3
python3 benchmark.py \
    --engine WHISPER_LARGE_V3 \
    --dataset MASRI_SPEECH \
    --dataset-folder ./masrispeech_benchmark \
    --language AR \
    --num-workers 4

# 2. Benchmark translation with Whisper Large-v3
python3 benchmark.py \
    --engine WHISPER_LARGE_V3_TRANSLATE \
    --dataset MASRI_SPEECH_TRANSLATE \
    --dataset-folder ./masrispeech_benchmark \
    --language AR \
    --num-workers 4

# 3. Compare different model sizes for transcription
for model in WHISPER_TINY WHISPER_BASE WHISPER_SMALL WHISPER_MEDIUM WHISPER_LARGE_V3; do
    python3 benchmark.py \
        --engine $model \
        --dataset MASRI_SPEECH \
        --dataset-folder ./masrispeech_benchmark \
        --language AR \
        --num-examples 500
done

# 4. Compare different model sizes for translation
for model in WHISPER_TINY_TRANSLATE WHISPER_BASE_TRANSLATE WHISPER_SMALL_TRANSLATE WHISPER_MEDIUM_TRANSLATE WHISPER_LARGE_V3_TRANSLATE; do
    python3 benchmark.py \
        --engine $model \
        --dataset MASRI_SPEECH_TRANSLATE \
        --dataset-folder ./masrispeech_benchmark \
        --language AR \
        --num-examples 500
done
```

## Notes

- **RTF (Real-time Factor)**:
  - < 1.0: Faster than real-time
  - = 1.0: Processes at real-time speed
  - > 1.0: Slower than real-time

- **WER (Word Error Rate)**:
  - Calculated using edit distance between prediction and reference
  - Lower is better
  - 0% = perfect transcription

- **BLEU Score**:
  - Measures n-gram overlap between prediction and reference
  - 0-100 scale (higher is better)
  - 40+ is typically considered good for machine translation

- **Model Size vs Performance**:
  - Tiny/Base: Fast but lower accuracy
  - Small/Medium: Balanced
  - Large/Large-v3: Highest accuracy but slower

## Troubleshooting

### Audio Format Issues
Ensure all audio files are:
- 16kHz sample rate
- Mono channel
- FLAC format

### Memory Issues
- Reduce `--num-workers`
- Process in smaller batches using `--num-examples`
- Use smaller Whisper models (Tiny/Base)

### Missing Translations
The translation dataset (`MASRI_SPEECH_TRANSLATE`) will skip entries without translations automatically.

## Credits

- Original benchmark framework: [Picovoice speech-to-text-benchmark](https://github.com/Picovoice/speech-to-text-benchmark)
- MasriSpeech dataset: [NightPrince/MasriSpeech-Full](https://huggingface.co/datasets/NightPrince/MasriSpeech-Full)
- Whisper model: [OpenAI Whisper](https://github.com/openai/whisper)

## License

Same as the original Picovoice benchmark framework (check their repository for details).
