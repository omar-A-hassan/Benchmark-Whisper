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

### Using Kaggle (Recommended)

The easiest way to prepare the MasriSpeech dataset is using the provided Kaggle notebook workflow:

1. **Load the dataset and add translations:**

```python
from datasets import load_dataset, Audio, Value
import pandas as pd

# Load dataset
dataset = load_dataset("NightPrince/MasriSpeech-Full")

# Load translations CSV
csv_path = "/kaggle/input/translated/Translate - 1 (3).csv"
df = pd.read_csv(csv_path, encoding="utf-8-sig")
translations = df.iloc[:, 1].tolist()

# Add translation column to train split
new_feats = dataset["train"].features.copy()
new_feats["translation"] = Value("string")

def add_translation(batch, indices):
    return {"translation": [translations[i] for i in indices]}

dataset["train"] = dataset["train"].map(
    add_translation,
    with_indices=True,
    batched=True,
    features=new_feats,
    desc="Adding translation column"
)

# Cast audio to not decode (returns bytes for efficient processing)
dataset["train"] = dataset["train"].cast_column("audio", Audio(decode=False))
```

2. **Export using the helper script:**

```python
from export_masrispeech_kaggle import export_masrispeech_to_csv

# Export configuration
OUTPUT_FOLDER = "./masrispeech_benchmark"
AUDIO_FOLDER = "./masrispeech_benchmark/audio"

metadata_path, count = export_masrispeech_to_csv(
    dataset=dataset,
    output_folder=OUTPUT_FOLDER,
    audio_folder=AUDIO_FOLDER,
    split="train",
    max_samples=100  # or None for all 50,715 samples
)

print(f"Exported {count} samples")
```

The export script automatically:
- Decodes audio from bytes (handles torchcodec issues)
- Resamples to 16kHz mono
- Converts to int16 FLAC format
- Saves **relative paths** in metadata.csv for portability
- Handles missing translations gracefully

3. **Clone this repository in Kaggle:**

```python
!git clone https://github.com/omar-A-hassan/Benchmark-Whisper.git
```

4. **Run benchmarks** (see Usage section below)

**Note:** See `talsem.ipynb` for the complete Kaggle notebook workflow.

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

- **BLEU**: n-gram error rate, 0-100 (lower is better)
  - Measures translation quality using n-gram overlap with reference
  - Returns percentage of non-matching n-grams (1-4 grams)
  - Lower values indicate better translation quality
  - ~20-30% error rate is good for machine translation
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
- Uses English normalizer for translation tasks (target language)
- Uses source language normalizer for transcription tasks
- Fixed results log filename generation (uses `.value` instead of `str()`)

### 7. Export Helper (`export_masrispeech_kaggle.py`)
- Added helper script for exporting MasriSpeech dataset in Kaggle
- Handles audio decoding from bytes (solves torchcodec issues)
- Resamples audio to 16kHz mono
- Saves relative paths for portability
- Handles missing translations gracefully

### 8. GPU Support (`engine.py`)
- Auto-detects CUDA availability
- Uses GPU when available for 5-10x speedup
- Falls back to CPU automatically

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
audio/masrispeech_train_00000.flac,على إنها عار في الوقت...,It was a shame that she...
audio/masrispeech_train_00001.flac,فأكيد ربنا عوضهم خير...,Surely God compensated them...
```

- `audio_path`: Path to audio file (relative to metadata.csv folder - recommended)
- `transcription`: Arabic transcription (required for MASRI_SPEECH)
- `translation`: English translation (required for MASRI_SPEECH_TRANSLATE, can be empty string if missing)

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

- **BLEU Metric** (as implemented):
  - Returns n-gram **error rate**, not traditional BLEU score
  - 0-100 scale (lower is better - opposite of traditional BLEU)
  - Measures percentage of non-matching n-grams (1-4 grams)
  - 20-30% error rate is good for machine translation
  - Note: This differs from traditional BLEU scoring for consistency with WER format

- **Model Size vs Performance**:
  - Tiny/Base: Fast but lower accuracy
  - Small/Medium: Balanced
  - Large/Large-v3: Highest accuracy but slower

- **GPU Acceleration**:
  - The framework automatically detects and uses GPU (CUDA) when available
  - Falls back to CPU if GPU is not available
  - GPU provides 5-10x speedup (RTF: ~0.1-0.3 vs 0.8-1.3 on CPU)
  - Enable GPU in Kaggle: Runtime → Change runtime type → GPU T4
  - Verify GPU usage:
    ```python
    import torch
    print(f"CUDA available: {torch.cuda.is_available()}")
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    ```

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
