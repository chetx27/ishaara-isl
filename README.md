# Ishaara (इशारा)

Ishaara is a research-grade, real-time Indian Sign Language (ISL) recognition and translation system. 

Unlike American Sign Language (ASL), which benefits from massive parallel corpora (WLASL, How2Sign) and numerous commercial applications, ISL severely lacks robust computational tooling. This project aims to address this accessibility gap for the 7+ million Deaf and Hard-of-Hearing individuals in India by providing a strictly evaluated, open-source translation pipeline.

## System Architecture
Ishaara is built end-to-end in Python 3.11 using PyTorch, FastAPI, and React.

- **Phase 1 (Perception Layer):** MediaPipe Holistic extracts 95 key landmarks (Pose, Hands, Face). Landmarks are spatially normalized to the shoulder midpoint and scaled by shoulder width to achieve distance invariance.
- **Phase 2 (Isolated Recognition):** A PyTorch BiLSTM encoder with temporal attention pooling (with a direct Transformer comparison) classifies individual signs.
- **Phase 3 (Live API):** A real-time FastAPI WebSocket ingests streaming webcam frames, extracts landmarks locally, maintains a sliding buffer, and triggers motion-heuristic based inference.
- **Phase 4 (Continuous Segmentation):** The BiLSTM is extended with CTC loss to recognize unsegmented continuous signing.
- **Phase 5 & 6 (Grammar Translation & Generation):** seq2seq models (T5) map ISL topic-comment glosses to grammatical English and vice versa. Text-to-Sign is achieved via lookup-and-stitch crossfade video generation.

## Known Limitations & Honest Constraints

To maintain research integrity, the following limitations are explicitly documented rather than obscured:

1. **Dataset Constraints**: The system is trained on the INCLUDE dataset (IIT Bombay / AI4Bharat), which contains ~4,000 isolated signs across 263 classes. This is too small for true zero-shot generalization.
2. **Dialect Diversity**: While the models use strictly signer-independent training splits, the 115 signers in INCLUDE do not comprehensively cover the vast regional variations (dialects) of ISL across India. 
3. **Synthetic Continuous Data (Co-articulation-naive)**: Because a massive continuous ISL-to-text corpus does not currently exist, the CTC models are trained on synthetically concatenated isolated clips. This data lacks natural *co-articulation* (the fluid blending of signs). As such, the Sign Error Rate (SER) reported is "co-articulation-naive".
4. **Synthetic Grammar Data**: The seq2seq translation models are trained on English sentences programmatically converted to ISL gloss structures based on linguistic rules.
5. **Generative Synthesis**: The Text-to-ISL generation is a deterministic clip-retrieval and stitching engine. True generative pose-synthesis is beyond the current scope.

## Setup & Execution

### Backend
```bash
uv venv
source venv/bin/activate
uv pip install -r requirements.txt
python api/main.py
```

### Frontend
```bash
cd frontend
npm install
npm run dev
```

## UI & Accessibility
The frontend is built with React + Vite + TypeScript.
- Strictly dark-themed using explicit CSS custom properties.
- High-contrast visual design, avoiding low-accessibility practices like drop-shadow elevation.
- Explicit and real error states (e.g., "NO_HAND_DETECTED", "NOT_CONFIDENT") instead of generic loading spinners.
- No emojis are used in the codebase, UI, or documentation to maintain professional strictness.

## License
Code is provided under the MIT License. Data used (INCLUDE) is subject to its respective AI4Bharat academic license.
