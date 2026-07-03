# Phase 0: Research & Data Audit (Ishaara ISL Project)

## 1. Primary Dataset: INCLUDE (IIT Bombay / AI4Bharat)
The INCLUDE dataset is a large-scale resource for Indian Sign Language (ISL) recognition.
- **Total Videos:** 4,287
- **Classes:** 263 isolated word signs across 15 categories (e.g., people, places, food, colors, days/months).
- **Video Specifications:** Approx. 0.27 million frames total. The videos typically range from 1 to 3 seconds.
- **Signer Diversity:** The dataset was recorded by multiple signers (the official paper notes 115 distinct signers). This is critical for our requirement of generalization.
- **Train/Val/Test Split Strategy:** 
  **Decision:** We will use a strictly **signer-independent split**. No signer appearing in the training set will appear in the validation or test sets. This ensures the model learns the sign articulation, not signer-specific appearance or motion idiosyncrasies. (For instance, 80% signers for train, 10% for val, 10% for test).

## 2. INCLUDE-50 Subset
- **Details:** A curated subset of 50 word signs designed for rapid evaluation and hyperparameter tuning.
- **Decision:** We will start with **INCLUDE-50** for Phase 1 and Phase 2 (BiLSTM baseline) to establish a working training loop, debug the MediaPipe extraction pipeline, and tune hyperparameters. Once the pipeline proves stable and accurately converges on INCLUDE-50, we will scale to the full 263-class INCLUDE dataset.

## 3. Secondary/Supplementary Sources
Our audit identified the following supplementary datasets:
- **iSign:** Developed by IIT Kanpur, a large-scale benchmark for continuous ISL processing (SignVideo2Text).
- **ISL-CSLTR:** A sentence-level dataset on Kaggle intended for continuous sign language translation.
- **Unlabeled Continuous Footage:** We will aggregate public ISL news broadcasts (e.g., DD News ISL bulletins) as unlabeled continuous footage.
- **Decision:** The ISL-CSLTR and iSign data will be evaluated qualitatively for Phase 4 (Continuous Signing). We will primarily use synthetic concatenation of INCLUDE clips for training CTC to maintain domain consistency, and use actual continuous videos strictly for out-of-domain qualitative testing.

## 4. Gap Analysis & Limitations
This project explicitly documents the following gaps in current ISL research resources:
1. **No Large Continuous-Signing Corpus:** Unlike ASL (which has How2Sign), there is no massive parallel corpus of continuous ISL mapped to English sentences suitable for deep sequence-to-sequence training.
2. **No Gloss-to-English Parallel Corpus:** Translation rules are linguistically defined but lack a large-scale parallel text corpus.
3. **Regional Dialect Limitations:** INCLUDE captures primarily urban/standardized ISL. Regional variations in signing (which are substantial in India) are not well-represented.
4. **Conclusion on Continuous Models:** Because of Gap #1, our Phase 4 and Phase 5 models will rely heavily on synthetic data (concatenated isolated signs and linguistically-generated gloss sequences). Results on these phases must be understood as "co-articulation-naive."

## 5. Data Licensing
- **INCLUDE License:** The INCLUDE dataset is released for academic and research purposes by the AI4Bharat initiative. Our use case (a research-grade translation system) falls under permitted usage. We will ensure proper attribution in our open-source release.

## Decision Log
1. **Dataset Choice:** We are starting with INCLUDE-50 for initial baseline validation to ensure rapid iteration, then scaling to the full INCLUDE dataset.
2. **Train/Test Protocol:** Signer-independent splits will be strictly enforced across all 4,287 videos.
3. **Continuous Data Strategy:** We will synthetically concatenate INCLUDE isolated clips with transition interpolation for CTC training, explicitly documenting the lack of natural co-articulation.
