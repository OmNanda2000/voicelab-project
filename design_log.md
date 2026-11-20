# Design Log – VoiceLab

One or two lines per part: what you tried, what failed, what you changed.

- **Part 0:** Recorded the words, sentences and room-noise blocks using the personal read-aloud script. Ran the splitting and audit workflow; all 35 clips passed the audit without flags.

- **Part 1:** Implemented MFCC extraction from first principles using NumPy and SciPy. Initially, Mel filterbank scaling differed from Librosa; corrected the frequency conversion and filterbank implementation. Implemented autocorrelation-based F0 estimation and compared it with pYIN, refining pitch selection using normalization and continuity checks.

- **Part 2:** Implemented diagonal-Gaussian log emissions, Forward and Viterbi algorithms in log space, and segmental K-means training for ten three-state HMMs. Achieved 90% held-out recognition accuracy (9/10); the word “think” was misclassified as “sink”.

- **Part 3:** Implemented the CTC forward algorithm from first principles using NumPy and log-space dynamic programming, including blank expansion and repeated-label transition restrictions. Verified against PyTorch CTCLoss: NumPy 0.3073 versus approximately 0.30729994.

- **Part 4:** Implemented add-one and interpolated Kneser–Ney language models, evaluated Brown corpus perplexity, integrated CMUdict with G2P fallback, and recorded three weighted personal pronunciation alternatives.

- **Part 5:** Built the toy lexicon with 10 states. Hand determinisation reduced it to 7 states, and minimisation reduced it to 5 states while preserving the three phone-to-word mappings and their costs. The notebook implementation of WFST composition and own-sentence L o G decoding still requires a clean-kernel rerun before these operations can be treated as fully verified.

- **Part 6:** Precomputed wav2vec2 posteriors and Whisper transcripts; implemented WER evaluation and CTC prefix beam search with language-model fusion. Selected alpha = 0.0 because the tested language-model weights produced the same tuning WER. Evaluated both recognisers under clean, 10 dB and 0 dB conditions. Latest beam-search timing: approximately 19.19 ms at beam width 1 and 174.86 ms at beam width 8.

- **Part 7:** Implemented written-to-spoken text normalisation and achieved exact matches on all four assignment sentences. Implemented Mel-spectrogram inversion using a pseudo-inverse Mel filterbank and Griffin–Lim reconstruction; spectral convergence decreased from 0.545613 (1 iteration) to 0.201625 (10 iterations) and 0.158171 (30 iterations). Evaluated Neural TTS using Tacotron2 with WaveRNN and assessed the generated recordings through listening. Griffin–Lim STOI was approximately 0.9204; direct Neural TTS STOI was not reported because the independently generated speech was not time-aligned with the reference.