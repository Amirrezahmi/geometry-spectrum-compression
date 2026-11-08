# The Geometry and Spectrum of Compression

Code and reproduction notebooks for *The Geometry and Spectrum of Compression: A Differential and Spectral Analysis of In-Context Compression Dynamics* (Amir Reza Hashemi, Department of Computer Science, Islamic Azad University, Science and Research Branch, Tehran, Iran).

Delétang et al. (ICLR 2024) showed that compression and prediction are the same thing, and that language models are surprisingly good general-purpose compressors. That work looks at the compression *rate*. This paper looks at the *shape* of the curve instead — the sequence of per-position code lengths a compressor produces as it sees more data — and asks what classical tools (curve geometry, Fourier analysis, Newton's method) can tell us about it.

Three tools, in short:
- **Curve geometry** — treat the compression curve as a plane curve, look at its velocity and curvature, and fit it locally with the osculating parabola from the MDL redundancy expansion of Rissanen (1984) and Grünwald (2004).
- **Fourier analysis** — take the DFT of the curve to recover hidden periodic structure, in the data itself or in the compressor's own architecture (e.g. a finite context window).
- **Newton-type model search** — a quadratic fit in log-parameter space that finds the compression-optimal model size from a handful of evaluations instead of a full grid search.

All three are validated on synthetic data with known ground truth, then applied to real text (enwik8), image (CIFAR-10, Olivetti Faces), audio (`waves_yesno` recordings), and to Transformer language models trained from scratch (up to 2.9M parameters, 50MB of enwik8).

## What held up, and what did not

- **Image row width** is recovered from the compression curve of a trained neural model (z ≈ 16–19), collapses under a pixel-shuffled control (z ≈ 0.5), and survives removal of the position-dependent mean cost, so it is not an artifact of the context window.
- **Speaker pitch** (≈ 124 Hz) is recovered by a classical online compressor and agrees with autocorrelation and YIN to within about 6 Hz (119.1–125.0 Hz in the run reproduced by `research_project_v4_multimodal_neural.ipynb`).
- **The neural audio pitch *line* did not hold up.** The apparent peak sits at 63.75 samples, exactly the fourth harmonic of the model's 255-sample context-reset period, grows rather than shrinks under shuffling (z 8.95 → 22.61), and disappears once the position-dependent mean is subtracted (|z| ≤ 1.63 at every candidate pitch period). What remains after the correction is a weaker, broad excess in the pitch range: in a band fixed in advance from the independent tracker (58–91 samples, 87.9–137.9 Hz) the spectral magnitude is 1.42× the rest of the 40–200-sample range on real data and 1.005× on shuffled data. We treat this as moderate, not definitive, evidence (one shuffled control; bin edges beyond 58 samples were chosen after looking). The run is `audio_reset_corrected_check.ipynb`; the band numbers are reproduced without a GPU by `python audio_band_test.py audio_losses_60_rerun.npy audio_shuf_losses_60_rerun.npy`.
- **The context-reset period** is an evaluation-window artifact that the spectral test detects correctly, and is not expected to change under shuffling.
- **CTW vs. backoff:** our simple backoff model beats our CTW implementation on enwik8 (3.13 vs. 3.55 bits/byte). Both implementations are re-derived and validated in `validate_ctw.ipynb`.

## Repository contents

**Analysis notebooks**
| Notebook | What it does |
|---|---|
| `research_project_v3.ipynb` | Main experimental notebook. Sections 0-5 validate the three tools on synthetic data; Sections 6-9 repeat them on real text, images, and audio, plus the text neural model. |
| `research_project_v4_multimodal_neural.ipynb` | Extends Section 9 of the above to the image and audio neural models, and reproduces the pitch cross-validation table. Read together with the note at its top on the neural audio peak. |
| `audio_reset_corrected_check.ipynb` | Self-contained run (retrains the small audio model, ~8 min on a T4, fixed seeds) that produces every number in the neural-audio paragraph of the paper: real and shuffled held-out losses, the harmonic sweep, the reset-corrected test at six candidate pitch periods, and the autocorrelation and YIN estimates. Executed outputs are included. |
| `audio_band_test.py` | CPU-only script for the band-level pitch check; takes the two stored loss files below. |
| `validate_ctw.ipynb` | Independent, from-scratch validation of every CTW-related claim in Section 3.4 / Appendix C: bit-identical check between the naive and optimized implementations, the depth-sweep validation table, the enwik8 rate, the backoff-vs-CTW comparison, and the matched-input speedup. No numbers are tuned to match the paper. |
| `additional_validation.ipynb` | Further checks: the 511 vs. 512 period, instantaneous vs. cumulative cost in Figure 6, shuffled controls, the matched held-out comparison, and the oracle-vs-online diagnostic. |

**Training notebooks** (each ends with shuffled-control and matched-range cells)
| Notebook | What it does |
|---|---|
| `train_neural_lm.ipynb` | Trains the small text Transformer (4 layers, context 256, ~0.9M params) on 5MB of enwik8. |
| `train_neural_lm_v2.ipynb` | Trains the large text Transformer (6 layers, context 512, ~2.9M params) on the full 55MB. |
| `train_neural_lm_image_audio.ipynb` | Trains the same small architecture on CIFAR-10 and `waves_yesno` audio, then runs the shuffled, harmonic, and reset-corrected tests. The image results in the paper come from this notebook. Its audio outputs are from an earlier exploratory run with different random seeds, so its audio numbers differ slightly from the paper's; the audio numbers in the paper come from `audio_reset_corrected_check.ipynb`. |

**Data**
| File | Contents |
|---|---|
| `colab_data_bundle_v3.zip` | 300KB of enwik8 (classical baselines), 60 CIFAR-10 images, 8 audio clips — the small-scale data used in `research_project_v3.ipynb`. |
| `image_bundle_full150.zip` | 150 CIFAR-10 images (128x128 grayscale) — full-scale image training set. |
| `audio_bundle_full60.zip` | 60 `waves_yesno` recordings — full-scale audio training set. |

Note: `enwik8_first_10MB.txt` inside `colab_data_bundle_v3.zip` is, despite its name, only the first 300,000 bytes of enwik8. The notebooks that read it expect that name, so it is kept as is.

The bundles are small subsets of public datasets, included for reproducibility. See *Data licenses* below.

**enwik8 (not included, 55MB).** The two text training notebooks expect `enwik8_first_55MB.txt` (the first 57,671,680 bytes of enwik8). Build it with:

```bash
wget http://mattmahoney.net/dc/enwik8.zip
unzip enwik8.zip
head -c 57671680 enwik8 > enwik8_first_55MB.txt
```

**Precomputed model outputs** (so you can run the analysis notebooks without retraining)
| File(s) | Model | Result |
|---|---|---|
| `neural_lm_losses.npy` / `neural_lm_meta.npy` | small text model | 2.10 bits/byte — used by `research_project_v3.ipynb`. The weights of this run were not kept; re-running `train_neural_lm.ipynb` at 7000 steps gives the next row. |
| `neural_lm_losses_7000steps.npy` / `neural_lm_meta_7000steps.npy` | small text model, 7000-step run | 2.28 bits/byte (training-run variance; the output of `train_neural_lm.ipynb`) |
| `neural_lm_losses_v2.npy` / `neural_lm_meta_v2.npy` | large text model | 1.68 bits/byte |
| `small_model_matched_losses.npy` / `large_model_matched_losses.npy` | small and large text models on an identical shared range (bytes 55,000,000–57,671,680, untouched by either model's training) | 2.33 vs. 1.65 bits/byte |
| `neural_lm_image_losses_{60,150}.npy` / `neural_lm_image_meta_{60,150}.npy` | image model, two dataset scales | 2.83 / 2.68 bits/byte |
| `neural_lm_audio_losses_{8,60}.npy` / `neural_lm_audio_meta_{8,60}.npy` | audio model, two dataset scales | 1.27 / 1.32 bits/byte |
| `audio_losses_60_rerun.npy` / `audio_shuf_losses_60_rerun.npy` | audio model (60 recordings): real and shuffled held-out losses from `audio_reset_corrected_check.ipynb` | 1.32 / 5.22 bits/byte; the real file is identical to `neural_lm_audio_losses_60.npy` |

## Reproducing the results

Everything runs on Google Colab with a free-tier T4 GPU (the training notebooks need one; the analysis notebooks don't).

1. Upload a notebook to Colab.
2. Run the cells top to bottom — each notebook's first code cell tells you which file(s) to upload when prompted.
3. To reproduce a figure from scratch rather than from the precomputed outputs above, run the matching training notebook first, then feed its output into the corresponding analysis notebook.

Suggested order if starting from nothing: `train_neural_lm.ipynb` → `train_neural_lm_v2.ipynb` → `train_neural_lm_image_audio.ipynb` → `research_project_v3.ipynb` → `research_project_v4_multimodal_neural.ipynb`.

## Data licenses

None of the four sources states a formal open-source license that we could confirm. The conditions each one does state:

| Dataset | Source | Stated terms |
|---|---|---|
| CIFAR-10 | Krizhevsky (2009), <https://www.cs.toronto.edu/~kriz/cifar.html> | The dataset page asks users to cite the tech report *Learning Multiple Layers of Features from Tiny Images*. No formal license is stated. |
| AT&T (Olivetti) Database of Faces | AT&T Laboratories Cambridge | "When using these images, please give credit to AT&T Laboratories Cambridge." No formal license. |
| `waves_yesno` | OpenSLR SLR1, created for the Kaldi project, <https://www.openslr.org/1/> | "No formal license but free to use for any purpose." |
| enwik8 | Hutter Prize / Large Text Compression Benchmark (Mahoney) | The first 10^8 bytes of the English Wikipedia XML dump of 3 March 2006. The text is subject to Wikipedia's copyright terms (<https://en.wikipedia.org/wiki/Wikipedia:Copyrights>). |

If you reuse the bundles, please cite and credit the original sources. The terms above were read from each source's own page at the time of writing and may change.

## Citation

```bibtex
@article{hashemi2026geometry,
  title   = {The Geometry and Spectrum of Compression: A Differential and Spectral Analysis of In-Context Compression Dynamics},
  author  = {Hashemi, Amir Reza},
  year    = {2026},
  note    = {Preprint}
}
```

## References

- Delétang, G. et al. *Language Modeling Is Compression.* ICLR 2024.
- Rissanen, J. *Universal Coding, Information, Prediction, and Estimation.* IEEE Trans. Information Theory, 1984.
- Grünwald, P. *A Tutorial Introduction to the Minimum Description Length Principle.* arXiv:math/0406077, 2004.
- de Cheveigné, A. and Kawahara, H. *YIN, a Fundamental Frequency Estimator for Speech and Music.* JASA, 2002.
- Hutter, M. *The Hutter Prize.* 2006.

## License

Code is released under the MIT License (see `LICENSE`). See the paper for the license terms on the text itself.
