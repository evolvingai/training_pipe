# COCO 2017 — Known Dataset Issues & Resolutions

*A practitioner's reference for diagnosing and mitigating dataset-level problems before and during custom detector training.*

**8 categories · 19 documented issues · COCO 2017 Detection track**

---

## Action Items (by Priority)

### Critical — fix before any training run

1. **Validate all images can be decoded.** Log and remove any that fail decoding. Run pHash deduplication and remove near-duplicates with similarity > 0.95.
3. **Assert train/val disjoint.** Hash image filenames (MD5 or SHA1) across train and val and assert zero overlap before training. Download from the official COCO source, not third-party mirrors.
4. **Set `maxDets=[1, 10, 300]`** in your COCO evaluator — the default cap of 100 silently drops predictions and suppresses reported performance on dense scenes. [fixed in SOTA]
5. **Use `pycocotools` for evaluation.** It correctly handles `ignore=1` regions; custom mAP implementations frequently skip this, inflating FP counts and depressing precision. [fixed in SOTA]
6. **Filter `iscrowd=1` from training and evaluation.** Assert in your loader that no crowd annotations reach the label pipeline. [fixed in SOTA]

### High — address before final benchmarking

7. **Run pseudo-labeling or inspect high-confidence predictions** on training images to surface the estimated 10–20% of unlabeled instances (false negatives). Use focal loss to reduce sensitivity to remaining label errors. [fixed in SOTA]
8. **Apply inverse-frequency loss weighting and a repeat factor sampler** to counter `person`-class dominance and the long-tail distribution. Monitor per-class AP — overall mAP will mask rare-class failures. [fixed in SOTA]
9. **Deduplicate overlapping boxes.** For each image and class, suppress annotations with IoU > 0.85, keeping the larger box. Log counts per class.
10. **Run a per-class confusion matrix on val** to identify ambiguous class pairs (cup/bowl, truck/car, handbag/backpack). Apply label smoothing (ε = 0.05–0.1) at noisy boundaries; merge classes if the distinction is irrelevant to your task. [fixed in SOTA]

### Medium — address for robustness and fair comparison

11. **Report AP_S, AP_M, AP_L separately.** Use multi-scale training (mosaic, random resize) and a high-resolution output stride. Calibrate expectations if your deployment domain has fewer small objects. [fixed in SOTA]
12. **Audit predictions on out-of-context test images** for target classes before deploying. Use copy-paste augmentation to break spurious context correlations (e.g., surfboard always with person). [fixed in SOTA]
13. **Use letterbox resizing, not naive stretch.** Log min/max/mean image dimensions during preprocessing and assert expected ranges before training. [fixed in SOTA]
14. **Always specify the COCO year** (2017 vs. 2014) in reports and code. When using pre-trained weights, verify they were trained on `train2017` if evaluating on `val2017`.
15. **Convert area thresholds proportionally** to your training resolution, or evaluate on original-resolution images when computing AP_S/AP_M/AP_L.

### Lower — documentation and domain-specific

16. **Document truncation policy.** Augment with aggressive random cropping to regularize inconsistent partial-object labels.
17. **Do not mix segmentation-derived boxes with manually drawn boxes.** Document the derivation method; they are not interchangeable for localization metrics.
18. **Always state which val split was used** when reporting numbers. Use `minival` only for fast iteration; use full `val2017` for benchmarking.
19. **Use `instances_train2017.json` for detection**, not the panoptic JSON. If both are required, compare on a sample and document discrepancies.
20. **Add domain-specific training images** if your deployment geography or environment differs from COCO's Western/indoor bias — even a few hundred well-annotated examples improve generalization.
21. **Build a single-object evaluation set** for critical classes to measure AP outside co-occurrence context.
22. **Plan test-dev submissions in advance.** Check current rate limits before your reporting deadline. Reserve test-dev for final paper or production numbers only.

---

## 01. Annotation Quality

### Missing annotations (false negatives)

**Issue:** COCO's crowd-sourced pipeline has significant gaps, especially for small or occluded instances. Studies estimate 10–20% of objects go unlabeled. The model sees these as background, receiving incorrect gradient signal during training.

**Resolution:**
- Use semi-supervised or pseudo-labeling techniques to surface unannotated instances.
- High-confidence model predictions on training images may reveal missed labels — inspect them.
- Treat this as irreducible noise; use robust loss functions (e.g. focal loss) to reduce sensitivity to individual label errors.

---

### Ambiguous crowd regions

**Issue:** `iscrowd=1` annotations use RLE masks and are excluded from mAP evaluation. If your loader includes them in training labels by mistake, the model trains on noisy polygon-approximated crowds.

**Resolution:**
- Filter `iscrowd=1` entries from both training and evaluation annotations.
- Assert in your loader that no crowd annotations reach the label pipeline.
- For eval, ensure predictions overlapping `iscrowd` regions are excluded — they should be neither TP nor FP.

---

### Duplicate and overlapping boxes

**Issue:** Some images contain near-identical boxes for the same instance from different annotators. These act as implicit upsampling, distorting loss computation for specific instances.

**Resolution:**
- Run a deduplication pass: for each image and class, suppress annotations with IoU > 0.85 against each other, keeping the box with the larger area.
- Log removed duplicate counts per class to quantify impact before deciding on a threshold.

---

### Class boundary ambiguity

**Issue:** Semantically adjacent pairs — *cup/bowl*, *truck/car*, *handbag/backpack*, *knife/fork* — were annotated inconsistently across annotators. This produces irreducible label noise at decision boundaries that more training data will not fix.

**Resolution:**
- Identify your highest-confusion class pairs by running a per-class confusion matrix on val predictions.
- For pairs that matter to your task, manually relabel a validation sample to establish ground truth.
- Merge classes if the distinction is unimportant to your deployment. Apply label smoothing (ε = 0.05–0.1) to reduce overconfidence on noisy boundaries.

---

### Truncation not standardized

**Issue:** No consistent rule governed how much of an object must be visible to be labeled. Some annotators labeled 10%-visible objects; others ignored them. The model receives inconsistent gradient signal on partially occluded instances.

**Resolution:**
- Augment training with aggressive random cropping to expose the model to partial objects deliberately — this regularizes the inconsistency.
- If deployment involves heavy occlusion, supplement with domain-specific data annotated under a controlled truncation policy.

---

### Segmentation-derived bbox inconsistency

**Issue:** For classes with polygon segmentations, the bounding box is derived as the tight polygon envelope — which can differ from a manually drawn box around the perceived object boundary, especially for irregular shapes.

**Resolution:**
- Do not mix segmentation-derived boxes with manually drawn boxes in the same dataset.
- Document the derivation method explicitly; tight-polygon and manually-drawn boxes are not interchangeable for localization metrics.

---

## 02. Class Imbalance

### Person class dominance

**Issue:** `person` accounts for ~250k instances. Rare classes like `toaster`, `hair drier`, and `parking meter` each have fewer than 5k. Without correction, the loss gradient is dominated by the most common class and rare-class AP collapses.

**Resolution:**
- Compute per-class instance counts and apply inverse-frequency weighting to the classification loss.
- Use a repeat factor sampler: set image sampling probability proportional to the rarest class present in that image.
- Monitor per-class AP explicitly — overall mAP will mask rare-class failures.

---

### Long tail distribution

**Issue:** The top 10 classes account for ~60% of all instances. Standard training over-fits the head of the distribution. The remaining 70 classes form a steep long tail where performance degrades rapidly with rarity.

**Resolution:**
- Use two-stage training: train normally, then fine-tune with class-balanced sampling applied only to the classifier head.
- If your target is a subset of classes, prune the others before training — irrelevant classes waste capacity and pollute decision boundaries.

---

## 03. Small Object Bias

### ~41% of objects are small (area < 32²)

**Issue:** COCO val is heavily weighted toward small objects. Models that don't handle small objects score disproportionately low on COCO mAP vs. real-world performance if the deployment domain has fewer tiny objects — and the reverse is true for dense small-object domains.

**Resolution:**
- Always report AP_S, AP_M, AP_L separately — not just overall mAP — to understand where performance actually breaks down.
- Use multi-scale training (mosaic, random resize) and ensure the neck retains a high-resolution output stride.
- Calibrate expectations: if your deployment domain has few small objects, COCO mAP systematically understates your real-world performance.

---

## 04. Image Quality & Scale

### High resolution variance

**Issue:** COCO images range from under 200px to over 4K wide. Without consistent resize handling, the pipeline will silently produce degenerate inputs for extreme cases — very small images become over-blurred, very large images get random cropping artifacts.

**Resolution:**
- Use letterbox (pad-to-aspect) resizing, not naive stretch — this preserves object aspect ratios and keeps annotation coordinates valid.
- Log min/max/mean image dimensions during preprocessing and assert expected ranges before training starts.
- Explicitly test a few very small and very large images through the full pipeline before the first training run.

---

### Corrupted and near-duplicate images

**Issue:** Roughly 1% of the training set contains corrupted or near-duplicate images. Duplicates cause overfitting to specific instances; corrupted images cause NaN losses or silent training stalls.

**Resolution:**
- Validate all images can be decoded before training starts — log and remove any that fail.
- Run perceptual hash (pHash) deduplication and remove near-duplicates with similarity > 0.95.

---

## 06. Split Integrity

### Train/val overlap in third-party datasets

**Issue:** COCO 2017 train and val are disjoint by design. Third-party pre-converted datasets (Roboflow, Kaggle, HuggingFace) frequently introduce silent overlap by re-splitting, making validation metrics unreliable.

**Resolution:**
- Hash image filenames (MD5 or SHA1) across train and val and assert zero overlap before training.
- Download COCO directly from the official source rather than third-party mirrors whenever possible.

---

### minival vs. full val — metric incompatibility

**Issue:** Many published results use the 5k-image `minival` subset rather than the full 40k val set. Results are not directly comparable, and minival overstates variance due to sample size.

**Resolution:**
- Always explicitly state which val split was used when reporting numbers.
- Use `minival` only for fast iteration during development; use full `val2017` for benchmarking.

---

### Test set has no public labels

**Issue:** `test-dev` (20k images) requires submitting to the COCO evaluation server. There is no way to evaluate locally, and the server has submission rate limits — this creates a bottleneck for final performance reporting.

**Resolution:**
- Plan test-dev submissions in advance; check current rate limits before your reporting deadline.
- Use full `val2017` as the primary benchmark during development. Reserve test-dev for final paper or production numbers only.

---

### COCO 2014 vs. 2017 split difference

**Issue:** Both versions use the same images but different train/val splits. Pre-trained weights or benchmarks labeled "COCO" without a year may have train/val contamination relative to your split — inflating transfer learning baselines.

**Resolution:**
- Always specify the year in reports and code comments.
- When using pre-trained weights, verify they were trained on `train2017` if evaluating on `val2017`.

---

### Panoptic vs. detection annotation divergence

**Issue:** Panoptic annotations (added later) don't always agree 1:1 with the original instance annotations. Mixing both types without reconciliation introduces label inconsistency.

**Resolution:**
- Use one annotation type per task — stick to `instances_train2017.json` for detection, not the panoptic JSON.
- If both types are required, compare on a sample and document any discrepancies explicitly.

---

## 07. Systematic Biases

### Context bias (spurious correlations)

**Issue:** `surfboard` almost always appears with `person`; `toilet` almost always appears indoors. Models learn to exploit context as a shortcut rather than recognizing the object itself. This causes failures when context breaks — a surfboard in a shop, a toilet outdoors.

**Resolution:**
- Audit predictions on out-of-context test images for your target classes before deploying.
- Use copy-paste augmentation to place objects in unexpected backgrounds during training.
- Add domain-specific data that breaks expected co-occurrences if deployment requires context-independent detection.

---

### Western and indoor scene bias

**Issue:** The dataset skews heavily toward North American and European scenes. Food, vehicles, street scenes, and household objects reflect a specific geographic and cultural context. Generalization to other geographies or environments requires supplementary data.

**Resolution:**
- If your deployment domain differs geographically, add domain-specific training images — even a few hundred well-annotated examples significantly improve generalization.
- Evaluate on deployment-representative data, not only COCO val, before declaring a model production-ready.

---

### Co-occurrence bias

**Issue:** Objects that frequently co-occur in COCO (`person` + `sports ball`, `dining table` + `chair`) may be learned as a bundle. Evaluated in isolation, these objects underperform relative to their COCO AP.

**Resolution:**
- Build a single-object evaluation set for critical classes and measure AP separately from the full co-occurrence context.
- Use aggressive random crop augmentation to force single-object detections during training.

---

## 08. Evaluation Protocol Gotchas

### maxDets cap silently drops predictions

**Issue:** COCO eval caps predictions at 100 per image by default. If your head produces more, the excess is silently dropped before scoring. This significantly hurts reported performance on dense scenes.

**Resolution:**
- Set `maxDets=[1, 10, 300]` in your COCO evaluator — 300 is the standard for modern detectors (RT-DETR, DINO, etc.).
- Log the distribution of predictions-per-image on val. If frequently above 100, the default cap is actively suppressing your reported numbers.

---

### Area thresholds calibrated to original image sizes

**Issue:** Small/medium/large splits use fixed pixel-area cutoffs (small: area < 32², large: area > 96²) calibrated to COCO's original image sizes. If you resize images during preprocessing, these thresholds no longer correspond to the same objects.

**Resolution:**
- Convert area thresholds proportionally to your training resolution, or evaluate on original-resolution images when computing AP_S/AP_M/AP_L.
- When comparing these metrics across papers, verify all experiments used the same input resolution.

---

### ignore=1 regions inflate false positives

**Issue:** Some annotations have `ignore=1` (distinct from `iscrowd`). Predictions overlapping these regions should be neither TP nor FP. Custom mAP implementations frequently skip this, inflating FP counts and depressing precision.

**Resolution:**
- Use the official `pycocotools` evaluator rather than a custom mAP implementation — it handles `ignore` regions correctly.
- If using a custom evaluator, validate it numerically against `pycocotools` on a known subset before trusting its output.
