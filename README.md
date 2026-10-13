# Steganalysis Lab: proposed method

Install: `python -m pip install -r requirements.txt`

Run: `streamlit run app.py`
Open the first sidebar option **"1. Proposed Method: Baseline vs Proposed"**.

Command-line evaluation requires at least ten distinct cover images and saves per-cover results to CSV:
`python evaluate_proposed.py cover01.png cover02.png cover03.png cover04.png cover05.png cover06.png cover07.png cover08.png cover09.png cover10.png --repeats 3 --output experiment_results.csv`

## Contribution and scope
Plain LSB replacement writes the message in raster order. The proposed **DIP-guided adaptive LSB** uses local variance, Sobel,
and Laplacian responses, averages their fused energy over 5x5 regions, then applies Otsu thresholding and morphology to choose
textured carriers first. The order is computed from the seven high bits, which embedding leaves unchanged, so extraction can
rebuild it without side information. This is a course-project adaptation of established image-processing operations, not a
claim of a new general steganographic primitive.

The method aims to shift changes into textured regions. It does not guarantee invisibility or resistance to steganalysis. Its
low-risk capacity is image-dependent; excess payload spills into the remaining samples. Password-protected text uses AES-GCM
with a Scrypt-derived key. Legacy XOR payloads can still be decoded, but new payloads are not created with that legacy cipher.

## Evaluation
The CLI requires ten unique image contents and evaluates sequential LSB plus four DIP masks: variance only, Sobel only, Laplacian
only, and all three combined. Otsu thresholding, 5x5 energy averaging, morphology, payload generation, and downstream embedding
remain fixed. It reports PSNR, SSIM, changed samples, changes inside smooth areas as defined by one fixed combined-map reference,
SPAM/LDA detector accuracy, and textured-region capacity from each variant's own map; it saves per-cover/per-payload metrics and
detector repeat statistics to CSV and prints mean +/- standard deviation tables. Use diverse, lossless covers and report the image
count, payload fractions, and repeat count. The fixed smooth-area reference keeps that comparison consistent across variants.

A single-image result is a demonstration, not evidence of general superiority. The program reports descriptive comparisons, not
a statistical-significance claim. Detection accuracy near 50% means this particular detector is near chance; it does not establish
resistance to other detectors. The receiver rebuilds each selected map from `img & 0xFE`; the regression suite verifies matching
embedding/extraction orders for all four feature sets.

## References
- N. Otsu, “A Threshold Selection Method from Gray-Level Histograms,” IEEE Transactions on Systems, Man, and Cybernetics, 1979.
- A. Westfeld and A. Pfitzmann, “Attacks on Steganographic Systems,” Information Hiding, 1999.
- J. Fridrich, M. Goljan, and R. Du, “Detecting LSB Steganography in Color and Gray-Scale Images,” IEEE Multimedia, 2001.

## Files
- `stego_core.py`: engine. New: `dip_stages`, `dip_order`, `secure_capacity_bytes`, `spam_features`, `detector_accuracy`; technique `"dip"`.
- `app.py`: Streamlit GUI. New page `page_proposed`.
- `evaluate_proposed.py`: multi-cover comparison and seeded SPAM/LDA detector evaluation.
- `test_stego_core.py`: standard-library regression tests. Run with `python -m unittest -v`.

## Honest limitations
- Safe capacity = size of the textured region; it depends on the image. Beyond it the advantage shrinks.
- Detector = simple SPAM-style linear classifier, not a state-of-the-art steganalyser.
- LSB replacement is still LSB replacement; stronger detectors (e.g. SRM + ensemble) would detect more.
