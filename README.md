# RhythmSense AF Inference Engine

Software item of **RhythmSense AI**, a fictional demo product used to showcase
[P4SaMD](https://mia-care.io). Not a real medical device.

RhythmSense AI is an over-the-counter software function that analyses
photoplethysmography (PPG) recorded by a wrist wearable, identifies episodes of
irregular heart rhythm suggestive of atrial fibrillation (AF) and notifies the
user. FDA device class II (21 CFR 870.2790, product code QDA), IEC 62304
software safety class B.

This service:

- detects systolic peaks and extracts inter-beat intervals (SRS-ALG-001, SRS-ALG-002);
- computes irregularity features (RMSSD, pNN50, CV, Shannon entropy);
- classifies each window with a locked logistic model, version `1.3.0` (SRS-ALG-003);
- refuses to classify low-quality windows (SRS-ALG-005);
- confirms a notification only when 5 of 6 consecutive windows are irregular (SRS-ALG-004).

## Known anomalies

| ID | Description | Status |
|----|-------------|--------|
| RS-ANOM-014 | Unclassifiable windows count as "regular" in the confirmation history, delaying notifications. `test_unclassifiable_windows_are_skipped_by_confirmation` fails until fixed. | Open |

## Build and test

```bash
python -m venv .venv && . .venv/bin/activate
pip install -r requirements-dev.txt
python -m pytest --junitxml=reports/junit.xml
```

CI (`.github/workflows/ci.yml`) publishes the JUnit report as the
`junit-report` artifact, which P4SaMD acquires automatically.
