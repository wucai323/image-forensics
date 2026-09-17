# Test fixtures

## UI betting-detail screenshots

| File | Role | Notes (verified by pixel inspection) |
|------|------|--------------------------------------|
| `ui_bet_edited.png` | Edited / re-exported | 1680×1173 RGBA PNG. Opaque alpha; black-text anti-alias is mostly grayscale (ClearType destroyed) — consistent with Photoshop-style re-export / processing of a UI table screenshot. |
| `ui_bet_original.png` | Untouched screenshot | 1680×797 RGB PNG. Black text retains colorful subpixel (ClearType-like) fringes — native screen-capture rendering. |

Content differs (different periods / rows). Labels follow **rendering / container forensics**, not whether bet amounts look "lucky".

## Mobile bank / Alipay-style receipt JPEGs

User-labeled ground truth (amount / text paste edits on mobile UI screenshots):

| File | Label |
|------|-------|
| `receipt_real_1.jpg` … `receipt_real_3.jpg` | Unedited authentic captures |
| `receipt_fake_1.jpg` … `receipt_fake_3.jpg` | Photoshopped amount / UI text edits |

Discriminative cues used by the engine: local ELA of amount digits vs neighboring flat UI, and chroma-noise wipe in flat panels — **not** desktop ClearType absence (Android/iOS use grayscale AA natively).
