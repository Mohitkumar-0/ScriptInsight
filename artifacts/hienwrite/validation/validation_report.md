# HiEnWrite validation report

Generated 2026-10-22T20:00:12+00:00 from `/Users/amitkumaryadav/Downloads/HiEnWrite-Dataset-main` (CSV sha256 `0fca5b5ef0972478`).

| Count | Value |
|---|---|
| csv rows | 690 |
| unique image references | 690 |
| image files on disk | 660 |
| csv rows matched to a file | 660 |
| missing image files | 30 |
| files on disk not in csv | 0 |
| unreadable images | 8 |
| too small images | 1 |
| duplicate images excluded | 73 |
| usable samples | 578 |
| participants in csv | 345 |
| participants usable | 303 |
| groups usable | 295 |
| usable by language | English: 298, Hindi: 280 |

## Exclusions

| Reason | Rows |
|---|---|
| duplicate_image | 5 |
| duplicate_image_conflicting_labels | 68 |
| missing_file | 30 |
| too_small(<128px) | 1 |
| unreadable_image:corrupt_image | 1 |
| unreadable_image:empty_file | 7 |

Excluded images: `24e.jpeg` (missing_file), `24h.jpeg` (missing_file), `26e.jpeg` (missing_file), `26h.jpeg` (missing_file), `41e.jpeg` (missing_file), `41h.jpeg` (missing_file), `52h.jpeg` (unreadable_image:corrupt_image), `57h.jpeg` (duplicate_image_conflicting_labels), `60e.jpeg` (unreadable_image:empty_file), `81h.jpeg` (duplicate_image_conflicting_labels), `82h.jpeg` (duplicate_image_conflicting_labels), `95h.jpeg` (missing_file), `115h.jpeg` (missing_file), `118h.jpeg` (missing_file), `119e.jpeg` (unreadable_image:empty_file), `119h.jpeg` (duplicate_image_conflicting_labels), `120e.jpeg` (missing_file), `120h.jpeg` (missing_file), `123e.jpeg` (duplicate_image_conflicting_labels), `123h.jpeg` (unreadable_image:empty_file), `125e.jpeg` (missing_file), `125h.jpeg` (missing_file), `134h.jpeg` (unreadable_image:empty_file), `136e.jpeg` (duplicate_image_conflicting_labels), `136h.jpeg` (duplicate_image_conflicting_labels), `137e.jpeg` (duplicate_image_conflicting_labels), `137h.jpeg` (duplicate_image_conflicting_labels), `138e.jpeg` (duplicate_image_conflicting_labels), `138h.jpeg` (duplicate_image_conflicting_labels), `139e.jpeg` (duplicate_image_conflicting_labels), `139h.jpeg` (duplicate_image_conflicting_labels), `142h.jpeg` (duplicate_image_conflicting_labels), `144h.jpeg` (duplicate_image), `150h.jpeg` (duplicate_image_conflicting_labels), `151e.jpeg` (unreadable_image:empty_file), `151h.jpeg` (missing_file), `160e.jpeg` (missing_file), `160h.jpeg` (missing_file), `162e.jpeg` (missing_file), `162h.jpeg` (missing_file), `164h.jpeg` (missing_file), `170e.jpeg` (duplicate_image_conflicting_labels), `170h.jpeg` (duplicate_image_conflicting_labels), `172e.jpeg` (duplicate_image), `172h.jpeg` (duplicate_image), `176h.jpeg` (too_small(<128px)), `179e.jpeg` (missing_file), `191e.jpeg` (missing_file), `191h.jpeg` (missing_file), `201h.jpeg` (duplicate_image_conflicting_labels), `219e.jpeg` (missing_file), `219h.jpeg` (missing_file), `226h.jpeg` (duplicate_image_conflicting_labels), `231e.jpeg` (duplicate_image_conflicting_labels), `231h.jpeg` (duplicate_image_conflicting_labels), `232e.jpeg` (duplicate_image_conflicting_labels), `232h.jpeg` (duplicate_image_conflicting_labels), `236e.jpeg` (duplicate_image_conflicting_labels), `236h.jpeg` (duplicate_image_conflicting_labels), `239e.jpeg` (duplicate_image_conflicting_labels), `239h.jpeg` (duplicate_image_conflicting_labels), `248e.jpeg` (duplicate_image_conflicting_labels), `248h.jpeg` (duplicate_image_conflicting_labels), `250e.jpeg` (duplicate_image_conflicting_labels), `250h.jpeg` (duplicate_image_conflicting_labels), `255e.jpeg` (duplicate_image_conflicting_labels), `255h.jpeg` (duplicate_image_conflicting_labels), `263e.jpeg` (duplicate_image_conflicting_labels), `263h.jpeg` (duplicate_image_conflicting_labels), `264h.jpeg` (duplicate_image_conflicting_labels), `268e.jpeg` (missing_file), `268h.jpeg` (missing_file), `272e.jpeg` (missing_file), `272h.jpeg` (missing_file), `277h.jpeg` (duplicate_image_conflicting_labels), `279e.jpeg` (duplicate_image_conflicting_labels), `279h.jpeg` (duplicate_image_conflicting_labels), `283e.jpeg` (duplicate_image_conflicting_labels), `283h.jpeg` (duplicate_image_conflicting_labels), `291e.jpeg` (unreadable_image:empty_file), `295h.jpeg` (duplicate_image), `296h.jpeg` (duplicate_image_conflicting_labels), `297e.jpeg` (duplicate_image_conflicting_labels), `297h.jpeg` (duplicate_image_conflicting_labels), `310e.jpeg` (duplicate_image_conflicting_labels), `310h.jpeg` (duplicate_image_conflicting_labels), `316h.jpeg` (duplicate_image_conflicting_labels), `322e.jpeg` (duplicate_image_conflicting_labels), `322h.jpeg` (duplicate_image_conflicting_labels), `333e.jpeg` (duplicate_image_conflicting_labels), `333h.jpeg` (duplicate_image_conflicting_labels), `337h.jpeg` (duplicate_image), `338e.jpeg` (duplicate_image_conflicting_labels), `338h.jpeg` (duplicate_image_conflicting_labels), `341e.jpeg` (duplicate_image_conflicting_labels), `341h.jpeg` (duplicate_image_conflicting_labels), `348e.jpeg` (unreadable_image:empty_file), `354e.jpeg` (duplicate_image_conflicting_labels), `354h.jpeg` (duplicate_image_conflicting_labels), `361h.jpeg` (duplicate_image_conflicting_labels), `363e.jpeg` (duplicate_image_conflicting_labels), `369h.jpeg` (duplicate_image_conflicting_labels), `372e.jpeg` (missing_file), `372h.jpeg` (missing_file), `386e.jpeg` (duplicate_image_conflicting_labels), `386h.jpeg` (duplicate_image_conflicting_labels), `387e.jpeg` (duplicate_image_conflicting_labels), `387h.jpeg` (duplicate_image_conflicting_labels), `402e.jpeg` (duplicate_image_conflicting_labels), `402h.jpeg` (duplicate_image_conflicting_labels), `403e.jpeg` (duplicate_image_conflicting_labels), `403h.jpeg` (duplicate_image_conflicting_labels)

## Masked targets

- **Openness** (`Openness to Experience`) is 24 for all 345 Hindi rows; treated as missing for those rows.

## Writer identity

inferred participant = numeric prefix of the image id (<number><e|h>.jpeg), merged through shared exact / near-duplicate images.

Evidence: {"participants_with_two_rows": 345, "pairs_identical_on_E_A_C_N_age_gender": 345, "paper_abstract": "handwritten samples collected from over 400 authors; approximately 800 images"}

Writer identity is not documented in the CSV or readme. If one person contributed under several numbers, their samples could fall in different splits.

## Duplicate images

Clusters of byte-identical files or near-identical photographs (dHash <= 6 of 256 bits; 40 such pairs). Participants that share a cluster are also merged into one split group.

- 57h.jpeg, 142h.jpeg, 296h.jpeg (incl. near-identical): labels CONFLICT (all excluded)
- 81h.jpeg, 82h.jpeg, 277h.jpeg (incl. near-identical): labels CONFLICT (all excluded)
- 119h.jpeg, 150h.jpeg (byte-identical): labels CONFLICT (all excluded)
- 123e.jpeg, 363e.jpeg (byte-identical): labels CONFLICT (all excluded)
- 136e.jpeg, 137e.jpeg, 138e.jpeg, 139e.jpeg (byte-identical): labels CONFLICT (all excluded)
- 136h.jpeg, 137h.jpeg, 138h.jpeg, 139h.jpeg (byte-identical): labels CONFLICT (all excluded)
- 144e.jpeg, 144h.jpeg (byte-identical): labels agree (kept 144e.jpeg)
- 146e.jpeg, 172e.jpeg (byte-identical): labels agree (kept 146e.jpeg)
- 146h.jpeg, 172h.jpeg (byte-identical): labels agree (kept 146h.jpeg)
- 157h.jpeg, 295h.jpeg (byte-identical): labels agree (kept 157h.jpeg)
- 170e.jpeg, 322e.jpeg (byte-identical): labels CONFLICT (all excluded)
- 170h.jpeg, 322h.jpeg (byte-identical): labels CONFLICT (all excluded)
- 201h.jpeg, 226h.jpeg (byte-identical): labels CONFLICT (all excluded)
- 231e.jpeg, 232e.jpeg, 236e.jpeg, 239e.jpeg, 248e.jpeg, 250e.jpeg, 283e.jpeg, 297e.jpeg, 297h.jpeg, 310e.jpeg (incl. near-identical): labels CONFLICT (all excluded)
- 231h.jpeg, 232h.jpeg, 236h.jpeg, 239h.jpeg, 248h.jpeg, 250h.jpeg, 263e.jpeg, 263h.jpeg, 264h.jpeg, 283h.jpeg, 310h.jpeg, 369h.jpeg (incl. near-identical): labels CONFLICT (all excluded)
- 255e.jpeg, 333e.jpeg (incl. near-identical): labels CONFLICT (all excluded)
- 255h.jpeg, 333h.jpeg (incl. near-identical): labels CONFLICT (all excluded)
- 279e.jpeg, 341e.jpeg (byte-identical): labels CONFLICT (all excluded)
- 279h.jpeg, 341h.jpeg (byte-identical): labels CONFLICT (all excluded)
- 316h.jpeg, 361h.jpeg (incl. near-identical): labels CONFLICT (all excluded)
- 337e.jpeg, 337h.jpeg (byte-identical): labels agree (kept 337e.jpeg)
- 338e.jpeg, 354e.jpeg (incl. near-identical): labels CONFLICT (all excluded)
- 338h.jpeg, 354h.jpeg (incl. near-identical): labels CONFLICT (all excluded)
- 386e.jpeg, 387e.jpeg (byte-identical): labels CONFLICT (all excluded)
- 386h.jpeg, 387h.jpeg (byte-identical): labels CONFLICT (all excluded)
- 402e.jpeg, 403e.jpeg (byte-identical): labels CONFLICT (all excluded)
- 402h.jpeg, 403h.jpeg (byte-identical): labels CONFLICT (all excluded)

## Target distributions (usable rows, valid values only)

| Target | n | mean | std | min | median | max |
|---|---|---|---|---|---|---|
| Extraversion | 578 | 19.36 | 6.606 | 0 | 20 | 40 |
| Agreeableness | 578 | 26.37 | 4.741 | 11 | 27 | 40 |
| Conscientiousness | 578 | 24.391 | 6.023 | 7 | 24 | 40 |
| Neuroticism | 578 | 20.715 | 6.131 | 3 | 20 | 37 |
| Openness | 298 | 25.534 | 5.394 | 8 | 25 | 40 |
