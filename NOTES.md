# Datasets
## CopCo
### Features and Adjustements
* Accounted for non-breaking white spaces in interest areas `\xa0`.
### Filtered
* Excluded practice trials indicated via `paragraph_id=-1` and `speech_id=1327`.
* Excluded participants without `RCS score` for `Reading Comprehension Skill` task.
### Missing values
* `NEXT_SAC_ANGLE` set to 0.
* `NEXT_FIX_ANGLE` set to 0.
* `NEXT_FIX_DISTANCE` set to 0.
* `PREVIOUS_FIX_ANGLE` set to 0.
* `CURRENT_FIX_PUPIL` set to 0.
* `NEXT_SAC_AVG_VELOCITY` set to 0.
## IITB-HGC
### Features and Adjustements
* Since the dataset only published `word indices`, approximated `x` coordinated via `word index`, set and set `y` to 0. 
* To match the `paragraph` and provided interest areas in the fixation report we adjusted:
  * 3: `with Andy` non-breaking whitespace in interest area
  * 5: `watch Jose` non-breaking whitespace in interest area
  * 9: `$1,750` vs `$__NBWS__1,750.00`
  * 25: `£3` vs `£__NBWS__3.00`
  * 26: `for Virgil` non-breaking whitespace in interest area
  * 33: `at FC` non-breaking whitespace in interest area
  * 53: `$5,000` vs `$__NBWS__5,000.00`
  * 74: `$20,000` vs `$__NBWS__20,000.00`
  * 82: `£750,000` vs `£__NBWS__7,50,000.00`
  * 99: `$5.3` vs `$__NBWS__5.30`
  * 102: `$10` vs `$__NBWS__10.00`, and `$9` vs `$__NBWS__9.00`
  * 130: `$50,000` vs `$__NBWS__50,000.00`
  * 257: `$10` vs `$__NBWS__10.00`
  * 280: `Jedinak twisted` vs `Jedinak__NBWS__twisted`
  * 288: `$2` vs `$__NBWS__2.00`
  * 298: `$200` vs `$__NBWS__200.00`
  * 325: `$9` vs `$__NBWS__9.00`
  * 357: `$1.8` vs `$__NBWS__1.80`, and `$2.6` vs `$__NBWS__2.60`
  * 365: `$25,000` vs `$__NBWS__25,000.00`
  * 373: `$260` vs `$__NBWS__260.00`, `$1.7` vs `$__NBWS__1.70`, and `$1.37` vs `$__NBWS__1.37`
  * 403: `hour-long` vs `hour-long hour-long`
  * 404: `$2.9` vs `$__NBWS__2.90`
  * 425: `- many` vs `-__NBWS__many`, `against Leicester City` vs `against__NBWS__Leicester__NBWS__City`, and `van Gaal's` vs `van__NBWS__Gaal's`
  * 441: `$10,000` vs `$__NBWS__10,000.00`
  * 460: `$1.6` vs `$__NBWS__1.60`
  * 468: `$105` vs `$__NBWS__105.00`
  * 483: `(£4,943)` vs `(£__NBWS__4,943.00)`
  * 485: `long-running` vs `long-running long-running`
### Missing values
* `NEXT_SAC_START_X` set to 0
* `NEXT_SAC_END_X` set to 0
* `NEXT_SAC_END_Y` set to 0
* `NEXT_SAC_START_Y` set to 0
* `PREVIOUS_FIX_DISTANCE` set to 0
* `NEXT_SAC_ANGLE` set to 0
* `NEXT_FIX_ANGLE` set to 0
* `NEXT_FIX_DISTANCE` set to 0
* `PREVIOUS_FIX_ANGLE` set to 0
## MECOL2
### Features and Adjustments
* Fixed misalignments for `participant_id` in paragraph id's (original: adjusted)
  * `sp_36` 
    * 11: 12
    * 10: 11
    * 9: 10
    * 8: 9
    * 7: 8
    * 6: 7
    * 5: 6
  * `gr_45`
    * 11: 12
    * 10: 11
    * 9: 10
    * 8: 9
    * 7: 8
  * `it_25`
    * 10: 11
    * 9: 10
    * 7: 8
    * 6: 7
  * `se_38`
    * 11: 12
    * 10: 11
    *  9: 10
    *  8: 9
    *  7: 8
    *  6: 7
    *  5: 6
    *  4: 5
### Filtered
* Excluded participants without Lextale score (label for VK).
## PoTeC
### Filtered
* Excluded participants without background questions.
### Missing values
* `NEXT_SAC_START_X` set to 0
* `NEXT_SAC_START_Y` set to 0
* `NEXT_SAC_END_X` set to 0
* `NEXT_SAC_END_Y` set to 0
* `NEXT_SAC_AVG_VELOCITY` set to 0
* `NEXT_SAC_AMPLITUDE` set to 0
## SBSAT
### Features and Adjustments
* Adjusted encoding inside of areas of interest to match statistical and machine learning models:
  * `\x92': `'`
  * `\x93`: `"`
  * `\x94`: `"`
  * `\x97`: `—`
* Matched paragraphs to areas of interest:
  * `reading-dickens-3`: Sempere & with non-breaking whitespace
  * `reading-dickens-5`: `Mr. Dickens` occupied one area of interest, split into `Mr.` and `Dickens`
  * `reading-flytrap-3`: `Burdon-Sanderson's` split across two lines lines but one area of interest defined by authors. Split into two: `Burdon-` and `Sanderson's`
  * `reading-genome-2`: `species—in` split across two lines but only one area of interest defined by authors. Split into two: `species—` and `in`.
  * `reading-genome-3`: `gee-whiz,` split across two lines but only one area of interest defined by authors. Split into two: `gee-` and `whiz,`.
  * Used OCR to get the text from the original stimuli pictures, afterwards manually corrected.  (@aarbeikop)
### Missing values
* `NEXT_SAC_DURATION` set to 0
* `start_of_line` set to 0
* `end_of_line` set to 0