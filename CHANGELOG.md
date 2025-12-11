EyeBench v1.0.0 - 15.12.2025
----------------------------
Initial release of EyeBench

* `NOTES.md` contains information about datasets and models. 

### Datasets
* CopCo
* IITB-HGC
* MECOL2
* OneStop
* PoTeC
* SBSAT

### Tasks
* Reading Comprehension [PoTeC, OneStop, SBSAT]
* Reading Comprehension Skill [CopCo]
* Dyslexia Detection [CopCo]
* Domain Expertise [PoTeC]
* Claim Verification [IITB-HGC]
* Subjective Text Difficulty [SBSAT]
* Vocabulary Knowledge [MECOL2]

### Models
#### Baselines
* Majority Class / Chance
* Reading Speed
* Text-only RoBERTa
#### Traditional Machine Learning
* Logistic Regression
* SVM
* Random Forest
#### Deep Learning
* AhnRNN
* AhnCNN
* BEyeLSTM
* PLM-AS
* PLM-AS-RM
* RoBERTEye-W
* RoBERTEye-F
* MAG-Eye
* PostFusion-Eye

### Evaluation Splits
* Unseen Reader
* Unseen Text
* Unseen Reader & Text