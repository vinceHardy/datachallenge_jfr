# 2018 JFR Data Challenge: Studies & Results
Repository used for collaborative work for the Data Med Team during the [JFR Data Challenge 2018][web] in Paris.

The team was composed of:
- Dali Ioudarene, CentraleSupelec
- Solene Lefevre, Faculte de Medecine du Kremlin-Bicetre
- Vincent Hardy, CentraleSupelec

The team decided to compete on the liver Data Challenge. Technical choices are summarized in the [.pdf file][relative_path].
Furthermore one can discover here the Notebooks which were used during the Challenge.

1. [Pre-processing scripts][pre]
2. [Training of the CNN][train]
3. [Inference and score computation][inference]

At the end the team got an AUC score of 0.72 on the test set. The winners managers managed to reach 0.89 on the same liver challenge.
Our performance remains quite far from the winners' but we managed to achieve an AUC score which is comparabke to the one a radiologist can have during daily pratices - which is quite encouraging !

[web]: http://jfr.radiologie.fr/les-jfr/villages-et-forum/forum-intelligence-artificielle

[relative_path]: technical_note.pdf

[pre]: https://nbviewer.jupyter.org/github/vinceHardy/datachallenge_jfr/blob/master/preprocessing_liver.ipynb
[train]: https://nbviewer.jupyter.org/github/vinceHardy/datachallenge_jfr/blob/master/train_liver.ipynb
[inference]: https://nbviewer.jupyter.org/github/vinceHardy/datachallenge_jfr/blob/master/predict_and_score_liver.ipynb
