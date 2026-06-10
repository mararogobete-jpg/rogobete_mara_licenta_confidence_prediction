# Predicția nivelului de încredere metacognitivă folosind Machine Learning

Acest repository conține codul sursă utilizat pentru lucrarea de licență privind predicția nivelului de încredere raportat de participanți pe baza unor indicatori comportamentali extrași din *The Confidence Database*.

Scopul proiectului este construirea unui set de date coerent pentru clasificare și evaluarea mai multor modele de Machine Learning pentru predicția clasei de încredere: încredere scăzută, medie sau ridicată.

## Sursa datelor

Datele utilizate provin din *The Confidence Database* https://osf.io/s46pr/overview (Rahnev et al., 2020).

Fișierele brute din baza de date și fișierul procesat `dataset_main_classification.csv` nu sunt incluse în acest repository din cauza dimensiunii ridicate.

Pentru reproducerea completă a analizei, fișierele brute trebuie descărcate separat și plasate local în folderul:

```text
data/raw/
```

Setul procesat `dataset_main_classification.csv` este generat automat prin rularea scriptului:

```bash
python src/02_preprocesare_dataset.py
```

## Structura proiectului

```text
src/              - scripturile Python utilizate în analiză
data/raw/         - folder pentru fișierele brute din Confidence Database
data/processed/   - folder pentru datasetul procesat generat automat
reports/          - rapoarte CSV și TXT generate de scripturi
figures/          - grafice generate automat
README.md         - descrierea proiectului și instrucțiuni de rulare
requirements.txt  - bibliotecile Python necesare
```

Folderele `data/raw/`, `data/processed/`, `reports/` și `figures/` sunt incluse în repository doar ca structură. Fișierele mari generate local nu sunt încărcate.

## Cerințe software

Proiectul a fost realizat în Python 3.x.

Bibliotecile necesare pot fi instalate folosind:

```bash
pip install -r requirements.txt
```

Bibliotecile principale utilizate sunt:

* pandas
* numpy
* matplotlib
* seaborn
* scipy
* scikit-learn
* statsmodels

## Instrucțiuni de rulare

Scripturile trebuie rulate din folderul principal al proiectului, nu direct din folderul `src/`.

Exemplu:

```bash
python src/01_explorare_date.py
```

Ordinea recomandată de rulare este următoarea:

### 1. Explorarea inițială a bazei de date

```bash
python src/01_explorare_date.py
```

Scriptul inventariază fișierele CSV și README, analizează frecvența coloanelor și verifică tipurile de scale folosite pentru variabila `Confidence`.

```bash
python src/01b_explorare_fisiere_incluse.py
```

Scriptul continuă analiza fișierelor incluse și verifică disponibilitatea predictorilor relevanți.

```bash
python src/01c_analiza_target_predictori.py
```

Scriptul analizează relațiile exploratorii dintre variabila țintă și predictorii candidați.

### 2. Preprocesarea datelor

```bash
python src/02_preprocesare_dataset.py
```

Scriptul construiește datasetul final pentru clasificare, prin:

* normalizarea și recodificarea variabilei `Confidence`;
* filtrarea observațiilor valide;
* curățarea timpilor de reacție;
* transformarea logaritmică și standardizarea pe participant;
* construirea predictorilor derivați;
* salvarea fișierului `dataset_main_classification.csv` în `data/processed/`.

```bash
python src/02a_grafice_distributie_variabile.py
```

Scriptul generează grafice descriptive și matricea de corelație Spearman pentru variabilele principale.

### 3. Modele predictive

```bash
python src/03a_baseline_clasificare.py
```

Evaluează modelul baseline `DummyClassifier`, care prezice clasa majoritară.

```bash
python src/03b2_regresie_logistica_multinomiala_curs.py
```

Estimează modelul MNLogit pentru interpretare statistică, incluzând coeficienți, valori p, odds ratio și VIF.

```bash
python src/03b3_mnlogit_groupkfold.py
```

Evaluează modelul MNLogit out-of-sample prin `GroupKFold`, folosind `uid_participant` ca variabilă de grupare.

```bash
python src/03c_decision_tree.py
```

Antrenează și evaluează modelul `DecisionTreeClassifier`.

```bash
python src/03c2_tuning_decision_tree.py
```

Realizează optimizarea hiperparametrilor pentru arborele de decizie prin nested `GroupKFold`.

```bash
python src/03d_knn.py
```

Antrenează și evaluează modelul K-Nearest Neighbors. Pentru reducerea costului computațional, tuning-ul este realizat pe un eșantion de participanți.

```bash
python src/03e_svm.py
```

Antrenează și evaluează modelul `LinearSVC`.

### 4. Compararea finală a modelelor

```bash
python src/04_comparatie_finala_metrici.py
```

Construiește tabelul final de comparație între modelele evaluate, pe baza matricilor de confuzie generate anterior.

```bash
python src/04_roc_auc_modele.py
```

Generează analiza ROC-AUC multiclasă folosind abordarea One-vs-Rest.

## Observații privind reproducerea rezultatelor

Rularea completă a tuturor scripturilor poate necesita timp ridicat, deoarece setul de date procesat conține peste un milion de observații.

Fișierul `dataset_main_classification.csv` nu este inclus în repository, dar poate fi regenerat local prin rularea scriptului `02_preprocesare_dataset.py`, după plasarea fișierelor brute în `data/raw/`.

Rezultatele generate de scripturi sunt salvate automat în folderele:

```text
reports/
figures/
```

## Modele evaluate

Modelele utilizate în analiza finală sunt:

* `DummyClassifier` — model baseline;
* `MNLogit` — model statistic interpretativ și predictiv out-of-sample;
* `DecisionTreeClassifier` — model non-liniar interpretabil;
* `KNeighborsClassifier` — model bazat pe similaritatea dintre observații;
* `LinearSVC` — model liniar scalabil pentru clasificare multiclasă.

## Autor

Rogobete Mara

Lucrare de licență, specializarea Informatică Economică.
