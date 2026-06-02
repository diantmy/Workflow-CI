# Telco Customer Churn Prediction - CI/CD Workflow

## Deskripsi Proyek

Proyek ini bertujuan untuk memprediksi kemungkinan pelanggan berhenti berlangganan (churn) menggunakan algoritma Random Forest. Selain proses pelatihan model, proyek ini juga mengimplementasikan Continuous Integration (CI) menggunakan GitHub Actions, MLflow Tracking, dan Docker Hub untuk otomatisasi proses machine learning.

---

## Dataset

Dataset yang digunakan adalah Telco Customer Churn Dataset yang telah melalui tahap preprocessing dan disimpan dalam file:

* `telco_preprocessed.csv`

Target yang diprediksi:

* `Churn`

---

## Algoritma

Model yang digunakan:

* Random Forest Classifier

Hyperparameter tuning dilakukan menggunakan RandomizedSearchCV untuk mendapatkan kombinasi parameter terbaik.

---

## MLflow Tracking

Eksperimen model dicatat menggunakan MLflow Tracking dengan manual logging.

Metrik yang dicatat meliputi:

### Training Metrics

* Accuracy
* Precision
* Recall
* F1 Score
* ROC AUC
* Log Loss

### Test Metrics

* Accuracy
* Precision
* Recall
* F1 Score

Parameter terbaik hasil tuning juga dicatat ke MLflow.

---

## Artifacts

Artifact yang dihasilkan:

* `confusion_matrix.png`
* `classification_report.txt`
* `model.pkl`

Artifact tersebut disimpan dan dapat diakses melalui GitHub Actions maupun MLflow Tracking UI.

---

## DagsHub Integration

Eksperimen MLflow disimpan secara online menggunakan DagsHub sehingga seluruh metrik, parameter, model, dan artifact dapat dipantau secara terpusat.

Repository DagsHub:

https://dagshub.com/dianatmoyo.s/telco-churn-mlops

---

## Continuous Integration (CI)

Workflow CI dibuat menggunakan GitHub Actions.

Trigger workflow:

```yaml
on:
  push:
    branches:
      - main
```

Setiap perubahan yang di-push ke branch `main` akan secara otomatis:

1. Mengunduh source code.
2. Menginstal dependency.
3. Menjalankan training model.
4. Menghasilkan artifact.
5. Mengunggah artifact ke GitHub Actions.
6. Membuat Docker Image.
7. Mengunggah Docker Image ke Docker Hub.

---

## Docker Hub

Docker image dipublikasikan ke Docker Hub dengan repository:

```text
diantmy/telco-churn
```

Image dapat digunakan kembali untuk deployment maupun inferensi model.

---

## Menjalankan Secara Lokal

Install dependency:

```bash
pip install -r requirements.txt
```

Menjalankan training:

```bash
python modelling.py
```

---

## Struktur Proyek

```text
Workflow-CI
│
├── .github
│   └── workflows
│       └── train.yml
│
└── MLProject
    ├── modelling.py
    ├── MLproject
    ├── conda.yaml
    ├── requirements.txt
    ├── Dockerfile
    └── telco_preprocessed.csv
```

---

## Author

Dian Atmoyo
