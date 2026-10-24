# 🏠 Bordeaux Real Estate Price Prediction

Predicting property values in Bordeaux using official French DVF open data and Machine Learning.

---

## 🚀 Overview
* **Dataset:** Official DVF transactions (`33.csv`).
* **Target:** `valeur_fonciere` (Price in €).
* **Features:** Surface area (`surface_reelle_bati`), room count, property type (`is_maison`), and GPS coordinates (latitude/longitude).

---

## 📊 Model Performance (MAE)
Comparison of models using Mean Absolute Error (MAE):

| Model | Target Transformation | MAE (Error) |
| :--- | :--- | :--- |
| **Linear Regression** | Log (`np.log1p`) | ~107,940 € |
| **Random Forest** | Log (`np.log1p`) | ~87,267 € |
| **XGBoost (Tuned)** | Log (`np.log1p`) | **~85,500 €** |

---

## 🔬 Key Technical Takeaway: Handling Luxury Bias
* **The Problem:** Real estate data is heavily skewed (many standard homes, few luxury properties). Training on raw prices causes models to ignore high-end properties.
* **The Solution:** Applied a log transformation (`np.log1p` / `np.expm1`) to optimize for **relative error (percentages)** instead of raw euros, successfully correcting the bias on luxury properties.

---

## 💻 Tech Stack
* **Language:** Python
* **Data Processing:** Pandas, NumPy
* **Machine Learning:** Scikit-Learn, XGBoost
* **Visualization:** Matplotlib
