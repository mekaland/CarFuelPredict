import pandas as pd
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
from scipy.stats import norm, skew
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression, Lasso
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.base import clone
import xgboost as xgb
import streamlit as st
import io
import warnings
warnings.filterwarnings('ignore')

# Streamlit App
st.title("Auto-MPG Regresyon Modeli Arayüzü")
st.write("Bu uygulama, Auto-MPG veri seti üzerinde regresyon modellerini eğitir ve değerlendirir.")

# Veri Yükleme
st.header("1. Veri Yükleme")
uploaded_file = st.file_uploader("CSV dosyasını yükleyin (örneğin, auto-mpg.data)", type=["csv", "txt"])
if uploaded_file is not None:
    # Veri setini yükle
    column_name = ["MPG", "Cylinders", "Displacement", "Horsepower", "Weight", "Acceleration", "Model Year", "Origin"]
    data = pd.read_csv(uploaded_file, names=column_name, na_values="?", comment="\t", sep=" ", skipinitialspace=True)
    data = data.rename(columns={"MPG": "target"})
    
    st.write("### Veri Seti Önizlemesi")
    st.write(data.head())
    st.write(f"Veri boyutu: {data.shape}")
    st.write("### Eksik Değerler")
    st.write(data.isna().sum())

    # Eksik değerlerin işlenmesi
    # Önce sayısal değere dönüştürün, hataları NaN olarak işleyin
    data["Horsepower"] = pd.to_numeric(data["Horsepower"], errors='coerce')
    # Şimdi güvenle ortalama alabilir ve NaN değerleri doldurabilirsiniz
    data["Horsepower"] = data["Horsepower"].fillna(data["Horsepower"].mean())

    # Keşifsel Veri Analizi (EDA)
    st.header("2. Keşifsel Veri Analizi (EDA)")
    st.write("### Korelasyon Matrisi")
    corr_matrix = data.corr()
    fig, ax = plt.subplots()
    sns.heatmap(corr_matrix, annot=True, fmt=".2f", ax=ax)
    plt.title("Correlation btw features")
    st.pyplot(fig)

    threshold = 0.75
    filtre = np.abs(corr_matrix["target"]) > threshold
    corr_features = corr_matrix.columns[filtre].tolist()
    if corr_features:
        st.write(f"### Yüksek Korelasyonlu Özellikler (Threshold: {threshold})")
        fig, ax = plt.subplots()
        sns.heatmap(data[corr_features].corr(), annot=True, fmt=".2f", ax=ax)
        plt.title("Correlation btw high-correlation features")
        st.pyplot(fig)

    st.write("### Box Plotlar")
    for c in data.columns:
        fig, ax = plt.subplots()
        sns.boxplot(x=c, data=data, orient="v", ax=ax)
        plt.title(f"Boxplot of {c}")
        st.pyplot(fig)

    # Aykırı Değerlerin Kaldırılması
    st.header("3. Aykırı Değerlerin Kaldırılması")
    thr = st.slider("Aykırı değer eşiği (threshold)", 1.0, 3.0, 2.0, 0.1)
    describe = data.describe()

    # Horsepower
    horsepower_desc = describe["Horsepower"]
    q3_hp, q1_hp = horsepower_desc[6], horsepower_desc[4]
    IQR_hp = q3_hp - q1_hp
    top_limit_hp = q3_hp + thr * IQR_hp
    bottom_limit_hp = q1_hp - thr * IQR_hp
    filter_hp = (bottom_limit_hp < data["Horsepower"]) & (data["Horsepower"] < top_limit_hp)
    data = data[filter_hp]

    # Acceleration
    acceleration_desc = describe["Acceleration"]
    q3_acc, q1_acc = acceleration_desc[6], acceleration_desc[4]
    IQR_acc = q3_acc - q1_acc
    top_limit_acc = q3_acc + thr * IQR_acc
    bottom_limit_acc = q1_acc - thr * IQR_acc
    filter_acc = (bottom_limit_acc < data["Acceleration"]) & (data["Acceleration"] < top_limit_acc)
    data = data[filter_acc]

    st.write(f"Aykırı değerler kaldırıldıktan sonra veri boyutu: {data.shape}")

    # Özellik Mühendisliği
    st.header("4. Özellik Mühendisliği")
    st.write("### Hedef Değişken (MPG) için Log Dönüşümü")
    col1, col2 = st.columns(2)
    with col1:
        st.write("#### Orijinal Dağılım")
        fig, ax = plt.subplots()
        sns.histplot(data.target, kde=True, stat='density', ax=ax)
        plt.title("Orijinal MPG Dağılımı")
        st.pyplot(fig)
    with col2:
        st.write("#### Q-Q Grafiği (Orijinal)")
        fig, ax = plt.subplots()
        stats.probplot(data["target"], plot=ax)
        plt.title("Orijinal MPG Q-Q Grafiği")
        st.pyplot(fig)

    data["target"] = np.log1p(data["target"])

    col1, col2 = st.columns(2)
    with col1:
        st.write("#### Log Dönüşümü Sonrası Dağılım")
        fig, ax = plt.subplots()
        sns.histplot(data.target, kde=True, stat='density', ax=ax)
        plt.title("Log Dönüşümü MPG Dağılımı")
        st.pyplot(fig)
    with col2:
        st.write("#### Q-Q Grafiği (Log Dönüşümü)")
        fig, ax = plt.subplots()
        stats.probplot(data["target"], plot=ax)
        plt.title("Log Dönüşümü MPG Q-Q Grafiği")
        st.pyplot(fig)

    # One-hot encoding
    data["Cylinders"] = data["Cylinders"].astype(str)
    data["Origin"] = data["Origin"].astype(str)
    data = pd.get_dummies(data, columns=["Cylinders", "Origin"])

    # Veri Bölme ve Standardizasyon
    x = data.drop(["target"], axis=1)
    y = data.target
    test_size = st.slider("Test seti oranı", 0.1, 0.5, 0.2, 0.05)
    X_train, X_test, Y_train, Y_test = train_test_split(x, y, test_size=test_size, random_state=42)
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    # Model Seçimi ve Eğitimi
    st.header("5. Model Seçimi ve Eğitimi")
    model_options = ["Linear Regression", "Lasso", "Random Forest", "Decision Tree", "XGBoost"]
    selected_models = st.multiselect("Eğitilecek modelleri seçin", model_options, default=model_options)

    models = {
        "Linear Regression": LinearRegression(),
        "Lasso": Lasso(random_state=42, max_iter=10000),
        "Random Forest": RandomForestRegressor(random_state=42),
        "Decision Tree": DecisionTreeRegressor(random_state=42),
        "XGBoost": xgb.XGBRegressor(random_state=42)
    }

    # Hiperparametre optimizasyonu
    st.write("### Hiperparametre Optimizasyonu")
    optimize_lasso = st.checkbox("Lasso için GridSearchCV çalıştır", value=False)
    optimize_xgb = st.checkbox("XGBoost için GridSearchCV çalıştır", value=False)

    if optimize_lasso and "Lasso" in selected_models:
        lasso_params = {'alpha': np.logspace(-4, -0.5, 30)}
        lasso_grid = GridSearchCV(models["Lasso"], lasso_params, cv=5, scoring='neg_mean_squared_error')
        lasso_grid.fit(X_train, Y_train)
        models["Lasso"] = lasso_grid.best_estimator_
        st.write(f"Lasso en iyi parametreler: {lasso_grid.best_params_}")

    if optimize_xgb and "XGBoost" in selected_models:
        xgb_params = {
            'nthread': [4], 'objective': ['reg:squarederror'], 'learning_rate': [0.03, 0.05],
            'max_depth': [5, 6], 'min_child_weight': [4], 'subsample': [0.7], 'colsample_bytree': [0.7],
            'n_estimators': [500]
        }
        xgb_grid = GridSearchCV(models["XGBoost"], xgb_params, cv=5, scoring='neg_mean_squared_error')
        xgb_grid.fit(X_train, Y_train)
        models["XGBoost"] = xgb_grid.best_estimator_
        st.write(f"XGBoost en iyi parametreler: {xgb_grid.best_params_}")

    # Model Değerlendirme
    st.header("6. Model Değerlendirme")
    results = []
    if st.button("Modelleri Çalıştır"):
        for name in selected_models:
            model = models[name]
            model.fit(X_train, Y_train)
            y_pred = model.predict(X_test)

            mse = mean_squared_error(Y_test, y_pred)
            r2 = r2_score(Y_test, y_pred)
            cv_scores = cross_val_score(model, X_train, Y_train, cv=5, scoring='neg_mean_squared_error')
            cv_mse = -cv_scores.mean()

            results.append({
                "Model": name,
                "MSE": mse,
                "R²": r2,
                "Cross-Val MSE": cv_mse
            })

            st.write(f"\n### {name}")
            st.write(f"MSE: {mse:.5f}")
            st.write(f"R²: {r2:.5f}")
            st.write(f"Cross-Validation MSE: {cv_mse:.5f}")

            # Feature Importance
            if name in ["Random Forest", "XGBoost"]:
                feature_importance = model.feature_importances_
                feature_names = x.columns
                sorted_idx = np.argsort(feature_importance)[::-1]
                fig, ax = plt.subplots(figsize=(10, 6))
                ax.bar(range(len(feature_importance)), feature_importance[sorted_idx], align="center")
                ax.set_xticks(range(len(feature_importance)))
                ax.set_xticklabels(feature_names[sorted_idx], rotation=45)
                ax.set_title(f"{name} Feature Importance")
                st.pyplot(fig)

            # Gerçek vs. Tahmin
            fig, ax = plt.subplots(figsize=(8, 6))
            ax.scatter(Y_test, y_pred, alpha=0.5)
            ax.plot([Y_test.min(), Y_test.max()], [Y_test.min(), Y_test.max()], 'r--', lw=2)
            ax.set_xlabel("Gerçek MPG (log)")
            ax.set_ylabel("Tahmin Edilen MPG (log)")
            ax.set_title(f"{name} - Gerçek vs. Tahmin")
            st.pyplot(fig)

        # Sonuç Tablosu
        if results:
            st.write("### Model Performans Özeti")
            results_df = pd.DataFrame(results)
            st.dataframe(results_df)

else:
    st.write("Lütfen bir veri seti yükleyin.")
