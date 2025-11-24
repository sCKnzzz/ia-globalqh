
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report
import warnings
warnings.filterwarnings('ignore')

# ================================================================
# 0. FUNCIONES AUXILIARES
# ================================================================

def func_pot(x, a, b):
    return a * x**b

def func_poly2(x, a, b, c):
    return a * x**2 + b * x + c

def func_poly3(x, a, b, c, d):
    return a * x**3 + b * x**2 + c * x + d

def calculate_r2(y_obs, y_pred):
    ss_tot = np.sum((y_obs - np.mean(y_obs))**2)
    ss_res = np.sum((y_obs - y_pred)**2)
    return 1 - (ss_res / ss_tot) if ss_tot != 0 else 1

def entrenar_modelo_grupo(df_grupo):
    """
    Ajusta Polinómico G2, Polinómico G3 y Potencial y elige el mejor según R².
    Devuelve un diccionario con información del mejor modelo o None.
    """
    if len(df_grupo) < 3:
        return None
    
    H = df_grupo['NIVEL DE AFORO (m)'].values
    Q = df_grupo['CAUDAL (m3/s)'].values
    
    # Ordenar por nivel
    idx = np.argsort(H)
    H_sorted = H[idx]
    Q_sorted = Q[idx]
    
    best_r2 = -np.inf
    best_model_name = None
    best_params = None
    best_func = None
    
    modelos = [
        ("Polinómico G2", func_poly2),
        ("Polinómico G3", func_poly3),
        ("Potencial",    func_pot),
    ]
    
    for nombre, f in modelos:
        try:
            if nombre in ["Polinómico G2", "Polinómico G3"]:
                params, _ = curve_fit(f, H_sorted, Q_sorted, maxfev=5000)
            else:  # Potencial
                p0 = [1.0, 2.0]
                params, _ = curve_fit(f, H_sorted, Q_sorted, p0=p0, maxfev=5000)
            
            Q_pred = f(H_sorted, *params)
            r2 = calculate_r2(Q_sorted, Q_pred)
            
            min_r2 = 0.7 if len(df_grupo) < 10 else 0.8
            if r2 > best_r2 and r2 > min_r2:
                best_r2 = r2
                best_model_name = nombre
                best_params = params
                best_func = f
        except Exception:
            continue
    
    if best_model_name is None:
        return None
    
    # Armar ecuación en texto
    if best_model_name == "Potencial":
        ecuacion = f"Q = {best_params[0]:.3f}·H^{best_params[1]:.3f}"
    elif best_model_name == "Polinómico G2":
        ecuacion = (f"Q = {best_params[0]:.3f}·H² "
                    f"+ {best_params[1]:.3f}·H "
                    f"+ {best_params[2]:.3f}")
    else:  # G3
        ecuacion = (f"Q = {best_params[0]:.3f}·H³ "
                    f"+ {best_params[1]:.3f}·H² "
                    f"+ {best_params[2]:.3f}·H "
                    f"+ {best_params[3]:.3f}")
    
    info = {
        "modelo": best_model_name,
        "ecuacion": ecuacion,
        "r2": best_r2,
        "n": len(df_grupo),
        "params": best_params,
        "func": best_func,
        "H_min": float(H_sorted.min()),
        "H_max": float(H_sorted.max()),
        "Q_min": float(Q_sorted.min()),
        "Q_max": float(Q_sorted.max()),
    }
    return info


class SistemaInteligenteMultiCaso:
    """
    - Clasifica BAJO / ALTO
    - Dentro de BAJO define subgrupo (2025 / alto RH / bajo RH)
    - Usa el modelo de curva correspondiente para estimar Q(H)
    """
    def __init__(self, clasificador, escalador, modelos_bajos, modelos_altos, features):
        self.clf = clasificador
        self.scaler = escalador
        self.modelos_bajos = modelos_bajos
        self.modelos_altos = modelos_altos
        self.features = features
    
    def predecir_tipo_rio(self, df_nuevos):
        X = df_nuevos[self.features]
        Xs = self.scaler.transform(X)
        return self.clf.predict(Xs)
    
    def predecir_caudal(self, df_nuevos):
        tipos = self.predecir_tipo_rio(df_nuevos)
        resultados = []
        
        for i, (_, fila) in enumerate(df_nuevos.iterrows()):
            tipo = tipos[i]
            H = fila["NIVEL DE AFORO (m)"]
            
            if tipo == "BAJO":
                rh = fila["RADIO HIDRAULICO (m)"]
                year = fila["YEAR"]
                
                if year == 2025:
                    sub = "GRUPO_2025"
                elif rh > 0.6:
                    sub = "GRUPO_ALTO_RH"
                else:
                    sub = "GRUPO_BAJO_RH"
                
                if sub in self.modelos_bajos:
                    m = self.modelos_bajos[sub]
                    Qp = m["func"](H, *m["params"])
                    modelo_usado = f"BAJO_{sub}"
                    r2 = m["r2"]
                else:
                    Qp = None
                    modelo_usado = "Sin modelo disponible"
                    r2 = None
            
            else:  # ALTO
                if self.modelos_altos:
                    m = list(self.modelos_altos.values())[0]
                    Qp = m["func"](H, *m["params"])
                    modelo_usado = "ALTO_PRINCIPAL"
                    r2 = m["r2"]
                else:
                    Qp = None
                    modelo_usado = "Sin modelo disponible"
                    r2 = None
            
            resultados.append({
                "NIVEL_DE_AFORO (m)": H,
                "TIPO_PREDICHO": tipo,
                "CAUDAL_PREDICHO (m3/s)": Qp,
                "MODELO_USADO": modelo_usado,
                "R2_MODELO": r2
            })
        
        return pd.DataFrame(resultados)

# ================================================================
# 1. CARGA Y PREPARACIÓN DE DATOS
# ================================================================

def cargar_datos():
    # --------- Datos caudales bajos ---------
    data_original = {
        'FECHA': ['2/10/2021', '2/24/2021', '4/13/2021', '5/11/2021', '9/26/2021', '11/25/2021', '3/10/2022', '4/6/2022',
                  '10/14/2022', '10/27/2022', '11/8/2022', '12/10/2022', '2/8/2023', '2/24/2023', '3/6/2023', '4/27/2023',
                  '10/26/2023', '11/9/2023', '2/15/2024', '2/22/2024', '3/1/2024', '3/7/2024', '6/22/2024', '8/16/2024',
                  '8/21/2024', '11/21/2024', '2/7/2025', '2/13/2025', '2/22/2025', '3/7/2025', '3/15/2025', '3/20/2025',
                  '4/2/2025', '4/9/2025'],
        'CAUDAL (m3/s)': [4.79, 1.89, 2.15, 1.11, 1.1, 1.36, 10.38, 1.22, 1.44, 1.71, 1.74, 3.03, 8.22, 2.56, 1.97, 1.84,
                          1.3, 1.12, 7.23, 7.69, 16.86, 12.38, 2.14, 1.72, 1.73, 1.18, 8.18, 15.36, 22.53, 13.08, 17.2, 4.79,
                          2.72, 3.98],
        'VELOCIDAD (m/s)': [1.38, 0.62, 0.74, 0.46, 0.48, 0.56, 2.19, 0.7, 0.62, 0.66, 0.65, 0.86, 0.83, 0.71, 0.63, 0.48,
                            0.49, 0.41, 1.04, 0.88, 1.09, 1.0, 0.54, 0.59, 0.63, 0.49, 0.71, 1.16, 1.38, 1.05, 1.22, 1.01,
                            0.72, 0.9],
        'AREA (m2)': [3.47, 2.76, 2.64, 2.07, 1.97, 2.07, 4.58, 1.64, 2.25, 2.47, 2.6, 3.4, 9.42, 3.43, 2.96, 3.77, 2.49,
                      2.32, 6.72, 8.67, 15.41, 12.41, 3.32, 2.78, 2.67, 2.27, 11.6, 13.02, 16.1, 12.26, 13.91, 4.42, 3.61, 4.39],
        'PERIMETRO (m)': [8.57, 8.57, 8.34, 7.54, 7.55, 8.35, 9.69, 7.02, 9.05, 9.35, 9.05, 9.36, 15.06, 10.58, 10.61, 13.42,
                          9.84, 9.86, 13.08, 13.08, 13.57, 13.06, 13.13, 11.56, 10.6, 11.06, 11.07, 11.09, 11.13, 11.12, 11.13,
                          10.2, 9.61, 8.67],
        'ANCHO RIO (m)': [8.5, 8.5, 8.3, 7.5, 7.5, 8.3, 9.5, 7.0, 9.0, 9.3, 9.0, 9.3, 15.0, 10.5, 10.5, 13.3, 9.8, 9.8, 13.0,
                          13.0, 13.0, 13.0, 13.0, 11.5, 10.5, 11.0, 11.0, 11.0, 11.0, 11.0, 11.0, 10.0, 9.5, 8.5],
        'NIVEL DE AFORO (m)': [1.2, 1.01, 1.05, 0.97, 0.96, 0.92, 1.38, 0.97, 0.96, 0.97, 0.98, 1.06, 1.38, 1.01, 0.94, 1.01,
                               0.96, 0.94, 1.41, 1.46, 1.92, 1.7, 0.87, 0.85, 0.86, 0.82, 1.1, 1.35, 1.6, 1.4, 1.18, 1.15,
                               0.97, 1.03]
    }

    df_bajos = pd.DataFrame(data_original)
    df_bajos["YEAR"] = pd.to_datetime(df_bajos["FECHA"]).dt.year
    df_bajos["RADIO HIDRAULICO (m)"] = df_bajos["AREA (m2)"] / df_bajos["PERIMETRO (m)"]
    df_bajos["TIRANTE MEDIO (m)"] = df_bajos["AREA (m2)"] / df_bajos["ANCHO RIO (m)"]
    df_bajos["CAUDAL/AREA"] = df_bajos["CAUDAL (m3/s)"] / df_bajos["AREA (m2)"]
    df_bajos["TIPO_RIO"] = "BAJO"

    # --------- Datos caudales altos ---------
    data_altos = {
        'FECHA': ['9/28/2021', '10/29/2019', '9/24/2019', '8/28/2019', '7/23/2019', '4/17/2019', 
                  '11/19/2019', '3/24/2021', '11/18/2021', '6/27/2019', '3/23/2022', '3/27/2025', '7/26/2024'],
        'CAUDAL (m3/s)': [312, 639.56, 806.02, 295.03, 392.67, 997.66, 786.09, 2026.2, 372.5, 300.26, 1521.1, 887.877, 197.791],
        'VELOCIDAD (m/s)': [0.56, 1.13, 1.37, 0.58, 0.71, 1.55, 1.17, 2.45, 0.71, 0.58, 2.19, 1.367, 0.402],
        'AREA (m2)': [521.6, 579.4, 595.7, 505.7, 549.6, 670.6, 677.4, 839.4, 507.2, 516.6, 708.5, 649.65, 475.08],
        'PERIMETRO (m)': [108.195, 115.0399, 116.59, 113.5599, 115.88, 122.17, 116.47, 138.2, 113.8, 110.86, 127.36,
                         111.63, 103.91],
        'ANCHO RIO (m)': [8.42, 9.18, 9.34, 8.5, 8.72, 9.78, 9.5, 10.56, 8.68, 8.5, 10.19, 9.8, 8.29],
        # Ojo: aquí NIVEL DE AFORO está igual que ANCHO (según tus datos originales)
        'NIVEL DE AFORO (m)': [8.42, 9.18, 9.34, 8.5, 8.72, 9.78, 9.5, 10.56, 8.68, 8.5, 10.19, 9.8, 8.29]
    }

    df_altos = pd.DataFrame(data_altos)
    df_altos["YEAR"] = pd.to_datetime(df_altos["FECHA"]).dt.year
    df_altos["RADIO HIDRAULICO (m)"] = df_altos["AREA (m2)"] / df_altos["PERIMETRO (m)"]
    df_altos["TIRANTE MEDIO (m)"] = df_altos["AREA (m2)"] / df_altos["ANCHO RIO (m)"]
    df_altos["CAUDAL/AREA"] = df_altos["CAUDAL (m3/s)"] / df_altos["AREA (m2)"]
    df_altos["TIPO_RIO"] = "ALTO"

    return df_bajos, df_altos


# ================================================================
# 2. ENTRENAR CLASIFICADOR Y MODELOS
# ================================================================

def entrenar_sistema(df_bajos, df_altos):
    df_comb = pd.concat([df_bajos, df_altos], ignore_index=True)

    features = [
        'NIVEL DE AFORO (m)', 'ANCHO RIO (m)', 'PERIMETRO (m)',
        'AREA (m2)', 'VELOCIDAD (m/s)', 'RADIO HIDRAULICO (m)',
        'TIRANTE MEDIO (m)', 'CAUDAL/AREA', 'YEAR'
    ]
    X = df_comb[features]
    y = df_comb["TIPO_RIO"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)

    clf = RandomForestClassifier(
        n_estimators=100,
        max_depth=6,
        min_samples_split=4,
        min_samples_leaf=2,
        random_state=42
    )
    clf.fit(X_train_s, y_train)

    y_pred = clf.predict(X_test_s)
    acc_train = clf.score(X_train_s, y_train)
    acc_test = clf.score(X_test_s, y_test)
    cv_scores = cross_val_score(clf, X_train_s, y_train, cv=5)

    # Reporte de clasificación
    report = classification_report(y_test, y_pred, output_dict=False)

    # Importancia de variables
    fi = pd.DataFrame({
        "feature": features,
        "importance": clf.feature_importances_
    }).sort_values("importance", ascending=False)

    # ---------------- Modelos por subgrupo (BAJO) ----------------
    df_bajos = df_bajos_raw.copy()
    df_bajos["GRUPO_MANUAL"] = df_bajos.apply(
        lambda row: "GRUPO_2025" if row["YEAR"] == 2025 else
                    "GRUPO_ALTO_RH" if row["RADIO HIDRAULICO (m)"] > 0.6 else
                    "GRUPO_BAJO_RH",
        axis=1
    )

    modelos_bajos = {}
    for g in df_bajos["GRUPO_MANUAL"].unique():
        sub = df_bajos[df_bajos["GRUPO_MANUAL"] == g]
        modelo_info = entrenar_modelo_grupo(sub)
        if modelo_info:
            modelos_bajos[g] = modelo_info

    # ---------------- Modelo para ALTOS ----------------
    modelos_altos = {}
    modelo_alto = entrenar_modelo_grupo(df_altos)
    if modelo_alto:
        modelos_altos["ALTO_PRINCIPAL"] = modelo_alto

    sistema = SistemaInteligenteMultiCaso(
        clasificador=clf,
        escalador=scaler,
        modelos_bajos=modelos_bajos,
        modelos_altos=modelos_altos,
        features=features
    )

    info = {
        "acc_train": acc_train,
        "acc_test": acc_test,
        "cv_mean": cv_scores.mean(),
        "cv_std": cv_scores.std(),
        "feature_importance": fi,
        "classification_report": report,
        "modelos_bajos": modelos_bajos,
        "modelos_altos": modelos_altos,
        "features": features
    }

    return sistema, info


# ================================================================
# 3. INTERFAZ STREAMLIT
# ================================================================

def main():
    st.title("💧 Sistema inteligente multi-caso: curvas altura–caudal")
    st.write(
        "App demo para un **sistema híbrido** que:\n\n"
        "- Clasifica si un aforo corresponde a **caudal BAJO** o **ALTO**.\n"
        "- Usa diferentes curvas altura–caudal según el grupo.\n"
        "- Permite probar un nuevo aforo y obtener un caudal estimado."
    )

    # ---------- Cargar y entrenar ----------
    global df_bajos_raw
    df_bajos_raw, df_altos = cargar_datos()
    sistema, info = entrenar_sistema(df_bajos_raw, df_altos)

    st.sidebar.header("Información rápida")
    st.sidebar.metric("Aforos BAJOS", len(df_bajos_raw))
    st.sidebar.metric("Aforos ALTOS", len(df_altos))
    st.sidebar.write(f"Precisión prueba clasificador: **{info['acc_test']:.3f}**")
    st.sidebar.write(f"CV (5-fold): **{info['cv_mean']:.3f} ± {info['cv_std']:.3f}**")

    # --------- Sección: resultados del entrenamiento ---------
    st.subheader("📊 Resultados del clasificador BAJO vs ALTO")
    col1, col2, col3 = st.columns(3)
    col1.metric("Precisión entrenamiento", f"{info['acc_train']:.3f}")
    col2.metric("Precisión prueba", f"{info['acc_test']:.3f}")
    col3.metric("CV (5-fold)", f"{info['cv_mean']:.3f} ± {info['cv_std']:.3f}")

    st.write("**Importancia de características:**")
    st.dataframe(info["feature_importance"])

    st.write("**Modelos por subgrupos (caudales bajos):**")
    if info["modelos_bajos"]:
        for g, m in info["modelos_bajos"].items():
            st.markdown(
                f"- **{g}**: {m['modelo']} (R² = {m['r2']:.3f})  \n"
                f"  Ecuación: `{m['ecuacion']}`  \n"
                f"  Rango H: {m['H_min']:.2f}–{m['H_max']:.2f} m, "
                f"Rango Q: {m['Q_min']:.1f}–{m['Q_max']:.1f} m³/s"
            )
    else:
        st.info("No se logró ajustar modelos confiables para caudales bajos.")

    st.write("**Modelo caudales altos:**")
    if info["modelos_altos"]:
        m = list(info["modelos_altos"].values())[0]
        st.markdown(
            f"- {m['modelo']} (R² = {m['r2']:.3f})  \n"
            f"  Ecuación: `{m['ecuacion']}`  \n"
            f"  Rango H: {m['H_min']:.2f}–{m['H_max']:.2f} m, "
            f"Rango Q: {m['Q_min']:.1f}–{m['Q_max']:.1f} m³/s"
        )
    else:
        st.info("No se logró ajustar modelo confiable para caudales altos.")

    # --------- Formulario: nuevo aforo ---------
    st.subheader("🔮 Probar con un nuevo aforo")

    with st.form("nuevo_aforo"):
        st.markdown("Introduce los datos hidráulicos del aforo a estimar:")

        c1, c2, c3 = st.columns(3)
        with c1:
            H = st.number_input("Nivel de aforo H (m)", value=1.20, step=0.01)
            ancho = st.number_input("Ancho del río (m)", value=11.0, step=0.1)
            perim = st.number_input("Perímetro mojado (m)", value=11.5, step=0.1)
        with c2:
            area = st.number_input("Área mojada (m²)", value=12.0, step=0.1)
            vel = st.number_input("Velocidad (m/s)", value=1.1, step=0.01)
            year = st.number_input("Año", value=2025, step=1)
        with c3:
            # Derivados aproximados
            rh = area / perim if perim != 0 else 0.0
            tirante = area / ancho if ancho != 0 else 0.0
            caudal_area = st.number_input("Caudal/Área (m³/s / m²)", value=0.92, step=0.01)

        st.caption(
            f"Radio hidráulico calculado: **{rh:.3f} m**  |  "
            f"Tirante medio calculado: **{tirante:.3f} m**"
        )

        enviado = st.form_submit_button("Calcular tipo de río y caudal")

    if enviado:
        df_nuevo = pd.DataFrame([{
            "NIVEL DE AFORO (m)": H,
            "ANCHO RIO (m)": ancho,
            "PERIMETRO (m)": perim,
            "AREA (m2)": area,
            "VELOCIDAD (m/s)": vel,
            "RADIO HIDRAULICO (m)": rh,
            "TIRANTE MEDIO (m)": tirante,
            "CAUDAL/AREA": caudal_area,
            "YEAR": int(year)
        }])

        resultado = sistema.predecir_caudal(df_nuevo)
        st.write("### Resultado de la predicción")
        st.dataframe(resultado)

        tipo = resultado.loc[0, "TIPO_PREDICHO"]
        Qp = resultado.loc[0, "CAUDAL_PREDICHO (m3/s)"]
        modelo_usado = resultado.loc[0, "MODELO_USADO"]
        r2 = resultado.loc[0, "R2_MODELO"]

        if pd.notna(Qp):
            st.success(
                f"Tipo de río predicho: **{tipo}**  \n"
                f"Caudal estimado: **{Qp:.2f} m³/s**  \n"
                f"Modelo usado: **{modelo_usado}** (R² ≈ {r2:.3f})"
            )
        else:
            st.warning(
                f"Tipo de río predicho: **{tipo}**, pero no hay modelo "
                f"de curva disponible para este caso."
            )

    # --------- Visualizaciones ---------
    st.subheader("📉 Visualización de datos y curvas")

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # 1) Distribución de tipos de río
    df_comb = pd.concat([df_bajos_raw, df_altos], ignore_index=True)
    counts = df_comb["TIPO_RIO"].value_counts()
    axes[0, 0].pie(counts.values, labels=counts.index, autopct="%1.1f%%", startangle=90)
    axes[0, 0].set_title("Distribución de tipos de río")

    # 2) Boxplot de caudales
    axes[0, 1].boxplot(
        [df_bajos_raw["CAUDAL (m3/s)"], df_altos["CAUDAL (m3/s)"]],
        labels=["BAJO", "ALTO"]
    )
    axes[0, 1].set_title("Comparación de caudales")
    axes[0, 1].set_ylabel("Q (m³/s)")
    axes[0, 1].grid(alpha=0.3)

    # 3) Curvas caudales bajos
    for g, m in info["modelos_bajos"].items():
        sub = df_bajos_raw.copy()
        sub["GRUPO_MANUAL"] = sub.apply(
            lambda row: "GRUPO_2025" if row["YEAR"] == 2025 else
                        "GRUPO_ALTO_RH" if row["RADIO HIDRAULICO (m)"] > 0.6 else
                        "GRUPO_BAJO_RH",
            axis=1
        )
        datos_g = sub[sub["GRUPO_MANUAL"] == g]
        axes[1, 0].scatter(
            datos_g["NIVEL DE AFORO (m)"], datos_g["CAUDAL (m3/s)"],
            label=f"{g}", alpha=0.7
        )
        H_range = np.linspace(m["H_min"]*0.9, m["H_max"]*1.1, 100)
        Q_curve = m["func"](H_range, *m["params"])
        axes[1, 0].plot(H_range, Q_curve, label=f"Curva {g} (R²={m['r2']:.2f})")

    axes[1, 0].set_xlabel("H (m)")
    axes[1, 0].set_ylabel("Q (m³/s)")
    axes[1, 0].set_title("Curvas altura–caudal (bajos)")
    axes[1, 0].legend()
    axes[1, 0].grid(alpha=0.3)

    # 4) Curva caudales altos
    if info["modelos_altos"]:
        m_alto = list(info["modelos_altos"].values())[0]
        axes[1, 1].scatter(
            df_altos["NIVEL DE AFORO (m)"], df_altos["CAUDAL (m3/s)"],
            color="red", alpha=0.7, label="Datos ALTOS"
        )
        H_range = np.linspace(m_alto["H_min"]*0.9, m_alto["H_max"]*1.1, 100)
        Q_curve = m_alto["func"](H_range, *m_alto["params"])
        axes[1, 1].plot(H_range, Q_curve, "r", label=f"Curva ALTO (R²={m_alto['r2']:.2f})")
        axes[1, 1].set_title("Curva altura–caudal (altos)")
    else:
        axes[1, 1].text(0.5, 0.5, "Sin modelo confiable\npara caudales altos",
                        ha="center", va="center", transform=axes[1, 1].transAxes)
        axes[1, 1].set_title("Curva altura–caudal (altos)")

    axes[1, 1].set_xlabel("H (m)")
    axes[1, 1].set_ylabel("Q (m³/s)")
    axes[1, 1].legend()
    axes[1, 1].grid(alpha=0.3)

    plt.tight_layout()
    st.pyplot(fig)


if __name__ == "__main__":
    main()
