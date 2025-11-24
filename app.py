# (pegar TODO lo de abajo)
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score
import warnings

warnings.filterwarnings("ignore")

# =============================================================================
# 0. CONFIGURACIÓN DE LA PÁGINA
# =============================================================================
st.set_page_config(
    page_title="Sistema Multi-Caso Curvas H-Q",
    page_icon="🌊",
    layout="wide"
)

st.markdown(
    "<h1 style='text-align: center;'>🌊 Sistema Inteligente Multi-Caso para Curvas Altura-Caudal</h1>",
    unsafe_allow_html=True
)

# =============================================================================
# 1. FUNCIONES DEL MODELO
# =============================================================================


def func_pot(x, a, b):
    return a * x ** b


def func_poly2(x, a, b, c):
    return a * x ** 2 + b * x + c


def func_poly3(x, a, b, c, d):
    return a * x ** 3 + b * x ** 2 + c * x + d


def calculate_r2(y_obs, y_pred):
    ss_tot = np.sum((y_obs - np.mean(y_obs)) ** 2)
    ss_res = np.sum((y_obs - y_pred) ** 2)
    return 1 - (ss_res / ss_tot) if ss_tot != 0 else 1


def entrenar_modelo_grupo(df_grupo, nombre_grupo):
    """Entrenar modelo de curva Q(H) para un grupo específico."""
    if len(df_grupo) < 3:
        return None

    H = df_grupo["NIVEL DE AFORO (m)"].values
    Q = df_grupo["CAUDAL (m3/s)"].values

    # Ordenar
    sort_idx = np.argsort(H)
    H_sorted = H[sort_idx]
    Q_sorted = Q[sort_idx]

    best_r2 = -np.inf
    best_model_name = None
    best_params = None
    best_model_func = None

    models = [
        ("Polinómico G2", func_poly2),
        ("Polinómico G3", func_poly3),
        ("Potencial", func_pot),
    ]

    for model_name, model_func in models:
        try:
            if model_name in ["Polinómico G2", "Polinómico G3"]:
                params, _ = curve_fit(model_func, H_sorted, Q_sorted, maxfev=5000)
            else:
                p0 = [1.0, 2.0]
                params, _ = curve_fit(model_func, H_sorted, Q_sorted, p0=p0, maxfev=5000)

            Q_pred = model_func(H_sorted, *params)
            r2 = calculate_r2(Q_sorted, Q_pred)

            min_r2 = 0.7 if len(df_grupo) < 10 else 0.8
            if r2 > best_r2 and r2 > min_r2:
                best_r2 = r2
                best_model_name = model_name
                best_params = params
                best_model_func = model_func
        except Exception:
            continue

    if best_model_name:
        return {
            "modelo": best_model_name,
            "ecuacion": (
                f"Q = {best_params[0]:.3f}H^{best_params[1]:.3f}"
                if best_model_name == "Potencial"
                else f"Q = {best_params[0]:.3f}H² + {best_params[1]:.3f}H + {best_params[2]:.3f}"
                if best_model_name == "Polinómico G2"
                else f"Q = {best_params[0]:.3f}H³ + {best_params[1]:.3f}H² + "
                     f"{best_params[2]:.3f}H + {best_params[3]:.3f}"
            ),
            "r2": best_r2,
            "n": len(df_grupo),
            "params": best_params,
            "func": best_model_func,
            "rango_niveles": f"{min(H_sorted):.2f}-{max(H_sorted):.2f} m",
            "rango_caudales": f"{min(Q_sorted):.1f}-{max(Q_sorted):.1f} m³/s",
            "H_min": float(min(H_sorted)),
            "H_max": float(max(H_sorted)),
        }
    return None


class SistemaInteligenteMultiCaso:
    """
    Sistema multi-caso:
    - Clasificador Random Forest (BAJO/ALTO).
    - Modelos de curvas por subgrupos.
    """

    def __init__(self, clasificador_principal, escalador_principal,
                 modelos_bajos, modelos_altos, features):
        self.clasificador_principal = clasificador_principal
        self.escalador_principal = escalador_principal
        self.modelos_bajos = modelos_bajos
        self.modelos_altos = modelos_altos
        self.features = features

    def predecir_tipo_rio(self, nuevos_datos):
        X_nuevo = nuevos_datos[self.features]
        X_nuevo_esc = self.escalador_principal.transform(X_nuevo)
        return self.clasificador_principal.predict(X_nuevo_esc)

    def predecir_caudal(self, nuevos_datos):
        resultados = []
        tipos_predichos = self.predecir_tipo_rio(nuevos_datos)

        for i, (_, fila) in enumerate(nuevos_datos.iterrows()):
            tipo_pred = tipos_predichos[i]
            nivel = fila["NIVEL DE AFORO (m)"]

            if tipo_pred == "BAJO":
                radio_hidraulico = fila["RADIO HIDRAULICO (m)"]
                year = fila["YEAR"]

                if year == 2025:
                    subgrupo = "GRUPO_2025"
                elif radio_hidraulico > 0.6:
                    subgrupo = "GRUPO_ALTO_RH"
                else:
                    subgrupo = "GRUPO_BAJO_RH"

                if subgrupo in self.modelos_bajos:
                    modelo_info = self.modelos_bajos[subgrupo]
                    caudal_pred = modelo_info["func"](nivel, *modelo_info["params"])
                    modelo_usado = f"BAJO_{subgrupo}"
                    r2_modelo = modelo_info["r2"]
                else:
                    caudal_pred = np.nan
                    modelo_usado = "No disponible"
                    r2_modelo = np.nan
            else:  # ALTO
                if self.modelos_altos:
                    modelo_info = list(self.modelos_altos.values())[0]
                    caudal_pred = modelo_info["func"](nivel, *modelo_info["params"])
                    modelo_usado = "ALTO_PRINCIPAL"
                    r2_modelo = modelo_info["r2"]
                else:
                    caudal_pred = np.nan
                    modelo_usado = "No disponible"
                    r2_modelo = np.nan

            resultados.append(
                {
                    "NIVEL DE AFORO (m)": nivel,
                    "TIPO_PREDICHO": tipo_pred,
                    "CAUDAL_PREDICHO (m3/s)": caudal_pred,
                    "MODELO_USADO": modelo_usado,
                    "R2_MODELO": r2_modelo,
                }
            )

        return pd.DataFrame(resultados)


# =============================================================================
# 2. ENTRENAR TODO EL SISTEMA (CACHEADO)
# =============================================================================
@st.cache_resource
def entrenar_sistema_completo():
    # ---------------------- Datos originales (BAJOS) ----------------------
    data_original = {
        "FECHA": [
            "2/10/2021", "2/24/2021", "4/13/2021", "5/11/2021", "9/26/2021",
            "11/25/2021", "3/10/2022", "4/6/2022", "10/14/2022", "10/27/2022",
            "11/8/2022", "12/10/2022", "2/8/2023", "2/24/2023", "3/6/2023",
            "4/27/2023", "10/26/2023", "11/9/2023", "2/15/2024", "2/22/2024",
            "3/1/2024", "3/7/2024", "6/22/2024", "8/16/2024", "8/21/2024",
            "11/21/2024", "2/7/2025", "2/13/2025", "2/22/2025", "3/7/2025",
            "3/15/2025", "3/20/2025", "4/2/2025", "4/9/2025",
        ],
        "CAUDAL (m3/s)": [
            4.79, 1.89, 2.15, 1.11, 1.1, 1.36, 10.38, 1.22, 1.44, 1.71, 1.74,
            3.03, 8.22, 2.56, 1.97, 1.84, 1.3, 1.12, 7.23, 7.69, 16.86, 12.38,
            2.14, 1.72, 1.73, 1.18, 8.18, 15.36, 22.53, 13.08, 17.2, 4.79,
            2.72, 3.98,
        ],
        "VELOCIDAD (m/s)": [
            1.38, 0.62, 0.74, 0.46, 0.48, 0.56, 2.19, 0.7, 0.62, 0.66, 0.65,
            0.86, 0.83, 0.71, 0.63, 0.48, 0.49, 0.41, 1.04, 0.88, 1.09, 1.0,
            0.54, 0.59, 0.63, 0.49, 0.71, 1.16, 1.38, 1.05, 1.22, 1.01, 0.72,
            0.9,
        ],
        "AREA (m2)": [
            3.47, 2.76, 2.64, 2.07, 1.97, 2.07, 4.58, 1.64, 2.25, 2.47, 2.6,
            3.4, 9.42, 3.43, 2.96, 3.77, 2.49, 2.32, 6.72, 8.67, 15.41, 12.41,
            3.32, 2.78, 2.67, 2.27, 11.6, 13.02, 16.1, 12.26, 13.91, 4.42,
            3.61, 4.39,
        ],
        "PERIMETRO (m)": [
            8.57, 8.57, 8.34, 7.54, 7.55, 8.35, 9.69, 7.02, 9.05, 9.35, 9.05,
            9.36, 15.06, 10.58, 10.61, 13.42, 9.84, 9.86, 13.08, 13.08, 13.57,
            13.06, 13.13, 11.56, 10.6, 11.06, 11.07, 11.09, 11.13, 11.12,
            11.13, 10.2, 9.61, 8.67,
        ],
        "ANCHO RIO (m)": [
            8.5, 8.5, 8.3, 7.5, 7.5, 8.3, 9.5, 7.0, 9.0, 9.3, 9.0, 9.3, 15.0,
            10.5, 10.5, 13.3, 9.8, 9.8, 13.0, 13.0, 13.0, 13.0, 13.0, 11.5,
            10.5, 11.0, 11.0, 11.0, 11.0, 11.0, 11.0, 10.0, 9.5, 8.5,
        ],
        "NIVEL DE AFORO (m)": [
            1.2, 1.01, 1.05, 0.97, 0.96, 0.92, 1.38, 0.97, 0.96, 0.97, 0.98,
            1.06, 1.38, 1.01, 0.94, 1.01, 0.96, 0.94, 1.41, 1.46, 1.92, 1.7,
            0.87, 0.85, 0.86, 0.82, 1.1, 1.35, 1.6, 1.4, 1.18, 1.15, 0.97,
            1.03,
        ],
    }

    df_original = pd.DataFrame(data_original)
    df_original["YEAR"] = pd.to_datetime(df_original["FECHA"]).dt.year
    df_original["RADIO HIDRAULICO (m)"] = df_original["AREA (m2)"] / df_original["PERIMETRO (m)"]
    df_original["TIRANTE MEDIO (m)"] = df_original["AREA (m2)"] / df_original["ANCHO RIO (m)"]
    df_original["CAUDAL/AREA"] = df_original["CAUDAL (m3/s)"] / df_original["AREA (m2)"]
    df_original["TIPO_RIO"] = "BAJO"

    # ---------------------- Datos nuevos (ALTOS) ----------------------
    data_nuevo = {
        "FECHA": [
            "9/28/2021", "10/29/2019", "9/24/2019", "8/28/2019", "7/23/2019",
            "4/17/2019", "11/19/2019", "3/24/2021", "11/18/2021", "6/27/2019",
            "3/23/2022", "3/27/2025", "7/26/2024",
        ],
        "CAUDAL (m3/s)": [
            312, 639.56, 806.02, 295.03, 392.67, 997.66, 786.09, 2026.2,
            372.5, 300.26, 1521.1, 887.877, 197.791,
        ],
        "VELOCIDAD (m/s)": [
            0.56, 1.13, 1.37, 0.58, 0.71, 1.55, 1.17, 2.45, 0.71, 0.58,
            2.19, 1.367, 0.402,
        ],
        "AREA (m2)": [
            521.6, 579.4, 595.7, 505.7, 549.6, 670.6, 677.4, 839.4, 507.2,
            516.6, 708.5, 649.65, 475.08,
        ],
        "PERIMETRO (m)": [
            108.195, 115.0399, 116.59, 113.5599, 115.88, 122.17, 116.47,
            138.2, 113.8, 110.86, 127.36, 111.63, 103.91,
        ],
        "ANCHO RIO (m)": [
            8.42, 9.18, 9.34, 8.5, 8.72, 9.78, 9.5, 10.56, 8.68, 8.5,
            10.19, 9.8, 8.29,
        ],
        "NIVEL DE AFORO (m)": [
            8.42, 9.18, 9.34, 8.5, 8.72, 9.78, 9.5, 10.56, 8.68, 8.5,
            10.19, 9.8, 8.29,
        ],
    }

    df_nuevo = pd.DataFrame(data_nuevo)
    df_nuevo["YEAR"] = pd.to_datetime(df_nuevo["FECHA"]).dt.year
    df_nuevo["RADIO HIDRAULICO (m)"] = df_nuevo["AREA (m2)"] / df_nuevo["PERIMETRO (m)"]
    df_nuevo["TIRANTE MEDIO (m)"] = df_nuevo["AREA (m2)"] / df_nuevo["ANCHO RIO (m)"]
    df_nuevo["CAUDAL/AREA"] = df_nuevo["CAUDAL (m3/s)"] / df_nuevo["AREA (m2)"]
    df_nuevo["TIPO_RIO"] = "ALTO"

    df_combinado = pd.concat([df_original, df_nuevo], ignore_index=True)

    features_comunes = [
        "NIVEL DE AFORO (m)",
        "ANCHO RIO (m)",
        "PERIMETRO (m)",
        "AREA (m2)",
        "VELOCIDAD (m/s)",
        "RADIO HIDRAULICO (m)",
        "TIRANTE MEDIO (m)",
        "CAUDAL/AREA",
        "YEAR",
    ]

    X_combined = df_combinado[features_comunes]
    y_combined = df_combinado["TIPO_RIO"]

    X_train_c, X_test_c, y_train_c, y_test_c = train_test_split(
        X_combined, y_combined, test_size=0.2, random_state=42, stratify=y_combined
    )

    scaler_combined = StandardScaler()
    X_train_scaled_c = scaler_combined.fit_transform(X_train_c)
    X_test_scaled_c = scaler_combined.transform(X_test_c)

    clf_principal = RandomForestClassifier(
        n_estimators=100,
        max_depth=6,
        min_samples_split=4,
        min_samples_leaf=2,
        random_state=42,
    )
    clf_principal.fit(X_train_scaled_c, y_train_c)

    train_score_c = clf_principal.score(X_train_scaled_c, y_train_c)
    test_score_c = clf_principal.score(X_test_scaled_c, y_test_c)
    cv_scores_c = cross_val_score(clf_principal, X_train_scaled_c, y_train_c, cv=5)

    # ------------------- Modelos por subgrupo BAJO -------------------
    df_original = df_original.copy()
    df_original["GRUPO_MANUAL"] = df_original.apply(
        lambda row: "GRUPO_2025"
        if row["YEAR"] == 2025
        else "GRUPO_ALTO_RH"
        if row["RADIO HIDRAULICO (m)"] > 0.6
        else "GRUPO_BAJO_RH",
        axis=1,
    )

    modelos_bajos = {}
    for grupo in df_original["GRUPO_MANUAL"].unique():
        grupo_data = df_original[df_original["GRUPO_MANUAL"] == grupo]
        modelo = entrenar_modelo_grupo(grupo_data, f"BAJO_{grupo}")
        if modelo:
            modelos_bajos[grupo] = modelo

    # ------------------- Modelo ALTOS -------------------
    modelo_alto = entrenar_modelo_grupo(df_nuevo, "ALTO")
    modelos_altos = {}
    if modelo_alto:
        modelos_altos["ALTO_PRINCIPAL"] = modelo_alto

    sistema_inteligente = SistemaInteligenteMultiCaso(
        clf_principal,
        scaler_combined,
        modelos_bajos,
        modelos_altos,
        features_comunes,
    )

    resumen = {
        "train_score": float(train_score_c),
        "test_score": float(test_score_c),
        "cv_mean": float(cv_scores_c.mean()),
        "cv_std": float(cv_scores_c.std()),
        "n_bajos": int(len(df_original)),
        "n_altos": int(len(df_nuevo)),
    }

    return sistema_inteligente, resumen, df_original, df_nuevo, modelos_bajos, modelos_altos


sistema_inteligente, resumen_modelo, df_bajos, df_altos, modelos_bajos, modelos_altos = (
    entrenar_sistema_completo()
)

# =============================================================================
# 3. FUNCIONES PARA PREPARAR DATOS DEL USUARIO Y GRÁFICOS
# =============================================================================


def preparar_datos_usuario(df):
    """
    Espera columnas:
    - NIVEL DE AFORO (m)
    - CAUDAL (m3/s)
    - AREA (m2)
    - ANCHO RIO (m)
    - PERIMETRO (m)
    - VELOCIDAD (m/s)
    - FECHA (opcional) o YEAR (opcional)
    """
    df = df.copy()

    # Año
    if "YEAR" not in df.columns:
        if "FECHA" in df.columns:
            df["YEAR"] = pd.to_datetime(df["FECHA"], errors="coerce").dt.year
            df["YEAR"] = df["YEAR"].fillna(2025).astype(int)
        else:
            df["YEAR"] = 2025

    requeridas = [
        "NIVEL DE AFORO (m)",
        "CAUDAL (m3/s)",
        "AREA (m2)",
        "ANCHO RIO (m)",
        "PERIMETRO (m)",
        "VELOCIDAD (m/s)",
    ]
    faltantes = [c for c in requeridas if c not in df.columns]
    if faltantes:
        raise ValueError("Faltan columnas obligatorias: " + ", ".join(faltantes))

    df["RADIO HIDRAULICO (m)"] = df["AREA (m2)"] / df["PERIMETRO (m)"]
    df["TIRANTE MEDIO (m)"] = df["AREA (m2)"] / df["ANCHO RIO (m)"]
    df["CAUDAL/AREA"] = df["CAUDAL (m3/s)"] / df["AREA (m2)"]

    return df


def grafico_curvas_internas():
    """Gráfico de curvas internas para caudales bajos y altos."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # BAJOS
    ax = axes[0]
    for grupo, modelo_info in modelos_bajos.items():
        grupo_data = df_bajos[df_bajos["GRUPO_MANUAL"] == grupo]
        ax.scatter(
            grupo_data["NIVEL DE AFORO (m)"],
            grupo_data["CAUDAL (m3/s)"],
            label=f"BAJO_{grupo}",
            alpha=0.7,
        )
        H_range = np.linspace(
            modelo_info["H_min"] * 0.9, modelo_info["H_max"] * 1.1, 100
        )
        Q_curve = modelo_info["func"](H_range, *modelo_info["params"])
        ax.plot(
            H_range,
            Q_curve,
            label=f"Curva {grupo} (R²={modelo_info['r2']:.3f})",
        )

    ax.set_xlabel("Nivel de Aforo (m)")
    ax.set_ylabel("Caudal (m³/s)")
    ax.set_title("Curvas para Caudales Bajos")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # ALTOS
    ax2 = axes[1]
    if modelos_altos:
        modelo_alto_info = list(modelos_altos.values())[0]
        ax2.scatter(
            df_altos["NIVEL DE AFORO (m)"],
            df_altos["CAUDAL (m3/s)"],
            color="red",
            label="Datos altos",
            alpha=0.7,
        )
        H_range = np.linspace(
            modelo_alto_info["H_min"] * 0.9, modelo_alto_info["H_max"] * 1.1, 100
        )
        Q_curve = modelo_alto_info["func"](H_range, *modelo_alto_info["params"])
        ax2.plot(
            H_range,
            Q_curve,
            color="red",
            label=f"Curva alta (R²={modelo_alto_info['r2']:.3f})",
        )

    ax2.set_xlabel("Nivel de Aforo (m)")
    ax2.set_ylabel("Caudal (m³/s)")
    ax2.set_title("Curvas para Caudales Altos")
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    return fig


def grafico_predicciones(df_usuario, predicciones):
    """Scatter de nivel–caudal con colores por BAJO/ALTO."""
    fig, ax = plt.subplots(figsize=(7, 5))

    if "CAUDAL (m3/s)" in df_usuario.columns:
        ax.scatter(
            df_usuario["NIVEL DE AFORO (m)"],
            df_usuario["CAUDAL (m3/s)"],
            label="Caudal observado",
            alpha=0.6,
            color="gray",
        )

    for tipo in ["BAJO", "ALTO"]:
        mask = predicciones["TIPO_PREDICHO"] == tipo
        if mask.any():
            ax.scatter(
                df_usuario.loc[mask, "NIVEL DE AFORO (m)"],
                predicciones.loc[mask, "CAUDAL_PREDICHO (m3/s)"],
                label=f"Predicho ({tipo})",
                alpha=0.8,
            )

    ax.set_xlabel("Nivel de Aforo (m)")
    ax.set_ylabel("Caudal (m³/s)")
    ax.set_title("Predicciones de caudal por tipo de río")
    ax.grid(True, alpha=0.3)
    ax.legend()

    return fig


# =============================================================================
# 4. INTERFAZ STREAMLIT
# =============================================================================

opcion = st.sidebar.radio(
    "Navegación:",
    ["🏠 Inicio", "📤 Subir Aforos", "📊 Ingreso Manual", "📈 Modelos Internos"],
)

# ------------------------- 4.1 INICIO -------------------------
if opcion == "🏠 Inicio":
    st.header("🏠 Inicio")
    st.markdown(
        """
Este sistema:

- Entrena un **clasificador Random Forest** que distingue entre **régimen BAJO** y **régimen ALTO**.
- Ajusta **curvas altura-caudal** distintas para:
  - Subgrupos de caudales **bajos** (`GRUPO_BAJO_RH`, `GRUPO_ALTO_RH`, `GRUPO_2025`).
  - Un grupo de caudales **altos**.
- Permite:
  - Cargar aforos desde un archivo CSV.
  - Ingresar aforos manualmente.
  - Visualizar las curvas internas del modelo.
"""
    )

    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Aforos BAJOS usados", resumen_modelo["n_bajos"])
    with col2:
        st.metric("Aforos ALTOS usados", resumen_modelo["n_altos"])
    with col3:
        st.metric("Precisión en prueba RF", f"{resumen_modelo['test_score']:.3f}")

    st.subheader("Notas")
    st.info(
        """
1. La clasificación BAJO/ALTO se basa en 9 variables hidráulicas y geométricas.
2. Las curvas Q(H) se construyen con modelos polinómicos o potenciales.
3. El sistema puede utilizarse como emulador para nuevos aforos dentro de los rangos observados.
"""
    )

# ------------------------- 4.2 SUBIR AFOROS -------------------------
elif opcion == "📤 Subir Aforos":
    st.header("📤 Subir Archivo de Aforos")

    st.markdown(
        """
**Formato esperado (CSV):**

Columnas mínimas con estos nombres exactos:

- `NIVEL DE AFORO (m)`
- `CAUDAL (m3/s)`
- `AREA (m2)`
- `ANCHO RIO (m)`
- `PERIMETRO (m)`
- `VELOCIDAD (m/s)`
- `FECHA` (opcional) o `YEAR` (opcional)
"""
    )

    archivo = st.file_uploader("Selecciona archivo CSV", type=["csv"])

    if archivo is not None:
        try:
            df_user = pd.read_csv(archivo)
            st.success(f"✅ Archivo cargado. Filas: {len(df_user)}")
            st.subheader("📋 Vista previa")
            st.dataframe(df_user.head())

            if st.button("🚀 Procesar aforos", type="primary"):
                with st.spinner("Procesando..."):
                    df_proc = preparar_datos_usuario(df_user)
                    pred = sistema_inteligente.predecir_caudal(df_proc)

                    resultado = pd.concat(
                        [df_proc.reset_index(drop=True), pred.reset_index(drop=True)],
                        axis=1,
                    )
                    # 🔹 Eliminar columnas duplicadas
                    resultado = resultado.loc[:, ~resultado.columns.duplicated()]

                    st.subheader("📊 Resultados de predicción")
                    st.dataframe(
                        resultado[
                            [
                                "NIVEL DE AFORO (m)",
                                "CAUDAL (m3/s)",
                                "TIPO_PREDICHO",
                                "CAUDAL_PREDICHO (m3/s)",
                                "MODELO_USADO",
                                "R2_MODELO",
                            ]
                        ]
                    )

                    # Métricas si hay caudal observado
                    if "CAUDAL (m3/s)" in df_proc.columns:
                        mse = mean_squared_error(
                            df_proc["CAUDAL (m3/s)"],
                            pred["CAUDAL_PREDICHO (m3/s)"],
                        )
                        r2 = r2_score(
                            df_proc["CAUDAL (m3/s)"],
                            pred["CAUDAL_PREDICHO (m3/s)"],
                        )
                        st.subheader("📏 Métricas vs caudal observado")
                        col1, col2 = st.columns(2)
                        with col1:
                            st.metric("MSE", f"{mse:.3f}")
                        with col2:
                            st.metric("R²", f"{r2:.3f}")

                    # Gráfico
                    st.subheader("📈 Gráfico Nivel-Caudal (observado vs predicho)")
                    fig_pred = grafico_predicciones(df_proc, pred)
                    st.pyplot(fig_pred)

        except Exception as e:
            st.error(f"❌ Error al procesar el archivo: {e}")

# ------------------------- 4.3 INGRESO MANUAL -------------------------
elif opcion == "📊 Ingreso Manual":
    st.header("📊 Ingreso Manual de Aforos")

    num = st.number_input(
        "Número de aforos a ingresar",
        min_value=1,
        max_value=20,
        value=3,
        step=1,
    )

    registros = []
    for i in range(num):
        with st.expander(f"Aforo {i+1}", expanded=(i == 0)):
            col1, col2, col3 = st.columns(3)
            with col1:
                nivel = st.number_input(
                    "Nivel de aforo (m)", 0.1, 15.0, 1.0, 0.05, key=f"nivel{i}"
                )
                caudal = st.number_input(
                    "Caudal (m3/s)", 0.1, 3000.0, 2.0, 0.1, key=f"caudal{i}"
                )
                year = st.number_input(
                    "Año del aforo", 2018, 2030, 2025, 1, key=f"year{i}"
                )
            with col2:
                area = st.number_input(
                    "Área mojada (m2)", 0.1, 2000.0, 3.0, 0.1, key=f"area{i}"
                )
                ancho = st.number_input(
                    "Ancho del río (m)", 0.1, 200.0, 8.0, 0.1, key=f"ancho{i}"
                )
            with col3:
                perimetro = st.number_input(
                    "Perímetro mojado (m)",
                    0.1,
                    300.0,
                    10.0,
                    0.1,
                    key=f"perim{i}",
                )
                velocidad = st.number_input(
                    "Velocidad (m/s)", 0.1, 10.0, 0.7, 0.05, key=f"vel{i}"
                )

            registros.append(
                {
                    "NIVEL DE AFORO (m)": nivel,
                    "CAUDAL (m3/s)": caudal,
                    "AREA (m2)": area,
                    "ANCHO RIO (m)": ancho,
                    "PERIMETRO (m)": perimetro,
                    "VELOCIDAD (m/s)": velocidad,
                    "YEAR": int(year),
                }
            )

    if st.button("🚀 Procesar datos manuales", type="primary"):
        with st.spinner("Procesando..."):
            df_manual = pd.DataFrame(registros)
            df_proc = preparar_datos_usuario(df_manual)
            pred = sistema_inteligente.predecir_caudal(df_proc)

            resultado = pd.concat(
                [df_proc.reset_index(drop=True), pred.reset_index(drop=True)],
                axis=1,
            )
            # 🔹 Eliminar columnas duplicadas
            resultado = resultado.loc[:, ~resultado.columns.duplicated()]

            st.subheader("📊 Resultados")
            st.dataframe(
                resultado[
                    [
                        "NIVEL DE AFORO (m)",
                        "CAUDAL (m3/s)",
                        "TIPO_PREDICHO",
                        "CAUDAL_PREDICHO (m3/s)",
                        "MODELO_USADO",
                        "R2_MODELO",
                    ]
                ]
            )

            st.subheader("📈 Gráfico Nivel-Caudal (observado vs predicho)")
            fig_pred = grafico_predicciones(df_proc, pred)
            st.pyplot(fig_pred)

# ------------------------- 4.4 MODELOS INTERNOS -------------------------
elif opcion == "📈 Modelos Internos":
    st.header("📈 Curvas internas del sistema")

    col1, col2 = st.columns(2)
    with col1:
        st.metric("Subgrupos BAJOS", len(modelos_bajos))
    with col2:
        st.metric("Modelos ALTOS", len(modelos_altos))

    st.subheader("Curvas altura-caudal utilizadas por el sistema")
    fig_curvas = grafico_curvas_internas()
    st.pyplot(fig_curvas)

    st.subheader("Detalles de modelos por subgrupo (BAJOS)")
    for nombre, info in modelos_bajos.items():
        with st.expander(f"BAJO_{nombre}  (R² = {info['r2']:.3f})"):
            st.write(f"**Modelo:** {info['modelo']}")
            st.write(f"**Ecuación:** {info['ecuacion']}")
            st.write(f"**N° de puntos:** {info['n']}")
            st.write(f"**Rango de niveles:** {info['rango_niveles']}")
            st.write(f"**Rango de caudales:** {info['rango_caudales']}")

    if modelos_altos:
        st.subheader("Modelo de caudales altos")
        info = list(modelos_altos.values())[0]
        with st.expander(f"ALTO_PRINCIPAL  (R² = {info['r2']:.3f})"):
            st.write(f"**Modelo:** {info['modelo']}")
            st.write(f"**Ecuación:** {info['ecuacion']}")
            st.write(f"**N° de puntos:** {info['n']}")
            st.write(f"**Rango de niveles:** {info['rango_niveles']}")
            st.write(f"**Rango de caudales:** {info['rango_caudales']}")

st.markdown("---")
st.markdown("**🌊 Sistema Inteligente Multi-Caso para Curvas H-Q**")
