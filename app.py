import pandas as pd
import numpy as np
import joblib
from scipy.optimize import curve_fit
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler

# Definir funciones (las mismas que en el archivo principal)
def func_poly2(x, a, b, c):
    return a * x**2 + b * x + c

def func_poly3(x, a, b, c, d):
    return a * x**3 + b * x**2 + c * x + d

def func_pot(x, a, b):
    return a * x**b

def calculate_r2(y_obs, y_pred):
    ss_tot = np.sum((y_obs - np.mean(y_obs))**2)
    ss_res = np.sum((y_obs - y_pred)**2)
    return 1 - (ss_res / ss_tot) if ss_tot != 0 else 1

def entrenar_modelo_grupo(df_grupo, nombre_grupo):
    if len(df_grupo) < 3:
        return None
    
    H = df_grupo['NIVEL_AFORO'].values
    Q = df_grupo['CAUDAL'].values
    
    sort_idx = np.argsort(H)
    H_sorted = H[sort_idx]
    Q_sorted = Q[sort_idx]
    
    best_r2 = -np.inf
    best_model_name = None
    best_params = None
    best_model_func = None
    
    models = [
        ('Polinómico G2', func_poly2),
        ('Polinómico G3', func_poly3),
        ('Potencial', func_pot)
    ]
    
    for model_name, model_func in models:
        try:
            if model_name in ['Polinómico G2', 'Polinómico G3']:
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
                
        except Exception as e:
            continue
    
    if best_model_name:
        if best_model_name == 'Potencial':
            ecuacion = f'Q = {best_params[0]:.4f}H^{{{best_params[1]:.4f}}}'
        elif best_model_name == 'Polinómico G2':
            ecuacion = f'Q = {best_params[0]:.4f}H² + {best_params[1]:.4f}H + {best_params[2]:.4f}'
        else:
            ecuacion = f'Q = {best_params[0]:.4f}H³ + {best_params[1]:.4f}H² + {best_params[2]:.4f}H + {best_params[3]:.4f}'
        
        return {
            'modelo': best_model_name,
            'ecuacion': ecuacion,
            'r2': best_r2,
            'n': len(df_grupo),
            'params': best_params,
            'func': best_model_func,
            'rango_niveles': (min(H_sorted), max(H_sorted)),
            'rango_caudales': (min(Q_sorted), max(Q_sorted))
        }
    return None

# Clase del sistema (la misma que en el archivo principal)
class SistemaInteligenteMultiCaso:
    def __init__(self):
        self.clasificador_principal = RandomForestClassifier(n_estimators=100, random_state=42)
        self.escalador_principal = StandardScaler()
        self.modelos_bajos = {}
        self.modelos_altos = {}
        self.features = [
            'NIVEL_AFORO', 'ANCHO_RIO', 'PERIMETRO', 
            'AREA', 'VELOCIDAD', 'RADIO_HIDRAULICO', 
            'TIRANTE_MEDIO', 'CAUDAL_AREA', 'YEAR'
        ]
    
    def entrenar_clasificador(self, X, y):
        X_esc = self.escalador_principal.fit_transform(X)
        self.clasificador_principal.fit(X_esc, y)
        return self
    
    def predecir_tipo_rio(self, X):
        X_esc = self.escalador_principal.transform(X)
        return self.clasificador_principal.predict(X_esc)
    
    def agregar_modelo_bajo(self, nombre, modelo_info):
        self.modelos_bajos[nombre] = modelo_info
    
    def agregar_modelo_alto(self, nombre, modelo_info):
        self.modelos_altos[nombre] = modelo_info

# Función para preparar datos
def preparar_datos(df):
    df_procesado = df.copy()
    
    mapeo_columnas = {
        'NIVEL DE AFORO (m)': 'NIVEL_AFORO',
        'CAUDAL (m3/s)': 'CAUDAL', 
        'AREA (m2)': 'AREA',
        'ANCHO RIO (m)': 'ANCHO_RIO',
        'PERIMETRO (m)': 'PERIMETRO',
        'VELOCIDAD (m/s)': 'VELOCIDAD',
        'FECHA AFORO': 'FECHA',
        'RADIO HIDRAULICO (m)': 'RADIO_HIDRAULICO',
        'TIRANTE MEDIO (m)': 'TIRANTE_MEDIO'
    }
    
    for col_original, col_nuevo in mapeo_columnas.items():
        if col_original in df_procesado.columns:
            df_procesado[col_nuevo] = df_procesado[col_original]
    
    if 'RADIO_HIDRAULICO' not in df_procesado.columns or df_procesado['RADIO_HIDRAULICO'].isna().any():
        df_procesado['RADIO_HIDRAULICO'] = df_procesado['AREA'] / df_procesado['PERIMETRO']
    
    if 'TIRANTE_MEDIO' not in df_procesado.columns or df_procesado['TIRANTE_MEDIO'].isna().any():
        df_procesado['TIRANTE_MEDIO'] = df_procesado['AREA'] / df_procesado['ANCHO_RIO']
    
    if 'CAUDAL_AREA' not in df_procesado.columns:
        df_procesado['CAUDAL_AREA'] = df_procesado['CAUDAL'] / df_procesado['AREA']
    
    if 'FECHA' in df_procesado.columns:
        try:
            df_procesado['FECHA'] = pd.to_datetime(df_procesado['FECHA'], errors='coerce')
            df_procesado['YEAR'] = df_procesado['FECHA'].dt.year.fillna(2024).astype(int)
        except:
            df_procesado['YEAR'] = 2024
    else:
        df_procesado['YEAR'] = 2024
    
    return df_procesado

# Entrenar y guardar el modelo
def entrenar_y_guardar_modelo():
    print("Entrenando modelo avanzado...")
    
    # Datos de entrenamiento (los mismos de tu Colab)
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
                               0.96, 0.94, 1.41, 1.46, 1.92, 1.7, 0.87, 0.85, 0.86, 0.82, 1.1, 1.35, 1.6, 1.4, 1.18, 1.15, 0.97, 1.03]
    }

    data_nuevo = {
        'FECHA': ['9/28/2021', '10/29/2019', '9/24/2019', '8/28/2019', '7/23/2019', '4/17/2019', 
                  '11/19/2019', '3/24/2021', '11/18/2021', '6/27/2019', '3/23/2022', '3/27/2025', '7/26/2024'],
        'CAUDAL (m3/s)': [312, 639.56, 806.02, 295.03, 392.67, 997.66, 786.09, 2026.2, 372.5, 300.26, 1521.1, 887.877, 197.791],
        'VELOCIDAD (m/s)': [0.56, 1.13, 1.37, 0.58, 0.71, 1.55, 1.17, 2.45, 0.71, 0.58, 2.19, 1.367, 0.402],
        'AREA (m2)': [521.6, 579.4, 595.7, 505.7, 549.6, 670.6, 677.4, 839.4, 507.2, 516.6, 708.5, 649.65, 475.08],
        'PERIMETRO (m)': [108.195, 115.0399, 116.59, 113.5599, 115.88, 122.17, 116.47, 138.2, 113.8, 110.86, 127.36, 111.63, 103.91],
        'ANCHO RIO (m)': [8.42, 9.18, 9.34, 8.5, 8.72, 9.78, 9.5, 10.56, 8.68, 8.5, 10.19, 9.8, 8.29],
        'NIVEL DE AFORO (m)': [8.42, 9.18, 9.34, 8.5, 8.72, 9.78, 9.5, 10.56, 8.68, 8.5, 10.19, 9.8, 8.29]
    }

    df_original = pd.DataFrame(data_original)
    df_nuevo = pd.DataFrame(data_nuevo)
    
    # Preparar datos
    df_original_proc = preparar_datos(df_original)
    df_nuevo_proc = preparar_datos(df_nuevo)
    
    # Agregar tipo de río
    df_original_proc['TIPO_RIO'] = 'BAJO'
    df_nuevo_proc['TIPO_RIO'] = 'ALTO'
    
    df_combinado = pd.concat([df_original_proc, df_nuevo_proc], ignore_index=True)
    
    # Crear y entrenar sistema
    sistema = SistemaInteligenteMultiCaso()
    
    # Entrenar clasificador principal
    X_combined = df_combinado[sistema.features]
    y_combined = df_combinado['TIPO_RIO']
    sistema.entrenar_clasificador(X_combined, y_combined)
    
    # Entrenar modelos para caudales bajos (subgrupos)
    df_original_proc['GRUPO_MANUAL'] = df_original_proc.apply(
        lambda row: 'GRUPO_2025' if row['YEAR'] == 2025 else 
                   'GRUPO_ALTO_RH' if row['RADIO_HIDRAULICO'] > 0.6 else 'GRUPO_BAJO_RH', 
        axis=1
    )
    
    for grupo in df_original_proc['GRUPO_MANUAL'].unique():
        grupo_data = df_original_proc[df_original_proc['GRUPO_MANUAL'] == grupo]
        modelo = entrenar_modelo_grupo(grupo_data, grupo)
        if modelo:
            sistema.agregar_modelo_bajo(grupo, modelo)
    
    # Entrenar modelo para caudales altos
    modelo_alto = entrenar_modelo_grupo(df_nuevo_proc, "ALTO_PRINCIPAL")
    if modelo_alto:
        sistema.agregar_modelo_alto("ALTO_PRINCIPAL", modelo_alto)
    
    # Guardar modelo
    joblib.dump(sistema, 'modelo_talapalca_avanzado.pkl')
    print("✅ Modelo avanzado guardado como 'modelo_talapalca_avanzado.pkl'")
    
    return sistema

if __name__ == "__main__":
    sistema = entrenar_y_guardar_modelo()