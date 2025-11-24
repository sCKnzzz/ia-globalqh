import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import joblib
from scipy.optimize import curve_fit
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
import io

# DEFINIR LA CLASE QUE FALTA
class SistemaCurvasAlturaCaudal:
    def __init__(self):
        self.clasificador = RandomForestClassifier(n_estimators=100, random_state=42)
        self.escalador = StandardScaler()
        self.curvas = {}
    
    def entrenar(self, X, y):
        X_esc = self.escalador.fit_transform(X)
        self.clasificador.fit(X_esc, y)
        return self
    
    def predecir_grupo(self, X):
        X_esc = self.escalador.transform(X)
        return self.clasificador.predict(X_esc)

# DEFINIR FUNCIONES GLOBALES
def func_poly2(x, a, b, c):
    return a * x**2 + b * x + c

def func_poly3(x, a, b, c, d):
    return a * x**3 + b * x**2 + c * x + d

def func_pot(x, a, b):
    return a * x**b

def func_exp(x, a, b):
    return a * np.exp(b * x)

def func_log(x, a, b):
    return a * np.log(x + b)

# FUNCIÓN PARA PREPARAR DATOS (MEJORADA)
def preparar_datos(df):
    df_procesado = df.copy()
    
    # Mapear nombres de columnas
    mapeo_columnas = {
        'NIVEL DE AFORO (m)': 'NIVEL_AFORO',
        'CAUDAL (m3/s)': 'CAUDAL', 
        'AREA (m2)': 'AREA',
        'ANCHO RIO (m)': 'ANCHO_RIO',
        'PERIMETRO (m)': 'PERIMETRO',
        'VELOCIDAD (m/s)': 'VELOCIDAD',
        'FECHA AFORO': 'FECHA'
    }
    
    for col_original, col_nuevo in mapeo_columnas.items():
        if col_original in df_procesado.columns:
            df_procesado[col_nuevo] = df_procesado[col_original]
    
    # Estimar perímetro si falta
    if 'PERIMETRO' not in df_procesado.columns or df_procesado['PERIMETRO'].isna().any():
        # Calcular tirante medio
        df_procesado['TIRANTE_MEDIO'] = df_procesado['AREA'] / df_procesado['ANCHO_RIO']
        # Estimar perímetro (aproximación para sección rectangular)
        df_procesado['PERIMETRO'] = 2 * df_procesado['TIRANTE_MEDIO'] + df_procesado['ANCHO_RIO']
        st.info("📏 Perímetro calculado automáticamente usando aproximación rectangular")
    
    # Calcular variables hidráulicas
    df_procesado['RADIO_HIDRAULICO'] = df_procesado['AREA'] / df_procesado['PERIMETRO']
    df_procesado['TIRANTE_MEDIO'] = df_procesado['AREA'] / df_procesado['ANCHO_RIO']
    df_procesado['CAUDAL_AREA'] = df_procesado['CAUDAL'] / df_procesado['AREA']
    
    # Año
    if 'FECHA' in df_procesado.columns:
        try:
            df_procesado['FECHA'] = pd.to_datetime(df_procesado['FECHA'], errors='coerce')
            df_procesado['YEAR'] = df_procesado['FECHA'].dt.year.fillna(2024).astype(int)
        except:
            df_procesado['YEAR'] = 2024
    else:
        df_procesado['YEAR'] = 2024
    
    return df_procesado

# FUNCIÓN PARA AJUSTAR CURVAS
def ajustar_curva(datos_grupo):
    H = datos_grupo['NIVEL_AFORO'].values
    Q = datos_grupo['CAUDAL'].values
    
    if len(H) < 3:
        return None
        
    sort_idx = np.argsort(H)
    H_sorted = H[sort_idx]
    Q_sorted = Q[sort_idx]
    
    modelos = [
        ('Polinómico G2', func_poly2),
        ('Polinómico G3', func_poly3),
        ('Potencial', func_pot)
    ]
    
    mejor_r2 = -np.inf
    mejor_modelo = None
    
    for nombre, funcion in modelos:
        try:
            if nombre == 'Potencial':
                params, _ = curve_fit(funcion, H_sorted, Q_sorted, p0=[1.0, 2.0], maxfev=5000)
            else:
                params, _ = curve_fit(funcion, H_sorted, Q_sorted, maxfev=5000)
            
            Q_pred = funcion(H_sorted, *params)
            r2 = 1 - np.sum((Q_sorted - Q_pred)**2) / np.sum((Q_sorted - np.mean(Q_sorted))**2)
            
            if r2 > mejor_r2 and r2 > 0.7:
                mejor_r2 = r2
                mejor_modelo = {
                    'nombre': nombre,
                    'funcion': funcion,
                    'parametros': params,
                    'r2': round(r2, 3),
                    'n_puntos': len(H_sorted),
                    'rango_niveles': (min(H_sorted), max(H_sorted)),
                    'rango_caudales': (min(Q_sorted), max(Q_sorted))
                }
        except:
            continue
    
    return mejor_modelo

# FUNCIÓN MEJORADA PARA AJUSTAR MODELOS A RELACIONES HIDRÁULICAS
def ajustar_modelo_relacion(x, y, nombre_relacion):
    """Ajustar diferentes modelos y seleccionar el mejor según R²"""
    
    modelos = [
        ('Lineal', lambda x, a, b: a * x + b),
        ('Polinómico G2', func_poly2),
        ('Exponencial', func_exp),
        ('Logarítmico', func_log),
        ('Potencial', func_pot)
    ]
    
    mejor_r2 = -np.inf
    mejor_modelo = None
    mejor_params = None
    
    for nombre, funcion in modelos:
        try:
            if nombre == 'Exponencial':
                params, _ = curve_fit(funcion, x, y, p0=[1.0, 0.1], maxfev=5000)
            elif nombre == 'Logarítmico':
                # Asegurar que x sea positivo para logaritmo
                x_positivo = x + 0.001  # Evitar log(0)
                params, _ = curve_fit(funcion, x_positivo, y, p0=[1.0, 1.0], maxfev=5000)
            elif nombre == 'Potencial':
                params, _ = curve_fit(funcion, x, y, p0=[1.0, 1.0], maxfev=5000)
            else:
                params, _ = curve_fit(funcion, x, y, maxfev=5000)
            
            y_pred = funcion(x, *params)
            r2 = 1 - np.sum((y - y_pred)**2) / np.sum((y - np.mean(y))**2)
            
            if r2 > mejor_r2:
                mejor_r2 = r2
                mejor_modelo = nombre
                mejor_params = params
                mejor_funcion = funcion
                
        except Exception as e:
            continue
    
    return mejor_modelo, mejor_params, round(mejor_r2, 3), mejor_funcion

# FUNCIONES PARA GRÁFICOS - ANÁLISIS HIDRÁULICO COMPLETO
def crear_grafico_principal(df, curvas, titulo):
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Graficar puntos
    ax.scatter(df['NIVEL_AFORO'], df['CAUDAL'], alpha=0.7, s=60, color='blue', label='Datos de aforo')
    
    # Graficar curvas ajustadas
    for grupo, curva in curvas.items():
        H_range = np.linspace(curva['rango_niveles'][0], curva['rango_niveles'][1], 100)
        Q_curve = curva['funcion'](H_range, *curva['parametros'])
        ax.plot(H_range, Q_curve, label=f'{grupo} (R²={curva["r2"]:.3f})', linewidth=2)
    
    ax.set_xlabel('Nivel (m)', fontsize=12, fontweight='bold')
    ax.set_ylabel('Caudal (m³/s)', fontsize=12, fontweight='bold')
    ax.set_title(titulo, fontsize=14, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    return fig

def crear_graficos_hidraulicos(df, titulo_sufijo=""):
    """Crear gráficos completos de análisis hidráulico"""
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle(f'Análisis de Relaciones Hidráulicas {titulo_sufijo}', fontsize=16, fontweight='bold')
    
    # 1. Altura vs Área
    ax1 = axes[0, 0]
    ax1.scatter(df['NIVEL_AFORO'], df['AREA'], alpha=0.7, s=50, color='blue')
    
    # Ajustar modelo
    x_area = df['NIVEL_AFORO'].values
    y_area = df['AREA'].values
    modelo_area, params_area, r2_area, funcion_area = ajustar_modelo_relacion(x_area, y_area, "Altura-Área")
    
    if modelo_area and r2_area > 0:
        x_range = np.linspace(min(x_area), max(x_area), 100)
        if modelo_area == 'Logarítmico':
            y_pred = funcion_area(x_range + 0.001, *params_area)
        else:
            y_pred = funcion_area(x_range, *params_area)
        ax1.plot(x_range, y_pred, 'red', linewidth=2, label=f'{modelo_area} (R²={r2_area:.3f})')
    
    ax1.set_xlabel('Nivel (m)', fontweight='bold')
    ax1.set_ylabel('Área (m²)', fontweight='bold')
    ax1.set_title(f'Altura vs Área\nMejor modelo: {modelo_area}', fontweight='bold')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # 2. Altura vs Velocidad
    ax2 = axes[0, 1]
    ax2.scatter(df['NIVEL_AFORO'], df['VELOCIDAD'], alpha=0.7, s=50, color='green')
    
    x_vel = df['NIVEL_AFORO'].values
    y_vel = df['VELOCIDAD'].values
    modelo_vel, params_vel, r2_vel, funcion_vel = ajustar_modelo_relacion(x_vel, y_vel, "Altura-Velocidad")
    
    if modelo_vel and r2_vel > 0:
        x_range = np.linspace(min(x_vel), max(x_vel), 100)
        if modelo_vel == 'Logarítmico':
            y_pred = funcion_vel(x_range + 0.001, *params_vel)
        else:
            y_pred = funcion_vel(x_range, *params_vel)
        ax2.plot(x_range, y_pred, 'red', linewidth=2, label=f'{modelo_vel} (R²={r2_vel:.3f})')
    
    ax2.set_xlabel('Nivel (m)', fontweight='bold')
    ax2.set_ylabel('Velocidad (m/s)', fontweight='bold')
    ax2.set_title(f'Altura vs Velocidad\nMejor modelo: {modelo_vel}', fontweight='bold')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # 3. Altura vs Perímetro
    ax3 = axes[0, 2]
    ax3.scatter(df['NIVEL_AFORO'], df['PERIMETRO'], alpha=0.7, s=50, color='orange')
    
    x_per = df['NIVEL_AFORO'].values
    y_per = df['PERIMETRO'].values
    modelo_per, params_per, r2_per, funcion_per = ajustar_modelo_relacion(x_per, y_per, "Altura-Perímetro")
    
    if modelo_per and r2_per > 0:
        x_range = np.linspace(min(x_per), max(x_per), 100)
        if modelo_per == 'Logarítmico':
            y_pred = funcion_per(x_range + 0.001, *params_per)
        else:
            y_pred = funcion_per(x_range, *params_per)
        ax3.plot(x_range, y_pred, 'red', linewidth=2, label=f'{modelo_per} (R²={r2_per:.3f})')
    
    ax3.set_xlabel('Nivel (m)', fontweight='bold')
    ax3.set_ylabel('Perímetro (m)', fontweight='bold')
    ax3.set_title(f'Altura vs Perímetro\nMejor modelo: {modelo_per}', fontweight='bold')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # 4. Altura vs Ancho
    ax4 = axes[1, 0]
    ax4.scatter(df['NIVEL_AFORO'], df['ANCHO_RIO'], alpha=0.7, s=50, color='purple')
    
    x_ancho = df['NIVEL_AFORO'].values
    y_ancho = df['ANCHO_RIO'].values
    modelo_ancho, params_ancho, r2_ancho, funcion_ancho = ajustar_modelo_relacion(x_ancho, y_ancho, "Altura-Ancho")
    
    if modelo_ancho and r2_ancho > 0:
        x_range = np.linspace(min(x_ancho), max(x_ancho), 100)
        if modelo_ancho == 'Logarítmico':
            y_pred = funcion_ancho(x_range + 0.001, *params_ancho)
        else:
            y_pred = funcion_ancho(x_range, *params_ancho)
        ax4.plot(x_range, y_pred, 'red', linewidth=2, label=f'{modelo_ancho} (R²={r2_ancho:.3f})')
    
    ax4.set_xlabel('Nivel (m)', fontweight='bold')
    ax4.set_ylabel('Ancho Río (m)', fontweight='bold')
    ax4.set_title(f'Altura vs Ancho\nMejor modelo: {modelo_ancho}', fontweight='bold')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    
    # 5. Altura vs Radio Hidráulico
    ax5 = axes[1, 1]
    ax5.scatter(df['NIVEL_AFORO'], df['RADIO_HIDRAULICO'], alpha=0.7, s=50, color='brown')
    
    x_rh = df['NIVEL_AFORO'].values
    y_rh = df['RADIO_HIDRAULICO'].values
    modelo_rh, params_rh, r2_rh, funcion_rh = ajustar_modelo_relacion(x_rh, y_rh, "Altura-Radio Hidráulico")
    
    if modelo_rh and r2_rh > 0:
        x_range = np.linspace(min(x_rh), max(x_rh), 100)
        if modelo_rh == 'Logarítmico':
            y_pred = funcion_rh(x_range + 0.001, *params_rh)
        else:
            y_pred = funcion_rh(x_range, *params_rh)
        ax5.plot(x_range, y_pred, 'red', linewidth=2, label=f'{modelo_rh} (R²={r2_rh:.3f})')
    
    ax5.set_xlabel('Nivel (m)', fontweight='bold')
    ax5.set_ylabel('Radio Hidráulico (m)', fontweight='bold')
    ax5.set_title(f'Altura vs Radio Hidráulico\nMejor modelo: {modelo_rh}', fontweight='bold')
    ax5.legend()
    ax5.grid(True, alpha=0.3)
    
    # 6. Caudal vs Velocidad
    ax6 = axes[1, 2]
    ax6.scatter(df['CAUDAL'], df['VELOCIDAD'], alpha=0.7, s=50, color='teal')
    
    x_caudal = df['CAUDAL'].values
    y_caudal = df['VELOCIDAD'].values
    modelo_caudal, params_caudal, r2_caudal, funcion_caudal = ajustar_modelo_relacion(x_caudal, y_caudal, "Caudal-Velocidad")
    
    if modelo_caudal and r2_caudal > 0:
        x_range = np.linspace(min(x_caudal), max(x_caudal), 100)
        if modelo_caudal == 'Logarítmico':
            y_pred = funcion_caudal(x_range + 0.001, *params_caudal)
        else:
            y_pred = funcion_caudal(x_range, *params_caudal)
        ax6.plot(x_range, y_pred, 'red', linewidth=2, label=f'{modelo_caudal} (R²={r2_caudal:.3f})')
    
    ax6.set_xlabel('Caudal (m³/s)', fontweight='bold')
    ax6.set_ylabel('Velocidad (m/s)', fontweight='bold')
    ax6.set_title(f'Caudal vs Velocidad\nMejor modelo: {modelo_caudal}', fontweight='bold')
    ax6.legend()
    ax6.grid(True, alpha=0.3)
    
    plt.tight_layout()
    return fig

# FUNCIÓN PARA PROCESAR CON MODELO (SIMPLIFICADA)
def procesar_con_modelo(df):
    """Procesar datos y generar curvas"""
    
    df_procesado = preparar_datos(df)
    
    # Generar curvas para todos los datos
    curva = ajustar_curva(df_procesado)
    
    resultados = {}
    if curva:
        resultados['CURVA_PRINCIPAL'] = curva
    
    return resultados, df_procesado

# CONFIGURACIÓN STREAMLIT
st.set_page_config(page_title="Sistema Talapalca", page_icon="🌊", layout="wide")
st.title("🌊 IA para la generación de Curvas Altura-Caudal")
st.markdown("**Análisis hidráulico completo con relaciones altura-área-velocidad-perímetro**")

# NAVEGACIÓN
opcion = st.sidebar.radio("Navegación:", ["🏠 Inicio", "📤 Subir Aforos", "📊 Ingreso Manual", "📈 Análisis Hidráulico"])

if opcion == "🏠 Inicio":
    st.header("Bienvenido al Sistema de Análisis Hidráulico")
    st.info("Aplicación IA para generar curvas altura-caudal y análisis de relaciones hidráulicas completas")
    
    st.subheader("📊 Funcionalidades principales:")
    st.markdown("""
    - **Curvas Altura-Caudal**: Generación automática de ecuaciones
    - **Análisis hidráulico completo**:
      - Altura vs Área
      - Altura vs Velocidad  
      - Altura vs Perímetro
      - Altura vs Ancho del río
      - Altura vs Radio hidráulico
      - Caudal vs Velocidad
    - **Cálculo automático** de perímetro y variables hidráulicas
    - **Múltiples modelos**: Lineal, Polinómico, Exponencial, Logarítmico, Potencial
    """)
    
    st.subheader("📋 Columnas requeridas en CSV:")
    st.markdown("""
    - `NIVEL DE AFORO (m)` - **Requerido**
    - `CAUDAL (m3/s)` - **Requerido**
    - `AREA (m2)` - **Requerido**
    - `ANCHO RIO (m)` - **Requerido**
    - `VELOCIDAD (m/s)` - **Requerido**
    - `PERIMETRO (m)` - Opcional (se calcula automáticamente)
    - `FECHA AFORO` - Opcional
    """)

elif opcion == "📤 Subir Aforos":
    st.header("📤 Subir Archivo de Aforos")
    
    archivo_subido = st.file_uploader("Selecciona archivo CSV", type=['csv'])
    
    if archivo_subido is not None:
        try:
            df = pd.read_csv(archivo_subido)
            st.success(f"✅ {len(df)} aforos cargados exitosamente")
            
            # Mostrar vista previa
            st.subheader("📋 Vista previa de datos")
            st.dataframe(df.head())
            
            # Verificar columnas básicas
            columnas_necesarias = ['CAUDAL (m3/s)', 'VELOCIDAD (m/s)', 'AREA (m2)', 'ANCHO RIO (m)', 'NIVEL DE AFORO (m)']
            columnas_faltantes = [col for col in columnas_necesarias if col not in df.columns]
            
            if not columnas_faltantes:
                st.success("✅ Todas las columnas necesarias están presentes")
                
                # Procesar datos
                if st.button("🚀 Procesar Aforos", type="primary"):
                    with st.spinner("Procesando datos y generando curvas..."):
                        curvas, df_procesado = procesar_con_modelo(df)
                        
                        if curvas:
                            st.success("✅ Análisis completado exitosamente")
                            
                            # Mostrar datos procesados
                            st.subheader("📊 Datos Procesados")
                            st.dataframe(df_procesado[['NIVEL_AFORO', 'CAUDAL', 'VELOCIDAD', 'AREA', 'ANCHO_RIO', 'PERIMETRO', 'RADIO_HIDRAULICO']].head(10))
                            
                            # Gráfico principal
                            st.subheader("📈 Curva Altura-Caudal")
                            fig_principal = crear_grafico_principal(df_procesado, curvas, "Curva Altura-Caudal")
                            st.pyplot(fig_principal)
                            
                            # Mostrar ecuaciones
                            st.subheader("📐 Ecuaciones Generadas")
                            for grupo, curva in curvas.items():
                                with st.expander(f"{grupo} - R² = {curva['r2']:.3f}"):
                                    st.write(f"**Tipo de modelo:** {curva['nombre']}")
                                    st.write(f"**Puntos utilizados:** {curva['n_puntos']}")
                                    st.write(f"**Rango de niveles:** {curva['rango_niveles'][0]:.2f} - {curva['rango_niveles'][1]:.2f} m")
                                    st.write(f"**Rango de caudales:** {curva['rango_caudales'][0]:.2f} - {curva['rango_caudales'][1]:.2f} m³/s")
                                    
                                    if curva['nombre'] == 'Polinómico G2':
                                        a, b, c = curva['parametros']
                                        st.latex(f"Q = {a:.4f}H^2 + {b:.4f}H + {c:.4f}")
                                    elif curva['nombre'] == 'Polinómico G3':
                                        a, b, c, d = curva['parametros']
                                        st.latex(f"Q = {a:.4f}H^3 + {b:.4f}H^2 + {c:.4f}H + {d:.4f}")
                                    elif curva['nombre'] == 'Potencial':
                                        a, b = curva['parametros']
                                        st.latex(f"Q = {a:.4f}H^{{{b:.4f}}}")
                            
                            # Guardar en session state para análisis hidráulico
                            st.session_state.df_procesado = df_procesado
                            st.session_state.curvas = curvas
                            
                        else:
                            st.warning("⚠️ No se pudieron generar curvas con los datos proporcionados")
                            
            else:
                st.error(f"❌ Faltan las siguientes columnas necesarias: {', '.join(columnas_faltantes)}")
                st.info("💡 Asegúrate de que tu archivo CSV tenga las columnas con los nombres exactos.")
                    
        except Exception as e:
            st.error(f"❌ Error al procesar el archivo: {e}")

elif opcion == "📊 Ingreso Manual":
    st.header("📊 Ingreso Manual de Aforos")
    
    st.info("💡 Ingresa los datos de aforo manualmente. El perímetro se calculará automáticamente si no se proporciona.")
    
    num_aforos = st.number_input("Número de aforos a ingresar:", min_value=1, max_value=20, value=3)
    datos_manual = []
    
    for i in range(num_aforos):
        with st.expander(f"Aforo {i+1}", expanded=True if i == 0 else False):
            col1, col2 = st.columns(2)
            with col1:
                nivel = st.number_input("Nivel (m)", min_value=0.1, max_value=10.0, value=1.0, step=0.1, key=f"n{i}")
                caudal = st.number_input("Caudal (m³/s)", min_value=0.1, max_value=50.0, value=2.0, step=0.1, key=f"q{i}")
                area = st.number_input("Área (m²)", min_value=0.1, max_value=50.0, value=3.0, step=0.1, key=f"a{i}")
            with col2:
                ancho = st.number_input("Ancho Río (m)", min_value=0.1, max_value=20.0, value=8.0, step=0.1, key=f"w{i}")
                perimetro = st.number_input("Perímetro (m)", min_value=0.1, max_value=30.0, value=0.0, step=0.1, 
                                          help="Dejar en 0 para cálculo automático", key=f"p{i}")
                velocidad = st.number_input("Velocidad (m/s)", min_value=0.1, max_value=5.0, value=0.7, step=0.1, key=f"v{i}")
            
            datos_manual.append({
                'FECHA AFORO': '2024-01-01',
                'NIVEL DE AFORO (m)': nivel,
                'CAUDAL (m3/s)': caudal,
                'AREA (m2)': area,
                'ANCHO RIO (m)': ancho,
                'PERIMETRO (m)': perimetro if perimetro > 0 else None,
                'VELOCIDAD (m/s)': velocidad
            })
    
    if st.button("🚀 Procesar Datos Manuales", type="primary") and datos_manual:
        with st.spinner("Procesando datos manuales..."):
            df_manual = pd.DataFrame(datos_manual)
            curvas, datos_procesados = procesar_con_modelo(df_manual)
            
            if curvas:
                st.success("✅ Datos procesados exitosamente")
                
                st.subheader("📊 Datos Procesados")
                st.dataframe(datos_procesados[['NIVEL_AFORO', 'CAUDAL', 'VELOCIDAD', 'AREA', 'ANCHO_RIO', 'PERIMETRO', 'RADIO_HIDRAULICO']])
                
                st.subheader("📈 Curva Generada")
                fig = crear_grafico_principal(datos_procesados, curvas, "Curva Altura-Caudal - Datos Manuales")
                st.pyplot(fig)
                
                # Mostrar ecuaciones
                st.subheader("📐 Ecuación Generada")
                for grupo, curva in curvas.items():
                    with st.expander(f"{grupo} - R² = {curva['r2']:.3f}"):
                        st.write(f"**Tipo de modelo:** {curva['nombre']}")
                        st.write(f"**Puntos utilizados:** {curva['n_puntos']}")
                        st.write(f"**Rango de niveles:** {curva['rango_niveles'][0]:.2f} - {curva['rango_niveles'][1]:.2f} m")
                        st.write(f"**Rango de caudales:** {curva['rango_caudales'][0]:.2f} - {curva['rango_caudales'][1]:.2f} m³/s")
                        
                        if curva['nombre'] == 'Polinómico G2':
                            a, b, c = curva['parametros']
                            st.latex(f"Q = {a:.4f}H^2 + {b:.4f}H + {c:.4f}")
                        elif curva['nombre'] == 'Polinómico G3':
                            a, b, c, d = curva['parametros']
                            st.latex(f"Q = {a:.4f}H^3 + {b:.4f}H^2 + {c:.4f}H + {d:.4f}")
                        elif curva['nombre'] == 'Potencial':
                            a, b = curva['parametros']
                            st.latex(f"Q = {a:.4f}H^{{{b:.4f}}}")
                
                # Guardar para análisis hidráulico
                st.session_state.df_procesado = datos_procesados
                st.session_state.curvas = curvas
                
            else:
                st.warning("⚠️ No se pudieron generar curvas con los datos ingresados. Intenta con más puntos o diferentes valores.")

elif opcion == "📈 Análisis Hidráulico":
    st.header("📈 Análisis Hidráulico Completo")
    
    if 'df_procesado' not in st.session_state:
        st.warning("⚠️ Primero debes procesar datos en las pestañas 'Subir Aforos' o 'Ingreso Manual'")
        st.info("💡 Usando datos de demostración para mostrar funcionalidades...")
        
        # Datos de demostración
        datos_demo = {
            'FECHA AFORO': ['2/10/2021', '2/24/2021', '4/13/2021', '5/11/2021', '9/26/2021'],
            'NIVEL DE AFORO (m)': [1.2, 1.01, 1.05, 0.97, 0.96],
            'CAUDAL (m3/s)': [4.79, 1.89, 2.15, 1.11, 1.1],
            'AREA (m2)': [3.47, 2.76, 2.64, 2.07, 1.97],
            'ANCHO RIO (m)': [8.5, 8.5, 8.3, 7.5, 7.5],
            'VELOCIDAD (m/s)': [1.38, 0.62, 0.74, 0.46, 0.48]
        }
        df_demo = pd.DataFrame(datos_demo)
        curvas_demo, df_procesado_demo = procesar_con_modelo(df_demo)
        st.session_state.df_procesado = df_procesado_demo
        st.session_state.curvas = curvas_demo
    
    df_procesado = st.session_state.df_procesado
    
    st.subheader("🔍 Relaciones Hidráulicas Complejas")
    
    # Gráficos de análisis hidráulico
    fig_hidraulico = crear_graficos_hidraulicos(df_procesado)
    st.pyplot(fig_hidraulico)
    
    # Resumen estadístico
    st.subheader("📊 Resumen Estadístico")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Número de aforos", len(df_procesado))
        st.metric("Nivel promedio", f"{df_procesado['NIVEL_AFORO'].mean():.2f} m")
    
    with col2:
        st.metric("Caudal promedio", f"{df_procesado['CAUDAL'].mean():.2f} m³/s")
        st.metric("Velocidad promedio", f"{df_procesado['VELOCIDAD'].mean():.2f} m/s")
    
    with col3:
        st.metric("Área promedio", f"{df_procesado['AREA'].mean():.2f} m²")
        st.metric("Radio hidráulico promedio", f"{df_procesado['RADIO_HIDRAULICO'].mean():.2f} m")
    
    # Tabla de correlaciones
    st.subheader("🔗 Matriz de Correlaciones")
    columnas_corr = ['NIVEL_AFORO', 'CAUDAL', 'AREA', 'VELOCIDAD', 'ANCHO_RIO', 'PERIMETRO', 'RADIO_HIDRAULICO']
    correlaciones = df_procesado[columnas_corr].corr()
    
    # Mostrar matriz de correlación
    fig_corr, ax_corr = plt.subplots(figsize=(10, 8))
    im = ax_corr.imshow(correlaciones, cmap='coolwarm', aspect='auto', vmin=-1, vmax=1)
    
    # Añadir valores
    for i in range(len(correlaciones)):
        for j in range(len(correlaciones)):
            text = ax_corr.text(j, i, f'{correlaciones.iloc[i, j]:.2f}',
                           ha="center", va="center", color="w", fontweight='bold')
    
    ax_corr.set_xticks(range(len(correlaciones.columns)))
    ax_corr.set_yticks(range(len(correlaciones.columns)))
    ax_corr.set_xticklabels(correlaciones.columns, rotation=45)
    ax_corr.set_yticklabels(correlaciones.columns)
    ax_corr.set_title('Matriz de Correlación', fontweight='bold')
    plt.colorbar(im)
    st.pyplot(fig_corr)

st.markdown("---")
st.markdown("**🌊 Sistema de Análisis Hidráulico - Curvas H-Q**")