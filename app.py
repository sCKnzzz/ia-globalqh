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
    
    # Estimar perímetro si falta o es cero
    if 'PERIMETRO' not in df_procesado.columns or df_procesado['PERIMETRO'].isna().any() or (df_procesado['PERIMETRO'] == 0).any():
        # Calcular tirante medio
        df_procesado['TIRANTE_MEDIO'] = df_procesado['AREA'] / df_procesado['ANCHO_RIO']
        # Estimar perímetro (aproximación para sección rectangular - USGS Standard)
        df_procesado['PERIMETRO'] = 2 * df_procesado['TIRANTE_MEDIO'] + df_procesado['ANCHO_RIO']
        st.info("📏 Perímetro calculado automáticamente usando aproximación rectangular (USGS Standard)")
    
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

# FUNCIÓN MEJORADA PARA AJUSTAR MODELOS SEGÚN LITERATURA USGS/WMO
def ajustar_modelo_hidraulico(x, y, tipo_relacion):
    """Ajustar modelos según literatura USGS/WMO para diferentes relaciones hidráulicas"""
    
    # Modelos recomendados por USGS/WMO para diferentes relaciones
    modelos_recomendados = {
        'altura_area': [
            ('Potencial', func_pot),  # USGS: Q = a * A^b común en secciones naturales
            ('Polinómico G2', func_poly2),
            ('Lineal', lambda x, a, b: a * x + b)
        ],
        'altura_velocidad': [
            ('Logarítmico', func_log),  # WMO: Velocidad sigue perfil logarítmico
            ('Potencial', func_pot),
            ('Polinómico G2', func_poly2)
        ],
        'altura_perimetro': [
            ('Lineal', lambda x, a, b: a * x + b),  # USGS: Aproximación lineal común
            ('Polinómico G2', func_poly2),
            ('Potencial', func_pot)
        ],
        'altura_ancho': [
            ('Lineal', lambda x, a, b: a * x + b),  # Para ríos con márgenes regulares
            ('Potencial', func_pot),
            ('Polinómico G2', func_poly2)
        ],
        'altura_radio_hidraulico': [
            ('Potencial', func_pot),  # USGS: Relación potencial común
            ('Lineal', lambda x, a, b: a * x + b),
            ('Logarítmico', func_log)
        ],
        'caudal_velocidad': [
            ('Potencial', func_pot),  # WMO: V = a * Q^b
            ('Lineal', lambda x, a, b: a * x + b),
            ('Polinómico G2', func_poly2)
        ]
    }
    
    mejor_r2 = -np.inf
    mejor_modelo = None
    mejor_params = None
    mejor_funcion = None
    
    modelos = modelos_recomendados.get(tipo_relacion, [
        ('Lineal', lambda x, a, b: a * x + b),
        ('Polinómico G2', func_poly2),
        ('Potencial', func_pot)
    ])
    
    for nombre, funcion in modelos:
        try:
            if nombre == 'Exponencial':
                params, _ = curve_fit(funcion, x, y, p0=[1.0, 0.1], maxfev=5000)
            elif nombre == 'Logarítmico':
                # Asegurar que x sea positivo para logaritmo
                x_positivo = x - min(x) + 0.001  # Evitar log(0)
                params, _ = curve_fit(funcion, x_positivo, y, p0=[1.0, 1.0], maxfev=5000)
            elif nombre == 'Potencial':
                # Evitar valores negativos o cero
                x_positivo = np.maximum(x, 0.001)
                y_positivo = np.maximum(y, 0.001)
                params, _ = curve_fit(funcion, x_positivo, y_positivo, p0=[1.0, 1.0], maxfev=5000)
            else:
                params, _ = curve_fit(funcion, x, y, maxfev=5000)
            
            # Predecir y calcular R²
            if nombre == 'Logarítmico':
                x_pred = x - min(x) + 0.001
                y_pred = funcion(x_pred, *params)
            elif nombre == 'Potencial':
                x_pred = np.maximum(x, 0.001)
                y_pred = funcion(x_pred, *params)
            else:
                y_pred = funcion(x, *params)
            
            r2 = 1 - np.sum((y - y_pred)**2) / np.sum((y - np.mean(y))**2)
            
            if r2 > mejor_r2 and r2 > 0:  # Aceptar modelos con R² positivo
                mejor_r2 = r2
                mejor_modelo = nombre
                mejor_params = params
                mejor_funcion = funcion
                
        except Exception as e:
            continue
    
    return mejor_modelo, mejor_params, round(mejor_r2, 3) if mejor_r2 > -np.inf else 0, mejor_funcion

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
    """Crear gráficos completos de análisis hidráulico basados en literatura USGS/WMO"""
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle(f'Análisis de Relaciones Hidráulicas {titulo_sufijo}\n(Basado en estándares USGS/WMO)', 
                 fontsize=16, fontweight='bold')
    
    relaciones_info = {
        'altura_area': {
            'ax': axes[0, 0], 'color': 'blue', 'ylabel': 'Área (m²)',
            'title': 'Altura vs Área\n(USGS: Relación Potencial común)'
        },
        'altura_velocidad': {
            'ax': axes[0, 1], 'color': 'green', 'ylabel': 'Velocidad (m/s)',
            'title': 'Altura vs Velocidad\n(WMO: Perfil Logarítmico)'
        },
        'altura_perimetro': {
            'ax': axes[0, 2], 'color': 'orange', 'ylabel': 'Perímetro (m)',
            'title': 'Altura vs Perímetro\n(USGS: Aproximación Lineal)'
        },
        'altura_ancho': {
            'ax': axes[1, 0], 'color': 'purple', 'ylabel': 'Ancho Río (m)',
            'title': 'Altura vs Ancho\n(USGS: Relación Lineal/Potencial)'
        },
        'altura_radio_hidraulico': {
            'ax': axes[1, 1], 'color': 'brown', 'ylabel': 'Radio Hidráulico (m)',
            'title': 'Altura vs Radio Hidráulico\n(USGS: Relación Potencial)'
        },
        'caudal_velocidad': {
            'ax': axes[1, 2], 'color': 'teal', 'ylabel': 'Velocidad (m/s)',
            'title': 'Caudal vs Velocidad\n(WMO: V = aQ^b)'
        }
    }
    
    for relacion, info in relaciones_info.items():
        ax = info['ax']
        color = info['color']
        
        # Determinar variables x e y según la relación
        if relacion == 'altura_area':
            x, y = df['NIVEL_AFORO'].values, df['AREA'].values
        elif relacion == 'altura_velocidad':
            x, y = df['NIVEL_AFORO'].values, df['VELOCIDAD'].values
        elif relacion == 'altura_perimetro':
            x, y = df['NIVEL_AFORO'].values, df['PERIMETRO'].values
        elif relacion == 'altura_ancho':
            x, y = df['NIVEL_AFORO'].values, df['ANCHO_RIO'].values
        elif relacion == 'altura_radio_hidraulico':
            x, y = df['NIVEL_AFORO'].values, df['RADIO_HIDRAULICO'].values
        elif relacion == 'caudal_velocidad':
            x, y = df['CAUDAL'].values, df['VELOCIDAD'].values
        
        # Graficar puntos
        ax.scatter(x, y, alpha=0.7, s=50, color=color)
        
        # Ajustar modelo según literatura
        modelo, params, r2, funcion = ajustar_modelo_hidraulico(x, y, relacion)
        
        # Graficar curva del mejor modelo
        if modelo and r2 > 0:
            x_range = np.linspace(min(x), max(x), 100)
            
            try:
                if modelo == 'Logarítmico':
                    y_pred = funcion(x_range - min(x_range) + 0.001, *params)
                elif modelo == 'Potencial':
                    x_range_pos = np.maximum(x_range, 0.001)
                    y_pred = funcion(x_range_pos, *params)
                else:
                    y_pred = funcion(x_range, *params)
                
                ax.plot(x_range, y_pred, 'red', linewidth=2, 
                       label=f'{modelo} (R²={r2:.3f})')
            except:
                pass
        
        # Configurar ejes según la relación
        if 'altura' in relacion:
            ax.set_xlabel('Nivel (m)', fontweight='bold')
        elif 'caudal' in relacion:
            ax.set_xlabel('Caudal (m³/s)', fontweight='bold')
        
        ax.set_ylabel(info['ylabel'], fontweight='bold')
        ax.set_title(info['title'], fontweight='bold')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
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
st.markdown("**Análisis hidráulico completo basado en estándares USGS/WMO**")

# NAVEGACIÓN
opcion = st.sidebar.radio("Navegación:", ["🏠 Inicio", "📤 Subir Aforos", "📊 Ingreso Manual"])

if opcion == "🏠 Inicio":
    st.header("Bienvenido al Sistema de Análisis Hidráulico")
    st.info("Aplicación IA para generar curvas altura-caudal y análisis de relaciones hidráulicas basado en estándares USGS/WMO")
    
    st.subheader("📊 Funcionalidades principales:")
    st.markdown("""
    - **Curvas Altura-Caudal**: Generación automática de ecuaciones
    - **Análisis hidráulico completo** basado en literatura USGS/WMO:
      - ⚡ **Altura vs Velocidad**: Modelo logarítmico (WMO - perfil de velocidad)
      - 📐 **Altura vs Área**: Modelo potencial (USGS - secciones naturales)
      - 📏 **Altura vs Perímetro**: Modelo lineal (USGS - aproximación)
      - 🌊 **Altura vs Ancho**: Modelo lineal/potencial (USGS - márgenes regulares)
      - 🔵 **Altura vs Radio hidráulico**: Modelo potencial (USGS)
      - 💨 **Caudal vs Velocidad**: Modelo potencial (WMO - V = aQ^b)
    """)
    
    st.subheader("🏛️ Basado en estándares internacionales:")
    st.markdown("""
    - **USGS** (United States Geological Survey)
    - **WMO** (World Meteorological Organization)
    - **Manuales de hidrometría internacional**
    """)
    
    st.subheader("📋 Columnas requeridas en CSV:")
    st.markdown("""
    - `NIVEL DE AFORO (m)` - **Requerido**
    - `CAUDAL (m3/s)` - **Requerido**
    - `AREA (m2)` - **Requerido**
    - `ANCHO RIO (m)` - **Requerido**
    - `VELOCIDAD (m/s)` - **Requerido**
    - `PERIMETRO (m)` - Opcional (se calcula automáticamente según USGS)
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
                if st.button("🚀 Procesar Aforos y Análisis Hidráulico", type="primary"):
                    with st.spinner("Procesando datos y generando análisis completo..."):
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
                            
                            # ANÁLISIS HIDRÁULICO COMPLETO
                            st.subheader("🔍 Análisis Hidráulico Completo (USGS/WMO)")
                            st.info("""
                            **Relaciones basadas en estándares internacionales:**
                            - ⚡ **Velocidad**: Perfil logarítmico (WMO)
                            - 📐 **Área**: Relación potencial (USGS)
                            - 📏 **Perímetro**: Aproximación lineal (USGS)
                            - 🌊 **Ancho**: Relación lineal/potencial (USGS)
                            - 🔵 **Radio hidráulico**: Relación potencial (USGS)
                            - 💨 **Caudal-Velocidad**: Ley potencial V = aQ^b (WMO)
                            """)
                            
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
                            
                        else:
                            st.warning("⚠️ No se pudieron generar curvas con los datos proporcionados")
                            
            else:
                st.error(f"❌ Faltan las siguientes columnas necesarias: {', '.join(columnas_faltantes)}")
                st.info("💡 Asegúrate de que tu archivo CSV tenga las columnas con los nombres exactos.")
                    
        except Exception as e:
            st.error(f"❌ Error al procesar el archivo: {e}")

elif opcion == "📊 Ingreso Manual":
    st.header("📊 Ingreso Manual de Aforos")
    
    st.info("💡 Ingresa los datos de aforo manualmente. El perímetro se calculará automáticamente según estándares USGS si no se proporciona.")
    
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
                # CORREGIDO: Sin min_value para permitir 0
                perimetro = st.number_input("Perímetro (m)", value=0.0, step=0.1, 
                                          help="Dejar en 0 para cálculo automático según USGS", key=f"p{i}")
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
                
                # ANÁLISIS HIDRÁULICO COMPLETO
                st.subheader("🔍 Análisis Hidráulico Completo (USGS/WMO)")
                fig_hidraulico = crear_graficos_hidraulicos(datos_procesados, "(Datos Manuales)")
                st.pyplot(fig_hidraulico)
                
            else:
                st.warning("⚠️ No se pudieron generar curvas con los datos ingresados. Intenta con más puntos o diferentes valores.")

st.markdown("---")
st.markdown("**🌊 Sistema de Análisis Hidráulico - Basado en estándares USGS/WMO**")