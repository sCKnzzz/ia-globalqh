import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import joblib
from scipy.optimize import curve_fit
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
import io
import plotly

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
    # Manejar valores <= 0 para logaritmo
    x_safe = np.maximum(x, 0.001)
    return a * np.log(x_safe + b)

# FUNCIÓN PARA PREPARAR DATOS
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
    
    # CALCULAR PERÍMETRO SI ES NECESARIO
    if 'PERIMETRO' not in df_procesado.columns or df_procesado['PERIMETRO'].isna().any() or (df_procesado['PERIMETRO'] <= 0).any():
        df_procesado['TIRANTE_MEDIO'] = df_procesado['AREA'] / df_procesado['ANCHO_RIO']
        df_procesado['PERIMETRO'] = df_procesado['ANCHO_RIO'] + 2 * df_procesado['TIRANTE_MEDIO']
    
    # Calcular variables hidráulicas
    df_procesado['RADIO_HIDRAULICO'] = df_procesado['AREA'] / df_procesado['PERIMETRO']
    df_procesado['TIRANTE_MEDIO'] = df_procesado['AREA'] / df_procesado['ANCHO_RIO']
    
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

# FUNCIÓN PARA CLASIFICAR GRUPOS
def clasificar_grupos(df):
    """Clasificar datos en grupos según características hidráulicas"""
    df_clasificado = df.copy()
    
    # Clasificación basada en radio hidráulico y año
    condiciones = [
        (df_clasificado['RADIO_HIDRAULICO'] > 0.6),  # GRUPO_ALTO_RH
        (df_clasificado['YEAR'] >= 2023),            # GRUPO_RECIENTE
    ]
    
    grupos = ['GRUPO_ALTO_RH', 'GRUPO_RECIENTE']
    df_clasificado['GRUPO_PREDICHO'] = np.select(condiciones, grupos, default='GRUPO_ESTANDAR')
    
    return df_clasificado

# FUNCIÓN PARA AJUSTAR CURVAS
def ajustar_curva_grupo(datos_grupo, nombre_grupo):
    try:
        H = datos_grupo['NIVEL_AFORO'].values
        Q = datos_grupo['CAUDAL'].values
        
        if len(H) < 3:
            return None
        
        # Ordenar datos
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
                    # Para función potencial, asegurar valores positivos
                    H_pos = np.maximum(H_sorted, 0.01)
                    Q_pos = np.maximum(Q_sorted, 0.01)
                    params, _ = curve_fit(funcion, H_pos, Q_pos, p0=[1.0, 2.0], maxfev=5000)
                    Q_pred = funcion(H_pos, *params)
                else:
                    params, _ = curve_fit(funcion, H_sorted, Q_sorted, maxfev=5000)
                    Q_pred = funcion(H_sorted, *params)
                
                # Calcular R²
                ss_res = np.sum((Q_sorted - Q_pred)**2)
                ss_tot = np.sum((Q_sorted - np.mean(Q_sorted))**2)
                
                if ss_tot > 0:
                    r2 = 1 - (ss_res / ss_tot)
                else:
                    r2 = 0
                
                # Aceptar modelos con R² razonable
                if r2 > mejor_r2 and r2 > 0.5:
                    mejor_r2 = r2
                    mejor_modelo = {
                        'nombre': nombre,
                        'funcion': funcion,
                        'parametros': params,
                        'r2': round(r2, 3),
                        'n_puntos': len(H_sorted),
                        'rango_niveles': (min(H_sorted), max(H_sorted)),
                        'rango_caudales': (min(Q_sorted), max(Q_sorted)),
                        'grupo': nombre_grupo
                    }
                    
            except Exception as e:
                continue
        
        return mejor_modelo
            
    except Exception as e:
        return None

# FUNCIÓN PARA PROCESAR CON CLASIFICACIÓN DE GRUPOS
def procesar_con_clasificacion(df, incluir_alto_rh=True):
    """Procesar datos con clasificación por grupos"""
    
    df_procesado = preparar_datos(df)
    df_clasificado = clasificar_grupos(df_procesado)
    
    # Filtrar si no incluir GRUPO_ALTO_RH
    if not incluir_alto_rh:
        df_filtrado = df_clasificado[df_clasificado['GRUPO_PREDICHO'] != 'GRUPO_ALTO_RH'].copy()
    else:
        df_filtrado = df_clasificado.copy()
    
    # Generar curvas para cada grupo
    resultados = {}
    
    for grupo in df_filtrado['GRUPO_PREDICHO'].unique():
        grupo_data = df_filtrado[df_filtrado['GRUPO_PREDICHO'] == grupo]
        
        if len(grupo_data) >= 3:
            curva = ajustar_curva_grupo(grupo_data, grupo)
            if curva:
                resultados[grupo] = curva
    
    return resultados, df_filtrado

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

# FUNCIÓN PARA GRÁFICO PRINCIPAL
def crear_grafico_curvas(df, curvas, titulo):
    fig, ax = plt.subplots(figsize=(10, 6))
    
    colores = {'GRUPO_ALTO_RH': 'red', 'GRUPO_RECIENTE': 'blue', 'GRUPO_ESTANDAR': 'green'}
    
    # Graficar puntos por grupo
    for grupo in df['GRUPO_PREDICHO'].unique():
        color = colores.get(grupo, 'orange')
        grupo_data = df[df['GRUPO_PREDICHO'] == grupo]
        
        ax.scatter(grupo_data['NIVEL_AFORO'], grupo_data['CAUDAL'], 
                  color=color, s=60, label=grupo, alpha=0.7)
    
    # Graficar curvas ajustadas
    for grupo, curva in curvas.items():
        color = colores.get(grupo, 'orange')
        H_range = np.linspace(curva['rango_niveles'][0], curva['rango_niveles'][1], 100)
        Q_curve = curva['funcion'](H_range, *curva['parametros'])
        
        ax.plot(H_range, Q_curve, color=color, linewidth=2, 
               label=f'{grupo} (R²={curva["r2"]:.3f})')
    
    ax.set_xlabel('Nivel (m)', fontsize=12, fontweight='bold')
    ax.set_ylabel('Caudal (m³/s)', fontsize=12, fontweight='bold')
    ax.set_title(titulo, fontsize=14, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    return fig

# FUNCIÓN PARA GRÁFICOS HIDRÁULICOS USGS/WMO
def crear_graficos_hidraulicos(df, titulo_sufijo=""):
    """Crear gráficos completos de análisis hidráulico basados en literatura USGS/WMO"""
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle(f'Análisis de Relaciones Hidráulicas {titulo_sufijo}\n(Basado en estándares USGS/WMO)', 
                 fontsize=16, fontweight='bold')
    
    colores = {'GRUPO_ALTO_RH': 'red', 'GRUPO_RECIENTE': 'blue', 'GRUPO_ESTANDAR': 'green'}
    
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
        
        # Graficar puntos por grupo
        for grupo in df['GRUPO_PREDICHO'].unique():
            color = colores.get(grupo, 'orange')
            grupo_data = df[df['GRUPO_PREDICHO'] == grupo]
            
            # Determinar variables x e y según la relación
            if relacion == 'altura_area':
                x, y = grupo_data['NIVEL_AFORO'].values, grupo_data['AREA'].values
            elif relacion == 'altura_velocidad':
                x, y = grupo_data['NIVEL_AFORO'].values, grupo_data['VELOCIDAD'].values
            elif relacion == 'altura_perimetro':
                x, y = grupo_data['NIVEL_AFORO'].values, grupo_data['PERIMETRO'].values
            elif relacion == 'altura_ancho':
                x, y = grupo_data['NIVEL_AFORO'].values, grupo_data['ANCHO_RIO'].values
            elif relacion == 'altura_radio_hidraulico':
                x, y = grupo_data['NIVEL_AFORO'].values, grupo_data['RADIO_HIDRAULICO'].values
            elif relacion == 'caudal_velocidad':
                x, y = grupo_data['CAUDAL'].values, grupo_data['VELOCIDAD'].values
            
            ax.scatter(x, y, alpha=0.7, s=50, color=color, label=grupo)
        
        # Ajustar modelo para todos los datos
        if relacion == 'altura_area':
            x_all, y_all = df['NIVEL_AFORO'].values, df['AREA'].values
        elif relacion == 'altura_velocidad':
            x_all, y_all = df['NIVEL_AFORO'].values, df['VELOCIDAD'].values
        elif relacion == 'altura_perimetro':
            x_all, y_all = df['NIVEL_AFORO'].values, df['PERIMETRO'].values
        elif relacion == 'altura_ancho':
            x_all, y_all = df['NIVEL_AFORO'].values, df['ANCHO_RIO'].values
        elif relacion == 'altura_radio_hidraulico':
            x_all, y_all = df['NIVEL_AFORO'].values, df['RADIO_HIDRAULICO'].values
        elif relacion == 'caudal_velocidad':
            x_all, y_all = df['CAUDAL'].values, df['VELOCIDAD'].values
        
        modelo, params, r2, funcion = ajustar_modelo_hidraulico(x_all, y_all, relacion)
        
        # Graficar curva del mejor modelo
        if modelo and r2 > 0:
            x_range = np.linspace(min(x_all), max(x_all), 100)
            
            try:
                if modelo == 'Logarítmico':
                    y_pred = funcion(x_range - min(x_range) + 0.001, *params)
                elif modelo == 'Potencial':
                    x_range_pos = np.maximum(x_range, 0.001)
                    y_pred = funcion(x_range_pos, *params)
                else:
                    y_pred = funcion(x_range, *params)
                
                ax.plot(x_range, y_pred, 'black', linewidth=2, linestyle='--',
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

# CONFIGURACIÓN STREAMLIT
st.set_page_config(page_title="Curvas Altura-Caudal", page_icon="🌊", layout="wide")
st.title("🌊 Generador de Curvas Altura-Caudal")
st.markdown("**Sistema inteligente con clasificación por grupos y análisis USGS/WMO**")

# NAVEGACIÓN
opcion = st.sidebar.radio("Navegación:", ["🏠 Inicio", "📤 Subir Aforos", "📊 Ingreso Manual"])

if opcion == "🏠 Inicio":
    st.header("Bienvenido al Sistema de Análisis Hidráulico Inteligente")
    st.info("Sistema IA con clasificación automática por grupos hidráulicos y análisis basado en estándares USGS/WMO")
    
    st.subheader("🎯 Clasificación por Grupos:")
    st.markdown("""
    - **🔴 GRUPO_ALTO_RH**: Datos con Radio Hidráulico > 0.6 m
    - **🔵 GRUPO_RECIENTE**: Datos del año 2023 en adelante  
    - **🟢 GRUPO_ESTANDAR**: Resto de los datos
    """)
    
    st.subheader("📊 Análisis hidráulico USGS/WMO:")
    st.markdown("""
    - ⚡ **Altura vs Velocidad**: Modelo logarítmico (WMO)
    - 📐 **Altura vs Área**: Modelo potencial (USGS)
    - 📏 **Altura vs Perímetro**: Modelo lineal (USGS)
    - 🌊 **Altura vs Ancho**: Modelo lineal/potencial (USGS)
    - 🔵 **Altura vs Radio hidráulico**: Modelo potencial (USGS)
    - 💨 **Caudal vs Velocidad**: Modelo potencial (WMO)
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
                
                # USAR STATE PARA CONTROLAR EL RECÁLCULO
                if 'procesamiento_realizado' not in st.session_state:
                    st.session_state.procesamiento_realizado = False
                if 'curvas_sin_alto_rh' not in st.session_state:
                    st.session_state.curvas_sin_alto_rh = None
                if 'datos_sin_alto_rh' not in st.session_state:
                    st.session_state.datos_sin_alto_rh = None
                if 'tiene_alto_rh' not in st.session_state:
                    st.session_state.tiene_alto_rh = False
                if 'datos_completos' not in st.session_state:
                    st.session_state.datos_completos = None
                
                # BOTÓN PRINCIPAL DE PROCESAMIENTO
                if st.button("🚀 Procesar Aforos con Clasificación", type="primary"):
                    with st.spinner("Procesando datos y clasificando grupos..."):
                        # PROCESAMIENTO INICIAL - SIN GRUPO_ALTO_RH
                        curvas_sin, datos_sin = procesar_con_clasificacion(df, incluir_alto_rh=False)
                        
                        if curvas_sin:
                            st.session_state.procesamiento_realizado = True
                            st.session_state.curvas_sin_alto_rh = curvas_sin
                            st.session_state.datos_sin_alto_rh = datos_sin
                            
                            # Verificar si hay GRUPO_ALTO_RH y guardar datos completos
                            _, datos_completos = procesar_con_clasificacion(df, incluir_alto_rh=True)
                            st.session_state.tiene_alto_rh = 'GRUPO_ALTO_RH' in datos_completos['GRUPO_PREDICHO'].values
                            st.session_state.datos_completos = datos_completos
                
                # MOSTRAR RESULTADOS SI EL PROCESAMIENTO SE REALIZÓ
                if st.session_state.procesamiento_realizado and st.session_state.curvas_sin_alto_rh is not None:
                    curvas_sin = st.session_state.curvas_sin_alto_rh
                    datos_sin = st.session_state.datos_sin_alto_rh
                    
                    st.success(f"✅ Procesado exitoso: {len(datos_sin)} aforos clasificados")
                    
                    # Mostrar distribución de grupos
                    st.subheader("📊 Distribución de Grupos")
                    distribucion = datos_sin['GRUPO_PREDICHO'].value_counts()
                    col1, col2, col3 = st.columns(3)
                    
                    with col1:
                        st.metric("GRUPO_ALTO_RH", distribucion.get('GRUPO_ALTO_RH', 0))
                    with col2:
                        st.metric("GRUPO_RECIENTE", distribucion.get('GRUPO_RECIENTE', 0))
                    with col3:
                        st.metric("GRUPO_ESTANDAR", distribucion.get('GRUPO_ESTANDAR', 0))
                    
                    # Mostrar datos clasificados
                    st.subheader("📋 Datos Clasificados")
                    datos_mostrar = datos_sin[datos_sin['GRUPO_PREDICHO'] != 'GRUPO_ESTANDAR']
                    st.dataframe(datos_mostrar[['NIVEL_AFORO', 'CAUDAL', 'VELOCIDAD', 'AREA', 'RADIO_HIDRAULICO', 'GRUPO_PREDICHO']])
                    
                    # Gráfico inicial
                    st.subheader("📈 Curvas Altura-Caudal por Grupo")
                    fig_sin = crear_grafico_curvas(datos_sin, curvas_sin, "Curvas por Grupo (sin GRUPO_ALTO_RH)")
                    st.pyplot(fig_sin)
                    
                    # Mostrar ecuaciones
                    st.subheader("📐 Ecuaciones por Grupo")
                    for grupo, curva in curvas_sin.items():
                        with st.expander(f"{grupo} - {curva['nombre']} - R² = {curva['r2']:.3f}"):
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
                    
                    # ANÁLISIS HIDRÁULICO COMPLETO USGS/WMO
                    st.subheader("🔍 Análisis Hidráulico Completo (USGS/WMO)")
                    fig_hidraulico = crear_graficos_hidraulicos(datos_sin, "(sin GRUPO_ALTO_RH)")
                    st.pyplot(fig_hidraulico)
                    
                    # VERIFICAR SI HAY GRUPO_ALTO_RH PARA OFRECER RECÁLCULO
                    if st.session_state.tiene_alto_rh:
                        st.subheader("⚙️ Opción de Re-análisis con GRUPO_ALTO_RH")
                        
                        # Mostrar información específica sobre GRUPO_ALTO_RH
                        datos_completos = st.session_state.datos_completos
                        alto_rh_data = datos_completos[datos_completos['GRUPO_PREDICHO'] == 'GRUPO_ALTO_RH']
                        
                        st.warning(f"🔴 Se detectaron {len(alto_rh_data)} aforos del GRUPO_ALTO_RH:")
                        st.dataframe(alto_rh_data[['NIVEL_AFORO', 'CAUDAL', 'VELOCIDAD', 'AREA', 'RADIO_HIDRAULICO']])
                        
                        st.info("¿Deseas recalcular INCLUYENDO el GRUPO_ALTO_RH?")
                        
                        # BOTÓN DE RECÁLCULO
                        if st.button("🔄 RECALCULAR con GRUPO_ALTO_RH", key="btn_recalcular"):
                            with st.spinner("Recalculando con GRUPO_ALTO_RH..."):
                                # RECÁLCULO REAL INCLUYENDO GRUPO_ALTO_RH
                                curvas_con, datos_con = procesar_con_clasificacion(df, incluir_alto_rh=True)
                                
                                if curvas_con:
                                    st.success(f"✅ RECÁLCULO EXITOSO: {len(datos_con)} aforos (CON GRUPO_ALTO_RH)")
                                    
                                    # Mostrar comparación
                                    st.subheader("📊 COMPARACIÓN: Con vs Sin GRUPO_ALTO_RH")
                                    col1, col2 = st.columns(2)
                                    
                                    with col1:
                                        st.metric("Aforos SIN GRUPO_ALTO_RH", len(datos_sin))
                                        st.metric("Curvas generadas", len(curvas_sin))
                                    
                                    with col2:
                                        st.metric("Aforos CON GRUPO_ALTO_RH", len(datos_con))
                                        st.metric("Curvas generadas", len(curvas_con))
                                    
                                    # NUEVO gráfico con GRUPO_ALTO_RH
                                    st.subheader("📈 NUEVAS Curvas Altura-Caudal (CON GRUPO_ALTO_RH)")
                                    fig_con = crear_grafico_curvas(datos_con, curvas_con, "Curvas CON GRUPO_ALTO_RH")
                                    st.pyplot(fig_con)
                                    
                                    # Mostrar NUEVAS ecuaciones
                                    st.subheader("📐 NUEVAS Ecuaciones por Grupo")
                                    for grupo, curva in curvas_con.items():
                                        with st.expander(f"{grupo} - {curva['nombre']} - R² = {curva['r2']:.3f}"):
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
                                    
                                    # ANÁLISIS HIDRÁULICO COMPLETO CON GRUPO_ALTO_RH
                                    st.subheader("🔍 Análisis Hidráulico Completo (CON GRUPO_ALTO_RH)")
                                    fig_hidraulico_con = crear_graficos_hidraulicos(datos_con, "(CON GRUPO_ALTO_RH)")
                                    st.pyplot(fig_hidraulico_con)
                                else:
                                    st.error("❌ No se pudieron generar curvas con GRUPO_ALTO_RH")
                    else:
                        st.info("✅ No se detectó GRUPO_ALTO_RH en los datos. Los resultados están completos.")
                        
                elif st.session_state.procesamiento_realizado:
                    st.error("❌ No se pudieron generar curvas con los datos proporcionados")
                        
            else:
                st.error(f"❌ Faltan columnas: {', '.join(columnas_faltantes)}")
                    
        except Exception as e:
            st.error(f"❌ Error al procesar el archivo: {e}")

elif opcion == "📊 Ingreso Manual":
    st.header("📊 Ingreso Manual de Aforos")
    
    st.info("💡 Ingresa los datos de aforo manualmente")
    
    num_aforos = st.number_input("Número de aforos:", min_value=1, max_value=20, value=3)
    datos_manual = []
    
    for i in range(num_aforos):
        with st.expander(f"Aforo {i+1}", expanded=True if i == 0 else False):
            col1, col2 = st.columns(2)
            with col1:
                nivel = st.number_input("Nivel (m)", min_value=0.1, value=1.0 + i*0.5, key=f"n{i}")
                caudal = st.number_input("Caudal (m³/s)", min_value=0.1, value=2.0 + i*1.0, key=f"q{i}")
                area = st.number_input("Área (m²)", min_value=0.1, value=3.0 + i*2.0, key=f"a{i}")
            with col2:
                ancho = st.number_input("Ancho Río (m)", min_value=0.1, value=8.0, key=f"w{i}")
                velocidad = st.number_input("Velocidad (m/s)", min_value=0.1, value=0.7, key=f"v{i}")
            
            datos_manual.append({
                'FECHA AFORO': '2024-01-01',
                'NIVEL DE AFORO (m)': nivel,
                'CAUDAL (m3/s)': caudal,
                'AREA (m2)': area,
                'ANCHO RIO (m)': ancho,
                'VELOCIDAD (m/s)': velocidad
            })
    
    if st.button("🚀 Generar Curvas con Datos Manuales", type="primary") and datos_manual:
        with st.spinner("Procesando..."):
            df_manual = pd.DataFrame(datos_manual)
            curvas, datos_procesados = procesar_con_clasificacion(df_manual, incluir_alto_rh=True)
            
            if curvas:
                st.success(f"✅ Se generaron {len(curvas)} curvas")
                
                # Mostrar distribución de grupos
                distribucion = datos_procesados['GRUPO_PREDICHO'].value_counts()
                st.subheader("📊 Distribución de Grupos")
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.metric("GRUPO_ALTO_RH", distribucion.get('GRUPO_ALTO_RH', 0))
                with col2:
                    st.metric("GRUPO_RECIENTE", distribucion.get('GRUPO_RECIENTE', 0))
                with col3:
                    st.metric("GRUPO_ESTANDAR", distribucion.get('GRUPO_ESTANDAR', 0))
                
                # Mostrar gráfico
                st.subheader("📈 Curvas Generadas")
                fig = crear_grafico_curvas(datos_procesados, curvas, "Curvas Altura-Caudal")
                st.pyplot(fig)
                
                # Mostrar ecuaciones
                st.subheader("📐 Ecuaciones")
                for grupo, curva in curvas.items():
                    with st.expander(f"{grupo} - {curva['nombre']} - R² = {curva['r2']:.3f}"):
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
                st.error("❌ No se pudieron generar curvas con los datos ingresados")

st.markdown("---")
st.markdown("**🌊 Sistema Inteligente de Análisis Hidráulico - Clasificación por Grupos**")