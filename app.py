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

# DEFINIR LA CLASE DEL SISTEMA
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

# FUNCIÓN PARA PREPARAR DATOS (SIMPLIFICADA)
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
    
    # CALCULAR PERÍMETRO SI ES NECESARIO (más simple)
    if 'PERIMETRO' not in df_procesado.columns or df_procesado['PERIMETRO'].isna().any() or (df_procesado['PERIMETRO'] <= 0).any():
        # Calcular tirante medio y perímetro para sección rectangular
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
        (df_clasificado['YEAR'] >= 2024),            # GRUPO_RECIENTE
        (True)                                       # GRUPO_ESTANDAR (default)
    ]
    
    grupos = ['GRUPO_ALTO_RH', 'GRUPO_RECIENTE', 'GRUPO_ESTANDAR']
    df_clasificado['GRUPO_PREDICHO'] = np.select(condiciones, grupos, default='GRUPO_ESTANDAR')
    
    return df_clasificado

# FUNCIÓN PARA AJUSTAR CURVAS POR GRUPO
def ajustar_curva_grupo(datos_grupo, nombre_grupo):
    H = datos_grupo['NIVEL_AFORO'].values
    Q = datos_grupo['CAUDAL'].values
    
    if len(H) < 3:
        st.warning(f"⚠️ Grupo {nombre_grupo}: No hay suficientes puntos ({len(H)}) para ajustar curva")
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
                # Para función potencial, usar valores iniciales apropiados
                params, _ = curve_fit(funcion, H_sorted, Q_sorted, p0=[1.0, 2.0], maxfev=5000)
            else:
                params, _ = curve_fit(funcion, H_sorted, Q_sorted, maxfev=5000)
            
            Q_pred = funcion(H_sorted, *params)
            ss_res = np.sum((Q_sorted - Q_pred)**2)
            ss_tot = np.sum((Q_sorted - np.mean(Q_sorted))**2)
            
            if ss_tot > 0:
                r2 = 1 - (ss_res / ss_tot)
            else:
                r2 = 0
            
            if r2 > mejor_r2 and r2 > 0.7:  # Solo aceptar modelos con R² > 0.7
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
    
    if mejor_modelo:
        st.success(f"✅ {nombre_grupo}: {mejor_modelo['nombre']} ajustado (R²={mejor_modelo['r2']:.3f})")
    else:
        st.warning(f"⚠️ {nombre_grupo}: No se pudo ajustar curva con R² > 0.7")
    
    return mejor_modelo

# FUNCIÓN PARA PROCESAR CON CLASIFICACIÓN DE GRUPOS
def procesar_con_clasificacion(df, incluir_alto_rh=True):
    """Procesar datos con clasificación por grupos"""
    
    with st.spinner("Preparando datos..."):
        df_procesado = preparar_datos(df)
    
    with st.spinner("Clasificando grupos..."):
        df_clasificado = clasificar_grupos(df_procesado)
    
    # Filtrar si no incluir GRUPO_ALTO_RH
    if not incluir_alto_rh:
        df_filtrado = df_clasificado[df_clasificado['GRUPO_PREDICHO'] != 'GRUPO_ALTO_RH'].copy()
    else:
        df_filtrado = df_clasificado.copy()
    
    # Generar curvas para cada grupo (EXCLUYENDO GRUPO_ESTANDAR)
    resultados = {}
    
    with st.spinner("Ajustando curvas por grupo..."):
        for grupo in df_filtrado['GRUPO_PREDICHO'].unique():
            # EXCLUIR GRUPO_ESTANDAR como en tu lógica original
            if grupo == 'GRUPO_ESTANDAR':
                continue
                
            grupo_data = df_filtrado[df_filtrado['GRUPO_PREDICHO'] == grupo]
            if len(grupo_data) >= 3:
                curva = ajustar_curva_grupo(grupo_data, grupo)
                if curva:
                    resultados[grupo] = curva
            else:
                st.warning(f"⚠️ {grupo}: Solo {len(grupo_data)} puntos, se necesitan al menos 3")
    
    return resultados, df_filtrado

# FUNCIONES PARA GRÁFICOS
def crear_grafico_principal(df, curvas, titulo):
    fig, ax = plt.subplots(figsize=(10, 6))
    
    colores = {'GRUPO_ALTO_RH': 'red', 'GRUPO_RECIENTE': 'blue', 'GRUPO_ESTANDAR': 'green'}
    marcadores = {'GRUPO_ALTO_RH': 's', 'GRUPO_RECIENTE': '^', 'GRUPO_ESTANDAR': 'o'}
    
    # Graficar puntos por grupo
    for grupo in df['GRUPO_PREDICHO'].unique():
        color = colores.get(grupo, 'orange')
        marcador = marcadores.get(grupo, 'o')
        grupo_data = df[df['GRUPO_PREDICHO'] == grupo]
        
        # EXCLUIR GRUPO_ESTANDAR en el gráfico como en tu lógica original
        if grupo == 'GRUPO_ESTANDAR':
            continue
            
        ax.scatter(grupo_data['NIVEL_AFORO'], grupo_data['CAUDAL'], 
                  color=color, marker=marcador, s=80, label=grupo, alpha=0.8,
                  edgecolors='black', linewidth=1)
    
    # Graficar curvas ajustadas
    for grupo, curva in curvas.items():
        color = colores.get(grupo, 'orange')
        H_range = np.linspace(curva['rango_niveles'][0], curva['rango_niveles'][1], 100)
        Q_curve = curva['funcion'](H_range, *curva['parametros'])
        
        # Hacer la línea más gruesa para GRUPO_ALTO_RH
        linewidth = 3 if grupo == 'GRUPO_ALTO_RH' else 2
        ax.plot(H_range, Q_curve, color=color, linewidth=linewidth, 
               label=f'{grupo} (R²={curva["r2"]:.3f})')
    
    ax.set_xlabel('Nivel (m)', fontsize=12, fontweight='bold')
    ax.set_ylabel('Caudal (m³/s)', fontsize=12, fontweight='bold')
    ax.set_title(titulo, fontsize=14, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    return fig

# CONFIGURACIÓN STREAMLIT
st.set_page_config(page_title="Sistema Curvas Altura-Caudal", page_icon="🌊", layout="wide")
st.title("🌊 Sistema de Generación de Curvas Altura-Caudal")
st.markdown("**Clasificación por grupos y ajuste de curvas**")

# NAVEGACIÓN
opcion = st.sidebar.radio("Navegación:", ["🏠 Inicio", "📤 Subir Aforos", "📊 Ingreso Manual"])

if opcion == "🏠 Inicio":
    st.header("Bienvenido al Sistema de Curvas Altura-Caudal")
    st.info("Sistema con clasificación automática por grupos hidráulicos y generación de curvas")
    
    st.subheader("🎯 Clasificación por Grupos:")
    st.markdown("""
    - **🔴 GRUPO_ALTO_RH**: Datos con Radio Hidráulico > 0.6 m
    - **🔵 GRUPO_RECIENTE**: Datos del año 2024 en adelante  
    - **🟢 GRUPO_ESTANDAR**: Resto de los datos
    """)
    
    st.subheader("📈 Modelos de Curvas:")
    st.markdown("""
    - **Polinómico Grado 2**: Q = aH² + bH + c
    - **Polinómico Grado 3**: Q = aH³ + bH² + cH + d  
    - **Potencial**: Q = aHᵇ
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
                
                # BOTÓN PRINCIPAL DE PROCESAMIENTO
                if st.button("🚀 Generar Curvas Altura-Caudal", type="primary"):
                    
                    # PROCESAMIENTO INICIAL - SIN GRUPO_ALTO_RH
                    curvas_sin, datos_sin = procesar_con_clasificacion(df, incluir_alto_rh=False)
                    
                    if curvas_sin:
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
                        
                        # Gráfico principal
                        st.subheader("📈 Curvas Altura-Caudal por Grupo")
                        fig_sin = crear_grafico_principal(datos_sin, curvas_sin, "Curvas por Grupo (sin GRUPO_ALTO_RH)")
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
                        
                        # Verificar si hay GRUPO_ALTO_RH para ofrecer recálculo
                        tiene_alto_rh = 'GRUPO_ALTO_RH' in datos_sin['GRUPO_PREDICHO'].values
                        
                        if tiene_alto_rh:
                            st.subheader("⚙️ Opción de Re-análisis con GRUPO_ALTO_RH")
                            st.info("Se detectaron datos del GRUPO_ALTO_RH. ¿Deseas recalcular INCLUYENDO este grupo?")
                            
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
                                        fig_con = crear_grafico_principal(datos_con, curvas_con, "Curvas CON GRUPO_ALTO_RH")
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
                                    else:
                                        st.error("❌ No se pudieron generar curvas con GRUPO_ALTO_RH")
                    else:
                        st.error("❌ No se pudieron generar curvas con los datos proporcionados")
                        
            else:
                st.error(f"❌ Faltan las siguientes columnas necesarias: {', '.join(columnas_faltantes)}")
                    
        except Exception as e:
            st.error(f"❌ Error al procesar el archivo: {e}")

elif opcion == "📊 Ingreso Manual":
    st.header("📊 Ingreso Manual de Aforos")
    
    st.info("💡 Ingresa los datos de aforo manualmente. El sistema clasificará automáticamente por grupos.")
    
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
                velocidad = st.number_input("Velocidad (m/s)", min_value=0.1, max_value=5.0, value=0.7, step=0.1, key=f"v{i}")
            
            datos_manual.append({
                'FECHA AFORO': '2024-01-01',
                'NIVEL DE AFORO (m)': nivel,
                'CAUDAL (m3/s)': caudal,
                'AREA (m2)': area,
                'ANCHO RIO (m)': ancho,
                'VELOCIDAD (m/s)': velocidad
            })
    
    if st.button("🚀 Generar Curvas con Datos Manuales", type="primary") and datos_manual:
        with st.spinner("Procesando datos manuales..."):
            df_manual = pd.DataFrame(datos_manual)
            curvas, datos_procesados = procesar_con_clasificacion(df_manual, incluir_alto_rh=True)
            
            if curvas:
                st.success("✅ Datos procesados y clasificados exitosamente")
                
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
                
                st.subheader("📋 Datos Clasificados")
                st.dataframe(datos_procesados[['NIVEL_AFORO', 'CAUDAL', 'VELOCIDAD', 'AREA', 'RADIO_HIDRAULICO', 'GRUPO_PREDICHO']])
                
                st.subheader("📈 Curvas por Grupo")
                fig = crear_grafico_principal(datos_procesados, curvas, "Curvas Altura-Caudal - Datos Manuales")
                st.pyplot(fig)
                
                # Mostrar ecuaciones
                st.subheader("📐 Ecuaciones por Grupo")
                for grupo, curva in curvas.items():
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
                
            else:
                st.warning("⚠️ No se pudieron generar curvas con los datos ingresados")

st.markdown("---")
st.markdown("**🌊 Sistema de Generación de Curvas Altura-Caudal**")