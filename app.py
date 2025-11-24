import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

# DEFINIR FUNCIONES GLOBALES SIMPLIFICADAS
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
    
    # CALCULAR PERÍMETRO SI ES NECESARIO
    if 'PERIMETRO' not in df_procesado.columns or df_procesado['PERIMETRO'].isna().any() or (df_procesado['PERIMETRO'] <= 0).any():
        df_procesado['TIRANTE_MEDIO'] = df_procesado['AREA'] / df_procesado['ANCHO_RIO']
        df_procesado['PERIMETRO'] = df_procesado['ANCHO_RIO'] + 2 * df_procesado['TIRANTE_MEDIO']
        st.info("📏 Perímetro calculado automáticamente")
    
    # Calcular variables hidráulicas
    df_procesado['RADIO_HIDRAULICO'] = df_procesado['AREA'] / df_procesado['PERIMETRO']
    
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

# FUNCIÓN PARA CLASIFICAR GRUPOS (SIMPLIFICADA)
def clasificar_grupos(df):
    """Clasificar datos en grupos según características hidráulicas"""
    df_clasificado = df.copy()
    
    # Clasificación simple
    condiciones = [
        (df_clasificado['RADIO_HIDRAULICO'] > 0.6),  # GRUPO_ALTO_RH
        (df_clasificado['YEAR'] >= 2023),            # GRUPO_RECIENTE
    ]
    
    grupos = ['GRUPO_ALTO_RH', 'GRUPO_RECIENTE']
    df_clasificado['GRUPO_PREDICHO'] = np.select(condiciones, grupos, default='GRUPO_ESTANDAR')
    
    return df_clasificado

# FUNCIÓN SIMPLIFICADA PARA AJUSTAR CURVAS
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
                    # Para función potencial, usar valores positivos
                    H_pos = np.maximum(H_sorted, 0.01)
                    Q_pos = np.maximum(Q_sorted, 0.01)
                    params, _ = curve_fit(funcion, H_pos, Q_pos, p0=[1.0, 2.0], maxfev=5000)
                else:
                    params, _ = curve_fit(funcion, H_sorted, Q_sorted, maxfev=5000)
                
                # Calcular predicciones
                if nombre == 'Potencial':
                    Q_pred = funcion(H_pos, *params)
                else:
                    Q_pred = funcion(H_sorted, *params)
                
                # Calcular R²
                ss_res = np.sum((Q_sorted - Q_pred)**2)
                ss_tot = np.sum((Q_sorted - np.mean(Q_sorted))**2)
                
                if ss_tot > 0:
                    r2 = 1 - (ss_res / ss_tot)
                else:
                    r2 = 0
                
                # Aceptar modelos con R² razonable
                if r2 > mejor_r2 and r2 > 0.3:  # Umbral más bajo
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

# FUNCIÓN SIMPLIFICADA PARA ANÁLISIS HIDRÁULICO
def crear_graficos_hidraulicos(df, titulo_sufijo=""):
    """Crear gráficos de análisis hidráulico simplificados"""
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle(f'Análisis de Relaciones Hidráulicas {titulo_sufijo}', 
                 fontsize=16, fontweight='bold')
    
    colores = {'GRUPO_ALTO_RH': 'red', 'GRUPO_RECIENTE': 'blue', 'GRUPO_ESTANDAR': 'green'}
    
    relaciones = [
        ('NIVEL_AFORO', 'AREA', 'Altura vs Área', 'Área (m²)', axes[0, 0]),
        ('NIVEL_AFORO', 'VELOCIDAD', 'Altura vs Velocidad', 'Velocidad (m/s)', axes[0, 1]),
        ('NIVEL_AFORO', 'PERIMETRO', 'Altura vs Perímetro', 'Perímetro (m)', axes[0, 2]),
        ('NIVEL_AFORO', 'ANCHO_RIO', 'Altura vs Ancho', 'Ancho Río (m)', axes[1, 0]),
        ('NIVEL_AFORO', 'RADIO_HIDRAULICO', 'Altura vs Radio Hidráulico', 'Radio Hidráulico (m)', axes[1, 1]),
        ('CAUDAL', 'VELOCIDAD', 'Caudal vs Velocidad', 'Velocidad (m/s)', axes[1, 2])
    ]
    
    for x_col, y_col, titulo, ylabel, ax in relaciones:
        # Graficar puntos por grupo
        for grupo in df['GRUPO_PREDICHO'].unique():
            color = colores.get(grupo, 'orange')
            grupo_data = df[df['GRUPO_PREDICHO'] == grupo]
            
            x = grupo_data[x_col].values
            y = grupo_data[y_col].values
            
            ax.scatter(x, y, alpha=0.7, s=50, color=color, label=grupo)
        
        # Intentar ajustar una curva simple
        try:
            x_all = df[x_col].values
            y_all = df[y_col].values
            
            # Solo ajustar si hay suficientes puntos
            if len(x_all) >= 3:
                # Intentar ajuste lineal simple
                params = np.polyfit(x_all, y_all, 1)
                funcion = np.poly1d(params)
                
                x_range = np.linspace(min(x_all), max(x_all), 100)
                y_pred = funcion(x_range)
                
                # Calcular R²
                y_mean = np.mean(y_all)
                ss_res = np.sum((y_all - funcion(x_all))**2)
                ss_tot = np.sum((y_all - y_mean)**2)
                r2 = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0
                
                ax.plot(x_range, y_pred, 'black', linewidth=2, linestyle='--',
                       label=f'Lineal (R²={r2:.3f})')
        except:
            pass
        
        ax.set_xlabel(x_col.replace('_', ' ').title(), fontweight='bold')
        ax.set_ylabel(ylabel, fontweight='bold')
        ax.set_title(titulo, fontweight='bold')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    return fig

# CONFIGURACIÓN STREAMLIT
st.set_page_config(page_title="Curvas Altura-Caudal", page_icon="🌊", layout="wide")
st.title("🌊 Generador de Curvas Altura-Caudal")
st.markdown("**Sistema con clasificación por grupos y análisis hidráulico**")

# NAVEGACIÓN
opcion = st.sidebar.radio("Navegación:", ["🏠 Inicio", "📤 Subir Aforos", "📊 Ingreso Manual"])

if opcion == "🏠 Inicio":
    st.header("Bienvenido al Sistema de Análisis Hidráulico")
    st.info("Sube tus datos de aforos o ingrésalos manualmente para generar curvas altura-caudal")
    
    st.subheader("🎯 Clasificación por Grupos:")
    st.markdown("""
    - **🔴 GRUPO_ALTO_RH**: Datos con Radio Hidráulico > 0.6 m
    - **🔵 GRUPO_RECIENTE**: Datos del año 2023 en adelante  
    - **🟢 GRUPO_ESTANDAR**: Resto de los datos
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
                if st.button("🚀 Procesar Aforos con Clasificación", type="primary"):
                    with st.spinner("Procesando datos..."):
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
                            
                            # ANÁLISIS HIDRÁULICO
                            st.subheader("🔍 Análisis Hidráulico")
                            fig_hidraulico = crear_graficos_hidraulicos(datos_sin, "(sin GRUPO_ALTO_RH)")
                            st.pyplot(fig_hidraulico)
                            
                            # VERIFICAR SI HAY GRUPO_ALTO_RH PARA OFRECER RECÁLCULO
                            _, datos_completos = procesar_con_clasificacion(df, incluir_alto_rh=True)
                            tiene_alto_rh = 'GRUPO_ALTO_RH' in datos_completos['GRUPO_PREDICHO'].values
                            
                            if tiene_alto_rh:
                                st.subheader("⚙️ Opción de Re-análisis con GRUPO_ALTO_RH")
                                
                                alto_rh_data = datos_completos[datos_completos['GRUPO_PREDICHO'] == 'GRUPO_ALTO_RH']
                                st.warning(f"🔴 Se detectaron {len(alto_rh_data)} aforos del GRUPO_ALTO_RH")
                                
                                if st.button("🔄 RECALCULAR con GRUPO_ALTO_RH", key="btn_recalcular"):
                                    with st.spinner("Recalculando con GRUPO_ALTO_RH..."):
                                        # RECÁLCULO REAL INCLUYENDO GRUPO_ALTO_RH
                                        curvas_con, datos_con = procesar_con_clasificacion(df, incluir_alto_rh=True)
                                        
                                        if curvas_con:
                                            st.success(f"✅ RECÁLCULO EXITOSO: {len(datos_con)} aforos")
                                            
                                            # NUEVO gráfico con GRUPO_ALTO_RH
                                            st.subheader("📈 NUEVAS Curvas Altura-Caudal (CON GRUPO_ALTO_RH)")
                                            fig_con = crear_grafico_curvas(datos_con, curvas_con, "Curvas CON GRUPO_ALTO_RH")
                                            st.pyplot(fig_con)
                                            
                                            # ANÁLISIS HIDRÁULICO COMPLETO
                                            st.subheader("🔍 Análisis Hidráulico Completo")
                                            fig_hidraulico_con = crear_graficos_hidraulicos(datos_con, "(CON GRUPO_ALTO_RH)")
                                            st.pyplot(fig_hidraulico_con)
                                        else:
                                            st.error("❌ No se pudieron generar curvas con GRUPO_ALTO_RH")
                            else:
                                st.info("✅ No se detectó GRUPO_ALTO_RH en los datos")
                                
                        else:
                            st.error("❌ No se pudieron generar curvas. Verifica que los datos tengan suficiente variación.")
                        
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
                
                # Mostrar gráfico
                st.subheader("📈 Curvas Generadas")
                fig = crear_grafico_curvas(datos_procesados, curvas, "Curvas Altura-Caudal")
                st.pyplot(fig)
                
                # ANÁLISIS HIDRÁULICO
                st.subheader("🔍 Análisis Hidráulico")
                fig_hidraulico = crear_graficos_hidraulicos(datos_procesados, "(Datos Manuales)")
                st.pyplot(fig_hidraulico)
                
            else:
                st.error("❌ No se pudieron generar curvas con los datos ingresados")

st.markdown("---")
st.markdown("**🌊 Sistema de Generación de Curvas Altura-Caudal**")