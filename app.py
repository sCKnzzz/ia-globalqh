import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

# DEFINIR FUNCIONES GLOBALES
def func_poly2(x, a, b, c):
    return a * x**2 + b * x + c

def func_poly3(x, a, b, c, d):
    return a * x**3 + b * x**2 + c * x + d

def func_pot(x, a, b):
    return a * x**b

def func_lineal(x, a, b):
    return a * x + b

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
        # Evitar división por cero
        df_procesado['ANCHO_RIO'] = df_procesado['ANCHO_RIO'].replace(0, np.nan)
        df_procesado['TIRANTE_MEDIO'] = df_procesado['AREA'] / df_procesado['ANCHO_RIO']
        df_procesado['PERIMETRO'] = df_procesado['ANCHO_RIO'] + 2 * df_procesado['TIRANTE_MEDIO']
        st.info("📏 Perímetro calculado automáticamente")
    
    # Calcular variables hidráulicas (evitar división por cero)
    df_procesado['PERIMETRO'] = df_procesado['PERIMETRO'].replace(0, np.nan)
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
    
    # Limpiar valores infinitos
    df_procesado = df_procesado.replace([np.inf, -np.inf], np.nan)
    df_procesado = df_procesado.dropna(subset=['NIVEL_AFORO', 'CAUDAL', 'AREA', 'ANCHO_RIO'])
    
    return df_procesado

# FUNCIÓN MEJORADA PARA CLASIFICAR GRUPOS
def clasificar_grupos(df):
    """Clasificar datos en grupos según características hidráulicas"""
    df_clasificado = df.copy()
    
    # Asegurar que tenemos datos suficientes para clasificar
    if len(df_clasificado) < 3:
        # Si hay pocos datos, todos van al mismo grupo
        df_clasificado['GRUPO_PREDICHO'] = 'GRUPO_ESTANDAR'
        return df_clasificado
    
    # Clasificación mejorada
    radio_medio = df_clasificado['RADIO_HIDRAULICO'].median()
    nivel_medio = df_clasificado['NIVEL_AFORO'].median()
    
    condiciones = []
    grupos = []
    
    # GRUPO_ALTO_RH - solo si hay suficientes datos
    alto_rh_mask = df_clasificado['RADIO_HIDRAULICO'] > radio_medio * 1.2
    if alto_rh_mask.sum() >= 2:  # Solo crear grupo si hay al menos 2 puntos
        condiciones.append(alto_rh_mask)
        grupos.append('GRUPO_ALTO_RH')
    
    # GRUPO_RECIENTE - solo si hay suficientes datos
    if 'YEAR' in df_clasificado.columns:
        reciente_mask = df_clasificado['YEAR'] >= 2023
        if reciente_mask.sum() >= 2:
            condiciones.append(reciente_mask)
            grupos.append('GRUPO_RECIENTE')
    
    # Si no hay condiciones o todos caen en default, usar clasificación por niveles
    if not condiciones:
        st.info("📊 Usando clasificación por niveles (no hay suficientes datos para otros criterios)")
        # Dividir en tercios por nivel
        tercil_33 = np.percentile(df_clasificado['NIVEL_AFORO'], 33)
        tercil_67 = np.percentile(df_clasificado['NIVEL_AFORO'], 67)
        
        condiciones = [
            (df_clasificado['NIVEL_AFORO'] <= tercil_33),
            (df_clasificado['NIVEL_AFORO'] <= tercil_67),
        ]
        grupos = ['GRUPO_BAJO', 'GRUPO_MEDIO']
        df_clasificado['GRUPO_PREDICHO'] = np.select(condiciones, grupos, default='GRUPO_ALTO')
    else:
        # Usar las condiciones que sí tienen suficientes datos
        df_clasificado['GRUPO_PREDICHO'] = np.select(condiciones, grupos, default='GRUPO_ESTANDAR')
    
    # VERIFICACIÓN FINAL: asegurar que todos los grupos tengan al menos 2 puntos
    distribucion = df_clasificado['GRUPO_PREDICHO'].value_counts()
    grupos_validos = [grupo for grupo in distribucion.index if distribucion[grupo] >= 2]
    
    if len(grupos_validos) == 0:
        # Si ningún grupo tiene suficientes puntos, usar solo GRUPO_ESTANDAR
        df_clasificado['GRUPO_PREDICHO'] = 'GRUPO_ESTANDAR'
    elif len(grupos_validos) < len(distribucion):
        # Reasignar puntos de grupos inválidos al grupo estándar
        st.warning(f"⚠️ Reasignando grupos con pocos datos a GRUPO_ESTANDAR")
        for grupo in distribucion.index:
            if distribucion[grupo] < 2:
                df_clasificado.loc[df_clasificado['GRUPO_PREDICHO'] == grupo, 'GRUPO_PREDICHO'] = 'GRUPO_ESTANDAR'
    
    return df_clasificado

# FUNCIÓN MEJORADA PARA AJUSTAR CURVAS CON MODELOS DE LITERATURA
def ajustar_curva_grupo(datos_grupo, nombre_grupo):
    try:
        H = datos_grupo['NIVEL_AFORO'].values
        Q = datos_grupo['CAUDAL'].values
        
        # Mostrar información del grupo
        debug_text = f"🔍 **Analizando {nombre_grupo}:** {len(H)} puntos\n"
        debug_text += f"- Rango niveles: {min(H):.3f} - {max(H):.3f} m\n"
        debug_text += f"- Rango caudales: {min(Q):.3f} - {max(Q):.3f} m³/s\n"
        st.write(debug_text)
        
        if len(H) < 2:
            st.warning(f"⚠️ {nombre_grupo}: Solo {len(H)} puntos (mínimo 2)")
            return None
        
        # Ordenar datos
        sort_idx = np.argsort(H)
        H_sorted = H[sort_idx]
        Q_sorted = Q[sort_idx]
        
        # MODELOS HIDRÁULICOS SEGÚN LITERATURA
        def func_manning(x, a, n):
            """Q = a * H^(5/3) - Fórmula de Manning simplificada"""
            return a * x**(5/3)
        
        def func_chezy(x, a, b):
            """Q = a * H^b - Fórmula de Chezy generalizada"""
            return a * x**b
        
        def func_logaritmica(x, a, b):
            """Q = a * log(H + b) - Relación logarítmica"""
            return a * np.log(x + b)
        
        def func_exponencial(x, a, b):
            """Q = a * exp(b*H) - Relación exponencial"""
            return a * np.exp(b * x)
        
        # Todos los modelos en orden de preferencia
        modelos = [
            ('Lineal', func_lineal),
            ('Potencial (Chezy)', func_pot),
            ('Manning (H^5/3)', func_manning),
            ('Polinómico G2', func_poly2),
            ('Exponencial', func_exponencial),
            ('Logarítmica', func_logaritmica)
        ]
        
        mejor_r2 = -np.inf
        mejor_modelo = None
        
        for nombre, funcion in modelos:
            try:
                st.write(f"🔧 **Probando modelo {nombre}**")
                
                # Parámetros iniciales según el modelo
                if nombre == 'Potencial (Chezy)':
                    H_pos = np.maximum(H_sorted, 0.001)
                    Q_pos = np.maximum(Q_sorted, 0.001)
                    params, _ = curve_fit(funcion, H_pos, Q_pos, p0=[1.0, 2.0], maxfev=5000)
                    Q_pred = funcion(H_pos, *params)
                elif nombre == 'Manning (H^5/3)':
                    H_pos = np.maximum(H_sorted, 0.001)
                    Q_pos = np.maximum(Q_sorted, 0.001)
                    params, _ = curve_fit(funcion, H_pos, Q_pos, p0=[1.0, 0.035], maxfev=5000)
                    Q_pred = funcion(H_pos, *params)
                elif nombre == 'Exponencial':
                    params, _ = curve_fit(funcion, H_sorted, Q_sorted, p0=[1.0, 1.0], maxfev=5000)
                    Q_pred = funcion(H_sorted, *params)
                elif nombre == 'Logarítmica':
                    H_pos = np.maximum(H_sorted, 0.001)
                    params, _ = curve_fit(funcion, H_pos, Q_sorted, p0=[1.0, 1.0], maxfev=5000)
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
                
                st.write(f"   - R² = {r2:.3f}")
                
                # Aceptar CUALQUIER modelo que converja
                if r2 > mejor_r2:
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
                    st.success(f"   ✅ Modelo aceptado")
                    
            except Exception as e:
                st.error(f"   ❌ Error en {nombre}: {str(e)}")
                continue
        
        if mejor_modelo:
            st.success(f"🎉 **{nombre_grupo}:** {mejor_modelo['nombre']} ajustado (R²={mejor_modelo['r2']:.3f})")
            return mejor_modelo
        else:
            st.error(f"💥 **{nombre_grupo}:** No se pudo ajustar ningún modelo")
            return None
            
    except Exception as e:
        st.error(f"💥 **Error crítico en {nombre_grupo}:** {str(e)}")
        return None

# FUNCIÓN CORREGIDA PARA PROCESAR CON CLASIFICACIÓN DE GRUPOS
def procesar_con_clasificacion(df, incluir_alto_rh=True):
    """Procesar datos con clasificación por grupos"""
    
    with st.spinner("📊 Preparando datos..."):
        df_procesado = preparar_datos(df)
    
    with st.spinner("📊 Clasificando grupos..."):
        df_clasificado = clasificar_grupos(df_procesado)
    
    # Mostrar distribución de grupos
    distribucion = df_clasificado['GRUPO_PREDICHO'].value_counts()
    st.subheader("📈 Distribución de Grupos")
    for grupo, count in distribucion.items():
        st.write(f"- **{grupo}:** {count} puntos")
    
    # CORRECIÓN: Si no incluir_alto_rh pero no hay otros grupos, procesar igual
    if not incluir_alto_rh:
        grupos_originales = df_clasificado['GRUPO_PREDICHO'].unique()
        grupos_sin_alto_rh = [g for g in grupos_originales if g != 'GRUPO_ALTO_RH']
        
        if len(grupos_sin_alto_rh) > 0:
            df_filtrado = df_clasificado[df_clasificado['GRUPO_PREDICHO'] != 'GRUPO_ALTO_RH'].copy()
            st.info("🔵 Procesamiento SIN GRUPO_ALTO_RH")
        else:
            # Si no hay otros grupos, procesar todos los datos de todos modos
            df_filtrado = df_clasificado.copy()
            st.warning("⚠️ No hay otros grupos aparte de GRUPO_ALTO_RH. Procesando todos los datos.")
    else:
        df_filtrado = df_clasificado.copy()
        st.info("🔴 Procesamiento CON GRUPO_ALTO_RH")
    
    # Generar curvas para cada grupo
    resultados = {}
    
    st.subheader("🔍 Ajuste de Curvas por Grupo")
    with st.spinner("🔧 Ajustando curvas..."):
        # Procesar TODOS los grupos disponibles
        grupos_procesados = list(df_filtrado['GRUPO_PREDICHO'].unique())
        st.write(f"**Grupos a procesar:** {grupos_procesados}")
        
        for grupo in grupos_procesados:
            grupo_data = df_filtrado[df_filtrado['GRUPO_PREDICHO'] == grupo]
            
            st.write(f"---")
            st.write(f"### Procesando: {grupo}")
            
            if len(grupo_data) >= 2:
                curva = ajustar_curva_grupo(grupo_data, grupo)
                if curva:
                    resultados[grupo] = curva
                else:
                    st.warning(f"❌ No se pudo generar curva para {grupo}")
            else:
                st.warning(f"⚠️ {grupo}: Solo {len(grupo_data)} puntos (mínimo 2)")
    
    st.write(f"---")
    st.write(f"**📊 Resumen:** Se generaron {len(resultados)} curvas de {len(grupos_procesados)} grupos")
    
    return resultados, df_filtrado

# NUEVA FUNCIÓN PARA REPROCESAMIENTO ESPECIAL
def reprocesar_con_alto_rh(df):
    """Reprocesamiento ESPECIAL que fuerza la inclusión de GRUPO_ALTO_RH"""
    
    st.info("🔄 INICIANDO REPROCESAMIENTO CON GRUPO_ALTO_RH")
    
    with st.spinner("📊 Preparando datos para reprocesamiento..."):
        df_procesado = preparar_datos(df)
    
    # CLASIFICACIÓN ESPECIAL para asegurar GRUPO_ALTO_RH
    df_clasificado = df_procesado.copy()
    
    # Forzar la creación de GRUPO_ALTO_RH incluso con pocos puntos
    radio_medio = df_clasificado['RADIO_HIDRAULICO'].median()
    alto_rh_mask = df_clasificado['RADIO_HIDRAULICO'] > radio_medio * 1.2
    
    if alto_rh_mask.any():
        df_clasificado['GRUPO_PREDICHO'] = np.where(
            alto_rh_mask, 'GRUPO_ALTO_RH', 'GRUPO_ESTANDAR'
        )
        
        # Si GRUPO_ALTO_RH tiene solo 1 punto, duplicarlo artificialmente para el ajuste
        alto_rh_data = df_clasificado[df_clasificado['GRUPO_PREDICHO'] == 'GRUPO_ALTO_RH']
        if len(alto_rh_data) == 1:
            st.warning("🔧 Creando punto artificial para GRUPO_ALTO_RH (solo 1 punto real)")
            # Duplicar el punto con pequeña variación
            punto_extra = alto_rh_data.copy()
            punto_extra['NIVEL_AFORO'] = punto_extra['NIVEL_AFORO'] * 1.01  # +1% de variación
            punto_extra['CAUDAL'] = punto_extra['CAUDAL'] * 1.01
            df_clasificado = pd.concat([df_clasificado, punto_extra], ignore_index=True)
    
    # Generar curvas para cada grupo
    resultados = {}
    
    st.subheader("🔍 REPROCESAMIENTO: Ajuste de Curvas por Grupo")
    with st.spinner("🔧 Ajustando curvas en reprocesamiento..."):
        grupos_procesados = list(df_clasificado['GRUPO_PREDICHO'].unique())
        st.write(f"**Grupos en reprocesamiento:** {grupos_procesados}")
        
        for grupo in grupos_procesados:
            grupo_data = df_clasificado[df_clasificado['GRUPO_PREDICHO'] == grupo]
            
            st.write(f"---")
            st.write(f"### REPROCESANDO: {grupo}")
            
            if len(grupo_data) >= 1:  # Reducido a 1 punto mínimo para reprocesamiento
                curva = ajustar_curva_grupo(grupo_data, grupo)
                if curva:
                    resultados[grupo] = curva
                else:
                    st.warning(f"❌ No se pudo generar curva para {grupo} en reprocesamiento")
            else:
                st.warning(f"⚠️ {grupo}: Sin puntos para reprocesamiento")
    
    return resultados, df_clasificado

# FUNCIÓN PARA GRÁFICO PRINCIPAL
def crear_grafico_curvas(df, curvas, titulo):
    fig, ax = plt.subplots(figsize=(10, 6))
    
    colores = {'GRUPO_ALTO_RH': 'red', 'GRUPO_RECIENTE': 'blue', 'GRUPO_ESTANDAR': 'green', 
               'GRUPO_BAJO': 'orange', 'GRUPO_MEDIO': 'purple', 'GRUPO_ALTO': 'brown'}
    
    # Graficar puntos por grupo
    for grupo in df['GRUPO_PREDICHO'].unique():
        color = colores.get(grupo, 'gray')
        grupo_data = df[df['GRUPO_PREDICHO'] == grupo]
        
        ax.scatter(grupo_data['NIVEL_AFORO'], grupo_data['CAUDAL'], 
                  color=color, s=60, label=grupo, alpha=0.7)
    
    # Graficar curvas ajustadas
    for grupo, curva in curvas.items():
        color = colores.get(grupo, 'gray')
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
    
    colores = {'GRUPO_ALTO_RH': 'red', 'GRUPO_RECIENTE': 'blue', 'GRUPO_ESTANDAR': 'green',
               'GRUPO_BAJO': 'orange', 'GRUPO_MEDIO': 'purple', 'GRUPO_ALTO': 'brown'}
    
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
            color = colores.get(grupo, 'gray')
            grupo_data = df[df['GRUPO_PREDICHO'] == grupo]
            
            x = grupo_data[x_col].values
            y = grupo_data[y_col].values
            
            ax.scatter(x, y, alpha=0.7, s=50, color=color, label=grupo)
        
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
                    
                    # PROCESAMIENTO INICIAL - SIN GRUPO_ALTO_RH
                    curvas_sin, datos_sin = procesar_con_clasificacion(df, incluir_alto_rh=False)
                    
                    if curvas_sin:
                        st.success(f"🎉 ¡ÉXITO! Se generaron {len(curvas_sin)} curvas")
                        
                        # Gráfico inicial
                        st.subheader("📈 Curvas Altura-Caudal por Grupo")
                        fig_sin = crear_grafico_curvas(datos_sin, curvas_sin, "Curvas por Grupo")
                        st.pyplot(fig_sin)
                        
                        # Mostrar ecuaciones
                        st.subheader("📐 Ecuaciones por Grupo")
                        for grupo, curva in curvas_sin.items():
                            with st.expander(f"{grupo} - {curva['nombre']} - R² = {curva['r2']:.3f}"):
                                st.write(f"**Puntos utilizados:** {curva['n_puntos']}")
                                st.write(f"**Rango de niveles:** {curva['rango_niveles'][0]:.2f} - {curva['rango_niveles'][1]:.2f} m")
                                st.write(f"**Rango de caudales:** {curva['rango_caudales'][0]:.2f} - {curva['rango_caudales'][1]:.2f} m³/s")
                                
                                if curva['nombre'] == 'Lineal':
                                    a, b = curva['parametros']
                                    st.latex(f"Q = {a:.4f}H + {b:.4f}")
                                elif curva['nombre'] == 'Polinómico G2':
                                    a, b, c = curva['parametros']
                                    st.latex(f"Q = {a:.4f}H^2 + {b:.4f}H + {c:.4f}")
                                elif curva['nombre'] == 'Potencial (Chezy)':
                                    a, b = curva['parametros']
                                    st.latex(f"Q = {a:.4f}H^{{{b:.4f}}}")
                                elif curva['nombre'] == 'Manning (H^5/3)':
                                    a, n = curva['parametros']
                                    st.latex(f"Q = {a:.4f}H^{{5/3}}")
                                elif curva['nombre'] == 'Exponencial':
                                    a, b = curva['parametros']
                                    st.latex(f"Q = {a:.4f}e^{{{b:.4f}H}}")
                                elif curva['nombre'] == 'Logarítmica':
                                    a, b = curva['parametros']
                                    st.latex(f"Q = {a:.4f} \\cdot \\ln(H + {b:.4f})")
                        
                        # ANÁLISIS HIDRÁULICO
                        st.subheader("🔍 Análisis Hidráulico")
                        fig_hidraulico = crear_graficos_hidraulicos(datos_sin)
                        st.pyplot(fig_hidraulico)
                        
                        # VERIFICAR SI HAY GRUPO_ALTO_RH PARA OFRECER RECÁLCULO
                        _, datos_completos = procesar_con_clasificacion(df, incluir_alto_rh=True)
                        tiene_alto_rh = 'GRUPO_ALTO_RH' in datos_completos['GRUPO_PREDICHO'].values
                        
                        if tiene_alto_rh:
                            st.subheader("⚙️ Opción de Re-análisis con GRUPO_ALTO_RH")
                            
                            alto_rh_data = datos_completos[datos_completos['GRUPO_PREDICHO'] == 'GRUPO_ALTO_RH']
                            st.warning(f"🔴 Se detectaron {len(alto_rh_data)} aforos del GRUPO_ALTO_RH")
                            
                            # BOTÓN CORREGIDO - AHORA SÍ DEFINIDO
                            if st.button("🔄 REPROCESAR INCLUYENDO GRUPO_ALTO_RH", key="btn_reprocesar"):
                                with st.spinner("Reprocesamiento ESPECIAL con GRUPO_ALTO_RH..."):
                                    # USAR LA NUEVA FUNCIÓN DE REPROCESAMIENTO
                                    curvas_con, datos_con = reprocesar_con_alto_rh(df)
                                    
                                    if curvas_con:
                                        st.success(f"✅ REPROCESAMIENTO EXITOSO: {len(curvas_con)} curvas generadas")
                                        
                                        # NUEVO gráfico con GRUPO_ALTO_RH
                                        st.subheader("📈 CURVAS REPROCESADAS (INCLUYENDO GRUPO_ALTO_RH)")
                                        fig_con = crear_grafico_curvas(datos_con, curvas_con, "Curvas REPROCESADAS con GRUPO_ALTO_RH")
                                        st.pyplot(fig_con)
                                        
                                        # Mostrar ecuaciones del reprocesamiento
                                        st.subheader("📐 ECUACIONES REPROCESADAS")
                                        for grupo, curva in curvas_con.items():
                                            with st.expander(f"REPROCESADO: {grupo} - {curva['nombre']} - R² = {curva['r2']:.3f}"):
                                                st.write(f"**Puntos utilizados:** {curva['n_puntos']}")
                                                st.write(f"**Rango de niveles:** {curva['rango_niveles'][0]:.2f} - {curva['rango_niveles'][1]:.2f} m")
                                                st.write(f"**Rango de caudales:** {curva['rango_caudales'][0]:.2f} - {curva['rango_caudales'][1]:.2f} m³/s")
                                                
                                                if curva['nombre'] == 'Lineal':
                                                    a, b = curva['parametros']
                                                    st.latex(f"Q = {a:.4f}H + {b:.4f}")
                                                elif curva['nombre'] == 'Polinómico G2':
                                                    a, b, c = curva['parametros']
                                                    st.latex(f"Q = {a:.4f}H^2 + {b:.4f}H + {c:.4f}")
                                                elif curva['nombre'] == 'Potencial (Chezy)':
                                                    a, b = curva['parametros']
                                                    st.latex(f"Q = {a:.4f}H^{{{b:.4f}}}")
                                                elif curva['nombre'] == 'Manning (H^5/3)':
                                                    a, n = curva['parametros']
                                                    st.latex(f"Q = {a:.4f}H^{{5/3}}")
                                                elif curva['nombre'] == 'Exponencial':
                                                    a, b = curva['parametros']
                                                    st.latex(f"Q = {a:.4f}e^{{{b:.4f}H}}")
                                                elif curva['nombre'] == 'Logarítmica':
                                                    a, b = curva['parametros']
                                                    st.latex(f"Q = {a:.4f} \\cdot \\ln(H + {b:.4f})")
                                        
                                        # ANÁLISIS HIDRÁULICO COMPLETO
                                        st.subheader("🔍 Análisis Hidráulico Completo (Reprocesado)")
                                        fig_hidraulico_con = crear_graficos_hidraulicos(datos_con, " - REPROCESADO")
                                        st.pyplot(fig_hidraulico_con)
                                    else:
                                        st.error("❌ No se pudieron generar curvas en el reprocesamiento con GRUPO_ALTO_RH")
                    else:
                        st.error("❌ No se pudieron generar curvas. Revisa los mensajes de arriba para más detalles.")
                        
            else:
                st.error(f"❌ Faltan columnas: {', '.join(columnas_faltantes)}")
                    
        except Exception as e:
            st.error(f"❌ Error al procesar el archivo: {e}")

elif opcion == "📊 Ingreso Manual":
    st.header("📊 Ingreso Manual de Aforos")
    
    st.info("💡 Ingresa datos VARIADOS para mejores resultados")
    
    num_aforos = st.number_input("Número de aforos:", min_value=2, max_value=20, value=5)
    datos_manual = []
    
    for i in range(num_aforos):
        with st.expander(f"Aforo {i+1}", expanded=True if i == 0 else False):
            col1, col2 = st.columns(2)
            with col1:
                nivel = st.number_input("Nivel (m)", min_value=0.1, value=0.5 + i*0.3, key=f"n{i}")
                caudal = st.number_input("Caudal (m³/s)", min_value=0.1, value=1.0 + i*2.0, key=f"q{i}")
                area = st.number_input("Área (m²)", min_value=0.1, value=2.0 + i*3.0, key=f"a{i}")
            with col2:
                ancho = st.number_input("Ancho Río (m)", min_value=0.1, value=5.0 + i*1.0, key=f"w{i}")
                velocidad = st.number_input("Velocidad (m/s)", min_value=0.1, value=0.5 + i*0.2, key=f"v{i}")
            
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
                fig_hidraulico = crear_graficos_hidraulicos(datos_procesados)
                st.pyplot(fig_hidraulico)
                
            else:
                st.error("❌ No se pudieron generar curvas con los datos ingresados")

st.markdown("---")
st.markdown("**🌊 Sistema de Generación de Curvas Altura-Caudal**")