# FUNCIÓN MEJORADA PARA CLASIFICAR GRUPOS (MÁS ROBUSTA)
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

# FUNCIÓN CORREGIDA PARA EL REPROCESAMIENTO CON GRUPO_ALTO_RH
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

# ACTUALIZAR LA SECCIÓN DEL BOTÓN DE REPROCESAMIENTO
# En la sección donde está el botón "RECALCULAR con GRUPO_ALTO_RH", reemplaza esta parte:

# BUSCA ESTA SECCIÓN EN TU CÓDIGO Y REEMPLÁZALA:
"""
if st.button("🔄 RECALCULAR con GRUPO_ALTO_RH", key="btn_recalcular"):
    with st.spinner("Recalculando con GRUPO_ALTO_RH..."):
        # RECÁLCULO REAL INCLUYENDO GRUPO_ALTO_RH
        curvas_con, datos_con = procesar_con_clasificacion(df, incluir_alto_rh=True)
"""

# REEMPLÁZALA CON ESTO:
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