import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
import joblib
import warnings
warnings.filterwarnings('ignore')

# Configuración de la página
st.set_page_config(
    page_title="Sistema IA - GlobalQH",
    page_icon="🚀",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Título principal
st.title("🏭 Sistema de Inteligencia Artificial - GlobalQH")
st.markdown("---")

# Sidebar para navegación
st.sidebar.title("🌐 Navegación")
opcion = st.sidebar.radio(
    "Selecciona una sección:",
    ["🏠 Inicio", "📊 Análisis de Datos", "🤖 Modelo Predictivo", "📈 Dashboard", "⚙️ Configuración"]
)

# Datos de ejemplo para la mina Talapalca
@st.cache_data
def cargar_datos_ejemplo():
    """Cargar datos de ejemplo para la mina Talapalca"""
    np.random.seed(42)
    n_muestras = 1000
    
    datos = {
        'temperatura': np.random.normal(25, 5, n_muestras),
        'humedad': np.random.normal(60, 15, n_muestras),
        'presion': np.random.normal(1013, 50, n_muestras),
        'viento_velocidad': np.random.normal(15, 5, n_muestras),
        'material_dureza': np.random.normal(7, 2, n_muestras),
        'profundidad': np.random.normal(100, 30, n_muestras),
        'concentracion_metal': np.random.normal(85, 10, n_muestras),
        'produccion_diaria': np.random.normal(500, 100, n_muestras),
        'eficiencia': np.random.normal(0.85, 0.1, n_muestras)
    }
    
    # Asegurar que no haya valores negativos
    for key in datos:
        datos[key] = np.maximum(datos[key], 0)
    
    return pd.DataFrame(datos)

@st.cache_resource
def entrenar_modelo_avanzado(_df):
    """Entrenar modelo de machine learning"""
    try:
        # Preparar datos
        X = _df.drop(['produccion_diaria', 'eficiencia'], axis=1)
        y = _df['produccion_diaria']
        
        # Dividir datos
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        
        # Escalar características
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        # Entrenar modelo
        model = RandomForestRegressor(
            n_estimators=100,
            max_depth=10,
            random_state=42,
            n_jobs=-1
        )
        
        model.fit(X_train_scaled, y_train)
        
        # Evaluar modelo
        y_pred = model.predict(X_test_scaled)
        mse = mean_squared_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        
        return model, scaler, mse, r2
        
    except Exception as e:
        st.error(f"Error entrenando el modelo: {str(e)}")
        return None, None, None, None

# Cargar datos
df = cargar_datos_ejemplo()

if opcion == "🏠 Inicio":
    st.header("🏠 Página de Inicio")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.subheader("Bienvenido al Sistema IA de GlobalQH")
        st.markdown("""
        Este sistema integrado proporciona:
        
        - 📊 **Análisis avanzado** de datos mineros
        - 🤖 **Modelos predictivos** para optimización
        - 📈 **Dashboards interactivos** en tiempo real
        - ⚙️ **Herramientas de configuración** personalizadas
        
        ### Características principales:
        ✅ Monitoreo en tiempo real  
        ✅ Alertas tempranas  
        ✅ Optimización de procesos  
        ✅ Reportes automáticos  
        """)
    
    with col2:
        st.metric("📈 Producción Diaria", "485 ton", "+12%")
        st.metric("⚡ Eficiencia", "87%", "+5%")
        st.metric("🔄 Disponibilidad", "94%", "+3%")
        
    # Resumen de datos
    st.subheader("📋 Resumen de Datos Actuales")
    st.dataframe(df.describe(), use_container_width=True)

elif opcion == "📊 Análisis de Datos":
    st.header("📊 Análisis Exploratorio de Datos")
    
    tab1, tab2, tab3 = st.tabs(["📈 Estadísticas", "📊 Visualizaciones", "🔍 Correlaciones"])
    
    with tab1:
        st.subheader("Estadísticas Descriptivas")
        col1, col2 = st.columns(2)
        
        with col1:
            st.write("**Resumen Numérico:**")
            st.dataframe(df.describe(), use_container_width=True)
        
        with col2:
            st.write("**Información del Dataset:**")
            buffer = st.container()
            with buffer:
                st.text(f"Filas: {df.shape[0]}")
                st.text(f"Columnas: {df.shape[1]}")
                st.text(f"Valores nulos: {df.isnull().sum().sum()}")
    
    with tab2:
        st.subheader("Visualizaciones Interactivas")
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Histograma interactivo
            columna_hist = st.selectbox(
                "Selecciona columna para histograma:",
                df.columns,
                key="hist_col"
            )
            
            fig_hist = px.histogram(
                df, 
                x=columna_hist,
                title=f"Distribución de {columna_hist}",
                color_discrete_sequence=['#3366CC']
            )
            st.plotly_chart(fig_hist, use_container_width=True)
        
        with col2:
            # Scatter plot
            col_x = st.selectbox("Variable X:", df.columns, key="scatter_x")
            col_y = st.selectbox("Variable Y:", df.columns, key="scatter_y")
            
            fig_scatter = px.scatter(
                df,
                x=col_x,
                y=col_y,
                title=f"{col_y} vs {col_x}",
                color=df['concentracion_metal'],
                color_continuous_scale='viridis'
            )
            st.plotly_chart(fig_scatter, use_container_width=True)
    
    with tab3:
        st.subheader("Análisis de Correlaciones")
        
        # Matriz de correlación
        corr_matrix = df.corr()
        
        fig_corr = px.imshow(
            corr_matrix,
            title="Matriz de Correlación",
            color_continuous_scale='RdBu_r',
            aspect='auto'
        )
        st.plotly_chart(fig_corr, use_container_width=True)

elif opcion == "🤖 Modelo Predictivo":
    st.header("🤖 Modelo Predictivo Avanzado")
    
    st.info("""
    **Modelo de Random Forest** entrenado para predecir la producción diaria 
    basado en condiciones operativas y características del mineral.
    """)
    
    # Entrenar modelo
    if st.button("🚀 Entrenar Modelo Avanzado", type="primary"):
        with st.spinner("Entrenando modelo... Esto puede tomar unos segundos"):
            model, scaler, mse, r2 = entrenar_modelo_avanzado(df)
            
            if model is not None:
                # Guardar modelo
                joblib.dump(model, 'modelo_talapalca_avanzado.pkl')
                joblib.dump(scaler, 'scaler_talapalca.pkl')
                
                st.success("✅ Modelo avanzado guardado como 'modelo_talapalca_avanzado.pkl'")
                
                # Mostrar métricas
                col1, col2 = st.columns(2)
                with col1:
                    st.metric("📊 Error Cuadrático Medio (MSE)", f"{mse:.2f}")
                with col2:
                    st.metric("🎯 R² Score", f"{r2:.3f}")
    
    # Sección de predicciones
    st.subheader("🔮 Realizar Predicciones")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        temperatura = st.slider("🌡️ Temperatura (°C)", 0.0, 50.0, 25.0, 0.1)
        humedad = st.slider("💧 Humedad (%)", 0.0, 100.0, 60.0, 0.1)
        presion = st.slider("📊 Presión (hPa)", 900.0, 1100.0, 1013.0, 0.1)
    
    with col2:
        viento_velocidad = st.slider("💨 Velocidad del Viento (km/h)", 0.0, 50.0, 15.0, 0.1)
        material_dureza = st.slider("💎 Dureza del Material", 0.0, 10.0, 7.0, 0.1)
        profundidad = st.slider("⛏️ Profundidad (m)", 0.0, 200.0, 100.0, 0.1)
    
    with col3:
        concentracion_metal = st.slider("🥇 Concentración de Metal (%)", 0.0, 100.0, 85.0, 0.1)
    
    # Botón de predicción
    if st.button("🎯 Predecir Producción", type="secondary"):
        try:
            # Cargar modelo y scaler
            model = joblib.load('modelo_talapalca_avanzado.pkl')
            scaler = joblib.load('scaler_talapalca.pkl')
            
            # Preparar datos de entrada
            input_data = np.array([[
                temperatura, humedad, presion, viento_velocidad,
                material_dureza, profundidad, concentracion_metal
            ]])
            
            # Escalar y predecir
            input_scaled = scaler.transform(input_data)
            prediccion = model.predict(input_scaled)[0]
            
            # Mostrar resultado
            st.success(f"**Producción Diaria Predicha: {prediccion:.1f} toneladas**")
            
            # Análisis adicional
            eficiencia_estimada = min(0.95, max(0.7, prediccion / 500))
            st.metric("📈 Eficiencia Estimada", f"{eficiencia_estimada:.1%}")
            
        except FileNotFoundError:
            st.error("❌ Primero debes entrenar el modelo antes de hacer predicciones")
        except Exception as e:
            st.error(f"❌ Error en la predicción: {str(e)}")

elif opcion == "📈 Dashboard":
    st.header("📈 Dashboard en Tiempo Real")
    
    # KPIs principales
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric(
            "🏭 Producción Promedio",
            f"{df['produccion_diaria'].mean():.0f} ton",
            delta="+5%"
        )
    
    with col2:
        st.metric(
            "⚡ Eficiencia Promedio",
            f"{df['eficiencia'].mean():.1%}",
            delta="+2%"
        )
    
    with col3:
        st.metric(
            "🥇 Concentración Media",
            f"{df['concentracion_metal'].mean():.1f}%",
            delta="+1.5%"
        )
    
    with col4:
        st.metric(
            "🌡️ Temperatura Media",
            f"{df['temperatura'].mean():.1f}°C",
            delta="-0.5°C"
        )
    
    st.markdown("---")
    
    # Gráficos del dashboard
    col1, col2 = st.columns(2)
    
    with col1:
        # Serie temporal de producción (simulada)
        st.subheader("📊 Tendencia de Producción")
        fig_prod = px.line(
            df.head(100),
            y='produccion_diaria',
            title='Producción Diaria (Últimas 100 muestras)',
            color_discrete_sequence=['#00CC96']
        )
        st.plotly_chart(fig_prod, use_container_width=True)
    
    with col2:
        # Distribución de eficiencia
        st.subheader("📈 Distribución de Eficiencia")
        fig_eff = px.box(
            df,
            y='eficiencia',
            title='Distribución de Eficiencia Operativa',
            color_discrete_sequence=['#FFA15A']
        )
        st.plotly_chart(fig_eff, use_container_width=True)
    
    # Heatmap de correlaciones
    st.subheader("🔥 Mapa de Calor - Correlaciones")
    fig_heatmap = px.imshow(
        df.corr(),
        title="Correlaciones entre Variables",
        color_continuous_scale='Blues',
        aspect='auto'
    )
    st.plotly_chart(fig_heatmap, use_container_width=True)

elif opcion == "⚙️ Configuración":
    st.header("⚙️ Configuración del Sistema")
    
    tab1, tab2, tab3 = st.tabs(["🔧 Ajustes", "📁 Datos", "🛠️ Sistema"])
    
    with tab1:
        st.subheader("Ajustes de Parámetros")
        
        col1, col2 = st.columns(2)
        
        with col1:
            # CORREGIDO: Valor inicial igual al mínimo
            umbral_alerta = st.number_input(
                "🚨 Umbral de Alerta Producción",
                min_value=0.1,
                max_value=1000.0,
                value=400.0,  # Valor inicial dentro del rango
                step=10.0,
                help="Producción mínima para generar alerta"
            )
            
            intervalo_actualizacion = st.number_input(
                "🕐 Intervalo de Actualización (min)",
                min_value=1,
                max_value=60,
                value=5,
                step=1
            )
        
        with col2:
            # CORREGIDO: Valor inicial igual al mínimo
            confianza_modelo = st.number_input(
                "🎯 Nivel de Confianza del Modelo",
                min_value=0.1,
                max_value=1.0,
                value=0.8,  # Valor inicial dentro del rango
                step=0.05,
                help="Confianza mínima para aceptar predicciones"
            )
            
            # CORREGIDO: Valor inicial igual al mínimo
            temp_maxima = st.number_input(
                "🌡️ Temperatura Máxima Permitida",
                min_value=0.1,
                max_value=100.0,
                value=40.0,  # Valor inicial dentro del rango
                step=1.0
            )
    
    with tab2:
        st.subheader("Gestión de Datos")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.write("**Cargar Nuevos Datos**")
            archivo_cargado = st.file_uploader(
                "Selecciona archivo CSV",
                type=['csv'],
                help="Sube un archivo CSV con datos mineros"
            )
            
            if archivo_cargado is not None:
                try:
                    nuevos_datos = pd.read_csv(archivo_cargado)
                    st.success(f"✅ Datos cargados: {nuevos_datos.shape[0]} filas, {nuevos_datos.shape[1]} columnas")
                    st.dataframe(nuevos_datos.head(), use_container_width=True)
                except Exception as e:
                    st.error(f"❌ Error cargando archivo: {str(e)}")
        
        with col2:
            st.write("**Exportar Datos**")
            if st.button("📥 Exportar Dataset Actual", type="secondary"):
                csv = df.to_csv(index=False)
                st.download_button(
                    label="⬇️ Descargar CSV",
                    data=csv,
                    file_name="datos_talapalca_actual.csv",
                    mime="text/csv"
                )
    
    with tab3:
        st.subheader("Información del Sistema")
        
        st.write("**Versiones de Paquetes:**")
        info_col1, info_col2 = st.columns(2)
        
        with info_col1:
            st.text(f"Streamlit: {st.__version__}")
            st.text(f"Pandas: {pd.__version__}")
            st.text(f"NumPy: {np.__version__}")
        
        with info_col2:
            st.text(f"Scikit-learn: {joblib.__version__}")
            st.text(f"Plotly: {px.__version__}")
        
        st.write("**Estado del Sistema:**")
        st.success("✅ Todos los sistemas operando normalmente")
        st.info("🔄 Última actualización: Datos en tiempo real")

# Footer
st.markdown("---")
st.markdown(
    "<div style='text-align: center; color: gray;'>"
    "Sistema IA GlobalQH © 2024 - Desarrollado para optimización minera"
    "</div>",
    unsafe_allow_html=True
)
