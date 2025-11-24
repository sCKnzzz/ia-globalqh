import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
import io

# Configuración inicial
st.set_page_config(page_title="Sistema Talapalca", page_icon="🌊", layout="wide")
st.title("🌊 IA para la generación de Curvas Altura-Caudal")
st.markdown("**Sistema inteligente para análisis hidráulico**")

# Funciones básicas
def func_poly2(x, a, b, c):
    return a * x**2 + b * x + c

def func_poly3(x, a, b, c, d):
    return a * x**3 + b * x**2 + c * x + d

def func_pot(x, a, b):
    return a * x**b

# Datos de ejemplo para demostración
def cargar_datos_demo():
    """Cargar datos de demostración"""
    data = {
        'FECHA': ['2/10/2021', '2/24/2021', '4/13/2021', '5/11/2021', '9/26/2021'],
        'NIVEL DE AFORO (m)': [1.2, 1.01, 1.05, 0.97, 0.96],
        'CAUDAL (m3/s)': [4.79, 1.89, 2.15, 1.11, 1.1],
        'AREA (m2)': [3.47, 2.76, 2.64, 2.07, 1.97],
        'ANCHO RIO (m)': [8.5, 8.5, 8.3, 7.5, 7.5],
        'VELOCIDAD (m/s)': [1.38, 0.62, 0.74, 0.46, 0.48]
    }
    return pd.DataFrame(data)

# Navegación
opcion = st.sidebar.radio("Navegación:", ["🏠 Inicio", "📤 Subir Aforos", "📊 Análisis"])

if opcion == "🏠 Inicio":
    st.header("Bienvenido al Sistema de Curvas Altura-Caudal")
    st.info("Esta aplicación utiliza IA para generar curvas altura-caudal a partir de datos de aforos.")
    
    st.subheader("Instrucciones:")
    st.markdown("""
    1. **📤 Subir Aforos**: Carga un archivo CSV con datos de aforos
    2. **📊 Análisis**: Visualiza y analiza los datos cargados
    
    **Columnas requeridas:**
    - NIVEL DE AFORO (m)
    - CAUDAL (m3/s) 
    - AREA (m2)
    - ANCHO RIO (m)
    - VELOCIDAD (m/s)
    """)
    
    # Mostrar datos de demo
    st.subheader("📋 Datos de Ejemplo")
    df_demo = cargar_datos_demo()
    st.dataframe(df_demo)

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
            
            if columnas_faltantes:
                st.error(f"❌ Faltan columnas: {', '.join(columnas_faltantes)}")
            else:
                st.success("✅ Todas las columnas necesarias están presentes")
                
                # Análisis básico
                st.subheader("📊 Análisis Básico")
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.metric("Nivel mínimo", f"{df['NIVEL DE AFORO (m)'].min():.2f} m")
                    st.metric("Nivel máximo", f"{df['NIVEL DE AFORO (m)'].max():.2f} m")
                
                with col2:
                    st.metric("Caudal mínimo", f"{df['CAUDAL (m3/s)'].min():.2f} m³/s")
                    st.metric("Caudal máximo", f"{df['CAUDAL (m3/s)'].max():.2f} m³/s")
                
                with col3:
                    st.metric("Velocidad promedio", f"{df['VELOCIDAD (m/s)'].mean():.2f} m/s")
                    st.metric("Área promedio", f"{df['AREA (m2)'].mean():.2f} m²")
                
                # Guardar en session state para usar en otras pestañas
                st.session_state.df_cargado = df
                
        except Exception as e:
            st.error(f"❌ Error al procesar el archivo: {e}")

elif opcion == "📊 Análisis":
    st.header("📊 Análisis de Datos")
    
    if 'df_cargado' not in st.session_state:
        st.warning("⚠️ Primero carga un archivo en la pestaña 'Subir Aforos'")
        st.info("💡 Usando datos de demostración para mostrar funcionalidades...")
        df = cargar_datos_demo()
    else:
        df = st.session_state.df_cargado
    
    # Gráfico básico
    st.subheader("📈 Gráfico Altura-Caudal")
    
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.scatter(df['NIVEL DE AFORO (m)'], df['CAUDAL (m3/s)'], alpha=0.7, s=60)
    ax.set_xlabel('Nivel (m)', fontweight='bold')
    ax.set_ylabel('Caudal (m³/s)', fontweight='bold')
    ax.set_title('Relación Altura-Caudal', fontweight='bold')
    ax.grid(True, alpha=0.3)
    
    st.pyplot(fig)
    
    # Estadísticas detalladas
    st.subheader("📋 Estadísticas Detalladas")
    st.dataframe(df.describe())
    
    # Análisis de correlación
    st.subheader("🔗 Correlaciones")
    
    # Calcular correlaciones
    columnas_numericas = ['NIVEL DE AFORO (m)', 'CAUDAL (m3/s)', 'AREA (m2)', 'VELOCIDAD (m/s)']
    correlaciones = df[columnas_numericas].corr()
    
    fig_corr, ax_corr = plt.subplots(figsize=(8, 6))
    im = ax_corr.imshow(correlaciones, cmap='coolwarm', aspect='auto')
    ax_corr.set_xticks(range(len(columnas_numericas)))
    ax_corr.set_yticks(range(len(columnas_numericas)))
    ax_corr.set_xticklabels(columnas_numericas, rotation=45)
    ax_corr.set_yticklabels(columnas_numericas)
    
    # Añadir valores de correlación
    for i in range(len(columnas_numericas)):
        for j in range(len(columnas_numericas)):
            text = ax_corr.text(j, i, f'{correlaciones.iloc[i, j]:.2f}',
                           ha="center", va="center", color="w", fontweight='bold')
    
    plt.colorbar(im)
    ax_corr.set_title('Matriz de Correlación', fontweight='bold')
    st.pyplot(fig_corr)

# Footer
st.markdown("---")
st.markdown("**🌊 Sistema de Análisis Hidráulico** • Desarrollado con Streamlit")