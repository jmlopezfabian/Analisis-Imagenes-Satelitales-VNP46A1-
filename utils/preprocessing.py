import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.tsa.seasonal import STL
from statsmodels.graphics.tsaplots import plot_acf
import matplotlib.ticker as mticker
from typing import List, Tuple, Dict, Any


def load_radiance_data(path: str, municipio: str) -> pd.DataFrame:
    """
    Carga y filtra datos de radianza para un municipio específico
    
    Parameters:
    -----------
    path : str
        Ruta al archivo CSV con datos de municipios completos
    municipio : str
        Nombre del municipio a filtrar (en minúsculas)
    
    Returns:
    --------
    pd.DataFrame
        DataFrame filtrado para el municipio especificado
    """
    data = pd.read_csv(path)
    data["Fecha"] = pd.to_datetime(data["Fecha"])
    municipio_data = data[data["Municipio"] == municipio].copy()
    return municipio_data


def remove_outliers_by_top_n(df: pd.DataFrame, column: str, n: int = 2, date_column: str = "Fecha") -> pd.DataFrame:
    """
    Elimina outliers identificando las top N fechas con valores más altos en una columna
    
    Parameters:
    -----------
    df : pd.DataFrame
        DataFrame con los datos
    column : str
        Columna por la cual identificar outliers
    n : int, default=2
        Número de fechas a eliminar (las de mayor valor)
    date_column : str, default="Fecha"
        Nombre de la columna de fechas
    
    Returns:
    --------
    pd.DataFrame
        DataFrame sin las fechas problemáticas
    """
    fechas = df.sort_values(by=column, ascending=False).iloc[:n][date_column].values
    df_clean = df[~df[date_column].isin(fechas)].copy()
    return df_clean


def apply_stl_decomposition(df: pd.DataFrame, columns: List[str], period: int, 
                           date_column: str = "Fecha", 
                           municipio_column: str = "Municipio",
                           pixel_column: str = "Cantidad_de_pixeles") -> pd.DataFrame:
    """
    Aplica descomposición STL (Seasonal and Trend decomposition using Loess) a múltiples columnas
    
    Parameters:
    -----------
    df : pd.DataFrame
        DataFrame con los datos de series de tiempo
    columns : List[str]
        Lista de columnas a las que aplicar STL
    period : int
        Periodo de estacionalidad (365 para datos diarios, 4 para datos trimestrales)
    date_column : str, default="Fecha"
        Nombre de la columna de fechas
    municipio_column : str, default="Municipio"
        Nombre de la columna de municipio
    pixel_column : str, default="Cantidad_de_pixeles"
        Nombre de la columna de cantidad de pixeles
    
    Returns:
    --------
    pd.DataFrame
        DataFrame con las tendencias extraídas y series desestacionalizadas
    """
    result_dict = {
        "fecha": df[date_column],
        "municipio": df[municipio_column],
    }
    
    if pixel_column in df.columns:
        result_dict["total_pixeles"] = df[pixel_column]
    
    for column in columns:
        stl = STL(df[column], period=period)
        res = stl.fit()
        trend = res.trend
        
        # Guardar el trend y la serie detrended
        result_dict[f"{column}_trend"] = trend.values
        result_dict[f"{column}_detrend"] = df[column].values - trend.values
    
    return pd.DataFrame(result_dict)


def aggregate_to_quarterly(df: pd.DataFrame, date_column: str = "fecha") -> pd.DataFrame:
    """
    Agrega datos diarios de radianza a nivel trimestral
    
    Parameters:
    -----------
    df : pd.DataFrame
        DataFrame con datos diarios de radianza (después de STL)
    date_column : str, default="fecha"
        Nombre de la columna de fechas
    
    Returns:
    --------
    pd.DataFrame
        DataFrame agregado a nivel trimestral
    """
    df = df.copy()
    df["quarter"] = df[date_column].dt.to_period("Q")
    
    # Definir agregaciones específicas para cada columna
    agg_dict = {
        "total_pixeles": "median",
        "Suma_de_radianza_trend": ["sum", "median", "std", "mean"],
        "Media_de_radianza_trend": ["sum", "median", "std", "mean"],
        "Desviacion_estandar_de_radianza_trend": ["sum", "median", "std", "mean"],
        "Maximo_de_radianza_trend": ["sum", "median", "std", "mean"],
        "Minimo_de_radianza_trend":["sum", "median", "std", "mean"],
        "Percentil_25_de_radianza_trend":["sum", "median", "std", "mean"],
        "Percentil_50_de_radianza_trend":["sum", "median", "std", "mean"],
        "Percentil_75_de_radianza_trend":["sum", "median", "std", "mean"],
    }
    
    # Filtrar solo las columnas que existen en el DataFrame
    agg_dict_filtered = {k: v for k, v in agg_dict.items() if k in df.columns}
    
    df_quarter = df.groupby("quarter").agg(agg_dict_filtered).reset_index()
    
    # Aplanar nombres de columnas multi-nivel
    columns = ["_".join(col) if isinstance(col, tuple) else col for col in df_quarter.columns.values]
    columns = [col.replace("quarter_", "quarter") for col in columns]
    df_quarter.columns = columns
    
    # Convertir quarter a string
    df_quarter["quarter"] = df_quarter["quarter"].astype(str)
    
    return df_quarter


def preprocessing_pib(path: str, period: int = 4, 
                     date_column: str = "fecha",
                     municipio_column: str = "municipio",
                     pib_column: str = "pib_mun") -> pd.DataFrame:
    """
    Preprocesamiento de la serie de PIB municipal
    
    Parameters:
    -----------
    path : str
        Ruta al archivo CSV con datos de PIB
    period : int, default=4
        Periodo de estacionalidad (4 para datos trimestrales)
    date_column : str, default="fecha"
        Nombre de la columna de fechas
    municipio_column : str, default="municipio"
        Nombre de la columna de municipio
    pib_column : str, default="pib_mun"
        Nombre de la columna de PIB
    
    Returns:
    --------
    pd.DataFrame
        DataFrame con PIB desestacionalizado y su tendencia
    """
    # Cargar datos
    pib_df = pd.read_csv(path)
    pib_df[date_column] = pd.to_datetime(pib_df[date_column])
    pib_df.sort_values(by=date_column, ascending=True, inplace=True)
    
    # Aplicar STL
    stl = STL(pib_df[pib_column], period=period)
    res = stl.fit()
    trend = res.trend
    
    # Crear DataFrame resultado
    pib_trend = pd.DataFrame({
        date_column: pib_df[date_column],
        municipio_column: pib_df[municipio_column],
        f"{pib_column}_desestacionalizado": pib_df[pib_column].values - res.seasonal.values,
        f"{pib_column}_trend": trend.values
    })
    
    return pib_trend


def merge_pib_radianza_quarterly(radianza_df: pd.DataFrame, 
                                 pib_df: pd.DataFrame,
                                 date_column: str = "fecha") -> pd.DataFrame:
    """
    Une datos de radianza y PIB a nivel trimestral
    
    Parameters:
    -----------
    radianza_df : pd.DataFrame
        DataFrame con datos de radianza agregados trimestralmente
    pib_df : pd.DataFrame
        DataFrame con datos de PIB
    date_column : str, default="fecha"
        Nombre de la columna de fechas en el DataFrame de PIB
    
    Returns:
    --------
    pd.DataFrame
        DataFrame con datos unidos por trimestre
    """
    pib_df = pib_df.copy()
    pib_df["quarter"] = pib_df[date_column].dt.to_period("Q")
    pib_df["quarter"] = pib_df["quarter"].astype(str)
    
    merged_df = pd.merge(
        radianza_df, 
        pib_df, 
        on="quarter", 
        how="inner", 
        suffixes=("_luz", "_pib")
    )
    
    return merged_df


def preprocessing_completo(municipio: str,
                          path_radianza: str = None,
                          path_pib: str = None,
                          data_dir: str = "../data_municipios",
                          output_trend_radianza: str = None,
                          output_trend_pib: str = None,
                          output_merged: str = None,
                          save_outputs: bool = True) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Pipeline completo de preprocesamiento para cualquier municipio
    
    Parameters:
    -----------
    municipio : str
        Nombre del municipio a procesar (en minúsculas, ej: "monterrey", "cdmx", "guadalajara")
    path_radianza : str, optional
        Ruta al archivo CSV con datos de radianza de municipios.
        Si no se proporciona, se usa: {data_dir}/municipios_completos.csv
    path_pib : str, optional
        Ruta al archivo CSV con datos de PIB.
        Si no se proporciona, se busca: {data_dir}/serie_PIB_{municipio[:3]}.csv
    data_dir : str, default="../data_municipios"
        Directorio donde se encuentran los datos (usado si path_radianza o path_pib no se especifican)
    output_trend_radianza : str, optional
        Ruta para guardar datos de trend de radianza.
        Si save_outputs=True y no se especifica, se usa: {data_dir}/{municipio}_trend.csv
    output_trend_pib : str, optional
        Ruta para guardar datos de trend de PIB.
        Si save_outputs=True y no se especifica, se usa: {data_dir}/pib_trend_{municipio[:3]}.csv
    output_merged : str, optional
        Ruta para guardar datos merged trimestrales.
        Si save_outputs=True y no se especifica, se usa: {data_dir}/serie_trimestral_{municipio}.csv
    save_outputs : bool, default=True
        Si True, guarda los archivos CSV de salida
    
    Returns:
    --------
    Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]
        (radianza_trend, pib_trend, merged_quarterly)
    
    Ejemplos de uso:
    ----------------
    # Uso básico con Monterrey
    >>> radianza, pib, merged = preprocessing_completo("monterrey")
    
    # Procesar CDMX
    >>> radianza, pib, merged = preprocessing_completo("cdmx")
    
    # Con rutas personalizadas
    >>> radianza, pib, merged = preprocessing_completo(
    ...     municipio="guadalajara",
    ...     path_radianza="./datos/radianza.csv",
    ...     path_pib="./datos/pib_gdl.csv",
    ...     save_outputs=False
    ... )
    """
    import os
    
    # Construir rutas de entrada si no se proporcionaron
    if path_radianza is None:
        path_radianza = os.path.join(data_dir, "municipios_completos.csv")
    
    if path_pib is None:
        # Usar las primeras 3 letras del municipio para el nombre del archivo
        municipio_prefix = municipio[:3].lower()
        path_pib = os.path.join(data_dir, f"serie_PIB_{municipio_prefix}.csv")
    
    # Construir rutas de salida si save_outputs=True y no se especificaron
    if save_outputs:
        if output_trend_radianza is None:
            output_trend_radianza = os.path.join(data_dir, f"{municipio}_trend.csv")
        if output_trend_pib is None:
            municipio_prefix = municipio[:3].lower()
            output_trend_pib = os.path.join(data_dir, f"pib_trend_{municipio_prefix}.csv")
        if output_merged is None:
            output_merged = os.path.join(data_dir, f"serie_trimestral_{municipio}.csv")
    else:
        output_trend_radianza = None
        output_trend_pib = None
        output_merged = None
    
    # Cargar y filtrar datos de radianza
    radianza_df = load_radiance_data(path_radianza, municipio)
    
    # Remover outliers
    radianza_df = remove_outliers_by_top_n(radianza_df, column="Suma_de_radianza", n=2)
    
    # Identificar columnas de medidas (todas excepto Fecha, Municipio, Cantidad_de_pixeles)
    exclude_cols = ["Fecha", "Municipio", "Cantidad_de_pixeles"]
    medidas = [col for col in radianza_df.columns if col not in exclude_cols]
    
    # Aplicar STL a datos de radianza
    radianza_trend = apply_stl_decomposition(
        radianza_df, 
        columns=medidas, 
        period=365,
        date_column="Fecha",
        municipio_column="Municipio",
        pixel_column="Cantidad_de_pixeles"
    )
    
    # Procesar PIB
    pib_trend = preprocessing_pib(path_pib, period=4)
    
    # Agregar radianza a nivel trimestral
    radianza_quarterly = aggregate_to_quarterly(radianza_trend)
    
    # Merge de PIB y radianza
    merged_quarterly = merge_pib_radianza_quarterly(radianza_quarterly, pib_trend)
    
    # Guardar resultados si se especifican rutas de salida
    if output_trend_radianza:
        radianza_trend.to_csv(output_trend_radianza, index=False)
    
    if output_trend_pib:
        pib_trend.to_csv(output_trend_pib, index=False)
    
    if output_merged:
        merged_quarterly.to_csv(output_merged, index=False)
    
    return radianza_trend, pib_trend, merged_quarterly


# Funciones de visualización auxiliares

def plot_series_temporal(df: pd.DataFrame, 
                        date_column: str,
                        value_columns: List[str],
                        title_prefix: str = "",
                        figsize: Tuple[int, int] = (12, 3),
                        color: str = "#301D82",
                        sharex: bool = True) -> None:
    """
    Grafica múltiples series temporales
    
    Parameters:
    -----------
    df : pd.DataFrame
        DataFrame con los datos
    date_column : str
        Nombre de la columna de fechas
    value_columns : List[str]
        Lista de columnas a graficar
    title_prefix : str, default=""
        Prefijo para los títulos de las gráficas
    figsize : Tuple[int, int], default=(12, 3)
        Tamaño de cada subplot
    color : str, default="#301D82"
        Color de las líneas
    sharex : bool, default=True
        Compartir eje x entre subplots
    """
    n_plots = len(value_columns)
    fig, axes = plt.subplots(n_plots, 1, 
                            figsize=(figsize[0], figsize[1] * n_plots), 
                            sharex=sharex)
    
    if n_plots == 1:
        axes = [axes]
    
    for i, column in enumerate(value_columns):
        sns.lineplot(ax=axes[i], data=df, x=date_column, y=column, color=color)
        axes[i].set_title(f"{title_prefix}{column}")
        axes[i].set_xlabel("Fecha")
        axes[i].set_ylabel(column)
    
    plt.tight_layout()
    plt.show()


def plot_acf_multiple(df: pd.DataFrame, 
                     value_columns: List[str],
                     lags: int = 400,
                     figsize: Tuple[int, int] = (20, 3),
                     title_suffix: str = "") -> None:
    """
    Grafica ACF para múltiples series
    
    Parameters:
    -----------
    df : pd.DataFrame
        DataFrame con los datos
    value_columns : List[str]
        Lista de columnas para calcular ACF
    lags : int, default=400
        Número de lags para ACF
    figsize : Tuple[int, int], default=(20, 3)
        Tamaño de cada subplot
    title_suffix : str, default=""
        Sufijo para los títulos
    """
    n_plots = len(value_columns)
    fig, axes = plt.subplots(n_plots, 1, 
                            figsize=(figsize[0], figsize[1] * n_plots), 
                            sharex=False)
    
    if n_plots == 1:
        axes = [axes]
    
    for i, column in enumerate(value_columns):
        plot_acf(df[column], ax=axes[i], lags=lags)
        axes[i].set_title(f"ACF de {column} {title_suffix}")
        axes[i].xaxis.set_major_locator(mticker.MaxNLocator(nbins=40, integer=True))
        axes[i].xaxis.set_minor_locator(mticker.AutoMinorLocator(2))
        axes[i].tick_params(axis='x', which='major', labelrotation=45)
        axes[i].grid(True)
    
    plt.tight_layout()
    plt.show()
