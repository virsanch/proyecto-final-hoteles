import pandas as pd
import os

def cargar_datos():
    
    ruta_archivo = "../data/hotel_bookings.csv"
    
    if os.path.exists(ruta_archivo):
        df = pd.read_csv(ruta_archivo)
        print("¡Éxito! Datos cargados correctamente.")
        return df
    else:
        print(f"Error: No se encontró el archivo en {ruta_archivo}")
        return None