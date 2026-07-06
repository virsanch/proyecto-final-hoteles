import sqlite3
import pandas as pd

def crear_db_y_tabla():
    # Se conecta 
    conexion = sqlite3.connect('mi_base_de_datos.db')
    cursor = conexion.cursor()
    
    cursor.execute("DROP TABLE IF EXISTS reservas")
    cursor.execute("""
        CREATE TABLE reservas (
            hotel TEXT,
            meal TEXT,
            lead_time INTEGER,
            arrival_date_year INTEGER
        )
    """)
    conexion.commit()
    conexion.close()
    print("Tabla creada.")

def cargar_datos_en_db(df):
    conexion = sqlite3.connect('mi_base_de_datos.db')
    df.to_sql('reservas', conexion, if_exists='replace', index=False)
    conexion.close()
    print("Datos cargados.")

def consultar_datos(query):
    conexion = sqlite3.connect('mi_base_de_datos.db')
    resultado = pd.read_sql(query, conexion)
    conexion.close()
    return resultado