#Importación módulos
import pandas as pd
import seaborn as sns
import matplotlib as plt
import matplotlib.pyplot as plt
 
def extract(ruta):
    ruta="https://drive.google.com/uc?id=1Bl6P4csc7Swwh8t8GNF2KZMVNxn72m3a"
    archivo = pd.read_csv(ruta)
    archivo.head()

extract()