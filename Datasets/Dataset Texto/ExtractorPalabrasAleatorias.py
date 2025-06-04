import requests
import re
from collections import Counter

# URL del archivo
url = "https://raw.githubusercontent.com/garnachod/TwitterSentimentDataset/refs/heads/master/tweets_pos_clean.txt"

# Descargar contenido del archivo
response = requests.get(url)
if response.status_code == 200:
    text = response.text
else:
    print("No se pudo descargar el archivo.")
    exit()

# Eliminar símbolos y mantener solo palabras en español
words = re.findall(r'\b[a-záéíóúñü]+\b', text.lower())

# Contar palabras únicas
unique_words = Counter(words)

# Extraer las 3000 palabras únicas más comunes
top_10000_words = [word for word, _ in unique_words.most_common(10000)]

# Guardar las palabras en un archivo
with open("palabras_unicas3.txt", "w", encoding="utf-8") as f:
    for word in top_10000_words:
        f.write(word + "\n")

print("Palabras extraídas y guardadas en 'palabras_unicas3.txt'")
