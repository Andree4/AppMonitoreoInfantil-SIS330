import os
import requests

# Nombre de la carpeta donde se guardarán las imágenes
output_folder = "Dataset +18"
os.makedirs(output_folder, exist_ok=True)

# Nombre del archivo con los enlaces
input_file = r"F:\Tareas Hechas\Inteligencia Artificial 3\Codigo +18\urldataset+18.txt"

# Leer los enlaces desde el archivo
with open(input_file, "r") as file:
    links = file.readlines()

for index, link in enumerate(links):
    link = link.strip()
    if not link:
        continue

    try:
        response = requests.get(link, stream=True, timeout=10)
        if response.status_code == 200:
            # Obtener la extensión del archivo (si es posible)
            ext = link.split(".")[-1]
            if len(ext) > 5 or "/" in ext:
                ext = "jpg"  # Extensión por defecto si no es válida

            image_path = os.path.join(output_folder, f"image_{index}.{ext}")

            with open(image_path, "wb") as img_file:
                for chunk in response.iter_content(1024):
                    img_file.write(chunk)

            print(f"Descargado: {image_path}")
        else:
            print(f"Error {response.status_code}: {link} (Saltando)")
    except requests.RequestException:
        print(f"Error de conexión: {link} (Saltando)")

print("Descarga completa.")
