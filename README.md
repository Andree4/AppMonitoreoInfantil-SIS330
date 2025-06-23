# 🌟 Software de Monitoreo de Contenido para Protección Infantil 🛡️

¡Bienvenidos a un proyecto que protege a los más pequeños en el mundo digital! 🚀 Esta aplicación Android utiliza inteligencia artificial (IA) para monitorear en tiempo real audio, imágenes y videos en dispositivos móviles, detectando contenido ofensivo, no apto para menores (NSFW) o violento con una precisión del **91-97%**. 📊 Emite alertas instantáneas a los padres y bloquea contenido con un PIN parental. 🔐

---

## 📝 **Descripción General**

La app captura audio 🎙️ y capturas de pantalla 📸 cada 0.2, enviándolos a un servidor Flask para analizarlos con modelos de IA. 🧠 Aquí te explicamos cómo funciona:

- **Reconocimiento de Voz** 🎵: API Speech-to-Text transcribe audio en tiempo real.
- **Procesamiento de Lenguaje Natural (PLN)** 💬: BERT detecta lenguaje ofensivo en español.
- **Visión por Computadora** 🖼️:
  - MobileNetV2 clasifica imágenes como NSFW.
  - YOLOv8 + TimesFormer identifica violencia en videos (secuencias de 8 frames).
- **Cliente Android** 📱: Captura pantalla con MediaProjection, graba audio con AudioRecord y envía datos vía WebSockets.
- **Servidor Flask** 🌐: Procesa datos en `/texto` y `/video`, ejecuta inferencias y envía alertas.
- **Seguridad** 🔒: Overlay con PIN parental bloquea contenido y notifica a los padres.

La arquitectura cliente-servidor asegura un análisis rápido en menos de 200ms. ⚡

---

## 🛠️ **Tecnologías y Dependencias**

### 📱 **Aplicación Android**

- **Lenguajes**: Kotlin/Java ☕
- **APIs**:
  - 📸 MediaProjection (captura de pantalla)
  - 🎙️ AudioRecord (captura audio)
  - 🗣️ Speech-to-Text (transcripción)
- **Requisitos**: Android 10+ (API 29) 📲
- **Dependencias**: WebSocket, Android SDK 🧩

### 🖥️ **Servidor Flask**

- **Lenguaje**: Python 3.12 🐍
- **Framework**: Flask (API + WebSockets) 🧪
- **Bibliotecas de IA**:
  - PyTorch 🔥
  - Transformers (HuggingFace) 🤗
  - Ultralytics (YOLOv8) 🎯
  - TorchVision, OpenCV 🖼️
- **Modelos Preentrenados**:
  - BERT: `dccuchile/bert-base-spanish-wwm-uncased` 📜
  - MobileNetV2: ImageNet 🖼️
  - YOLOv8: COCO 🕵️‍♂️
  - TimesFormer: `timesformer-base-finetuned-k400` 🎥
- **Hardware**:
  - CPU: 4+ núcleos 🧮
  - RAM: 16GB 💾
  - GPU: NVIDIA CUDA (recomendado) 🚀

### 📊 **Datasets**

- **Texto Ofensivo**: 4,278 frases (864 ofensivas, 3,414 no ofensivas) de [spanlp](https://github.com/jfreddypuentes/spanlp). 📝
- **Contenido NSFW**: 9,910 imágenes (4,254 NSFW, 5,656 aptas) de [nsfw_data_scraper](https://github.com/alex000kim/nsfw_data_scraper) y Kaggle. 🖼️
- **Contenido Violento**: 2,000 videos (1,000 violentos, 1,000 no violentos) de [Real-Life Violence Situations](https://www.kaggle.com/datasets/mohamedmustafa/real-life-violence-situations-dataset). 🎬

---

## 🚀 **Cómo Funciona**

1. **Captura de Datos** 📸🎙️:

   - La app toma capturas de pantalla cada 0.2s con MediaProjection.
   - Graba audio con AudioRecord y lo transcribe con Speech-to-Text.
   - Envía imágenes y texto al servidor vía WebSockets (`/texto`, `/video`).

2. **Procesamiento en el Servidor** 🧠:

   - 💬 **Texto**: BERT detecta lenguaje ofensivo.
   - 🖼️ **Imágenes**: MobileNetV2 clasifica NSFW.
   - 🎥 **Videos**: YOLOv8 detecta personas/colisiones; TimesFormer confirma violencia.
   - Genera alertas si detecta contenido inapropiado.

3. **Acciones** 🚨:

   - Notifica a la app si encuentra contenido ofensivo/NSFW/violento.
   - Muestra un overlay con PIN parental para bloquear el contenido. 🔐
   - Envía notificaciones a los padres con detalles. 📩

4. **Seguridad y Configuración** 🔧:
   - Solicita permisos al inicio (pantalla, micrófono).
   - Los padres configuran un PIN de seguridad.

---

## 🛠️ **Tutorial para Desplegar la Aplicación**

### 📋 **Prerrequisitos**

- **Dispositivo Android**: Android 10+ 📱
- **Servidor**:
  - SO: Windows 🖥️
  - Python 3.12 🐍
  - GPU NVIDIA con CUDA (opcional) 🖥️
- **Herramientas**:
  - Android Studio 🛠️
  - Git 📂
  - pip 🐍

### 🕹️ **Paso 1: Clonar el Repositorio**

```bash
git clone https://github.com/usuario/nombre-repositorio.git
cd nombre-repositorio
```

### 🌐 **Paso 2: Configurar el Servidor Flask**

1. **Instalar Dependencias** 📦:

   ```bash
   python -m venv venv
   source venv/bin/activate  # Windows: venv\Scripts\activate
   pip install flask torch torchvision transformers ultralytics opencv-python websocket-client
   ```

2. **Iniciar el Servidor** 🚀:
   - Configura IP/puerto en `server.py` (predeterminado: `0.0.0.0:5000`).
   - Ejecuta:
     ```bash
     python server.py
     ```

### 📱 **Paso 3: Configurar la Aplicación Android**

1. **Abrir el Proyecto** 🛠️:

   - Abre `android_app/` en Android Studio.
   - Sincroniza con Gradle.

2. **Configurar Conexión al Servidor** 🌐:

   - Edita `app/src/main/res/values/strings.xml`:
     ```xml
     <string name="server_url">ws://<IP_DEL_SERVIDOR>:5000</string>
     ```

3. **Compilar e Instalar** 📲:

   - Conecta un dispositivo Android o usa un emulador.
   - Compila: `Run > Run 'app'`.

4. **Configurar Permisos** 🔐:
   - Concede permisos de pantalla y micrófono.
   - Configura el PIN parental en la app.

### 🧪 **Paso 4: Probar la Aplicación**

1. **Iniciar el Servidor** 🌐:

   - Verifica que Flask esté corriendo.

2. **Ejecutar la App** 📱:

   - Abre la app y reproduce contenido (YouTube, música, etc.).
   - Confirma que detecta contenido ofensivo/NSFW/violento y muestra el overlay.

3. **Monitorear Logs** 📜:
   - Usa Logcat en Android Studio para depurar la app.
   - Revisa los logs del servidor Flask.

---

## ⚠️ **Notas de Despliegue**

- **Rendimiento**: Dispositivos de gama baja pueden tener retrasos con conexiones inestables. Optimiza la compresión si es necesario. 🐢
- **Escalabilidad**: Usa AWS/GCP para múltiples usuarios. ☁️
- **Mantenimiento**: Actualiza modelos y dependencias regularmente. 🛠️
