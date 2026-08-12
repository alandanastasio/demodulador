# Demodulador SDR Multicanal 📻

Una plataforma avanzada de Procesamiento Digital de Señales (DSP) en tiempo real con una interfaz gráfica basada en PyQt6. Este software permite la conexión directa a múltiples periféricos SDR (Software Defined Radio) para visualizar el espectro, realizar análisis forense de señales y demodular en tiempo real distintos protocolos y modulaciones de RF.

## Características Principales

*   **Arquitectura Modular:** Diseñado con un sistema de plugins de DSP, lo que permite agregar nuevos demoduladores de manera rápida y eficiente sin modificar el núcleo de la interfaz.
*   **Aceleración en Tiempo Real:** Interfaz optimizada con `pyqtgraph` y subprocesamiento multi-hilo en Python para garantizar análisis libre de congelamientos (GIL yield & direct-yield pipelines).
*   **Análisis Visual:** 
    *   Analizador de Espectro interactivo (densidad espectral de potencia en dB).
    *   Gráfico de Cascada (Waterfall) de alto rendimiento.
*   **Demoduladores Integrados:**
    *   **WBFM (Audio Estéreo):** Radio FM comercial con decodificación completa del subcanal estéreo (Tono Piloto de 19kHz y subportadora L-R de 38kHz) a través de un lazo de seguimiento de fase (PLL) y audio en vivo usando `sounddevice`.
    *   **WBFM (Mono):** Demodulación simple para métricas crudas.
    *   **AM:** Demodulación de Amplitud Modulada clásica.
    *   **WiFi 802.11a/g (OFDM):** Sincronización fina Schmidl & Cox, estimación de canal, ecualización Zero-Forcing, visualización de la constelación OFDM, y decodificación Viterbi usando algoritmos matemáticos crudos sin librerías externas.
    *   **LTE:** Sincronizador de Trama LTE con detección de CFO y correlación en frecuencia cruzada de secuencias Zadoff-Chu (PSS).
*   **Modo Laboratorio:** Posibilidad de grabar el stream de I/Q crudo directamente a un archivo de disco (Raw IQ Sink) y reproducirlo posteriormente (Offline mode) simulando la tasa de muestreo original para facilitar el desarrollo de algoritmos de DSP.

## Hardware Soportado

El demodulador integra de forma nativa los siguientes equipos SDR (y descarga sus FPGAs/firmwares correspondientes):
*   **Ettus Research USRP B200** (a través de UHD)
*   **Nuand bladeRF x40** (a través de libbladerf)
*   **HackRF One** (a través de python-hackrf)
*   **RTL-SDR** (a través de librtlsdr)

## Instalación y Puesta en Marcha

Este proyecto está diseñado para ejecutarse en entornos Linux (testeado en Ubuntu/Debian).

### 1. Clonar el repositorio
```bash
git clone https://github.com/alandanastasio/demodulador.git
cd demodulador
```

### 2. Ejecutar el Script de Configuración del Sistema
Este script instalará todos los drivers a nivel de sistema operativo (UHD, bladeRF, RTL, HackRF, dependencias de audio de PortAudio) y descargará las imágenes de FPGA correspondientes para tu hardware SDR.
```bash
chmod +x setup.sh
./setup.sh
```

### 3. Crear el entorno virtual con UV
El proyecto utiliza [uv](https://github.com/astral-sh/uv) como gestor rápido de dependencias. 
> ⚠️ **IMPORTANTE:** Dado que el driver oficial de Ettus (UHD) se instaló a nivel de sistema operativo con `apt`, es de vital importancia crear el entorno virtual de Python permitiéndole leer los paquetes del sistema, de lo contrario fallará al importar `uhd`.

Ejecutá los siguientes comandos para crear y sincronizar el entorno virtual:
```bash
uv venv --system-site-packages
uv sync
```

### 4. Arrancar el programa
Con los drivers en orden y el hardware conectado (se recomienda encarecidamente utilizar un puerto y cable **USB 3.0** para anchos de banda mayores a 5 MHz), simplemente iniciá la aplicación gráfica:
```bash
uv run python demodulador/startup_window.py
```

## Solución de Problemas

*   **Console spam con letras "O" u "Overflow":** Esto significa que la tasa de muestreo (Sample Rate) configurada excede la capacidad de transferencia de tu puerto USB (muy común si usas USB 2.0) o de procesamiento de tu CPU. Baja el ancho de banda o el sample rate en el panel superior, o asegurate de usar un puerto SuperSpeed azul.
*   **ModuleNotFoundError: No module named 'uhd':** Olvidaste crear el entorno de `uv` con la bandera `--system-site-packages`. Borrá la carpeta `.venv` oculta del proyecto y volvé al Paso 3.
*   **La bladeRF tira error al iniciar:** Asegurate de que descargaste el firmware e instalaste las reglas `udev` (que están empaquetadas en el Paso 2 del `setup.sh`). Luego, desenchufa y vuelve a enchufar el SDR.
