# Demodulador SDR

Aplicación de recepción y análisis de señales de radio con interfaz gráfica en PyQt6. Toma muestras I/Q de un SDR, muestra el espectro y el espectrograma, y permite seleccionar distintos modos de demodulación. También puede grabar muestras y volver a analizarlas desde la interfaz.

## Funciones disponibles

| Modo | Funciones implementadas |
| --- | --- |
| Analizador de espectro | Espectro interactivo, espectrograma, marcadores, trazas y controles de visualización. |
| FM: medición | Medición de la señal multiplex, piloto estéreo y canales de audio, con un espectro multiplex de mayor resolución y lectura RDS. |
| FM: escucha | Gráficos y métricas FM, reproducción de audio estéreo en vivo, controles para los canales izquierdo y derecho y lectura RDS. |
| WiFi 802.11a/g | Detección y sincronización de tramas OFDM, estimación de desplazamiento de frecuencia y canal, constelación, EVM y decodificación del campo SIGNAL. |
| Bluetooth Low Energy | Detección y métricas de ráfagas LE 1M y LE 2M; en LE 1M, decodificación de anuncios *legacy* con verificación de CRC e historial de paquetes. |
| LoRa | Detección de preámbulo y decodificación de tramas de la capa física, con encabezado, carga útil, CRC cuando está presente, símbolos y *sync word*. La interfaz ofrece BW de 125, 250 y 500 kHz y SF7 a SF12. |
| LTE | Análisis de señales downlink y uplink, con sincronización, constelaciones, métricas y visualizaciones de recursos. Las opciones de ancho de banda dependen de la tasa de muestreo que admita el SDR. |

El proyecto es un receptor y analizador. El controlador de HackRF de esta aplicación trabaja en recepción. El modo LoRa analiza tramas de la capa física.

Ambos modos FM decodifican la subportadora RDS de 57 kHz. El panel RDS muestra PI, el nombre corto de la emisora (PS), el código PTY y RadioText cuando la estación los transmite y se han recibido grupos válidos. PS y RadioText se forman a partir de varios grupos, por lo que pueden tardar unos segundos en aparecer. Se admiten los grupos 0A/0B y 2A/2B; el juego de caracteres extendidos todavía se muestra de forma aproximada como Latin-1. La estructura de la señal y los grupos se describen en [UIT-R BS.643-3](https://www.itu.int/dms_pubrec/itu-r/rec/bs/R-REC-BS.643-3-201105-S!!PDF-E.pdf).

## Hardware y capturas

La ventana inicial detecta estos SDR mediante sus controladores correspondientes:

| Equipo | Integración |
| --- | --- |
| HackRF One | `python-hackrf` / `libhackrf` |
| RTL-SDR | `pyrtlsdr` / `librtlsdr` |
| Nuand bladeRF x40 | Bindings de `libbladeRF` |
| Ettus USRP B200 | UHD |

La grabación desde **Rec/Play** escribe secuencialmente `muestras_iq_*.iq` en el directorio desde el que se ejecutó el programa. El archivo `.json` del mismo nombre guarda el tipo de muestra, la frecuencia central, la tasa de muestreo, el número de bloques y muestras, y una suma SHA-256; conservá ambos archivos juntos. Al reproducir una captura nueva se verifican su tamaño y suma; también se admiten las capturas `.npz` anteriores, que no contienen esa verificación. La reproducción conserva el orden de las muestras y puede ejecutarse una vez, en bucle o detenerse. Si el procesamiento tarda más que el tiempo real, la reproducción puede retrasarse sin saltar muestras.

La ventana inicial ofrece **Abrir demodulador sin SDR**, separado de los equipos USB. En ese modo elegí el demodulador y luego **Rec/Play → Reproducir Archivo** (o reproducción en bucle). La frecuencia central y la tasa se toman de la captura al abrirla. La grabación queda deshabilitada porque no hay recepción en vivo; para grabar, seleccioná una SDR real.

Durante la grabación se pausan los gráficos y se bloquean los cambios de modo, frecuencia y tasa para priorizar la recepción. Si el SDR informa un corte, falla la escritura o se llena la cola de disco, la captura se descarta y la interfaz muestra el error. Algunos controladores no informan todas las pérdidas que puedan ocurrir antes de entregar un bloque a la aplicación; la integridad del archivo verifica las muestras recibidas por el programa, no las que el hardware haya perdido previamente.

## Instalación

El proyecto está orientado a Linux con `apt` y requiere Python 3.10 o posterior. Para usar un SDR, también hacen falta sus bibliotecas y permisos de acceso USB.

```bash
cd ~
git clone https://github.com/alandanastasio/demodulador.git
cd demodulador
bash setup.sh
```

`setup.sh` instala paquetes del sistema, instala `uv` si falta, compila `libhackrf`, descarga imágenes para bladeRF y UHD, configura reglas USB y crea `.venv` con las dependencias de Python. Ofrece opcionalmente actualizar el firmware de bladeRF. El script presupone que el repositorio está en `~/demodulador` y recrea `.venv`; revisalo antes de ejecutarlo si ya tenés un entorno preparado.

Con las dependencias instaladas:

```bash
uv run python demodulador/startup_window.py
```

Elegí un equipo detectado o la opción sin SDR en la ventana inicial. En la ventana principal, seleccioná un modo desde **Demodulación**. Con un SDR, ajustá frecuencia y ganancia según corresponda. Para grabar o reproducir I/Q, usá **Rec/Play**.

## Estructura del proyecto

- `demodulador/hardware/`: controladores SDR y contrato común de recepción.
- `demodulador/dsp/demoduladores/`: analizador y procesadores de cada modo.
- `demodulador/main.py`, `ui_builder.py` y `startup_window.py`: flujo de recepción e interfaz gráfica.
- `demodulador/research/`: notebooks y experimentos de desarrollo.
- `demodulador/tests/`: pruebas automáticas de DSP, controladores y componentes de interfaz.

Para ejecutar las pruebas con las dependencias de desarrollo instaladas:

```bash
uv run pytest demodulador/tests -q
```
