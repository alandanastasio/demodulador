# Demodulador SDR

Aplicación de recepción y análisis de señales de radio con interfaz gráfica en PyQt6. Toma muestras I/Q de un SDR, muestra el espectro y el espectrograma, y permite seleccionar distintos modos de demodulación. También puede grabar muestras y volver a analizarlas desde la interfaz.

## Funciones disponibles

| Modo | Funciones implementadas |
| --- | --- |
| Analizador de espectro | Espectro interactivo (FFT de 1024 a 65536 puntos, RBW y 6 ventanas seleccionables), espectrograma, hasta 4 trazas (*Clear & Write*, *Max Hold*, *Min Hold*, *Average*) y hasta 6 marcadores (*Normal*, *Delta*, *Peak Search*, *Next Peak*, *Track Peak*). |
| FM: medición | Medición de la señal multiplex (desviación pico máx./mín./RMS, piloto de 19 kHz, portadora de 38 kHz y canales L/R), espectro multiplex de mayor resolución con soporte de marcadores y lectura RDS. |
| FM: escucha | Gráficos y métricas FM, reproducción de audio estéreo en vivo a 48 kHz con selección de desénfasis (75 µs / 50 µs), controles para los canales izquierdo y derecho y lectura RDS. |
| WiFi 802.11a/g | Detección y sincronización de tramas OFDM legacy, estimación de CFO, SNR y canal, constelación ecualizada, EVM RMS (% y dB) y decodificación del símbolo `SIGNAL`. |
| Bluetooth Low Energy | Detección y métricas físicas GFSK (`df_avg`, `f1_avg`, `f2_avg`, `f2/f1`, desviación RMS y CFO) en LE 1M y LE 2M; en LE 1M, decodificación de anuncios *legacy* en canales primarios (37/38/39) con *dewhitening*, verificación de CRC-24 e historial deduplicado de paquetes. |
| LoRa | Detección de preámbulo y decodificación de tramas de la capa física (CFO, SNR, RSSI, encabezado explícito, carga útil, validación de CRC, símbolos y *sync word*). La interfaz ofrece BW de 125, 250 y 500 kHz y SF7 a SF12. |
| LTE | Análisis de señales downlink (PSS/SSS, `N_ID_cell`, detección ciega de 1/2/4 antenas TX por PBCH, MIB, RSRP/RSRQ/RSSI/SNR y EVM por canal) y uplink (flujo asistido DL→UL con identificación automática de banda FDD 3GPP, adquisición de `Cell ID` en DL y conmutación automática a UL para DMRS y PUSCH). |

El proyecto es un receptor y analizador. El controlador de HackRF de esta aplicación trabaja en recepción. El modo LoRa analiza tramas de la capa física.

Ambos modos FM decodifican la subportadora RDS de 57 kHz. El panel RDS muestra PI, el nombre corto de la emisora (PS), el código PTY y RadioText cuando la estación los transmite y se han recibido grupos válidos. PS y RadioText se forman a partir de varios grupos, por lo que pueden tardar unos segundos en aparecer. Se admiten los grupos 0A/0B y 2A/2B, decodificando los textos con la tabla de caracteres básica RDS (IEC 62106 Anexo E, tabla G0). La estructura de la señal y los grupos se describen en [UIT-R BS.643-3](https://www.itu.int/dms_pubrec/itu-r/rec/bs/R-REC-BS.643-3-201105-S!!PDF-E.pdf).

## Hardware y capturas

La ventana inicial detecta estos SDR mediante sus controladores correspondientes:

| Equipo | Integración | Tasa máx. y modos en vivo | Controles de ganancia |
| --- | --- | --- | --- |
| HackRF One | `python-hackrf` / `libhackrf` | Hasta 20 Ms/s — Todos los modos | `LNA` (0–40 dB), `VGA` (0–62 dB) y `AMP` |
| RTL-SDR | `pyrtlsdr` / `librtlsdr` | Hasta ~2,4–3,2 Ms/s — SA, FM (2,4 Ms/s), LoRa y LTE 1,4 MHz (no admite WiFi ni BLE en vivo) | Ganancia única (dB) y `AGC` |
| Nuand bladeRF x40 | Bindings de `libbladeRF` | Hasta 20+ Ms/s — Todos los modos | `LNA`, `RX VGA1` y `RX VGA2` |
| Ettus USRP B200 | UHD | Hasta 20+ Ms/s — Todos los modos | Ganancia global RX UHD |

La grabación desde **Rec/Play** escribe secuencialmente `muestras_iq_*.iq` en el directorio desde el que se ejecutó el programa. El archivo `.json` del mismo nombre guarda el tipo de muestra, la frecuencia central, la tasa de muestreo, el número de bloques y muestras, y una suma SHA-256; conservá ambos archivos juntos. Al reproducir una captura nueva se verifican su tamaño y suma; también se admiten las capturas `.npz` anteriores, que no contienen esa verificación. La reproducción conserva el orden de las muestras y puede ejecutarse una vez, en bucle o detenerse. Si el procesamiento tarda más que el tiempo real, la reproducción puede retrasarse sin saltar muestras.

Además, cuando el espectrograma está activo, la aplicación mantiene un buffer circular en memoria (`IQRingBuffer`) con el historial visible en la cascada. El botón **✂️ Recortar Señal** pausa el espectrograma, permite delimitar un intervalo temporal directamente sobre la cascada y guarda ese segmento como archivo `.npz` (`raw_iq`, `center_freq` y `sample_rate`). También se puede exportar la vista actual del espectrograma como imagen `.png` mediante **Guardar Espectrograma**.

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

- `demodulador/hardware/`: controladores SDR (`sdr_base.py`, HackRF, RTL-SDR, bladeRF, USRP B200, Mock) y transiciones de stream (`stream_transition.py`).
- `demodulador/dsp/`: procesadores de cada modo (`demoduladores/`), decodificador RDS (`rds.py`), remuestreador continuo (`streaming_resampler.py`) y reinicio de flujos (`stream_reset.py`).
- `demodulador/main.py`, `ui_builder.py`, `plot_renderer.py` y `startup_window.py`: ventana principal, construcción de paneles, renderizado y selección inicial de dispositivo.
- `demodulador/trace_manager.py`, `marker_manager.py` y `btle_history.py`: gestión de trazas, marcadores interactivos e historial deduplicado de anuncios BLE.
- `demodulador/playback_manager.py`, `iq_capture.py`, `iq_ring_buffer.py` y `audio_manager.py`: grabación/reproducción verificada con SHA-256, buffer circular para recortes desde el espectrograma y salida de audio estéreo a 48 kHz.
- `demodulador/research/`: notebooks y experimentos de desarrollo.
- `demodulador/tests/`: pruebas automáticas de DSP, controladores y componentes de interfaz.

Para ejecutar las pruebas con las dependencias de desarrollo instaladas:

```bash
uv run pytest -q
```

Para más detalles sobre la arquitectura interna, el contrato DSP, la validación y la resolución de problemas, consultá [DOCUMENTACION.md](DOCUMENTACION.md).


