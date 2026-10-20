# Demodulador SDR

Aplicación de recepción y análisis de señales de radio con interfaz gráfica en PyQt6. Toma muestras I/Q de un SDR, muestra el espectro y el espectrograma, y permite seleccionar distintos modos de demodulación. También puede grabar muestras y volver a analizarlas desde la interfaz.

## Funciones disponibles

| Modo | Funciones implementadas |
| --- | --- |
| Analizador de espectro | Espectro interactivo, espectrograma, marcadores, trazas y controles de visualización. |
| FM comercial (WBFM) | Medición de la señal multiplex, piloto estéreo y canales de audio; modo separado de reproducción de audio estéreo en vivo. |
| WiFi 802.11a/g | Detección y sincronización de tramas OFDM, estimación de desplazamiento de frecuencia y canal, constelación, EVM y decodificación del campo SIGNAL. |
| Bluetooth Low Energy | Detección y métricas de ráfagas LE 1M y LE 2M; en LE 1M, decodificación de anuncios *legacy* con verificación de CRC e historial de paquetes. |
| LoRa | Detección de preámbulo y decodificación de tramas de la capa física, con encabezado, carga útil, CRC cuando está presente, símbolos y *sync word*. La interfaz ofrece BW de 125, 250 y 500 kHz y SF7 a SF12. |
| LTE | Análisis de señales downlink y uplink, con sincronización, constelaciones, métricas y visualizaciones de recursos. Las opciones de ancho de banda dependen de la tasa de muestreo que admita el SDR. |

El proyecto es un receptor y analizador. El controlador de HackRF de esta aplicación trabaja en recepción. El modo LoRa analiza tramas de la capa física. El elemento «Custom FM» figura en el menú, pero todavía no tiene una acción de demodulación conectada.

## Hardware y capturas

La ventana inicial detecta estos SDR mediante sus controladores correspondientes:

| Equipo | Integración |
| --- | --- |
| HackRF One | `python-hackrf` / `libhackrf` |
| RTL-SDR | `pyrtlsdr` / `librtlsdr` |
| Nuand bladeRF x40 | Bindings de `libbladeRF` |
| Ettus USRP B200 | UHD |

La grabación desde **Rec/Play** guarda un archivo `muestras_iq_*.npz` en el directorio desde el que se ejecutó el programa. Contiene `raw_iq`, `center_freq` y `sample_rate`. El mismo menú permite reproducirlo una vez, en bucle o detener la reproducción. Las muestras se acumulan en memoria durante la grabación y se escriben al detenerla; conviene vigilar la RAM en capturas largas. La interfaz de reproducción se abre después de seleccionar e inicializar un SDR.

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

Elegí un equipo detectado en la ventana inicial. En la ventana principal, ajustá frecuencia y ganancia, y seleccioná un modo desde **Demodulación**. Para grabar o reproducir I/Q, usá **Rec/Play**.

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
