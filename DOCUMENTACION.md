# Documentación técnica del demodulador SDR

Este documento reúne el funcionamiento, el alcance y la evidencia de validación de la aplicación.

## Arquitectura del demodulador

### Recorrido de las muestras

- **Recepción en vivo:** SDR real → callback IQ → entrada de [`MainWindow`](demodulador/main.py).
- **Reproducción:** captura IQ → [`PlaybackManager`](demodulador/playback_manager.py) → entrada de [`MainWindow`](demodulador/main.py).

La entrada dirige las muestras a [`IQRecorder`](demodulador/iq_capture.py) durante la grabación o al demodulador activo durante el análisis. Los resultados llegan a los gráficos y métricas mediante [`SignalEmitter`](demodulador/main.py); el audio se entrega a [`AudioManager`](demodulador/audio_manager.py) a 48 kHz.

La grabación toma prioridad en [`procesar_muestras_iq`](demodulador/main.py): mientras está activa, cada bloque recibido va al escritor y no pasa por el DSP ni por los gráficos. Así se evita que una visualización pesada retrase la entrega de bloques al archivo. La ruta de reproducción entrega las muestras al mismo método de entrada que usa el SDR, sin encender el receptor.

### Piezas principales

| Componente | Responsabilidad |
| --- | --- |
| [`startup_window.py`](demodulador/startup_window.py) | Busca equipos, permite abrir el modo sin SDR y crea la ventana principal. |
| [`hardware/`](demodulador/hardware) | Adapta cada SDR al contrato de recepción, configuración y cierre definido en [`sdr_base.py`](demodulador/hardware/sdr_base.py). [`MockSDRHandler`](demodulador/hardware/mock_sdr_handler.py) permite reproducir sin equipo. |
| [`hardware/stream_transition.py`](demodulador/hardware/stream_transition.py) | Pausa y reanuda la recepción de los SDR físicos mediante `@receiver_transition` durante cambios de configuración anidados. |
| [`main.py`](demodulador/main.py) | Conserva el estado de la interfaz, selecciona el demodulador y conecta recepción, resultados, audio y controles. |
| [`dsp/demoduladores/`](demodulador/dsp/demoduladores) | Implementa el analizador y los modos FM, WiFi, BLE, LoRa y LTE. |
| [`dsp/stream_reset.py`](demodulador/dsp/stream_reset.py) | Descarta estado DSP anterior después de un corte o cambio de flujo. |
| [`dsp/rds.py`](demodulador/dsp/rds.py) y [`dsp/streaming_resampler.py`](demodulador/dsp/streaming_resampler.py) | Decodificador RDS de 57 kHz (PLL, síndrome CRC de 10 bits, grupos 0A/0B y 2A/2B) y remuestreador polifásico con preservación de estado entre bloques. |
| [`playback_manager.py`](demodulador/playback_manager.py) | Coordina grabación, reproducción y restauración de la recepción en vivo. |
| [`iq_capture.py`](demodulador/iq_capture.py) | Escribe y verifica los archivos IQ y sus metadatos. |
| [`iq_ring_buffer.py`](demodulador/iq_ring_buffer.py) | Buffer circular sincronizado (`IQRingBuffer`) que conserva el historial de muestras del espectrograma para recortes retrospectivos. |
| [`audio_manager.py`](demodulador/audio_manager.py) | Entrega audio estéreo a la salida de sonido y controla pausa, cola y *underruns*. |
| [`ui_builder.py`](demodulador/ui_builder.py) y [`plot_renderer.py`](demodulador/plot_renderer.py) | Construyen los paneles y actualizan gráficos y métricas. |
| [`trace_manager.py`](demodulador/trace_manager.py) y [`marker_manager.py`](demodulador/marker_manager.py) | Administran hasta 4 trazas (*Clear & Write*, *Max Hold*, *Min Hold*, *Average*) y hasta 6 marcadores (*Normal*, *Delta*, *Peak Search*, *Next Peak*, *Track Peak*) tanto en RF como en el espectro MPX FM. |
| [`btle_history.py`](demodulador/btle_history.py) y [`btle_frames.py`](demodulador/dsp/demoduladores/btle_frames.py) | Decodifican anuncios BLE *legacy* (PDUs, estructuras AD, *dewhitening* y CRC-24), deduplican direcciones y mantienen el historial de paquetes. |

### Contrato DSP

Cada modo implementa `configurar(sample_rate, fft_size, ...)` y `procesar(muestras_iq)` mediante [`DemoduladorBase`](demodulador/dsp/demoduladores/base.py). `procesar` puede devolver `None` mientras acumula muestras o espera a un trabajador. Cuando hay resultado, [`MainWindow`](demodulador/main.py) lo publica mediante una señal Qt; la interfaz actualiza los gráficos en su hilo. Algunos modos procesan bloques en segundo plano y **FM: escucha** entrega audio mediante un callback de resultados.

La frecuencia central y la tasa de muestreo forman parte del estado de la sesión. Al abrir una captura, se toman de sus metadatos para crear un procesador nuevo. Al terminar, se restablecen los valores y el procesador de la recepción en vivo. Un corte de IQ reinicia el estado DSP. Al resintonizar o cambiar la tasa de muestreo, todos los modos reinician su flujo mediante [`reset_demodulator_stream`](demodulador/dsp/stream_reset.py), descartando muestras y trabajadores en vuelo pero conservando la configuración del modo (incluyendo el canal BLE y el estado de barrido LTE Uplink).

### Límites de continuidad

El archivo de grabación conserva, en orden, todos los bloques entregados a la aplicación que el escritor aceptó. Una captura válida exige escritura completa, tamaño coherente y SHA-256 correcto. Ninguno de esos controles puede reconstruir muestras que el controlador o el USB hayan perdido antes del callback. Los modos de análisis en vivo tienen sus propias políticas de carga; por ejemplo, la cola de audio puede descartar bloques antiguos si se atrasa. Estos límites se detallan en [Grabación y reproducción IQ](#grabación-y-reproducción-iq).

---

## Modos de análisis

Esta sección delimita lo implementado en la interfaz actual. Todos los modos reciben IQ complejo de una SDR o de una captura; la disponibilidad en vivo depende de que el equipo admita la tasa solicitada.

| Modo | Resultado principal | Alcance |
| --- | --- | --- |
| Analizador de espectro | Espectro, espectrograma, trazas y marcadores | Visualización de RF sin demodular un protocolo. |
| FM: medición | Multiplex FM, piloto, canales L/R y RDS | Prioriza la resolución del espectro multiplex; no reproduce audio en vivo. |
| FM: escucha | Las mismas vistas FM y audio estéreo en vivo | Procesa bloques para mantener la reproducción de los canales elegidos. |
| WiFi 802.11a/g | Sincronización OFDM, SIGNAL, constelación y EVM | Análisis físico de tramas *legacy*; no ofrece una pila WiFi ni tráfico IP. |
| BLE | Preámbulo y métricas LE 1M/2M; anuncios *legacy* LE 1M | Los anuncios mostrados se validan con CRC; LE 2M se usa para métricas físicas. |
| LoRa | Encabezado, payload físico, CRC cuando existe y símbolos | Análisis de tramas LoRa de la capa física. |
| LTE downlink/uplink | Sincronización, métricas, constelaciones y recursos | Análisis de señales radio; no es un módem LTE completo. |

### Analizador de espectro

[`SpectrumAnalyzer`](demodulador/dsp/demoduladores/sa.py) calcula la potencia espectral del IQ con tamaños de FFT de 1024 a 65536 puntos, indicación de RBW y ventanas seleccionables (*Rectangular*, *Hanning*, *Hamming*, *Blackman*, *Blackman-Harris* y *Flat Top*). La interfaz agrega espectrograma, hasta 4 trazas simultáneas (*Clear & Write*, *Max Hold*, *Min Hold*, *Average*) y hasta 6 marcadores (*Normal*, *Delta*, *Peak Search*, *Next Peak* y *Track Peak*).

### FM: medición y FM: escucha

Los dos modos comparten la visualización del espectro RF, el espectro multiplex, la desviación temporal y los canales izquierdo y derecho. [`wbfm.py`](demodulador/dsp/demoduladores/wbfm.py) procesa ventanas de medición y presenta el espectro multiplex con mayor resolución. [`wbfm_audio.py`](demodulador/dsp/demoduladores/wbfm_audio.py) procesa bloques más cortos y entrega estéreo a [`AudioManager`](demodulador/audio_manager.py), que genera la salida a 48 kHz con selección de desénfasis (75 µs / 50 µs). Si no detecta un piloto estéreo utilizable, la ruta de escucha entrega mono a ambos canales. El panel informa desviación pico máxima, mínima y RMS (kHz), offset DC, nivel de piloto de 19 kHz (dB), enganche del PLL de 38 kHz y porcentaje de *stereo blend*, además de admitir marcadores sobre el espectro multiplex MPX.

Ambos usan [`RDSDecoder`](demodulador/dsp/rds.py) sobre la subportadora multiplex de 57 kHz. El panel muestra PI, PS, el código PTY y RadioText al reunir grupos válidos 0A/0B o 2A/2B, decodificando los caracteres mediante el repertorio RDS básico oficial (IEC 62106 Anexo E, tabla EBU G0). Un piloto de 19 kHz detectable no garantiza que la emisora transmita RDS; el panel permanece en espera si la estación no transmite esos grupos o la señal no alcanza para validarlos.

### WiFi 802.11a/g

[`DemoduladorWiFiAG`](demodulador/dsp/demoduladores/wifi_ag.py) detecta el preámbulo, ajusta tiempo y desplazamiento de frecuencia con las secuencias STS/LTS, estima el canal y decodifica el campo `SIGNAL`. Muestra métricas de CFO (kHz), SNR (dB), EVM RMS (% y dB), constelación ecualizada y los campos decodificados del símbolo `SIGNAL` (esquema de modulación, tasa en Mbps y longitud de PSDU). La ruta está centrada en tramas *legacy* 802.11a/g; una ráfaga HT *mixed* se descarta y el detector continúa buscando la siguiente trama *legacy*.

### Bluetooth Low Energy

[`DemoduladorBTLE`](demodulador/dsp/demoduladores/btle.py) busca ráfagas GFSK y sincroniza el preámbulo para LE 1M y LE 2M. En LE 1M, [`btle_frames.py`](demodulador/dsp/demoduladores/btle_frames.py) interpreta anuncios *legacy* de los canales primarios, aplica *dewhitening* y verifica CRC-24 antes de publicarlos. Calcula las métricas de desviación de frecuencia GFSK (`df_avg`, `f1_avg`, `f2_avg`, relación `f2/f1`, desviación RMS y CFO) e identifica automáticamente el canal primario de advertising (37, 38 o 39) a partir de la frecuencia central para inicializar el LFSR de *dewhitening*. La interfaz conserva un historial deduplicado de paquetes y una vista de estructuras AD legibles. La vista LE 2M se limita a métricas de la capa física.

### LoRa

[`DemoduladorLoRa`](demodulador/dsp/demoduladores/lora.py) administra el flujo de muestras y entrega ventanas a [`lora_core.py`](demodulador/dsp/demoduladores/lora_core.py). El núcleo busca preámbulo y *sync word*, corrige desplazamientos de tiempo y frecuencia, y reconstruye el encabezado explícito y la carga útil. Reporta CFO, SNR, RSSI, *Spreading Factor* (SF7 a SF12), *Coding Rate* (CR), *Sync Word*, longitud de payload y validación del CRC de encabezado y de carga útil. La interfaz permite BW de 125, 250 o 500 kHz y SF7 a SF12. Los bytes reconstruidos corresponden directamente a la trama de la capa física.

### LTE downlink y uplink

[`lte_downlink.py`](demodulador/dsp/demoduladores/lte_downlink.py) analiza sincronización PSS/SSS, estimación de canal mediante CRS, PBCH y otros recursos downlink para las vistas de constelación, métricas y grilla. [`lte_uplink.py`](demodulador/dsp/demoduladores/lte_uplink.py) identifica automáticamente la banda FDD 3GPP y el par de frecuencia Uplink a partir de la portadora Downlink, operando en dos etapas (`SEARCHING_DL` → `WAITING_UL`): primero adquiere el `N_ID_cell` mediante PSS/SSS en Downlink y luego conmuta automáticamente la sintonía del SDR a la frecuencia Uplink conservando la `Cell ID` para generar las secuencias DMRS y demodular PUSCH. En Downlink se detectan ciegamente 1, 2 o 4 puertos de antena TX por máscara CRC de PBCH (con combinación Alamouti SFBC), se decodifica el MIB (`N_RB_DL`, `SFN`, PHICH) y se reportan CFO, SNR, RSRP, RSRQ, RSSI y EVM por canal (PBCH, PDCCH, PDSCH / DMRS, PUSCH). Estas vistas sirven para inspección física; no reconstruyen una sesión de datos LTE de extremo a extremo.

### Tasas en recepción en vivo

La interfaz solicita 2,4 Ms/s para FM con RTL-SDR y 3 Ms/s para FM con otros equipos; LoRa usa 2 Ms/s, y WiFi y BLE solicitan 20 Ms/s. LTE selecciona la tasa según el ancho de banda y las capacidades del equipo. Al reproducir un archivo, el procesador toma la tasa de los metadatos de esa captura. Elegir un modo no hace que una SDR admita tasas fuera de sus capacidades.

### Compatibilidad por SDR y controles de ganancia

| Equipo SDR | Tasa máx. y modos en vivo | Controles de ganancia en la interfaz |
| --- | --- | --- |
| HackRF One | Hasta 20 Ms/s — Todos los modos (SA, FM, LoRa, WiFi, BLE, LTE). | Etapas independientes `LNA` (0–40 dB), `VGA` (0–62 dB) y amplificador RF `AMP`. |
| RTL-SDR | Hasta ~2,4–3,2 Ms/s — SA, FM (2,4 Ms/s), LoRa y LTE 1,4 MHz. No admite WiFi ni BLE en vivo. | Ganancia única de sintonizador (dB) y control automático `AGC`. |
| Nuand bladeRF x40 | Hasta 20+ Ms/s — Todos los modos (SA, FM, LoRa, WiFi, BLE, LTE). | Etapas independientes `LNA`, `RX VGA1` y `RX VGA2`. |
| Ettus USRP B200 | Hasta 20+ Ms/s — Todos los modos (SA, FM, LoRa, WiFi, BLE, LTE). | Ganancia global RX gestionada mediante UHD. |

---

## Grabación y reproducción IQ

El menú **Rec/Play** guarda muestras complejas tal como llegan al callback de la SDR y permite volver a analizarlas, incluso en una computadora sin receptor conectado.

### Grabar

1. Abrí la aplicación con una SDR real y seleccioná frecuencia y tasa de muestreo.
2. Elegí **Rec/Play → Iniciar Grabación**. Durante la captura se pausan los gráficos y se bloquean los controles que cambiarían la señal.
3. Detené la grabación y esperá el mensaje de guardado. El archivo aparece en el directorio desde el que ejecutaste el programa.

Cada captura nueva tiene dos archivos con el mismo nombre base:

- `muestras_iq_*.iq`: secuencia binaria de muestras IQ, sin encabezado ni saltos entre bloques.
- `muestras_iq_*.json`: descripción del formato y comprobación de integridad.

Ejemplo de metadatos:

```json
{
  "format": "demodulador-iq-v1",
  "dtype": "<c16",
  "sample_count": 11567104,
  "block_count": 353,
  "sample_rate": 2400000.0,
  "center_freq": 99900000.0,
  "sha256": "..."
}
```

`dtype` indica cómo interpretar los bytes; el ejemplo `<c16` representa muestras complejas de 16 bytes. `sample_count` permite calcular la duración como `sample_count / sample_rate`. `block_count` cuenta los bloques recibidos, aunque el archivo no conserva sus tamaños individuales. Guardá siempre el `.iq` junto con su `.json`.

[`IQRecorder`](demodulador/iq_capture.py) copia cada bloque antes de encolarlo, lo escribe en orden desde un hilo separado y calcula SHA-256 mientras escribe. Si la cola de disco supera su límite, ocurre un error de escritura o el SDR informa una discontinuidad, la captura se invalida y no queda un archivo final que parezca completo. El valor de SHA-256 verifica el archivo guardado; no detecta pérdidas previas al callback de recepción.

### Recorte retrospectivo desde el espectrograma y exportación PNG

Además de la grabación secuencial desde el menú **Rec/Play**, el panel del espectrograma permite capturar señales que acaban de aparecer en pantalla. Mientras la cascada está activa, [`IQRingBuffer`](demodulador/iq_ring_buffer.py) conserva en memoria las muestras complejas correspondientes a la ventana temporal visible en el espectrograma.

Al pulsar **✂️ Recortar Señal**, la interfaz pausa la cascada y habilita una región horizontal sobre el eje de tiempo. Al confirmar el recorte, se extraen del buffer circular las muestras del intervalo seleccionado y se guardan en un archivo `.npz` (con `raw_iq`, `center_freq` y `sample_rate`), que puede volver a abrirse desde **Rec/Play → Reproducir Archivo**. El panel incluye también el botón **Guardar Espectrograma** para exportar la vista actual de la cascada como imagen `.png`.

### Reproducir

1. En la ventana inicial elegí una SDR o **Abrir demodulador sin SDR**.
2. Seleccioná el modo de análisis antes de abrir la captura.
3. Usá **Rec/Play → Reproducir Archivo** o la opción en bucle, y elegí un `.iq` o `.npz`.

Al abrir un `.iq`, [`load_capture`](demodulador/iq_capture.py) comprueba que el tamaño corresponda al `dtype` y a `sample_count`, y verifica SHA-256. Los `.npz` antiguos también se leen si contienen `raw_iq`, `center_freq` y `sample_rate`, pero no tienen esa comprobación de integridad.

Durante la reproducción se detiene la SDR en vivo, si la había. El reproductor entrega bloques en orden y usa los metadatos de la captura para configurar el procesador. Si la interfaz se retrasa, la reproducción se atrasa y recupera bloques pendientes sin saltar índices. En bucle se entrega primero el final del archivo, se marca el corte y después empieza la siguiente vuelta. Al detenerla, vuelven la frecuencia, la tasa y el procesador de la sesión en vivo.

El modo sin SDR usa [`MockSDRHandler`](demodulador/hardware/mock_sdr_handler.py): permite analizar archivos, pero no recibir ni grabar señales nuevas. La reproducción no permite resintonizar dentro del ancho de banda de una captura desde el control de frecuencia; ese control muestra la frecuencia central guardada.

### Qué significa una captura íntegra

Una captura válida prueba que los bytes escritos coinciden con los recibidos por el programa y con los metadatos. La continuidad física anterior al callback depende del controlador y del enlace USB. Para un análisis reproducible, conservá ambos archivos, la versión del código, el modo seleccionado y cualquier ajuste relevante de ganancia o antena que no figure en el `.json`.

---

## Validación y evidencia

Esta sección separa las pruebas automatizadas, las capturas reales y lo que todavía depende de una comprobación con hardware. Una prueba sintética verifica una ruta concreta del código; no demuestra por sí sola recepción fiable de todas las señales reales.

### Ejecutar las pruebas

Con el entorno del proyecto instalado, desde la raíz del repositorio:

```bash
QT_QPA_PLATFORM=offscreen PYTHONPATH=demodulador .venv/bin/python -m pytest demodulador/tests -q
```

`QT_QPA_PLATFORM=offscreen` permite probar componentes de PyQt sin abrir ventanas (con el entorno activo también basta ejecutar `pytest -q` o `uv run pytest -q` desde la raíz gracias a la configuración en [`pyproject.toml`](pyproject.toml)). Para una prueba puntual, reemplazá `demodulador/tests` por el archivo correspondiente. La ejecución local del 28 de octubre de 2026 terminó con **133 pruebas aprobadas**; ese número es una referencia fechada, no un valor fijo del proyecto.

| Área | Pruebas de referencia | Qué comprueban |
| --- | --- | --- |
| Espectro y trazas | [`test_dsp.py`](demodulador/tests/test_dsp.py) | Pico espectral, promedio y retención de máximos. |
| FM y audio | [`test_wbfm_audio.py`](demodulador/tests/test_wbfm_audio.py), [`test_audio_manager.py`](demodulador/tests/test_audio_manager.py) | Separación estéreo, mono sin piloto, continuidad de bloques, pausa y cola de audio. |
| RDS | [`test_rds.py`](demodulador/tests/test_rds.py), [`test_offline_startup.py`](demodulador/tests/test_offline_startup.py) | CRC de grupos, PS/RadioText, tabla de caracteres EBU G0, versiones A/B, recorrido IQ en ambos modos FM y panel de interfaz. |
| WiFi | [`test_wifi_ag.py`](demodulador/tests/test_wifi_ag.py) | Sincronización, CFO, SIGNAL, EVM, tramas sintéticas y límites entre callbacks. |
| BLE | [`test_btle.py`](demodulador/tests/test_btle.py), [`test_btle_frames.py`](demodulador/tests/test_btle_frames.py) | Preámbulo LE 1M/2M, anuncios *legacy*, *dewhitening*, CRC y rechazo de señales falsas. |
| LoRa | [`test_lora_core.py`](demodulador/tests/test_lora_core.py), [`test_lora_stream.py`](demodulador/tests/test_lora_stream.py) | Decodificación física, CRC, flujo continuo y un paquete RF real recortado. |
| Capturas y reproducción | [`test_iq_capture.py`](demodulador/tests/test_iq_capture.py), [`test_playback_manager.py`](demodulador/tests/test_playback_manager.py) | Orden, muestras finales, SHA-256, errores, bucle y restauración de la sesión. |
| Hardware y cortes | [`test_hackrf_handler.py`](demodulador/tests/test_hackrf_handler.py), [`test_rtlsdr_handler.py`](demodulador/tests/test_rtlsdr_handler.py), [`test_stream_reset.py`](demodulador/tests/test_stream_reset.py) | Transiciones y fallas simuladas; no sustituye una prueba USB con el equipo real. |

Las rutas LTE tienen pruebas de integración del reinicio de flujo y publicación de resultados pendientes, pero todavía no tienen una batería equivalente de regresiones de protocolo downlink/uplink con capturas de referencia. Esa diferencia de cobertura importa al interpretar cambios en esos modos.

### Evidencia con capturas reales

El fixture versionado [`lora_sf8_500k.npy`](demodulador/tests/fixtures/lora_sf8_500k.npy) es un recorte IQ de un paquete RF real. La prueba verifica payload `INTI`, *sync word* `0x34`, encabezado y CRC físico válido. El archivo de prueba documenta el origen y el recorte de muestras.

También se examinó una captura local FM del 20 de octubre de 2026: 4,82 s, 2,4 Ms/s, frecuencia central 99,9 MHz. A 99,9 MHz no se recuperaron grupos RDS válidos. La misma banda contenía una emisora a 100,7 MHz; al desplazar 800 kHz el IQ para analizar esa señal, **FM: medición** y **FM: escucha** recuperaron PI `1007` y PS `BLUE FM` con grupos válidos. El archivo completo no está incluido en el repositorio, por lo que esta observación es evidencia manual y no una prueba automática distribuida. La interfaz de reproducción muestra la frecuencia central guardada y no ofrece ese desplazamiento de sintonía dentro del archivo.

### Alcance de las conclusiones

- El formato `.iq` verifica integridad de los bytes recibidos por la aplicación; no certifica que el SDR no haya perdido muestras antes del callback.
- Las pruebas de controladores usan dispositivos simulados. Errores USB, antenas, ganancia y estabilidad térmica requieren comprobación con hardware.
- Los resultados de DSP dependen de nivel de señal, interferencia, offset de frecuencia y parámetros elegidos. La ausencia de una trama válida en una captura corta no prueba que la estación nunca la transmita.
- Para añadir una regresión real, conservá una captura pequeña con origen, frecuencia, tasa, ajuste de canal y resultado esperado. Incluí solo los datos necesarios para repetir la prueba.

---

## Decisiones y trabajo pendiente

Este registro explica elecciones visibles en el código y se actualiza cuando cambia una decisión.

### Decisiones vigentes

- **Mantener la aplicación como receptor y analizador.** El HackRF se usa para recepción; el modo LoRa analiza la capa física.
- **Conservar dos modos FM.** **FM: medición** prioriza el análisis del multiplex; **FM: escucha** mantiene la reproducción estéreo en vivo. Ambos muestran RDS.
- **Guardar IQ secuencial con metadatos separados.** El archivo binario conserva el `dtype` original y el orden de muestras; el JSON permite comprobar tamaño, frecuencia, tasa y SHA-256.
- **Admitir reproducción sin SDR.** [`MockSDRHandler`](demodulador/hardware/mock_sdr_handler.py) inicia la misma interfaz sin recepción USB. El reproductor usa la tasa y frecuencia de la captura.
- **Publicar RDS solo tras grupos válidos.** El decodificador comprueba los cuatro bloques de cada grupo antes de actualizar PI, PS o RadioText.
- **Reiniciar el estado DSP al cortar o reemplazar el flujo IQ.** Se descartan muestras y resultados antiguos para que el panel no mezcle sesiones, tanto en cortes de flujo como al resintonizar o cambiar la tasa de muestreo en cualquier modo.

### Límites y siguientes comprobaciones

- Incorporar capturas pequeñas y pruebas de regresión específicas para LTE downlink y uplink.
- Repetir las pruebas de cambios de modo y sintonía con cada SDR física; las pruebas automáticas de controladores usan dobles de prueba.
- Documentar ajustes de antena, ganancia y entorno junto a futuras capturas RF de referencia; esos datos no se guardan en el JSON actual.

---

## Resolución de problemas frecuentes

- **El SDR no aparece en la ventana de inicio:** verificá que las reglas `udev` estén instaladas en `/etc/udev/rules.d/` y recargadas (`sudo udevadm control --reload-rules && sudo udevadm trigger`), y desconectá/reconectá el equipo USB.
- **Error de FPGA o imágenes al abrir bladeRF o USRP B200:** bladeRF x40 requiere el bitstream `/usr/share/Nuand/bladeRF/hostedx40.rbf` y USRP B200 requiere las imágenes descargadas mediante `uhd_images_downloader` (ambos pasos están automatizados en [`setup.sh`](setup.sh)).
- **Cortes de audio (*underruns*) en FM: escucha:** reducí el tamaño de FFT o desactivá temporalmente el espectrograma si la CPU se satura; [`AudioManager`](demodulador/audio_manager.py) reporta los *underruns* y descarta bloques antiguos si la cola supera su latencia máxima.
- **Grabación descartada automáticamente:** durante la grabación, [`IQRecorder`](demodulador/iq_capture.py) invalida la captura si el controlador SDR informa pérdida de continuidad o si la cola de escritura en disco se llena. Usá un disco local rápido y evitá concentradores USB compartidos a 20 Ms/s.
