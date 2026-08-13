#!/bin/bash

echo "📻 Iniciando setup del Demodulador SDR..."

# 1. Instalar dependencias del sistema
echo "📦 Instalando drivers y librerías base..."
sudo apt update
sudo apt install -y build-essential python3-dev curl cmake libusb-1.0-0-dev pkg-config bladerf libbladerf-dev rtl-sdr librtlsdr-dev libportaudio2 uhd-host libuhd-dev python3-uhd

# 1.5. Instalar 'uv' si no existe
if ! command -v uv &> /dev/null; then
    echo "⚙️  Instalando 'uv' (Gestor de dependencias ultra-rápido para Python)..."
    curl -LsSf https://astral.sh/uv/install.sh | sh
    export PATH="$HOME/.local/bin:$PATH"
fi

# 1.6. Compilar libhackrf desde el código fuente (requerido para python-hackrf >= 1.5.0.1)
echo "🛠️  Compilando libhackrf oficial desde código fuente..."
sudo apt remove -y hackrf libhackrf-dev &>/dev/null || true
cd /tmp
rm -rf hackrf
git clone https://github.com/greatscottgadgets/hackrf.git
cd hackrf/host
mkdir build && cd build
cmake ..
make -j$(nproc)
sudo make install
sudo ldconfig
cd ~/demodulador || cd -

# 2. Configurar el bitstream de la FPGA para BladeRF y UHD para Ettus USRP
echo "🧠 Descargando bitstream (FPGA) de la bladeRF x40..."
sudo mkdir -p /usr/share/Nuand/bladeRF/
sudo wget -q https://www.nuand.com/fpga/hostedx40-latest.rbf -O /usr/share/Nuand/bladeRF/hostedx40.rbf

echo "🧠 Descargando imágenes firmware/FPGA para Ettus USRP..."
if command -v uhd_images_downloader &> /dev/null; then
    sudo uhd_images_downloader
else
    DOWNLOADER=$(find /usr /lib /opt -type f -name "uhd_images_downloader.py" 2>/dev/null | head -n 1)
    if [ -n "$DOWNLOADER" ]; then
        sudo python3 "$DOWNLOADER"
    else
        echo "⚠️  No se encontró el descargador de imágenes de UHD en tu sistema. Es posible que debas descargarlas manualmente."
    fi
fi

# 3. Reglas udev (Para no tener que usar sudo al abrir el puerto USB)
echo "🔑 Configurando permisos USB (udev rules)..."
sudo wget -q https://raw.githubusercontent.com/Nuand/bladeRF/master/host/misc/udev/88-nuand-bladerf1.rules -P /etc/udev/rules.d/
sudo udevadm control --reload-rules && sudo udevadm trigger

# 4. Verificación y Flasheo de Firmware
echo ""
echo "--- ACTUALIZACIÓN DE FIRMWARE BLADERF ---"
echo "⚠️  Atención: Solo es necesario si la placa tira error de versión al iniciar."
read -p "¿Querés descargar y flashear el último firmware en la bladeRF ahora? (Asegurate de que esté conectada) [s/N]: " flash_fw

if [[ "$flash_fw" =~ ^[sS]$ ]]; then
    echo "⬇️ Descargando firmware oficial..."
    wget -q https://www.nuand.com/fx3/bladeRF_fw_latest.img -O /tmp/bladeRF_fw_latest.img
    
    echo "⚡ Flasheando... POR FAVOR NO DESCONECTES EL CABLE USB."
    bladeRF-cli -f /tmp/bladeRF_fw_latest.img
    
    echo "✅ Flasheo terminado. Desconectá y volvé a conectar el cable USB de la placa para que reinicie."
else
    echo "⏭️  Flasheo omitido."
fi

# 3. Crear entorno virtual automáticamente (Hardcodeado al Python del sistema operativo)
echo "📦 Creando entorno virtual e instalando librerías de Python..."
rm -f .python-version
rm -rf .venv
uv venv --python /usr/bin/python3 --system-site-packages
uv sync

echo ""
echo "🚀 ¡Setup terminado!"
echo "Para arrancar la aplicación:"
echo "    uv run python demodulador/startup_window.py"