#!/bin/bash

# 1. Instalar dependencias del sistema
sudo apt update
sudo apt install -y bladerf libbladerf-dev rtl-sdr librtlsdr-dev hackrf libhackrf-dev

# 2. Configurar el bitstream de la bladeRF (Para x40)
sudo mkdir -p /usr/share/Nuand/bladeRF/
sudo wget https://www.nuand.com/fpga/hostedx40-latest.rbf -O /usr/share/Nuand/bladeRF/hostedx40.rbf

# 3. Reglas udev (Para que no haga falta usar sudo para la radio)
# Esto es opcional pero muy recomendado para usuarios de Linux
sudo wget https://raw.githubusercontent.com/Nuand/bladeRF/master/host/misc/udev/88-nuand-bladerf1.rules -P /etc/sudoers.d/
sudo udevadm control --reload-rules && sudo udevadm trigger

echo "Configuración terminada. Ya podés usar uv run."