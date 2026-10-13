"""Decodificación de paquetes BLE legacy en canales de advertising."""

from uuid import UUID

import numpy as np


ADVERTISING_ACCESS_ADDRESS = 0x8E89BED6
ADVERTISING_ACCESS_BITS = np.unpackbits(
    np.frombuffer(ADVERTISING_ACCESS_ADDRESS.to_bytes(4, 'little'), dtype=np.uint8),
    bitorder='little',
)

PDU_NAMES = {
    0: 'ADV_IND',
    1: 'ADV_DIRECT_IND',
    2: 'ADV_NONCONN_IND',
    3: 'SCAN_REQ',
    4: 'SCAN_RSP',
    5: 'CONNECT_IND',
    6: 'ADV_SCAN_IND',
}
_DATA_PDUS = {0, 2, 4, 6}


def dewhiten(bits, channel):
    """Aplica la secuencia de whitening BLE a PDU y CRC, en orden de aire."""
    if channel not in (37, 38, 39):
        raise ValueError('El canal debe ser 37, 38 o 39.')
    result = np.asarray(bits, dtype=np.uint8).copy()
    lfsr = channel | 0x40
    for index in range(len(result)):
        if lfsr & 1:
            result[index] ^= 1
            lfsr ^= 0x88
        lfsr >>= 1
    return result


def crc24_ble(pdu):
    """CRC-24/BLE con CRCInit de advertising; devuelve los 3 bytes de aire."""
    init = 0x555555
    state = sum(
        int(f'{(init >> (8 * i)) & 0xff:08b}'[::-1], 2) << (8 * i)
        for i in range(3)
    )
    for octet in pdu:
        for _ in range(8):
            feedback = (state ^ octet) & 1
            octet >>= 1
            state >>= 1
            if feedback:
                state = (state | (1 << 23)) ^ 0x5a6000
    return state.to_bytes(3, 'little')


def _address(raw):
    return ':'.join(f'{octet:02X}' for octet in raw[::-1])


def _ad_fields(data):
    fields = []
    offset = 0
    while offset < len(data):
        length = data[offset]
        if length == 0:
            break
        if offset + 1 + length > len(data):
            fields.append({'type': None, 'name': 'AD incompleto',
                           'hex': data[offset:].hex(' ').upper()})
            break
        ad_type = data[offset + 1]
        value = data[offset + 2:offset + 1 + length]
        field = {'type': ad_type, 'name': f'AD 0x{ad_type:02X}',
                 'hex': value.hex(' ').upper()}
        if ad_type in (0x08, 0x09):
            field['name'] = 'Nombre corto' if ad_type == 0x08 else 'Nombre completo'
            field['text'] = value.decode('utf-8', errors='replace')
        elif ad_type == 0x01:
            field['name'] = 'Flags'
        elif ad_type == 0x0A and len(value) == 1:
            field['name'] = 'Potencia TX'
            field['text'] = f'{int.from_bytes(value, "little", signed=True)} dBm'
        elif ad_type == 0x21 and len(value) >= 16:
            field['name'] = 'Service Data (UUID 128 bits)'
            field['uuid'] = str(UUID(bytes=value[:16][::-1]))
            field['data_hex'] = value[16:].hex(' ').upper()
        fields.append(field)
        offset += length + 1
    return fields


def decode_legacy_pdu_bits(bits, channel):
    """Devuelve una trama con CRC válido, o None si no está completa o falla."""
    if len(bits) < 16 + 24:
        return None
    header = np.packbits(dewhiten(bits[:16], channel), bitorder='little').tobytes()
    pdu_type = header[0] & 0x0f
    payload_length = header[1] & 0x3f
    if (pdu_type not in PDU_NAMES or header[0] & 0x10
            or header[1] & 0xc0 or payload_length > 37):
        return None
    if pdu_type in _DATA_PDUS and payload_length < 6:
        return None
    if pdu_type in (1, 3) and payload_length != 12:
        return None
    if pdu_type == 5 and payload_length != 34:
        return None
    total_bits = (2 + payload_length + 3) * 8
    if len(bits) < total_bits:
        return None
    frame = np.packbits(dewhiten(bits[:total_bits], channel), bitorder='little').tobytes()
    pdu, received_crc = frame[:-3], frame[-3:]
    if crc24_ble(pdu) != received_crc:
        return None
    payload = pdu[2:]
    result = {
        'channel': channel,
        'pdu_type': PDU_NAMES[pdu_type],
        'length': payload_length,
        'tx_address_random': bool(header[0] & 0x40),
        'rx_address_random': bool(header[0] & 0x80),
        'crc_valid': True,
        'crc_hex': received_crc.hex(' ').upper(),
        'pdu_hex': pdu.hex(' ').upper(),
        'payload_hex': payload.hex(' ').upper(),
        'ad_fields': [],
    }
    if pdu_type in _DATA_PDUS:
        result['advertiser_address'] = _address(payload[:6])
        result['ad_fields'] = _ad_fields(payload[6:])
    elif pdu_type == 1:
        result['advertiser_address'] = _address(payload[:6])
        result['target_address'] = _address(payload[6:12])
    elif pdu_type == 3:
        result['scanner_address'] = _address(payload[:6])
        result['advertiser_address'] = _address(payload[6:12])
    elif pdu_type == 5:
        result['initiator_address'] = _address(payload[:6])
        result['advertiser_address'] = _address(payload[6:12])
    return result
