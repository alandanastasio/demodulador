"""El historial BLE conserva tramas y permite inspeccionarlas en vivo."""

import os

os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')

from PyQt6.QtWidgets import QApplication

from btle_history import BtleHistoryDialog, BtleHistoryModel


def _packet(number, pdu_type='ADV_SCAN_IND'):
    return {
        'count': number,
        'time_s': number * 0.1,
        'channel': 37,
        'pdu_type': pdu_type,
        'advertiser_address': '11:22:33:44:55:66',
        'scanner_address': 'AA:BB:CC:DD:EE:FF',
        'tx_address_random': True,
        'rx_address_random': False,
        'length': 12,
        'crc_hex': '01 02 03',
        'pdu_hex': '03 0C',
        'payload_hex': 'AA BB',
        'ad_fields': [],
    }


def test_full_history_keeps_all_packets_and_selected_details():
    app = QApplication.instance() or QApplication([])
    model = BtleHistoryModel()
    dialog = BtleHistoryDialog(model)
    try:
        model.append_packets(_packet(i) for i in range(1, 21))
        assert model.rowCount() == 20
        assert len(model.recent_lines()) == 12
        assert model.recent_lines()[0].startswith('#20')
        assert dialog.count_label.text() == 'Tramas con CRC válido: 20'

        dialog.table.selectRow(0)
        assert 'Paquete: 1' in dialog.details.toPlainText()
        assert 'AdvA: 11:22:33:44:55:66' in dialog.details.toPlainText()

        model.append_packets([_packet(1, 'SCAN_REQ')])
        assert model.rowCount() == 21
        assert 'Paquete: 1' in dialog.details.toPlainText()
        dialog.table.selectRow(20)
        assert 'Paquete: 21' in dialog.details.toPlainText()
        assert 'ScanA: AA:BB:CC:DD:EE:FF' in dialog.details.toPlainText()
    finally:
        dialog.close()
        app.processEvents()
