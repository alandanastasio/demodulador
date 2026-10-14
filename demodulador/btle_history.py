"""Historial de tramas BLE válidas para la interfaz Qt."""

from PyQt6.QtCore import QAbstractTableModel, QModelIndex, Qt
from PyQt6.QtWidgets import (
    QAbstractItemView, QCheckBox, QDialog, QHBoxLayout, QLabel,
    QPlainTextEdit, QPushButton, QSplitter, QTableView, QVBoxLayout,
)

from plot_renderer import format_btle_packet_details


class BtleHistoryModel(QAbstractTableModel):
    COLUMNS = ('N.º', 'Tiempo [s]', 'Canal', 'Tipo', 'AdvA', 'ScanA', 'Bytes')

    def __init__(self, parent=None):
        super().__init__(parent)
        self.packets = []

    def rowCount(self, parent=QModelIndex()):
        return 0 if parent.isValid() else len(self.packets)

    def columnCount(self, parent=QModelIndex()):
        return 0 if parent.isValid() else len(self.COLUMNS)

    def headerData(self, section, orientation, role=Qt.ItemDataRole.DisplayRole):
        if role == Qt.ItemDataRole.DisplayRole and orientation == Qt.Orientation.Horizontal:
            return self.COLUMNS[section]
        return None

    def data(self, index, role=Qt.ItemDataRole.DisplayRole):
        if not index.isValid() or role != Qt.ItemDataRole.DisplayRole:
            return None
        packet = self.packets[index.row()]
        return (
            packet['count'],
            f"{packet['time_s']:.6f}",
            packet['channel'],
            packet['pdu_type'],
            packet.get('advertiser_address', '—'),
            packet.get('scanner_address', '—'),
            packet['length'],
        )[index.column()]

    def append_packets(self, packets):
        """Conserva cada PDU con CRC válido, incluso al cambiar de modo/canal."""
        packets = list(packets)
        if not packets:
            return []
        first = len(self.packets)
        records = [dict(packet, count=first + offset + 1)
                   for offset, packet in enumerate(packets)]
        self.beginInsertRows(QModelIndex(), first, first + len(records) - 1)
        self.packets.extend(records)
        self.endInsertRows()
        return records

    def recent_lines(self, limit=12):
        return [
            f"#{packet['count']}  {packet['time_s']:.3f} s  "
            f"CH{packet['channel']}  {packet['pdu_type']}  "
            f"{packet.get('advertiser_address', '')}  CRC OK"
            for packet in reversed(self.packets[-limit:])
        ]


class BtleHistoryDialog(QDialog):
    """Lista completa, consultable mientras siguen llegando tramas."""

    def __init__(self, model, parent=None):
        super().__init__(parent)
        self.model = model
        self.setWindowTitle('Historial completo de tramas BLE')
        self.resize(1100, 650)
        layout = QVBoxLayout(self)

        self.count_label = QLabel()
        layout.addWidget(self.count_label)
        splitter = QSplitter(Qt.Orientation.Vertical)
        self.table = QTableView()
        self.table.setModel(model)
        self.table.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        self.table.setSelectionMode(QAbstractItemView.SelectionMode.SingleSelection)
        self.table.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        self.table.setAlternatingRowColors(True)
        self.table.setSortingEnabled(False)
        self.table.setStyleSheet("""
            QTableView {
                background-color: #202020;
                alternate-background-color: #303030;
                color: #F2F2F2;
                gridline-color: #454545;
                border: 1px solid #444;
                selection-background-color: #176B8A;
                selection-color: #FFFFFF;
            }
            QTableView::item:selected {
                background-color: #176B8A;
                color: #FFFFFF;
            }
            QTableView::item:selected:!active {
                background-color: #28566B;
                color: #FFFFFF;
            }
        """)
        self.table.verticalHeader().hide()
        self.table.horizontalHeader().setStretchLastSection(True)
        self.table.setColumnWidth(0, 65)
        self.table.setColumnWidth(1, 115)
        self.table.setColumnWidth(2, 65)
        self.table.setColumnWidth(3, 155)
        self.table.setColumnWidth(4, 175)
        self.table.setColumnWidth(5, 175)
        splitter.addWidget(self.table)

        self.details = QPlainTextEdit()
        self.details.setReadOnly(True)
        self.details.setPlaceholderText('Seleccioná una trama para ver todos sus campos.')
        self.details.setStyleSheet(
            'QPlainTextEdit { background: #101010; color: #E8E8E8; '
            'border: 1px solid #444; font-family: monospace; }')
        splitter.addWidget(self.details)
        splitter.setSizes([400, 200])
        layout.addWidget(splitter)

        bottom = QHBoxLayout()
        self.readable_checkbox = QCheckBox('Ver texto legible')
        bottom.addWidget(self.readable_checkbox)
        bottom.addStretch()
        close_button = QPushButton('Cerrar')
        close_button.clicked.connect(self.close)
        bottom.addWidget(close_button)
        layout.addLayout(bottom)

        self.table.selectionModel().currentRowChanged.connect(self._show_selected)
        self.readable_checkbox.toggled.connect(self._refresh_details)
        self.model.rowsInserted.connect(self._update_count)
        self._update_count()

    def _update_count(self, *args):
        self.count_label.setText(f'Tramas con CRC válido: {self.model.rowCount()}')

    def _show_selected(self, current, previous):
        self._refresh_details()

    def _refresh_details(self):
        row = self.table.currentIndex().row()
        if row < 0 or row >= self.model.rowCount():
            self.details.clear()
            return
        self.details.setPlainText(format_btle_packet_details(
            self.model.packets[row], self.readable_checkbox.isChecked()))

    def open_history(self, readable=False):
        self.readable_checkbox.setChecked(readable)
        if not self.isVisible() and self.model.rowCount():
            last = self.model.rowCount() - 1
            self.table.selectRow(last)
            self.table.scrollTo(self.model.index(last, 0))
        self.show()
        self.raise_()
        self.activateWindow()
