"""Comunicação BLE (Nordic UART) entre o PC e o ESP32 do robô.

Uso:
    robo = BLERobo("CuboRobo")   # procura e conecta
    robo.enviar("1C")            # só retorna quando o ESP32 responde "OK"
    robo.fechar()

Requer: pip install bleak
"""
import asyncio
import threading

from bleak import BleakClient, BleakScanner

RX_UUID = "6E400002-B5A3-F393-E0A9-E50E24DCCA9E"   # PC -> ESP32 (write)
TX_UUID = "6E400003-B5A3-F393-E0A9-E50E24DCCA9E"   # ESP32 -> PC (notify)


class BLERobo:
    def __init__(self, nome="CuboRobo", timeout_scan=10):
        # o bleak é assíncrono; roda num event loop próprio em outra thread
        # para o main.py (síncrono) poder chamar enviar() normalmente
        self.loop = asyncio.new_event_loop()
        threading.Thread(target=self.loop.run_forever, daemon=True).start()
        self.ok = threading.Event()
        self.client = None
        self._run(self._conectar(nome, timeout_scan), timeout=timeout_scan + 20)

    def _run(self, coro, timeout=None):
        return asyncio.run_coroutine_threadsafe(coro, self.loop).result(timeout)

    async def _conectar(self, nome, timeout_scan):
        print(f"Procurando '{nome}' por BLE...")
        dev = await BleakScanner.find_device_by_name(nome, timeout=timeout_scan)
        if dev is None:
            raise RuntimeError(f"ESP32 '{nome}' não encontrado. Está ligado e anunciando?")
        self.client = BleakClient(dev)
        await self.client.connect()
        await self.client.start_notify(TX_UUID, lambda _, data: self.ok.set())
        print(f"Conectado a '{nome}'.")

    def enviar(self, cmd, timeout=5):
        """Envia um comando de 2 caracteres e espera a confirmação do ESP32."""
        self.ok.clear()
        self._run(self.client.write_gatt_char(RX_UUID, (cmd + "\n").encode(), response=True),
                  timeout=timeout)
        if not self.ok.wait(timeout):
            raise TimeoutError(f"ESP32 não confirmou o comando '{cmd}'")

    def fechar(self):
        if self.client is not None:
            try:
                self._run(self.client.disconnect(), timeout=5)
            except Exception:
                pass
        self.loop.call_soon_threadsafe(self.loop.stop)
