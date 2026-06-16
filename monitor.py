"""
Monitor en vivo de las senales del nuevo setup ESP32 + ADS1256:
  4 EMG diferenciales del ADS1256 (emg0..emg3)
+ 2 ADC interno joysticks (joy_a, joy_b) — ambos miden el mismo dedo
  (indice) para redundancia. Solo se cablea una salida por joystick.

Schema del stream (debe coincidir con el firmware ESP32):
    t_micros, emg0, emg1, emg2, emg3, joy_a, joy_b

Convencion de canales EMG (ver docs/hardware_setup.md):
    emg0 = extensor del indice  (AIN0+ AIN1-)
    emg1 = flexor del indice    (AIN2+ AIN3-)
    emg2 = extensor del medio   (AIN4+ AIN5-)
    emg3 = flexor del medio     (AIN6+ AIN7-)

Visualizacion en rolling window con stats por canal. NO graba a CSV
(ese rol queda en collect.py). Si necesitas grabar datos, corre el
script de captura por separado y no este monitor.

Stream interface:
    StreamReader es la abstraccion. MockStream genera senales sinteticas
    a la fs objetivo; SerialStream parsea frames binarios del puerto
    serial cuando el firmware ESP32 esta corriendo.

Uso (mock data):
    python monitor.py
    python monitor.py --fs 1000 --window-seconds 5

Uso (firmware real):
    python monitor.py --source serial --port /dev/cu.usbmodem21101 --fs 2000
"""
from __future__ import annotations

import argparse
import struct
import time
from dataclasses import dataclass
from typing import List, Optional

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from scipy.ndimage import gaussian_filter1d
from scipy.signal import butter, iirnotch, sosfilt, sosfilt_zi, tf2sos


# ----- Schema del stream -----
# Mismo orden que el firmware ESP32 emitira por serial.
# Etiquetas semanticas (flexor/extensor de indice/medio) son convencion;
# la asignacion fisica depende del cableado (ver docs/hardware_setup.md).
CHANNELS = [
    ("emg0 (ext_ind)", "tab:blue"),
    ("emg1 (flx_ind)", "tab:red"),
    ("emg2 (ext_med)", "tab:cyan"),
    ("emg3 (flx_med)", "tab:orange"),
    ("joy_a",          "tab:green"),
    ("joy_b",          "tab:purple"),
]
N_CH = len(CHANNELS)
# Indices logicos
EMG_IDX = slice(0, 4)
JOY_IDX = slice(4, 6)


# ----- Stream readers -----

class StreamReader:
    """Interfaz comun. read() devuelve (t_s, [N_CH] float array) o None si no hay."""

    def read(self) -> Optional[tuple]:
        raise NotImplementedError

    def close(self) -> None:
        pass


# ----- Binary protocol del firmware ESP32 -----
# Frame de 27 bytes (ver sketchs/esp32/src/main.cpp):
#   [0..1]   magic 0xAA 0x55
#   [2..5]   t_micros uint32 LE
#   [6..9]   emg0     int32  LE
#   [10..13] emg1     int32  LE
#   [14..17] emg2     int32  LE
#   [18..21] emg3     int32  LE
#   [22..23] joy_a    uint16 LE
#   [24..25] joy_b    uint16 LE
#   [26]     CRC-8 (poly 0x07, init 0x00) sobre [2..25]
FRAME_SIZE = 27
MAGIC = b"\xAA\x55"
PAYLOAD_FMT = "<I4iHH"


def _crc8_table_init() -> bytes:
    """Tabla pre-computada de CRC-8 (poly 0x07)."""
    out = bytearray(256)
    for i in range(256):
        c = i
        for _ in range(8):
            c = ((c << 1) ^ 0x07) & 0xFF if (c & 0x80) else (c << 1) & 0xFF
        out[i] = c
    return bytes(out)


_CRC8_TABLE = _crc8_table_init()


def crc8(data: bytes) -> int:
    c = 0
    for b in data:
        c = _CRC8_TABLE[c ^ b]
    return c


class SerialStream(StreamReader):
    """Lee frames binarios del firmware ESP32 por puerto serial."""

    def __init__(self, port: str, baud: int = 2000000, timeout: float = 0.01):
        import serial
        self.ser = serial.Serial(port, baud, timeout=timeout)
        self.buf = bytearray()
        self.t0 = time.monotonic()
        # Stats internas
        self.frames_ok = 0
        self.frames_crc_bad = 0
        self.frames_dropped_bytes = 0

    def _find_frame(self) -> Optional[bytes]:
        """Busca un frame valido (magic + CRC ok) en self.buf y lo consume."""
        while True:
            idx = self.buf.find(MAGIC)
            if idx < 0:
                if len(self.buf) > 1024:
                    self.frames_dropped_bytes += len(self.buf) - 1
                    del self.buf[:-1]
                return None
            if idx > 0:
                self.frames_dropped_bytes += idx
                del self.buf[:idx]
            if len(self.buf) < FRAME_SIZE:
                return None
            frame = bytes(self.buf[:FRAME_SIZE])
            expected = frame[26]
            actual = crc8(frame[2:26])
            if expected == actual:
                del self.buf[:FRAME_SIZE]
                self.frames_ok += 1
                return frame
            # CRC malo: el magic match fue casual o el frame esta corrupto.
            # Avanzar un byte y reintentar la busqueda de magic.
            self.frames_crc_bad += 1
            del self.buf[0]

    def read(self) -> Optional[tuple]:
        nwait = self.ser.in_waiting
        if nwait:
            self.buf.extend(self.ser.read(nwait))
        frame = self._find_frame()
        if frame is None:
            return None
        t_us, e0, e1, e2, e3, j_a, j_b = struct.unpack(PAYLOAD_FMT, frame[2:26])
        sample = np.array([e0, e1, e2, e3, j_a, j_b], dtype=np.float32)
        t_now = time.monotonic() - self.t0
        return (t_now, sample)

    def close(self) -> None:
        try:
            self.ser.close()
        except Exception:
            pass


class MockStream(StreamReader):
    """
    Genera muestras sinteticas al formato esperado. Las senales son:
        EMG ind/med: burst gaussiano modulado por sin(0.5 Hz), con ruido.
        joy_a / joy_b: sinusoide lenta + ruido, simulando topes mecanicos.
    """

    def __init__(self, fs: float = 1000.0, seed: int = 0):
        self.fs = float(fs)
        self.dt = 1.0 / self.fs
        self.t0 = time.monotonic()
        self.n = 0
        self.rng = np.random.default_rng(seed)
        self._env_phase = self.rng.random(4) * 2 * np.pi

    def read(self) -> Optional[tuple]:
        t_now = time.monotonic() - self.t0
        expected_n = int(t_now * self.fs)
        if expected_n <= self.n:
            return None
        t = self.n * self.dt
        self.n += 1

        # EMG sintetico
        env_freq = np.array([0.5, 0.5, 0.4, 0.4])
        env = 0.4 + 0.4 * np.sin(2 * np.pi * env_freq * t + self._env_phase)
        gain = np.array([1.4, 0.9, 1.2, 1.0])
        noise_em = self.rng.standard_normal(4)
        emg = gain * env * (1.0 + 0.5 * noise_em)
        if self.rng.random() < 0.001:
            emg += self.rng.standard_normal(4) * 2.0

        # Joysticks sinteticos
        joy_freq = np.array([0.25, 0.22])
        joy_phase = np.array([0.0, np.pi / 3])
        joy_raw = 512 + 480 * np.sin(2 * np.pi * joy_freq * t + joy_phase)
        joy = np.clip(joy_raw, 0, 1023)
        joy += self.rng.standard_normal(2) * 1.5

        sample = np.concatenate([emg, joy]).astype(np.float32)
        return (t, sample)


# ----- Rolling buffer -----

@dataclass
class RollingBuffer:
    """Buffer circular para los ultimos N samples + timestamps."""
    capacity: int
    n_channels: int

    def __post_init__(self):
        self.t = np.zeros(self.capacity, dtype=np.float64)
        self.x = np.zeros((self.capacity, self.n_channels), dtype=np.float32)
        self.head = 0
        self.size = 0

    def push(self, t: float, sample: np.ndarray) -> None:
        self.t[self.head] = t
        self.x[self.head, :] = sample
        self.head = (self.head + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def view(self):
        if self.size < self.capacity:
            return self.t[: self.size].copy(), self.x[: self.size, :].copy()
        idx = (np.arange(self.capacity) + self.head) % self.capacity
        return self.t[idx], self.x[idx, :]


# ----- Stats helpers -----

def channel_stats(x: np.ndarray) -> dict:
    """Stats sobre la ventana actual. RMS y p99 utiles para senales AC
    (EMG post-bandpass); mean tiene sentido para senales DC (joysticks)."""
    if x.size == 0:
        return {"rms": 0.0, "std": 0.0, "p99": 0.0,
                "mean": 0.0, "min": 0.0, "max": 0.0}
    return {
        "rms": float(np.sqrt(np.mean(x ** 2))),
        "std": float(x.std()),
        "p99": float(np.percentile(np.abs(x), 99)),
        "mean": float(x.mean()),
        "min": float(x.min()),
        "max": float(x.max()),
    }


# ----- Monitor (matplotlib animation) -----

class Monitor:
    """Plot de N_CH traces verticales con stats overlay. Usa matplotlib animation."""

    def __init__(self, stream: StreamReader, fs: float, window_seconds: float,
                 fps: int = 30,
                 emg_bp_low: float = 70.0,
                 emg_bp_high: float = 300.0,
                 emg_bp_order: int = 4,
                 notch_base: float = 50.0,
                 notch_q: float = 30.0,
                 joy_sigma_ms: float = 50.0,
                 apply_filters: bool = True):
        self.stream = stream
        self.fs = fs
        self.window_seconds = window_seconds
        self.fps = fps
        # Guardamos los params originales para poder rebuild si fs_real
        # difiere significativamente de fs (auto-ajuste de filtros).
        self._emg_bp_low = emg_bp_low
        self._emg_bp_high = emg_bp_high
        self._emg_bp_order = emg_bp_order
        self._notch_base = notch_base
        self._notch_q = notch_q
        self._joy_sigma_ms = joy_sigma_ms
        self.buf = RollingBuffer(
            capacity=int(window_seconds * fs * 1.2),
            n_channels=N_CH,
        )

        self.apply_filters = apply_filters
        self._build_filters(fs)
        self._build_display(fs)

        # Figure: N_CH axes verticales + status text por canal
        self.fig, self.axes = plt.subplots(
            N_CH, 1, sharex=True, figsize=(11, 7.5),
        )
        self.fig.subplots_adjust(left=0.07, right=0.78, top=0.95, bottom=0.06,
                                 hspace=0.22)
        self.lines = []
        self.stat_texts = []
        for i, (ax, (name, color)) in enumerate(zip(self.axes, CHANNELS)):
            (ln,) = ax.plot(self.x_axis, self.display[:, i],
                            color=color, lw=0.9)
            self.lines.append(ln)
            ax.set_ylabel(name, fontsize=8)
            ax.set_xlim(-self.window_seconds, 0)
            ax.tick_params(labelsize=7)
            ax.grid(True, alpha=0.25)
            txt = self.fig.text(
                0.80, 1.0 - 0.06 - (i + 0.5) * (0.90 / N_CH),
                "", fontsize=8, va="center",
                family="monospace",
            )
            self.stat_texts.append(txt)
        self.axes[-1].set_xlabel("time (s)", fontsize=9)

        self.title = self.fig.suptitle(
            f"Monitor — fs_target={fs:.0f} Hz, window={window_seconds:.1f}s",
            fontsize=10,
        )

        self.last_fps_check = time.monotonic()
        self.samples_since_check = 0
        self.fs_real = 0.0
        self._fs_filter_design = fs
        self._last_rebuild = 0.0
        self._ylim_state = [None] * N_CH

    def _build_display(self, fs: float) -> None:
        """Inicializa el display buffer (fixed-size, NaN = nada que mostrar)
        y el eje X temporal estatico."""
        self.display_samples = max(2, int(self.window_seconds * fs))
        self.x_axis = np.linspace(-self.window_seconds, 0.0,
                                  self.display_samples)
        self.display = np.full((self.display_samples, N_CH), np.nan,
                               dtype=np.float32)

    def _build_filters(self, fs: float) -> None:
        """Construye / reconstruye SOS de bandpass y notch cascade."""
        emg_bp_low = self._emg_bp_low
        emg_bp_high = self._emg_bp_high
        notch_base = self._notch_base
        notch_q = self._notch_q

        if emg_bp_low <= 0 or emg_bp_high <= 0:
            self.emg_sos = None
        else:
            safe_high = min(emg_bp_high, 0.45 * fs)
            if safe_high < emg_bp_low + 5:
                self.emg_sos = None
                print(f"WARN: bandpass {emg_bp_low}-{emg_bp_high}Hz no viable a fs={fs:.0f}Hz "
                      f"(safe high = {safe_high:.1f}). Filtros EMG desactivados.")
            else:
                self.emg_sos = butter(self._emg_bp_order, [emg_bp_low, safe_high],
                                      btype="band", fs=fs, output="sos")
        if notch_base > 0:
            nyq = fs / 2
            sections = []
            f = notch_base
            while f < nyq * 0.95:
                b, a = iirnotch(f, notch_q, fs=fs)
                sections.append(tf2sos(b, a))
                f += notch_base
            self.notch_sos = np.vstack(sections) if sections else None
            if self.notch_sos is not None:
                print(f"[fs={fs:.0f}Hz] Notch cascade @ "
                      f"{[notch_base * (i+1) for i in range(len(sections))]} Hz "
                      f"(Q={notch_q})")
        else:
            self.notch_sos = None
        self.joy_sigma_samples = (self._joy_sigma_ms * fs / 1000.0
                                  if self._joy_sigma_ms > 0 else 0.0)
        self._fs_filter_design = fs

    def _maybe_rebuild_filters(self) -> None:
        """Si fs_real difiere >10% de la fs usada al disenar filtros,
        rebuild filtros + buffer + display."""
        if self.fs_real <= 0:
            return
        now = time.monotonic()
        if now - self._last_rebuild < 2.0:
            return
        ratio = self.fs_real / max(self._fs_filter_design, 1e-6)
        if ratio < 0.9 or ratio > 1.1:
            print(f"[auto] fs_real={self.fs_real:.0f}Hz difiere de "
                  f"fs_filter={self._fs_filter_design:.0f}Hz (ratio={ratio:.2f}). "
                  f"Reconstruyendo filtros y reset buffer.")
            self._build_filters(self.fs_real)
            new_capacity = int(self.window_seconds * self.fs_real * 1.2)
            self.buf = RollingBuffer(capacity=new_capacity, n_channels=N_CH)
            self._build_display(self.fs_real)
            self._ylim_state = [None] * N_CH
            for i, ln in enumerate(self.lines):
                ln.set_data(self.x_axis, self.display[:, i])
            self._last_rebuild = now

    def _drain_stream(self) -> None:
        while True:
            r = self.stream.read()
            if r is None:
                break
            t, x = r
            self.buf.push(t, x)
            self.samples_since_check += 1

    def _update_fs_real(self) -> None:
        now = time.monotonic()
        dt = now - self.last_fps_check
        if dt >= 0.5:
            self.fs_real = self.samples_since_check / dt
            self.samples_since_check = 0
            self.last_fps_check = now

    def update(self, frame):
        self._drain_stream()
        self._update_fs_real()
        self._maybe_rebuild_filters()

        if self.buf.size == 0:
            return self.lines + self.stat_texts + [self.title]

        t, x_full = self.buf.view()
        x = x_full.astype(np.float32, copy=True)
        t_end = t[-1]
        t_start = t_end - self.window_seconds
        x_w = x

        if self.apply_filters and x_w.shape[0] > 0:
            # Pipeline EMG: detrend -> notch -> bandpass.
            if self.notch_sos is not None or self.emg_sos is not None:
                bp_zi_template = (sosfilt_zi(self.emg_sos)
                                  if self.emg_sos is not None else None)
                notch_zi_template = (sosfilt_zi(self.notch_sos)
                                     if self.notch_sos is not None else None)
                for i in range(4):
                    sig = x_w[:, i]
                    sig_dc = sig.mean()
                    sig_ac = sig - sig_dc
                    if self.notch_sos is not None:
                        zi = notch_zi_template * sig_ac[0]
                        sig_ac, _ = sosfilt(self.notch_sos, sig_ac, zi=zi)
                    if self.emg_sos is not None:
                        zi = bp_zi_template * sig_ac[0]
                        sig_ac, _ = sosfilt(self.emg_sos, sig_ac, zi=zi)
                    x_w[:, i] = sig_ac
            if self.joy_sigma_samples > 0.5:
                for i in range(4, 6):
                    x_w[:, i] = gaussian_filter1d(x_w[:, i],
                                                  sigma=self.joy_sigma_samples,
                                                  mode="nearest")

        mask = t >= t_start
        t_w = t[mask] - t_end
        x_w = x_w[mask, :]

        # Resample/copy del rolling buffer al display buffer (tamano fijo).
        n_src = x_w.shape[0]
        n_dst = self.display_samples
        if n_src >= n_dst:
            idx = np.linspace(0, n_src - 1, n_dst).astype(np.int64)
            self.display[:, :] = x_w[idx, :]
        else:
            self.display[:-n_src, :] = np.nan
            self.display[-n_src:, :] = x_w

        # Update traces solo con set_ydata (eje X estatico)
        for i, ln in enumerate(self.lines):
            ln.set_ydata(self.display[:, i])

        # Update axes limits con EMA suave (alpha=0.15)
        alpha = 0.15
        for i, ax in enumerate(self.axes):
            valid = self.display[:, i]
            valid = valid[~np.isnan(valid)]
            if valid.size < 4:
                continue
            if i < 4:  # EMG
                mid = float(valid.mean())
                sd = float(valid.std())
                half = max(3.0 * sd, 1.0)
                target = (mid - half, mid + half)
            else:      # joysticks
                lo = float(np.percentile(valid, 1))
                hi = float(np.percentile(valid, 99))
                pad = max(1.0 * (hi - lo), 1e-3)
                target = (lo - pad, hi + pad)
            if self._ylim_state[i] is None:
                smoothed = target
            else:
                prev_lo, prev_hi = self._ylim_state[i]
                smoothed = (prev_lo * (1 - alpha) + target[0] * alpha,
                            prev_hi * (1 - alpha) + target[1] * alpha)
            self._ylim_state[i] = smoothed
            ax.set_ylim(*smoothed)

        # Update stats text
        for i, (name, _) in enumerate(CHANNELS):
            valid = self.display[:, i]
            valid = valid[~np.isnan(valid)]
            s = channel_stats(valid) if valid.size > 0 else channel_stats(np.array([0.0]))
            if i < 4:   # EMG
                self.stat_texts[i].set_text(
                    f"{name:<14}\n"
                    f"  RMS: {s['rms']:8.2f}\n"
                    f"  p99: {s['p99']:8.2f}\n"
                    f"  std: {s['std']:8.2f}\n"
                    f"  range: [{s['min']:+7.1f}, {s['max']:+7.1f}]"
                )
            else:       # joysticks
                self.stat_texts[i].set_text(
                    f"{name:<14}\n"
                    f"  mean: {s['mean']:8.1f}\n"
                    f"  std:  {s['std']:8.2f}\n"
                    f"  range: [{s['min']:+7.1f}, {s['max']:+7.1f}]"
                )

        self.title.set_text(
            f"Monitor — fs_target={self.fs:.0f} Hz   "
            f"fs_real={self.fs_real:.0f} Hz   "
            f"buffer={self.buf.size:,}/{self.buf.capacity:,}"
        )

        return self.lines + self.stat_texts + [self.title]

    def run(self) -> None:
        interval_ms = int(1000.0 / self.fps)
        self.anim = FuncAnimation(
            self.fig, self.update, interval=interval_ms,
            blit=False, cache_frame_data=False,
        )
        try:
            plt.show()
        finally:
            self.stream.close()


# ----- CLI -----

def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--source", choices=["mock", "serial"], default="mock",
                   help="origen del stream.")
    p.add_argument("--port", default=None,
                   help="puerto serial (solo si --source serial)")
    p.add_argument("--fs", type=float, default=1000.0,
                   help="fs objetivo (Hz). Si la fs real medida difiere "
                        ">10%%, el monitor reconstruye filtros automaticamente.")
    p.add_argument("--window-seconds", type=float, default=5.0,
                   help="duracion de la rolling window visualizada (s)")
    p.add_argument("--fps", type=int, default=30,
                   help="frame rate de update del plot")
    p.add_argument("--seed", type=int, default=0,
                   help="seed del mock (solo si --source mock)")
    p.add_argument("--no-filter", action="store_true",
                   help="desactivar TODOS los filtros (raw, EMG y joy)")
    p.add_argument("--no-emg-filter", action="store_true",
                   help="desactivar solo el bandpass + notch de EMG")
    p.add_argument("--no-joy-filter", action="store_true",
                   help="desactivar solo el gaussiano de joysticks")
    p.add_argument("--emg-low", type=float, default=70.0,
                   help="bandpass EMG cutoff bajo (Hz)")
    p.add_argument("--emg-high", type=float, default=300.0,
                   help="bandpass EMG cutoff alto (Hz). Se limita "
                        "a 0.45*fs si excede Nyquist.")
    p.add_argument("--notch-base", type=float, default=50.0,
                   help="frecuencia base del notch (50 en EU/AR, 60 en US). "
                        "Se aplican notches en base + harmonics hasta Nyquist. "
                        "Usar 0 para desactivar.")
    p.add_argument("--notch-q", type=float, default=30.0,
                   help="Q del notch (mayor = mas estrecho)")
    p.add_argument("--joy-sigma-ms", type=float, default=50.0,
                   help="sigma del gaussiano sobre joysticks (ms)")
    return p.parse_args()


def main():
    args = parse_args()

    if args.source == "mock":
        print(f"Mock stream: fs={args.fs:.0f} Hz, seed={args.seed}")
        stream: StreamReader = MockStream(fs=args.fs, seed=args.seed)
    elif args.source == "serial":
        if args.port is None:
            raise SystemExit("--port es obligatorio con --source serial")
        print(f"Serial stream: port={args.port}, baud=2000000")
        stream = SerialStream(port=args.port, baud=2000000)
    else:
        raise SystemExit(f"unknown source: {args.source}")

    if args.no_filter:
        emg_low, emg_high, joy_sigma, notch_base = 0.0, 0.0, 0.0, 0.0
    else:
        emg_low = 0.0 if args.no_emg_filter else args.emg_low
        emg_high = 0.0 if args.no_emg_filter else args.emg_high
        notch_base = 0.0 if args.no_emg_filter else args.notch_base
        joy_sigma = 0.0 if args.no_joy_filter else args.joy_sigma_ms

    monitor = Monitor(
        stream=stream,
        fs=args.fs,
        window_seconds=args.window_seconds,
        fps=args.fps,
        emg_bp_low=emg_low,
        emg_bp_high=emg_high,
        notch_base=notch_base,
        notch_q=args.notch_q,
        joy_sigma_ms=joy_sigma,
        apply_filters=True,
    )
    print(f"Window: {args.window_seconds:.1f} s")
    print(f"FPS: {args.fps}")
    print("Cerrar la ventana para salir.")
    monitor.run()


if __name__ == "__main__":
    main()
