# Hardware setup — ESP32-S3 + ADS1256

Setup actual del proyecto (post-migración desde Arduino Uno). 4 EMG
diferenciales por el ADS1256 + 2 joysticks por el ADC interno del
ESP32-S3.

## Componentes

| Componente              | Modelo                       | Función                          |
|-------------------------|------------------------------|----------------------------------|
| MCU                     | ESP32-S3 DevKitC-1 (N16R8V)  | Adquisición + serial             |
| ADC externo             | ADS1256 (24-bit, 8 canales)  | EMG diferencial low-noise        |
| Joysticks               | 2 × analógico (1 eje útil)   | Posición del dedo índice         |
| EMG pills               | 4 × módulo                   | Flexor / extensor índice + medio |

## Pinout — ESP32-S3 ↔ ADS1256 (SPI)

| ESP32-S3 GPIO | ADS1256 pin  | Función SPI       |
|---------------|--------------|-------------------|
| 38            | SCK          | SPI clock         |
| 41            | DOUT (MISO)  | Data ADS → MCU    |
| 1             | DIN  (MOSI)  | Data MCU → ADS    |
| 2             | CS           | Chip select       |
| 4             | DRDY         | Data ready (IRQ)  |

SPI a 1.92 MHz, modo SPI_MODE1. Se usa el periférico `HSPI` del
ESP32-S3 (no FSPI).

## Pinout — ESP32-S3 ↔ Joysticks (ADC interno)

| ESP32-S3 GPIO | Joystick    | Notas                              |
|---------------|-------------|------------------------------------|
| 5             | joy_a       | ADC1 channel, 12-bit, 0–3.3 V      |
| 6             | joy_b       | ADC1 channel, 12-bit, 0–3.3 V      |

Ambos joysticks miden el mismo dedo (índice) para redundancia. **Se
usa solo una salida por joystick** (la otra dirección no se cablea).

> Si las salidas del joystick van más allá de 3.3 V (típico: 5 V),
> usar un divisor de tensión o módulo nivelador. El ADC del ESP32-S3
> satura por arriba de 3.3 V.

GPIO libres en ADC1 para futuras expansiones: 3, 7, 8, 9, 10.

## Asignación de canales EMG ↔ músculo (cableado actual)

Los EMG pills se cablean **single-ended** al ADS: el `OUT` del pill va
a un AIN del ADS, y el `GND` del pill va al **GND común del sistema**
(no a un AIN). El ADS mide `AINn - AINCOM`, donde AINCOM está conectada
internamente al GND del módulo (que comparte el riel común).

| Stream field | AIN del ADS | Músculo                |
|--------------|-------------|------------------------|
| `emg0`       | AIN0        | **Extensor** del índice |
| `emg1`       | AIN1        | **Flexor** del índice   |
| `emg2`       | AIN5        | **Extensor** del medio  |
| `emg3`       | AIN6        | **Flexor** del medio    |

> AIN2, AIN3, AIN4 y AIN7 quedan libres para expansión (más canales o sensores).

Cableado físico de cada pill:

```
EMG pill OUT  →  AINn  (n = 0, 1, 5, 6 según el músculo)
EMG pill GND  →  GND común del sistema (NO a un AIN)
EMG pill VCC  →  3.3 V del ESP32 (rail interno regulado, no la fuente
                                  directa, para evitar ruido de línea)
```

> **Historia**: se intentó primero diferencial (OUT a AIN par, GND a
> AIN impar) y después single-ended con GND a AIN impar. Ninguna
> funcionó porque el GND del pill quedaba "flotando" al estar conectado
> a una entrada de alta impedancia del ADS. Cuando se conectó el GND al
> rail común del sistema (igual que con Arduino), funcionó.

> AINCOM no se cablea explícitamente (el módulo no expone ese pin; está
> internamente unida al GND principal del módulo).

El orden importa porque downstream (`collect.py`, dataset) usa esa
asignación para construir las features. Si se cambia el cableado, hay
que actualizar el schema del CSV o renombrar al ingestar.

> **Cambio respecto al setup Arduino-Uno**: el dataset histórico tiene
> 2 canales EMG (flx + ext del índice solamente). Al pasar a 4 canales,
> los datasets nuevos NO son retrocompatibles con modelos entrenados
> sobre el formato viejo. Hay que reentrenar.

## Formato del stream serial

```
t_micros, emg0, emg1, emg2, emg3, joy_a, joy_b
```

- `t_micros`: timestamp (unsigned long) desde `micros()`.
- `emg{0..3}`: int32 con extensión de signo, 24-bit raw del ADS1256.
- `joy_a`, `joy_b`: int 0–4095 del ADC interno (12-bit).

Baudrate: **921600**. CR/LF: solo `\n`. Líneas que empiezan con `#`
son comentarios / banner / errores; se descartan al parsear.

## Configuración del ADS1256

| Registro | Valor | Significado                                       |
|----------|-------|---------------------------------------------------|
| STATUS   | 0x00  | BUFEN=off, ACAL=off, ORDER=MSB                    |
| ADCON    | 0x00  | PGA gain = 1, CLKOUT off, SDCS off                |
| DRATE    | 0xE0  | 15000 SPS (ver alternativas abajo)                |
| MUX      | rotando | 4 pares diferenciales round-robin               |

> **Importante**: el buffer del ADS se controla en el registro **STATUS**
> (bit 1, `BUFEN`), NO en ADCON. Un error común es confundir `ADCON = 0x02`
> con "buffer ON" cuando en realidad es **PGA gain = 4** (bits 2-0 del
> ADCON). El buffer se queda OFF en esta config porque los EXG Pills ya
> tienen buffer interno y amplifican la señal; agregar el del ADS introduce
> ruido extra y limita el rango.

Calibración self-cal después de configurar DRATE (recomendado por
datasheet).

### Trade-off DRATE ↔ noise ↔ fs efectiva

Con multiplexado round-robin de 4 canales y settling de 1/DRATE por
cambio de MUX:

| DRATE setting | SPS interna | fs efectiva por canal | Noise (RMS) |
|---------------|-------------|-----------------------|-------------|
| 0xD0          | 7,500       | ~1 kHz                | bajo        |
| 0xE0          | 15,000      | ~2 kHz                | medio       |
| 0xF0          | 30,000      | ~3 kHz                | alto        |

Default: 0xD0 (7500 SPS). Si el monitor reporta fs efectiva consistente
> 900 Hz, mantener. Subir solo si necesitás más banda pasante (los EMG
útiles están en 70–300 Hz, así que 1 kHz es suficiente por Nyquist).

## Flow de captura del firmware (round-robin)

Para cada loop:

1. Para cada canal `ch` en 0..3:
   - Esperar `DRDY` (active low).
   - Leer 24-bit del ADS.
   - Cambiar MUX al canal siguiente, emitir `SYNC` + `WAKEUP`.
2. Leer `joy_a` y `joy_b` del ADC interno.
3. Emitir línea CSV.

La conversión del próximo canal arranca **mientras** estamos leyendo
el actual, así que el settling time del MUX queda absorbido en el
loop (no agrega overhead extra).

## Compilación / flash (PlatformIO)

```bash
cd sketchs/esp32
pio run                          # compila
pio run --target upload          # flash
pio device monitor -b 921600     # serial monitor
```

`platformio.ini` configurado para `esp32-s3-devkitc-1-n16r8v` con
USB CDC en boot.

## Smoke test

Después de flashear:

```bash
# Verificar que el stream sale
pio device monitor -b 921600

# Esperás ver:
#   # READY: t_micros,emg0,emg1,emg2,emg3,joy_a,joy_b
#   # ...comentarios de canales...
#   12345678,123,-456,789,-23,2048,2050
#   12345789,124,-455,790,-24,2049,2051
#   ...
```

Validaciones rápidas:

| Check                                          | Valor esperado                            |
|------------------------------------------------|-------------------------------------------|
| `t_micros` incrementa                          | sí, monotónico                            |
| Cadencia entre líneas                          | ~1000 μs (DRATE 7500 SPS)                 |
| `emg{0..3}` sin EMG conectado (entradas en GND)| valores chicos, ruido térmico ~±100       |
| `joy_a`, `joy_b` sin joystick conectado        | flotantes (cualquier valor)               |
| `joy_a`, `joy_b` con joystick a rest           | ~2048 (mitad del ADC 12-bit)              |
| `joy_a` con joystick a tope                    | 0 o 4095 (según dirección elegida)        |

## Ruido eléctrico de línea (PLI) y alimentación

**Hallazgo (2026-06-15)** durante la puesta a punto del setup nuevo:

Con los EXG Pills alimentados a una **fuente DC conectada a la red
eléctrica** (la típica MB102 con plug a 220 V), las capturas mostraron
ruido masivo en banda 50 Hz + harmonics (100, 150, 200, 250, 300 Hz),
con amplitud comparable o mayor a la señal EMG real. La FFT mostraba el
patrón "peine" clásico de PLI (power line interference).

### Diagnóstico

- El ruido aparecía en el espectro como picos claros a 50, 100, 150,
  200, 250 Hz... todos múltiplos de la frecuencia de línea local
  (50 Hz en AR/EU; sería 60 Hz en US).
- Con `bandpass 70-300 Hz` sin notch, los harmónicos a 100, 150, 200,
  250 Hz caen **dentro de la banda EMG** y dominaban la señal,
  ocultando los bursts musculares reales.
- El SINC⁵ embebido del ADS1256 **no ayuda en este régimen** porque
  sus ceros caen en múltiplos del DRATE configurado (15 kSPS), no en
  50/60 Hz.

### Solución que confirma el diagnóstico

Aplicar un **notch IIR en cascada** (50 Hz + harmónicos hasta Nyquist,
Q=30) **antes del bandpass**. Implementado tanto en `plot_raw.py` (flag
`--notch-base`/`--notch-q`, default 50/30) como en `monitor.py` (mismos
flags). Tras el notch, los bursts musculares se vuelven visibles en el
gráfico offline. Confirma que **abajo del PLI sí hay señal EMG real**.

### Solución de hardware (preferida)

Mover la alimentación **fuera de la red**:

- **Powerbank USB → ESP32**: el ESP32 alimenta pills y ADS por sus
  rieles internos. Es la opción más simple.
- **Batería 9 V o pack 6×AA → MB102**: la MB102 acepta 6.5-12 V por
  el jack DC. Regula a 5 V y 3.3 V internos.
  - 6 AA alkalinas (9 V, ~2 Ah) duran ~1-2 horas con todo el setup.
  - Batería 9 V (~500 mAh) dura ~30 min — útil para sesiones cortas.

Con alimentación de batería, los pills y el ADC quedan **galvánicamente
aislados de la red**, y el comb de 50 Hz desaparece (o se reduce 20-40 dB).
El notch software queda como red de seguridad, no como obligación.

### Workaround si seguís enchufado a la red

`plot_raw.py` y `monitor.py` aplican notch por defecto (`--notch-base 50`).
Cubre la mayoría del PLI pero **come ~5 Hz de banda EMG en cada
harmonic**. Para EMG genérico es tolerable; para análisis de frecuencia
específica (e.g. detección de fatiga muscular por shift espectral)
puede ser problemático.

## Cosas pendientes (TODO)

- Actualizar `collect.py` y `dataset/emg_dataset.py` al nuevo schema
  (7 campos en vez de 5).
- Calibración por sujeto del rango efectivo de los joysticks (no
  siempre llegan a 0/4095 por mecánica).
- Banner de versión / firmware ID en el stream para tracking.
