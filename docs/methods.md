# Métodos — decodificación EMG → velocidad del dedo índice con TCN causal

Documento de referencia para informe. Cubre hardware, protocolo de captura,
preprocesamiento offline, arquitectura del modelo, entrenamiento, métricas y
gotchas observados en la primera iteración (sujeto P01, sesión 1, 2026-05-24).

---

## 1. Hardware y adquisición

### MCU y canales

- **Microcontrolador**: Arduino Uno (ADC 10 bits, rango 0–1023).
- **Comunicación**: Serial USB a 230400 baud.
- **Sketch**: `sketchs/arduino/arduino.ino`. Loop libre sin `delay()` ni timer fijo.
- **Línea serial emitida por iteración** (5 campos enteros, CSV):
  `t_micros, y_top, y_bottom, emg_extensor, emg_flexor`.

| Pin | Señal | Tipo |
|-----|-------|------|
| A0  | EMG extensor digitorum | analógico, banda EMG cruda |
| A1  | EMG flexor digitorum    | analógico, banda EMG cruda |
| A4  | Joystick superior (`y_top`) | analógico, posición |
| A5  | Joystick inferior (`y_bottom`) | analógico, posición |

### Mecánica de la medición de posición

Dos joysticks lineales verticales con el dedo índice colocado entre ambos.
Extensión empuja `y_top` (desde reposo ~520 hacia 0). Flexión empuja `y_bottom`
(desde reposo ~528 hacia 1023). La posición instantánea del dedo se reconstruye
como `(-y_top + offset_top) + (-y_bottom + offset_bottom)` para mantener un
signo consistente (extensión positiva, flexión negativa).

### Frecuencia de muestreo observada

| Condición | fs efectivo |
|-----------|-------------|
| Dry-run (sin escritura a CSV) | ~641 Hz |
| Sesión real (escribiendo CSV) | **~605 Hz** |

La I/O al disco bloquea el loop unos microsegundos por línea. **El pipeline
estima `fs` por sesión desde `1 / mean(diff(t_micros))`; no se asume 800 Hz.**

### Plan de migración

ESP32 contemplado para iteraciones futuras: ADC de 12 bits, mayor throughput
(fs esperado > 1 kHz), posibilidad de cómputo de features on-board. Datos
pre/post migración no son intercambiables sin tratamiento explícito (anotar
en `optional_notes` o crear `placement_id` separado).

---

## 2. Protocolo de captura

Una sesión = 11 bloques contiguos (≈ 20 min de datos taggeados + ~33 s de
transiciones). Cada bloque tiene una `task` etiquetada. El vocabulario es
cerrado: los strings son inmutables entre sesiones (renombrar rompería
sesiones viejas).

| # | task                  | duración base | qué hace el sujeto |
|---|-----------------------|---------------|--------------------|
| 1 | `rest_baseline`       | 90 s          | Mano relajada, sin contracción. |
| 2 | `slow_flex`           | 120 s         | Flexión lenta (~3 s) + suelta. Repetir. |
| 3 | `slow_ext`            | 120 s         | Extensión lenta + suelta. Repetir. |
| 4 | `slow_cycles`         | 150 s         | Ciclos continuos flex↔ext (~0.25 Hz). |
| 5 | `fast_flex`           | 90 s          | Flex rápida + pausa 2 s. |
| 6 | `fast_ext`            | 90 s          | Ext rápida + pausa 2 s. |
| 7 | `holds_intermediate`  | 150 s         | Mantener 4 s en 25/50/75 % del rango. |
| 8 | `microcorrections`    | 120 s         | Oscilaciones de 5–10 % alrededor del centro. |
| 9 | `noise_other_fingers` | 100 s         | Mover pulgar, medio, anular, meñique. **Sin** mover índice. |
| 10| `noise_wrist_grip`    | 100 s         | Flex/ext muñeca, pronación, agarre. **Sin** mover índice. |
| 11| `free_controlled`     | 70 s          | Movimiento libre del índice cubriendo todo el rango. |

Entre bloques: 3 s de countdown grabados con `task="transition"` (se filtran
en training).

### Variantes por sesión (plan 1–10)

| Sesión | Variante | Propósito |
|--------|----------|-----------|
| 1–4    | Protocolo base, `placement_id = P0?-A` | Train principal |
| 5–6    | Mismo protocolo, placement levemente movido (`-B`) | Robustez a placement |
| 7–8    | `noise_*` extendidos a 180 s cada uno | Más negativos |
| 9      | Placement bastante diferente (`-C`) | Robustez fuerte |
| 10     | Protocolo reducido y mixto, `split="test"` | **No usar para entrenar** |

---

## 3. CSV schema

Por cada muestra recibida del serial, una fila:

```
pc_timestamp, t_micros, y_top, y_bottom, flexor, extensor,
trial_id, task, session_id, subject_id, placement_id, optional_notes
```

- `pc_timestamp`: `time.time()` del host al recibir la línea.
- `t_micros`: timestamp del Arduino (en microsegundos desde reset).
- `y_top, y_bottom, flexor, extensor`: int 0–1023 directos del ADC.
- `trial_id`: índice secuencial del bloque (1..11).
- `task`: string del vocabulario cerrado (incluye `transition`).
- `session_id = subject_S{NN}_{YYYYMMDD_HHMMSS}` (generado automáticamente).
- `placement_id`: convención `<subject>-A/B/C` (ver §2).
- `optional_notes`: libre. Recomendado anotar `mcu=arduino_uno`.

Junto al CSV se escribe un `<session_id>.json` con metadata del protocolo,
`split`, `started_at/ended_at`, `good_lines / bad_lines`, etc.

---

## 4. Preprocesamiento offline — target

### 4.1 Estimación de `fs`

```
fs = 1 / mean(diff(t_micros / 1e6))
```

Se hace por sesión. No se asume nada.

### 4.2 Smoothing de los joysticks

Cada joystick se invierte y se centra en su rest:

```
y_top_centered    = -y_top    + 500
y_bottom_centered = -y_bottom + 550
```

Sobre cada uno se aplica un filtro **gaussiano 1D** (no causal, sólo offline):

```
sigma_samples = sigma_ms / 1000 * fs       (sigma_ms = 50 ← calibrado, ver §10.8)
y_top_s    = scipy.ndimage.gaussian_filter1d(y_top_centered,   sigma=sigma_samples, mode='nearest')
y_bottom_s = scipy.ndimage.gaussian_filter1d(y_bottom_centered,sigma=sigma_samples, mode='nearest')
```

> **El valor de `sigma_ms` es crítico** y fue recalibrado experimentalmente
> desde el inicial 150 ms al actual 50 ms tras un sweep `σ ∈ {10, 30, 50, 150}`.
> Ver §10.8 para el análisis completo del impacto sobre la decodificabilidad
> del target y los próximos refinamientos planeados. Sigue siendo configurable
> vía `--sigma-ms` en `train.py`.

### 4.3 Posición y velocidad

```
joy      = y_top_s + y_bottom_s                       # posición compuesta
velocity = numpy.gradient(joy, dt)                    # diferencia central (no causal)
```

La velocidad cruda en sesión 1 tiene |p99| ≈ 1369 unidades/s.

### 4.4 Normalización del target

Escala robusta por percentil:

```
target_scale = percentile(|velocity|, 99) = 1368.97
target_norm  = velocity / target_scale
```

Resultado: target con `std ≈ 0.31`, rango ≈ [-1.8, +1.9]. Centrado en 0,
con la mayoría de la masa en [-1, +1].

> **Nota**: el clipping mecánico de los joysticks (`y_top=0`, `y_bottom=1023`)
> NO se enmascara — refleja un límite físico real del dedo, asimilable a la
> biomecánica. Se compensa en post con un factor de dilatación si hace falta.

---

## 5. Preprocesamiento — features EMG (todo causal)

Pipeline por canal (flexor y extensor en paralelo):

### 5.1 Bandpass causal

Butterworth orden 4, banda [70, 300] Hz, implementado como SOS en una pasada
(`scipy.signal.sosfilt`):

```
sos = butter(4, [70/nyq, 300/nyq], btype='bandpass', output='sos')
x_bp = sosfilt(sos, x_raw)
```

Una pasada (no `filtfilt`) → **causal**. Tiene phase delay propio del IIR,
aceptable porque el target también va con su delay (entrenamos pares
sincronizados).

### 5.2 RMS causal

Ventana rectangular móvil de **40 ms**. Implementada con `cumsum` para evitar
el bucle:

```
win = round(40/1000 * fs)
sq  = x_bp ** 2
csum = cumsum(sq)
avg[:win] = csum[:win] / arange(1, win+1)      # warmup
avg[win:] = (csum[win:] - csum[:-win]) / win
rms = sqrt(avg)
```

### 5.3 Normalización EMA

Línea de base adaptativa con constante de tiempo 1 s, vía `lfilter` (IIR
de primer orden):

```
alpha = 1 - exp(-dt / 1.0)
ema   = lfilter([alpha], [1, -(1-alpha)], rms)
rms_norm = rms / (ema + 1e-6)
```

Interpretación: `rms_norm` mide el RMS instantáneo en unidades de "veces
el baseline reciente". Durante una contracción aguda salta a >1; durante
el silencio cae a <1.

### 5.4 Derivada causal

```
d(rms_norm)/dt = (rms_norm[t] - rms_norm[t-1]) / dt    # backward diff
```

### 5.5 Vector final de features

4 canales por timestep:

| idx | nombre           | qué es |
|-----|------------------|--------|
| 0   | `flx_rms_norm`   | RMS flexor normalizado por EMA |
| 1   | `ext_rms_norm`   | RMS extensor normalizado por EMA |
| 2   | `flx_d`          | derivada causal de `flx_rms_norm` |
| 3   | `ext_d`          | derivada causal de `ext_rms_norm` |

**Clipping** de seguridad: `clip(features, -20, 20)`. Acota spikes del
cociente `RMS/EMA` cuando `EMA → 0` y mantiene gradientes bien condicionados.

### 5.6 Estadísticas observadas (sesión 1)

| feature        | mean   | std    | min     | max     |
|----------------|--------|--------|---------|---------|
| `flx_rms_norm` | +1.000 | 0.607  | +0.027  | +20.000 (clip) |
| `ext_rms_norm` | +1.000 | 0.974  | +0.016  | +20.000 (clip) |
| `flx_d`        | −0.137 | 12.604 | −20.000 (clip) | +20.000 (clip) |
| `ext_d`        | −0.361 | 12.293 | −20.000 (clip) | +20.000 (clip) |

---

## 6. Arquitectura del modelo — `EMGTCN`

TCN causal compacta. Diseñada para tiempo real: una sola pasada de
convoluciones causales sobre una ventana de `RF` muestras devuelve la
predicción para el último timestep.

### 6.1 Componentes

**`Chomp1d(chomp_size)`** — recorta los últimos `chomp_size` samples del
tensor `[B, C, T]`. Indispensable para preservar causalidad cuando se padea
la conv simétricamente.

**`CausalConv1d(in, out, k, dilation)`**:

```
pad = (k - 1) * dilation
Conv1d(in, out, kernel=k, padding=pad, dilation=dilation)  + Chomp1d(pad)
```

Conv1d en PyTorch padea simétricamente (k-1)*d a cada lado; Chomp1d quita
los pad samples del final. Equivale a left-padding puro. Opcionalmente
WeightNorm sobre el Conv (ver §10).

**`CausalConvBlock(in, out, k, [d1, d2], dropout)`** — bloque con residual:

```
h = drop(GELU(CausalConv1d(in,  out, k, d1)(x)))
h = drop(GELU(CausalConv1d(out, out, k, d2)(h)))
return h + residual(x)
```

`residual = Conv1d(in, out, k=1)` si `in != out`, else `Identity`.

### 6.2 Estructura completa

```
x: [B, in_channels=4, T]
   │
   └─► CausalConvBlock(4 → 32, dilations=[1, 3])           # bloque residual 1
        │
        └─► CausalConvBlock(32 → 32, dilations=[9, 27])    # bloque residual 2
             │
             └─► CausalConv1d(32 → 32, dilation=81)        # capa final
                  └─► GELU + Dropout
                       │
                       └─► Conv1d(32 → 1, kernel=1)        # head
                            │
                            └─► return [:, :, -1]          # último timestep
```

Forward devuelve `[B, output_dim=1]` (escalar por ventana).

### 6.3 Hiperparámetros del modelo

| HP                | valor   | nota |
|-------------------|---------|------|
| `in_channels`     | 4       | features EMG (ver §5.5) |
| `hidden_channels` | 32      | chico a propósito, poca data |
| `output_dim`      | 1       | velocidad escalar |
| `kernel_size`     | 3       | |
| `dilations`       | `[1, 3, 9, 27, 81]` | |
| `dropout`         | 0.1     | después de cada conv |
| `activation`      | GELU    | |
| `weight_norm`     | desactivado en MPS (ver §10) | |
| parámetros        | **13,025** (sin WN) / 13,185 (con WN) | |

### 6.4 Campo receptivo

```
RF = 1 + (k - 1) * sum(dilations)
   = 1 + 2 * (1 + 3 + 9 + 27 + 81)
   = 243 samples
```

A `fs = 605 Hz` (sesión 1): **RF ≈ 401 ms**. Mayor que el target original
de 300 ms por la diferencia entre fs nominal (800 Hz) y real (605 Hz).

### 6.5 Causalidad verificada empíricamente

Test en `models/tcn.py:__main__`: con input de longitud `T = 343 > RF`, se
perturban los samples `[0 : T - RF]` (fuera del campo receptivo del último
timestep). Diff esperado: 0. Resultado medido: `max |Δy| = 0.00e+00`.

---

## 7. Pipeline de training

### 7.1 Dataset

`EMGSessionDataset` (en `dataset/emg_dataset.py`):

- Carga el CSV, computa `fs`, deriva target y features (§ 4–5).
- Aplica clip a features.
- Calcula `target_scale` (p99 de |velocity|), normaliza el target.
- Construye índices válidos:
  - excluye `task == "transition"` (countdowns).
  - excluye warmup del bandpass/RMS/EMA: primer 1 s.
  - cada ventana cubre `[last - RF + 1, last]`, supervisada en `target[last]`.
- `__getitem__(i)` devuelve `(x: [4, 243], y: scalar)`.

### 7.2 Split train / val

**Estratificado por bloque de task** (default, recomendado):

```
para cada bloque contiguo de la misma task:
    último 20% → val
    los `gap = 243` windows previos al val → descartados (anti-leakage)
    el resto → train
```

Garantiza que cada task aparece en train y val. Alternativa `--split-mode
temporal` (último 20% de la sesión) sólo sirve como sanity check — en la
sesión 1 el val terminaba siendo 100 % `noise_*` + `free_controlled`.

### 7.3 Loss

**Huber** con `delta = 1.0`:

```
L(e) = 0.5 * e²              si |e| < delta
L(e) = delta * (|e| - 0.5*delta)  si no
```

Justificación: target normalizado tiene std ≈ 0.31. La mayoría del rango
[−1, +1] cae en zona cuadrática (sensible, gradientes ricos). Los picos
extremos durante `fast_flex` / `fast_ext` superan el delta y caen en zona
lineal (no dominan el gradiente).

### 7.4 Optimizador

```
AdamW(lr=1e-3, weight_decay=1e-4)
gradient clipping: norm 1.0 (torch.nn.utils.clip_grad_norm_)
sin scheduler en esta iteración
```

### 7.5 Hiperparámetros de training

| HP            | valor |
|---------------|-------|
| batch_size    | 256   |
| epochs        | 20    |
| val_fraction  | 0.2   |
| gap_windows   | 243   |
| train_stride  | 4     |
| val_stride    | 4     |
| seed          | 42    |

`stride = 4` reduce el solapamiento entre ventanas consecutivas
(redundancia muy alta con stride=1). De 729k windows totales → 145k train
+ 36k val.

### 7.6 Regularización por suavidad de la velocidad

**Problema.** Las predicciones de velocidad del modelo single-output, aun
con buen R² agregado, son **ruidosas a alta frecuencia**. Si se las usa
como comando directo de un controlador, ese ruido se traduce en jitter en
la posición. El uso clínico real requiere predicciones suaves.

Exploramos tres aproximaciones sucesivas. Cada una corrigió un problema
estructural de la anterior.

#### 7.6.1 TVD cross-batch (legacy, no usar)

**Idea**: penalizar la primera derivada de las predicciones consecutivas
en el tiempo, usando la formulación clásica de **Total Variation
Denoising** (Rudin, Osher, Fatemi 1992):

```
L_smooth = (1/N) · Σ_t | pred[t+1] - pred[t] |
L        = L_data + λ · L_smooth
```

**Implementación**: `--smooth-weight λ` en `train.py`. Como cada muestra
del batch corresponde a un timestep distinto, se requiere
`shuffle = False` para que el orden temporal se preserve dentro del
batch, más un estado cross-batch (`prev_pred_vel`) que conecta el último
output del batch anterior con el primero del actual.

**Resultado**: **colapso del modelo en epoch 1** con `λ ∈ {0.1, 0.5, 1.0}`.
Razones identificadas:

1. `shuffle = False` sesga los gradientes a tareas específicas (los
   primeros batches son solo `rest`, los siguientes solo `slow_flex`, etc.).
2. Penalizar el módulo absoluto de la derivada empuja al modelo hacia
   predicciones constantes — el camino más fácil de minimizar el término
   de smoothness es no moverse nunca.
3. El fine-tuning desde un checkpoint entrenado con `shuffle = True` rompe
   las estadísticas de batch.

**Conclusión**: la formulación es estructuralmente incorrecta para este
setting. El flag se conserva por compatibilidad pero genera error si se
combina con `--horizon > 1`.

#### 7.6.2 Adaptive TV / anisotropic diffusion (descartado en diseño)

**Idea**: peso variable por timestep, inversamente proporcional a la
magnitud de la derivada del target. Donde el target es plano, penalizar
fuerte cualquier salto en pred. Donde el target tiene una transición,
relajar la penalidad para que el modelo pueda seguirla. Formalización
clásica de **Perona & Malik (1990)** y **Black et al. (1998)** sobre
"anisotropic diffusion" / "edge-preserving smoothness":

```
w[t]     = 1 / (1 + α · |Δtarget[t]|²)         (Lorentziana, Black 1998)
       o = exp(-β · |Δtarget[t]|²)             (Gaussiana, Perona-Malik original)
L_smooth = Σ w[t] · |Δpred[t]|
```

**Por qué se descartó**: el paradigma de Perona-Malik nace en *image
denoising*, donde no hay ground truth de la derivada y el peso es un
*proxy* que intenta adivinar dónde están los edges reales. En nuestro
caso sí tenemos `Δtarget` exactamente (proviene del joystick). Usarlo
solo como peso es subutilizarlo.

Además presenta una asimetría: penaliza el caso "target plano, pred salta"
pero **no penaliza** el caso "target salta, pred plano". El modelo puede
quedarse subreactivo sin costo en el término de smoothness.

#### 7.6.3 Sobolev / H¹ loss (implementación actual)

**Idea**: en lugar de regularizar la derivada de `pred` con un peso, hay
que **supervisar directamente** la derivada de `pred` contra la derivada
de `target`. Formalización: norma del espacio de Sobolev H¹, que incluye
función + primera derivada. Referencia formal: **Czarnecki, Osindero,
Jaderberg, Świrszcz, Pascanu (2017)**, *Sobolev Training for Neural
Networks*, NeurIPS.

```
L_data  = Σ_t Huber( pred[t] - target[t] )
L_accel = Σ_t ( Δpred[t] - Δtarget[t] )²
L       = L_data + λ · L_accel
```

con `Δx[t] = x[t+1] - x[t]`.

**Por qué resuelve los dos problemas**:

1. **Simetría**. El gradiente `∂L_accel/∂Δpred = 2·(Δpred - Δtarget)` es
   el *error sobre la derivada* — penaliza con magnitud proporcional al
   gap en cualquiera de las dos direcciones:

   | Caso                                  | Δtarget | Δpred  | Penalidad |
   |---------------------------------------|---------|--------|-----------|
   | Target plano, pred salta              | ≈ 0     | grande | alta      |
   | Target salta, pred plana              | grande  | ≈ 0    | alta      |
   | Target plano, pred plana              | ≈ 0     | ≈ 0    | baja      |
   | Target salta, pred sigue              | grande  | grande | baja      |

2. **Sin hiperparámetro de sensibilidad**. El "β adaptativo" de
   Perona-Malik se convierte aquí en el error sobre la derivada mismo. Un
   solo λ controla cuánto pesa el término.

3. **Información extra "gratis"**. Expandiendo el cuadrado:

   ```
   (Δpred - Δtarget)² = Δpred² - 2·Δpred·Δtarget + Δtarget²
   ```

   El término cruzado `-2·Δpred·Δtarget` *premia* al modelo cuando
   predice una transición en la misma dirección que el target. Es
   supervisión sobre la **aceleración** del joystick — información que el
   modelo single-output ignora.

**Implementación**:

- `models/tcn.py` — `forward(x, horizon=K)` devuelve los últimos `K`
  timesteps en lugar de solo el último. Backward compatible: con `K=1`
  el shape de salida es `[B, output_dim]` exactamente como antes.
- `dataset/emg_dataset.py` — parámetro `horizon`. La ventana de input se
  extiende a `window_length + horizon - 1` para que los `K` outputs vean
  receptive field completo (sin padding artificial). `__getitem__`
  devuelve target `[K, 2]` cuando `K > 1`. El filtrado de tasks
  excluidas se aplica sobre los `K` timesteps del horizonte: si alguno
  cae en una task excluida, la ventana se descarta.
- `train.py` — flags `--horizon K` y `--accel-weight λ`.
  Validaciones: `--accel-weight > 0` requiere `--horizon >= 2`;
  `--smooth-weight > 0` es incompatible con `--horizon > 1`.
- **Eval reporta siempre las métricas sobre el último timestep
  predicho** (lo que se usaría en deployment), independiente de `K`.
  Esto mantiene MAE / RMSE / R² comparables entre runs con horizon
  distinto.

#### 7.6.4 Por qué predecir solo los últimos K (y no toda la ventana)

Predicir la velocidad en cada uno de los 243 timesteps de la ventana de
input es subóptimo:

1. **Outputs con contexto insuficiente**. Solo el último timestep ve RF
   completo. Los anteriores ven progresivamente más zero-padding en su
   contexto, así que el modelo aprende a "rellenar con ruido" en
   esos timesteps — señal que no aporta a deployment.

2. **Marginalidad decreciente**. Para `L_accel` solo necesitamos
   *diferencias* consecutivas. Con `K = 10` hay `K-1 = 9` diferencias
   por sample, suficiente para un estimador estable. Pasar de 9 a 242
   diferencias es más cómputo, mismo prior.

3. **Costo cero de cómputo**. La TCN es fully convolutional, ya computa
   todos los timesteps internamente. Tomar `pred[:, :, -K:]` no agrega
   ops, solo cambia el slice del head.

El único cambio real es extender la ventana de input por `K - 1` samples
para que cada uno de los `K` outputs vea RF completo. Con `K = 10` y
`RF = 243`, input crece de 243 → 252 samples (+ 3.7 %).

#### 7.6.5 Protocolo de búsqueda de hiperparámetros

**Setup base**:

- Datos: `P01_S02 + P01_S03` (mismo placement, plan estándar).
- Features: `rms_norm d_rms_norm rms_slow_norm d_rms_slow_norm`
  (8 canales).
- Arquitectura: `hidden=32`, `kernel=3`, `dilations=[1,3,9,27,81]`,
  `RF=243`.
- Epochs: 30 (suficiente para distinguir runs sin gastar tiempo).
- Split: estratificado, `gap=243`, `val_fraction=0.2`.
- Seed: 42 fijo.

**Eje 1 — `K` (horizon)**. Fijar `λ = 0.5`, barrer
`K ∈ {2, 5, 10, 20}`. Verificar que el gradiente del término accel se
mantiene estable (no domina ni desaparece) y que la pred final es
visiblemente más suave en replay offline.

- `K = 2`: regularización mínima, baseline.
- `K = 10`: punto de partida razonable (≈ 16 ms de horizonte a 605 Hz —
  por debajo de la escala motora del dedo).
- `K = 20`: casi el doble — testear si aporta o satura.

**Eje 2 — `λ` (accel-weight)**. Fijar `K = 10`, barrer
`λ ∈ {0.0, 0.1, 0.5, 1.0, 5.0}`. `λ = 0.0` con `K = 10` es el ablation
crítico — separar la contribución de "supervisión densa en K timesteps"
(viene gratis al cambiar el head) de la contribución de "regularización
sobre la derivada" (viene del término `L_accel`).

**Métricas a reportar** por run (sobre val, último timestep):

- MAE, RMSE, R, R² (estándar).
- **MAE-task**: desglose por task. Importa que `holds_intermediate` y
  `microcorrections` no se vuelvan catastróficos (ver §10.7).
- **Smoothness empírica**: `mean(|Δpred|)` sobre val. Debería bajar
  monotónicamente con `λ`. Es la métrica que motivó la regularización.
- **Lag inducido**: chequear que la pred no se "atrase" respecto al target
  en transiciones rápidas (`fast_flex` / `fast_ext`). El término accel
  podría inducir conservadurismo.

**Criterio de selección**: el `(K, λ)` que minimice
`MAE + γ · mean(|Δpred|)` con `γ` calibrado para que un controlador
con esa pred sea visualmente estable en replay offline (ver §11.1).

**Comando de referencia**:

```bash
python train.py --csv data/raw/P01/P01_S02_*.csv data/raw/P01/P01_S03_*.csv \
                --features rms_norm d_rms_norm rms_slow_norm d_rms_slow_norm \
                --horizon 10 --accel-weight 0.5 \
                --epochs 30 --batch-size 256 \
                --train-stride 4 --val-stride 4 \
                --run-name tcn_s02s03_K10_l0.5
```

---

## 8. Métricas

Computadas sobre las predicciones de val al final de cada epoch:

- **Loss** (Huber): la misma usada para train, comparable cross-epoch.
- **MAE** = mean(|pred − target|), unidades de target normalizado.
- **RMSE** = sqrt(mean((pred − target)²)).
- **Pearson R** = corr(pred, target).
- **R²** = 1 − SS_res / SS_tot.

Baseline de control: predecir la **media del target en train** (constante).
Da MAE ≈ MAD, R = 0, R² = 0.

---

## 9. Resultados — sesión 1 (P01, placement A)

### 9.1 Split temporal (sanity, primera corrida)

Val 100 % compuesto por bloques no vistos en train (último 20 % cronológico
= `noise_other_fingers` + `noise_wrist_grip` + `free_controlled`).

| epoch best | val Huber | val MAE | val RMSE | R     | R²    |
|------------|-----------|---------|----------|-------|-------|
| 1          | 0.0639    | 0.240   | 0.361    | +0.40 | +0.13 |

Síntoma: train sigue bajando, val plateau. Diagnóstico: el modelo no había
visto esos bloques en absoluto.

### 9.2 Split estratificado (default, recomendado)

Cada bloque contribuye su 20 % final al val.

Resultados con 20 epochs:

| epoch best | val Huber | val MAE | val RMSE | R     | R²    |
|------------|-----------|---------|----------|-------|-------|
| 19         | 0.0271    | 0.152   | 0.233    | +0.647 | +0.418 |

Resultados con 50 epochs (corrida extendida):

| epoch best | val Huber | val MAE | val RMSE | R     | R²    |
|------------|-----------|---------|----------|-------|-------|
| **36**     | **0.0269** | **0.149** | **0.232** | **+0.657** | **+0.423** |

Train: 0.047 → 0.023. Val: 0.036 → 0.027. **Train sigue bajando lentamente,
val plateau marcado desde el epoch ~12.** Mejora marginal entre 20 y 50 epochs
(ΔR² = +0.005, ΔHuber = −0.0002). A partir de epoch ~30 train < val:
overfitting muy leve.

**Diagnóstico**: el modelo está capacity-saturated en datos, no en parámetros.
Una sola sesión limita lo aprendible. El plateau es informativo: dice que
con 1 sesión + 13k params + features actuales, el ceiling efectivo de la
TCN está en R² ≈ 0.42.

### 9.3 Distribución del val estratificado

| task                 | windows en val |
|----------------------|----------------|
| holds_intermediate   | 4,597          |
| slow_cycles          | 4,591          |
| slow_ext             | 3,746          |
| microcorrections     | 3,708          |
| slow_flex            | 3,589          |
| noise_other_fingers  | 2,965          |
| noise_wrist_grip     | 2,931          |
| rest_baseline        | 2,802          |
| fast_ext             | 2,745          |
| fast_flex            | 2,711          |
| free_controlled      | 2,073          |
| **total**            | **36,458**     |

### 9.4 Multi-sesión con placement consistente (S02 + S03)

Tras archivar S01 (placement distinto, ver § 10.X), se reentrena con S02+S03
sobre el mismo placement.

| | S02 sola (50 ep) | S02+S03 (50 ep) | **S02+S03 (100 ep)** |
|---|---|---|---|
| Best epoch | 26 | 47 | **86** |
| Val Huber | 0.0223 | 0.0220 | **0.0217** |
| Val MAE | 0.131 | 0.128 | **0.123** |
| Val RMSE | 0.211 | 0.210 | **0.208** |
| Val R | +0.724 | +0.777 | **+0.783** |
| Val R² | 0.517 | 0.604 | **0.611** |

Plateau marcado a partir del epoch ~50. Más data (S02→S02+S03) sí ayuda
con placement consistente, pero los rendimientos son decrecientes y el
gain principal viene de la primera sesión, no de las subsiguientes.

### 9.4-bis Sweep de features (s02+s03, 100 epochs)

Manteniendo arquitectura baseline (k=3, dilations=[1,3,9,27,81], hidden=32),
se prueban variantes de feature set para identificar si más features
extraen más info de los datos.

| feature set | n features | Best Huber | R | R² | Δ vs baseline |
|---|---|---|---|---|---|
| **Baseline** (`rms_norm`, `d_rms_norm`) | 4 | 0.0217 | +0.783 | 0.611 | — |
| `+ rms_slow_norm, d_rms_slow_norm` | 8 | **0.0213** | **+0.786** | **0.616** | **+0.005** ← campeón |
| `+ rms_slow + ssc_norm` | 10 | 0.0221 | +0.777 | 0.602 | −0.009 |

**Observaciones**:

- **rms_slow** (150 ms window) suma un poquito. Correlación 0.81 con rms (40 ms) pero su derivada `d_rms_slow_norm` es casi ortogonal a `d_rms_norm` (corr 0.24-0.43) y es lo que aporta.
- **SSC** (slope sign changes con threshold relativo) **no ayudó**, incluso empeoró ligeramente. A pesar de tener baja correlación con RMS (0.22-0.55), la información espectral que captura no parece útil para regresión de velocidad continua. Esto contradice la intuición clásica del set Hudgins, pero ese set fue diseñado para **clasificación de gestos**, no regresión continua.
- Train loss sigue mejorando con más features (0.019 → 0.018) pero val no — síntoma de overfitting silencioso. Más features = más capacidad para memorizar.

**Implicación**: con 2 canales EMG el cap efectivo es ~R²=0.61. Las siguientes mejoras requieren más canales (ADS1256, post-migración a ESP32) o un controlador aprendido que estructure mejor la salida temporal.

> **Nota retrospectiva (§10.8)**: ese "cap" resultó ser artefacto del
> `sigma_ms=150` con que se construía el target. Bajándolo a 50 ms el
> mismo modelo con los mismos 2 canales y 8 features sube a R²=0.699
> en val (+0.083) sin tocar arquitectura. Y un oracle ensemble σ=50
> + σ=150 alcanza R²=0.734. El cap de 0.61 era el cap del **target**,
> no del modelo ni de las features.

### 9.5 Sweep arquitectónico (s02+s03, 100 epochs cada uno)

Manteniendo los 4 features actuales, se prueban variantes de geometría
del modelo para identificar si el bottleneck es la arquitectura.

| Config | k | dilations | hidden | RF (ms) | params | Best Huber | Best epoch |
|---|---|---|---|---|---|---|---|
| **Baseline** | 3 | [1,3,9,27,81] | 32 | 402 | 13k | **0.0217** | 86 |
| Wider (h=64) | 3 | [1,3,9,27,81] | 64 | 402 | 50k | 0.0225 | 80 |
| k=5 + RF corto | 5 | [1,2,4,8,16] | 32 | 207 | 21k | 0.0312 | **4** |
| k=5 + RF similar | 5 | [1,2,5,15,45] | 32 | 451 | 21k | 0.0236 | 51 |

**Conclusiones del sweep**:

1. **RF importa más que la densidad de sampling**. `k=5 + dilataciones
   chicas` colapsa con RF=207 ms (best epoch 4 → overfit inmediato). El
   modelo necesita ~300-400 ms para capturar la dinámica EMG → velocidad,
   probablemente para integrar el delay electromecánico (~50-100 ms) +
   contexto de actividad pre-movimiento.

2. **Densidad local NO mejora si se mantiene el RF**. `k=5 + dilataciones
   intermedias` con mismo RF (451 ms) dió **peor** (0.0236 vs 0.0217). El
   sampling esparcido del baseline ([1,3,9,27,81]) tilea el RF sin
   redundancia y resulta más eficiente. Las features ya cargan info local
   densa (RMS / EMA τ=1s), así que sampling esparcido a nivel TCN no
   pierde nada.

3. **Capacidad no es el bottleneck**. `hidden=64` (4× params) no ayudó
   (-3.7%). El modelo con 13k params ya satura lo que se puede extraer
   de las features actuales.

**Implicación**: el bottleneck es **la información que entra al modelo**,
no la arquitectura. Las siguientes mejoras vienen de (a) más features
EMG (WL, ZC, band-power), (b) más canales (4 con ADS1256 cuando llegue),
no de modificar el TCN. El baseline (k=3, dilations=[1,3,9,27,81],
hidden=32) queda fijado como la geometría apropiada para este set de
features y este hardware.

### 9.6 Análisis por task (50-epoch checkpoint, sesión completa)

Inferencia causal de una pasada sobre los 749 k samples, métricas por task
(excluyendo `transition`):

| task                 | n       | MAE    | RMSE   | R       | R²      |
|----------------------|---------|--------|--------|---------|---------|
| fast_flex            | 54,216  | 0.139  | 0.198  | +0.874  | +0.754  |
| fast_ext             | 54,907  | 0.140  | 0.201  | +0.846  | +0.713  |
| free_controlled      | 41,476  | 0.342  | 0.439  | +0.818  | +0.642  |
| slow_flex            | 71,794  | 0.150  | 0.205  | +0.724  | +0.514  |
| slow_cycles          | 91,826  | 0.209  | 0.276  | +0.716  | +0.487  |
| slow_ext             | 74,916  | 0.173  | 0.252  | +0.649  | +0.405  |
| microcorrections     | 74,163  | 0.134  | 0.188  | +0.416  | +0.090  |
| holds_intermediate   | 91,925  | 0.090  | 0.135  | +0.443  | +0.017  |
| noise_wrist_grip     | 58,619  | 0.060  | 0.117  | +0.212  | **−2.97**   |
| rest_baseline        | 56,023  | 0.020  | 0.035  | +0.051  | **−45,493** |
| noise_other_fingers  | 59,296  | 0.033  | 0.064  | +0.005  | **−103,031**|
| **ALL**              | **729,161** | **0.133** | **0.211** | **+0.740** | **+0.544** |

Tres grupos de comportamiento:

1. **Movimientos voluntarios → buen decoding** (R² 0.4–0.75). Las contracciones
   rápidas son las mejor decodificadas (R=0.87 en `fast_flex`). Los movimientos
   lentos son más difíciles porque el EMG asociado es de menor amplitud.

2. **Holds y microcorrecciones**: MAE pequeño (0.09 y 0.13) pero R² ≈ 0. El
   modelo predice valores cercanos al target pero no captura la varianza
   pequeña residual. Esperable: las features tienen ruido del orden de la
   señal en estos rangos.

3. **Bloques de "rest" o "negativos"** (`rest_baseline`, `noise_wrist_grip`,
   `noise_other_fingers`): R² **catastróficamente negativo**, pero MAE
   pequeñísimo (0.02–0.06). Esto NO es un fallo del modelo — es R² mal
   condicionado cuando `var(target) → 0`. El target verdadero es prácticamente
   constante en 0 en estos bloques; cualquier error mínimo del modelo da
   `SS_res / SS_tot → ∞`. El comportamiento real es bueno: **el modelo se
   queda cerca de 0** cuando no hay movimiento de índice. La métrica
   confiable para estos bloques es MAE en unidades absolutas.

Conclusión: el modelo separa correctamente "moverse vs. no moverse" y
decodifica bien las contracciones rápidas y libres. Los puntos débiles son
las microcorrecciones (probablemente requieren features de banda más alta o
más data) y la velocidad fina durante holds.

---

## 10. Gotchas y decisiones de implementación

### 10.1 MPS + WeightNorm produce NaN

PyTorch 2.12 sobre MPS (Apple Silicon) genera gradientes NaN dentro de los
primeros batches cuando `WeightNorm` está activo sobre las `Conv1d` causales.
Reproducible con `--device mps` y `--no-weight-norm` desactivado. En CPU el
mismo modelo entrena correctamente. **Workaround actual**: training se hace
con `--no-weight-norm` en MPS (`use_weight_norm=False` en `EMGTCN`). Sin WN
sobre MPS: ~23k samples/s, ~7 s/epoch. CPU con WN: ~1k samples/s, ~165 s/epoch
— 23× más lento.

### 10.2 Sampling rate por debajo del nominal

Diseño asumía 800 Hz; en la práctica el Uno entrega 605 Hz durante la
sesión (la I/O al disco bloquea). El pipeline no asume `fs` — estima por
sesión desde `t_micros`. El RF en milisegundos varía con `fs`; el receptive
field en samples (243) es lo que importa para el modelo.

### 10.3 Clipping mecánico de los joysticks

`y_top` llega a 0 en extensión máxima, `y_bottom` llega a 1023 en flexión
máxima. **No se enmascara** el clipping mecánico. Refleja la biomecánica
real del dedo (también tiene topes anatómicos) y se compensará en post-
procesamiento si la salida del controlador lo requiere.

### 10.4 Ganancia EMG asimétrica

El flexor satura el ADC (= 1023) durante contracciones máximas (visible en
dry-run con fuerza voluntaria máxima). Durante el protocolo natural NO
satura. Si en sesiones futuras el flexor satura > 0 % en los bloques del
protocolo, considerar reducir gain por hardware o recolocar el electrodo.

### 10.5 Per-feature derivative computation

`d(RMS)/dt` se calcula sobre `RMS_norm` (después de la EMA), NO sobre RMS
crudo. Razón: mantiene las 4 features en la misma escala numérica. Si se
usara RMS crudo, las derivadas estarían en unidades ADC²/s y dominarían
numéricamente sobre las features normalizadas.

### 10.6 Convención de signo del target

`joy = (-y_top + 500) + (-y_bottom + 550)`. Convención: extensión positiva,
flexión negativa, rest ≈ 0. Velocidad heredada: extender el dedo → velocity
> 0, flexionar → velocity < 0.

### 10.7 Ambigüedad mecánica del joystick a resorte en holds y microcorrecciones

**Observación empírica.** Los bloques `holds_intermediate` y
`microcorrections` decodifican consistentemente peor que el resto
(R² ≤ 0.45 vs R² ≥ 0.70 en `slow_*`/`fast_*`), independientemente del
modelo o set de features que se use. Esto NO es un fallo del modelo: es
una limitación intrínseca del setup mecánico con joystick a resorte.

**Explicación causal.** Durante un hold a posición intermedia o una
microcorrección, hay un equilibrio mecánico:

- El sujeto aplica una fuerza muscular F_músculo (visible en la EMG).
- El resorte del joystick aplica F_resorte = k · (pos − rest) en sentido
  opuesto.
- En equilibrio: F_músculo = F_resorte → velocidad observada = 0.

Esto significa que **la misma EMG sostenida puede corresponder a velocidad
0 (durante un hold) o a movimiento real contra otro resorte de menor
stiffness**. Sin medir k explícitamente, el mapeo EMG → velocidad es
estructuralmente ambiguo durante posiciones sostenidas.

**Evidencia cuantitativa.** Fracción de tiempo que el sujeto pasa en
posición ≠ reposo (= bajo la ambigüedad) por task (P01, S02):

| task | % en posición ≠ 0 | R² típico |
|------|-------------------|-----------|
| rest_baseline | 0 % | (var ≈ 0) |
| fast_flex | 17 % | 0.77 |
| fast_ext | 31 % | 0.77 |
| slow_ext | 55 % | 0.76 |
| slow_flex | 59 % | 0.72 |
| free_controlled | 69 % | 0.77 |
| **holds_intermediate** | **80 %** | **0.45** |
| **microcorrections** | **82 %** | **0.30** |

Correlación clara entre fracción bajo ambigüedad y dificultad de decoding.

**Por qué multi-task no rescató estos bloques.** Se probó multi-task con un
head auxiliar de posición (`alpha_pos = 0.25, 0.5, 0.75, 1.0`) sobre la
hipótesis de que el head de posición daría gradiente correcto donde el de
velocidad miente. Resultado: la TCN con RF = 243 samples (~400 ms) ya
aprende implícitamente a manejar el clipping mecánico de los joysticks
porque ve la historia que llevó al tope dentro de la ventana. El head
explícito de posición no aporta información que el backbone no infiera ya.
Detalle del experimento en sección 9.

**Experimento de exclusión (2026-05-28).** Se evaluó excluir holds y micro
del training. Resultado:

| | Con holds/micro (`slowrms`) | Sin holds/micro (`noholds`) |
|---|---|---|
| Val R² (con cambio de val) | 0.616 | 0.648 |
| R² sobre movements en S02 | 0.755 | 0.756 |
| R² sobre movements en S03 | 0.770 | 0.778 |
| R² sobre holds_intermediate S02 | 0.452 | **−0.308** |
| R² sobre microcorrections S02 | 0.299 | **−0.543** |

**Decisión revisada (2026-05-29).** Se mantiene el protocolo y el default
de training originales (holds y micro INCLUIDOS). Razonamiento:

1. La mejora real en movimientos es solo **+0.005 R² promedio**.
2. La degradación en holds y micro es **catastrófica** (R² pasa de
   ~0.30-0.45 a ~−0.30 a −0.54). El modelo se vuelve inutilizable en
   esas condiciones.
3. La hipótesis de la ambigüedad mecánica sigue siendo válida pero el
   problema **no es necesariamente irresoluble**: con suficiente data,
   un joystick de mejor resolución (menos dead zone, sin hysteresis),
   y RF=400 ms, el modelo podría aprender a distinguir hold vs movement
   por la historia temporal y la amplitud sostenida.
4. Para deployment (controlar un dedo prostético) **los holds son
   funcionalmente críticos** — el usuario va a querer sostener posiciones.

**Configurabilidad mantenida.** El parámetro `exclude_tasks` de
`EMGSessionDataset` sigue aceptando cualquier lista; el modelo `noholds`
queda como prueba de concepto archivada. Cuando se cambie el joystick por
un sensor sin resorte (idealmente flex sensor sobre el dedo), volver a
re-evaluar si excluir tiene sentido.

**Solución estructural.** Reemplazar el joystick a resorte por un sensor
de posición sin retorno mecánico — un **flex sensor sobre el dedo** o un
deslizante sin recuperación — elimina la ambigüedad de raíz. Es la
mejora más impactante de hardware después de los 4 canales del ADS1256.

### 10.8 Calibración del `sigma_ms` del smoothing del target

**Resumen**: el `sigma_ms=150` con que se filtraba la posición del
joystick antes de derivar (§4.2) estaba aplastando los picos rápidos del
target. Era el cap principal del R² alcanzable por el modelo, no la
arquitectura ni las features.

#### Observación que disparó el experimento

Después de probar regularización Sobolev (§7.6.3) con `λ ∈ {0, 0.1, 0.5,
1.0, 2.0, 5.0}` y arquitecturas variadas, **todos los modelos convergían
a R² ≈ 0.61** sobre val S02+S03 sin importar el λ ni los epochs. Las
trayectorias de loss eran casi idénticas. Eso sugería un cap estructural
del **target**, no del modelo.

#### Hipótesis

El smoothing gaussiano con `σ=150ms` colapsa picos cortos (< 100 ms)
hasta hacerlos indistinguibles de movimientos lentos. Verificado con el
script `analyze_target_sigma.py` (ver §Apéndice — archivos clave) que
reconstruye la velocidad con varios `σ` y mide std por task.

#### Resultado del experimento de stats

Sobre `P01_S03_20260525_194322.csv`, std de la velocidad por task y `σ`:

| task                  | σ=10  | σ=30  | σ=50  | σ=100 | σ=150 (era default) |
|-----------------------|-------|-------|-------|-------|---------------------|
| fast_flex             | 2404  | 1490  | 1161  | 726   | **470**             |
| slow_flex             | 1443  | 1073  | 890   | 679   | **575**             |
| slow_cycles           | 1425  | 1143  | 980   | 767   | **659**             |
| holds_intermediate    | 869   | 657   | 554   | 430   | **365**             |
| rest_baseline         | 10.97 | 2.25  | 1.08  | 0.42  | **0.25**            |

> Con `σ=150ms`, **`fast_flex` tiene `std=470` mientras `slow_flex` tiene
> `std=575`**. El target le está diciendo al modelo que los movimientos
> rápidos tienen menos amplitud que los lentos — lo cual es falso, los
> picos rápidos son los más altos. El gaussiano ancho los atenúa más por
> tener menor duración temporal. Esto es un sesgo estructural del target.

> Pasar de `σ=10` a `σ=150` reduce los picos de `fast_flex` por **5×**
> para reducir el ruido en `rest` por **44×**. El tradeoff está
> calibrado en contra de los movimientos, que son lo que queremos
> decodificar.

#### Sweep de entrenamiento con distintos σ

Mismo setup que el champion previo `slowrms` (single-task K=1, sin
regularización, 8 features `rms+d_rms+rms_slow+d_rms_slow`, seed=42,
batch=256, stride=4), sobre S02+S03:

| Run                                | epochs | best ep | val R²   | Pearson R | val Huber |
|------------------------------------|--------|---------|----------|-----------|-----------|
| baseline `σ=150` (`slowrms`)       | 100    | 100     | 0.616    | 0.776     | 0.0217    |
| `σ=10`                             | 200    | 129     | 0.384    | 0.621     | 0.0086    |
| `σ=30`                             | 100    | 90      | 0.655    | 0.810     | 0.0079    |
| `σ=30`                             | 200    | 168     | 0.663    | 0.815     | 0.0078    |
| `σ=30` + Sobolev λ=1 (`K=10`)      | 200    | 173     | 0.672    | 0.820     | 0.0075    |
| **`σ=50`** ← champion              | 200    | 197     | **0.699**| **0.837** | 0.0096    |

**Interpretación**:

- **`σ=50` supera al baseline `σ=150` en R² por +0.083 y en Pearson por
  +0.061** (con la misma arquitectura, mismas features). Mejora del 13 %
  relativo en R² solo cambiando el suavizado del target.
- **`σ=10` plateauea en R² ≈ 0.38** desde el epoch 30 — la trayectoria
  es 0.298 (ep10) → 0.361 (ep30) → 0.367 (ep100) → 0.354 (ep200). No
  sube con más entrenamiento. La alta frecuencia que `σ=10` preserva
  **es ruido del joystick irreducible desde EMG**.
- **`σ=30` se queda corto en tasks lentas** porque retiene ruido de
  joystick durante movimientos suaves que el EMG no puede explicar.
- **`σ=50` es el sweet spot real**: balance entre preservar picos
  rápidos y suavizar el ruido en regiones lentas.
- **Sobolev λ=1 con σ=30 ayuda marginalmente** (R² 0.663 → 0.672) pero
  no resuelve el tradeoff fast/slow. **El smoothing del target fue el
  factor dominante**, la regularización es secundaria.
- Las tres mejores trayectorias plateauean entre epoch 80-100; las
  mejoras posteriores (ep 100→200) suman +0.008-0.011 en R² —
  marginales, posible ruido del val.

#### Tradeoff por task — sweet spot (R² sobre VAL)

Sweep refinado `σ ∈ {30, 40, 50, 75, 100, 150}`. R² por task sobre el
val set (last 20 % de cada bloque, estratificado, no incluye train):

| task                | σ=30  | σ=40  | **σ=50** | σ=75  | σ=100 | σ=150 |
|---------------------|-------|-------|----------|-------|-------|-------|
| fast_flex           | 0.808 | 0.852 | **0.892**| 0.863 | 0.839 | 0.729 |
| fast_ext            | 0.622 | 0.669 | **0.677**| 0.594 | 0.535 | 0.407 |
| free_controlled     | 0.742 | 0.789 | **0.797**| 0.740 | 0.647 | 0.522 |
| microcorrections    | 0.415 | 0.420 | **0.465**| 0.416 | 0.376 | 0.252 |
| slow_flex           | 0.522 | 0.520 | 0.634    | 0.602 | 0.666 | **0.780** |
| slow_ext            | 0.501 | 0.555 | 0.674    | 0.668 | 0.677 | **0.783** |
| slow_cycles         | 0.340 | 0.403 | 0.565    | 0.578 | 0.630 | **0.762** |
| holds_intermediate  | 0.139 | 0.190 | 0.310    | 0.207 | 0.254 | **0.419** |
| **R² agregado**     | 0.629 | 0.661 | **0.699**| 0.642 | 0.607 | 0.616 |
| **Pearson R**       | 0.794 | 0.814 | **0.837**| 0.801 | 0.781 | 0.786 |

> **Nota importante de comparación**: estos números son sobre **val
> only** (best_val_preds.npz, no incluye train). Las tablas de
> versiones previas de este doc usaban predicciones sobre el CSV
> completo (train + val), lo cual inflaba el R² por +0.05 a +0.15
> según task. Los números reportados aquí son los honestos.

**Patrón claro: dos curvas opuestas según régimen**:

- **Fast / micro / free**: curva cóncava con pico en σ=50. Decae
  rápido por encima (σ≥75) y más suave por debajo (σ=30/40).
- **Slow / holds**: curva monotónicamente creciente con σ. **σ=150
  sigue siendo el mejor** para slow_*, σ=100 segundo, σ=50 tercero.

**`σ=50` es el champion único** pero **NO es el óptimo universal**:
gana en agregado porque equilibra los dos regímenes, pero para slow
tasks específicamente σ=100-150 le ganan por márgenes amplios. Esto
motiva la idea del multi-target / oracle ensemble (siguiente
subsección).

#### Por qué `σ=50` funciona mejor que `σ=30` en slow / holds

El target con `σ=30` retiene ruido de joystick durante movimientos
suaves (jitter de quantización del ADC, micro-temblores del dedo) que
**no es predecible desde EMG**. `σ=50` filtra ese ruido sin destruir
los picos rápidos. Es el balance entre detalle preservable y ruido
filtrable.

#### Hipótesis arquitectónicas — descartadas

Para entender el tradeoff residual en `slow_*` / `holds` vs el baseline
`σ=150`, se testearon dos hipótesis arquitectónicas. **Ambas
descartadas**:

##### a) Capacity del backbone (`hidden_channels`)

Sweep `hidden ∈ {64, 128}` con `σ=50` y 100 epochs sobre S02+S03:

| Run                              | best ep | val R²    | Pearson R |
|----------------------------------|---------|-----------|-----------|
| sig050 `hidden=32` 100ep         | 95      | 0.691     | 0.832     |
| sig050 `hidden=32` 200ep (champ) | 197     | **0.699** | **0.837** |
| sig050 `hidden=64` 100ep         | 57      | 0.694     | 0.834     |
| sig050 `hidden=128` (cortado)    | ~41     | 0.684     | 0.828     |

Más capacity **no ayuda**. `hidden=64` empata al baseline; `hidden=128`
overfitea más rápido (train loss baja a 0.0046 vs val ~0.010, gap 2×)
y plateauea **peor**. Las 32 channels son suficientes para representar
la info del target σ=50.

##### b) Capacity temporal (receptive field)

Run con dilations `[1, 3, 9, 27, 81, 162]` (RF = 567 samples ≈ 935 ms,
vs 243 / 400 ms del baseline; arquitectura extendida a 3 bloques
residuales pareados, sin conv final separada). `σ=50`, 100 epochs.

| Run                            | best ep | val R²    | Pearson R |
|--------------------------------|---------|-----------|-----------|
| sig050 RF=243 200ep (champion) | 197     | 0.699     | 0.837     |
| **sig050 RF=567 100ep**        | 68      | **0.704** | 0.840     |

Mejora **marginal en agregado** (+0.005 R²). Pero por task la mejora
**no sigue el patrón predicho** (si la hipótesis fuera correcta, slow
debería mejorar y fast quedarse igual):

| task                | RF=243 | RF=567 | Δ      |
|---------------------|--------|--------|--------|
| slow_flex           | 0.844  | 0.867  | +0.023 |
| slow_ext            | 0.835  | 0.829  | -0.005 |
| slow_cycles         | 0.791  | 0.792  | +0.001 |
| holds_intermediate  | 0.657  | 0.628  | **-0.029** |
| fast_flex           | 0.965  | 0.961  | -0.004 |
| fast_ext            | 0.904  | 0.911  | +0.007 |

Las diferencias son ruido del val, no señal estructural. **El RF
adicional acelera la convergencia** (alcanza 0.70 en 68 epochs vs 197
del baseline) **pero no desbloquea info nueva**. El bottleneck residual
**no es temporal**.

#### Oracle ensemble σ=50 + σ=150 (sobre VAL)

Dado que `σ=50` y `σ=150` se especializan en regímenes **distintos y
complementarios** (rápidos vs suaves), se computó el **oracle
ensemble**: para cada task, tomar las predicciones del modelo con mejor
R² en esa task.

Especialización observada (val):

| Régimen           | Best model | Tasks                                  |
|-------------------|------------|----------------------------------------|
| Movimiento rápido | **σ=50**   | fast_flex, fast_ext, free, micro       |
| Movimiento suave  | **σ=150**  | slow_flex, slow_ext, slow_cycles, holds |

Resultado del oracle agregado sobre val (todos los sigmas disponibles
en el sweep):

| Modelo                       | Pearson R  | R² agregado |
|------------------------------|------------|-------------|
| baseline σ=150 100ep         | 0.786      | 0.616       |
| sig050 200ep (champion)      | 0.837      | 0.699       |
| **Oracle (mejor por task)**  | **0.857**  | **0.734**   |

**+0.035 R² disponibles** si pudiéramos seleccionar el modelo correcto
por régimen. Es modesto en absoluto pero **cierra ~14 % del gap entre
el champion y R²=1.0**. Más relevante: confirma que el gap residual no
es del modelo en sí sino de su **single-target training** — cada `σ`
aprende correctamente su régimen, solo falta combinarlos.

#### Próximos pasos identificados

1. **Refinar σ ∈ {40, 60, 75, 100}** (en curso): confirmar que 50 es el
   óptimo y mapear bien la curva R² vs σ. Resultados parciales
   pendientes.
2. **Multi-target loss** (más prometedor): una sola TCN con **dos heads**
   — una entrenada contra `target σ=150`, otra contra `target σ=50` —
   compartiendo backbone. En deployment, combinar las dos preds con un
   peso (heurístico o aprendido). Es el camino más directo para
   capturar el oracle.
3. **Gating heurístico**: usar `|RMS reciente|` como detector de
   régimen. Activación muscular alta → usar pred de σ=50; reposo →
   usar pred de σ=150. Aproximación rápida sin entrenar gating
   adicional.
4. **Loss adaptativo por timestep**: peso del Huber proporcional a la
   actividad muscular (`sigmoid` sobre `|target_smooth|`). Alternativa
   al multi-target — quita presión del slow sin perder fast en un
   único modelo.

**Lo que NO mueve la aguja**: capacity del backbone (hidden_channels),
capacity temporal (RF), regularización Sobolev. Estos quedan
descartados como bottleneck principal.

> **Implementación**: `--sigma-ms` es flag CLI de `train.py`. Default
> revertido a 50 ms tras este experimento. Compatibilidad: checkpoints
> antiguos (sin `sigma_ms` en config) se asume `σ=150` para
> reconstrucción del target en `visualize.py` y `analyze.py`.

---

## 11. Visualización (offline replay y plan online)

### 11.1 Replay offline

`visualize.py` reproduce una sesión grabada con la predicción superpuesta.

Layout: una cajita vertical que representa el rango de movimiento del dedo
(percentil 1–99 de `joy`, con margen). Línea horizontal en y=0 marca el
reposo. Dos círculos:

- **Verde (`real`)**: posición del dedo desde los joysticks (`ds.joy`).
- **Rojo (`predicho`)**: integración de la velocidad predicha por el TCN
  vía leaky integrator:

  ```
  pos[t] = pos[t-1] * (1 - leak) + pred_vel[t] * dt
  leak  = dt / leak_tau     (default leak_tau = 5 s)
  ```

El integrador con leak tiende a 0 (reposo) con time constant `leak_tau`.
Necesario porque el controlador no está implementado todavía y la
integración pura acumula error sin cota.

Comando típico:
```
python visualize.py --checkpoint checkpoints/<run>/best.pt \
                    --csv data/raw/<subject>/<session>.csv \
                    --speed 2 --fps 30
```

Flags relevantes: `--start <s>` y `--duration <s>` para zoom temporal,
`--leak-tau <s>` para tunear el anclaje (más chico = mejor en rest pero
peor en holds), `--save <archivo.gif|mp4>` para exportar.

### 11.2 Arquitectura para online (pendiente de implementar)

El mismo render (cajita + un círculo) se reusa. Cambios necesarios:

1. **`OnlinePredictor` con filtros stateful**:
   - Bandpass IIR: usar `scipy.signal.sosfilt_zi` y pasar `zi` actualizado
     entre llamadas (una muestra por vez).
   - RMS rolling: buffer circular de `win_samples`.
   - EMA: estado escalar `ema_t = alpha * x + (1 - alpha) * ema_{t-1}`.
   - Buffer de features circular de `window_length=243` samples, 4 canales.
2. **Loop**: lectura serial → `OnlinePredictor.update(flex, ext)` → predicción
   escalar → leaky integrator → actualizar `pred_circle.center`.
3. Sin círculo verde (no hay ground truth online).
4. Para no bloquear el render con la lectura serial, dos threads + cola
   thread-safe, o un solo loop con `select()` no-bloqueante.

## 12. Referencias bibliográficas

### TCN — fundacionales

- **Bai, Kolter & Koltun (2018)**. *An Empirical Evaluation of Generic Convolutional and Recurrent Networks for Sequence Modeling*. arXiv:1803.01271.
- **van den Oord et al. (2016)**. *WaveNet: A Generative Model for Raw Audio*. arXiv:1609.03499.

### EMG clásico — features y pattern recognition

- **Hudgins, Parker & Scott (1993)**. *A new strategy for multifunction myoelectric control*. IEEE TBME 40(1).
- **Englehart & Hudgins (2003)**. *A robust, real-time control scheme for multifunction myoelectric control*. IEEE TBME 50(7).
- **Phinyomark, Phukpattaranont & Limsakul (2012)**. *Feature reduction and selection for EMG signal classification*. Expert Systems with Applications 39(8).

### Deep learning sobre EMG (clasificación)

- **Atzori, Cognolato & Müller (2016)**. *Deep learning with convolutional neural networks applied to electromyography data*. Frontiers in Neurorobotics.
- **Geng et al. (2016)**. *Gesture recognition by instantaneous surface EMG images*. Scientific Reports 6.
- **Côté-Allard et al. (2019)**. *Deep learning for electromyographic hand gesture signal classification using transfer learning*. IEEE TNSRE 27(4).

### TCN específicamente para EMG

- **Tsinganos et al. (2019)**. *Improved gesture recognition based on sEMG signals and TCN*. ICASSP 2019.
- **Tsinganos et al. (2020)**. *Hilbert sEMG data scanning for hand gesture recognition based on deep learning*. Neural Computing and Applications.

### EMG regresión / decodificación continua

- **Hahne et al. (2014)**. *Linear and nonlinear regression techniques for simultaneous and proportional myoelectric control*. IEEE TNSRE 22(2).
- **Smith, Kuiken & Hargrove (2014)**. *Real-time simultaneous and proportional myoelectric control using intramuscular EMG*. JNE 11(3).
- **Ngeo, Tamei & Shibata (2014)**. *Continuous and simultaneous estimation of finger kinematics using inputs from an EMG-to-muscle activation model*. JNER 11.
- **Quivira et al. (2018)**. *Translating sEMG signals to continuous hand poses using recurrent neural networks*. IEEE BHI.
- **Krasoulis et al. (2017)**. *Improved prosthetic hand control with concurrent use of myoelectric and inertial measurements*. JNER 14.

### EEG / BCI con deep learning

- **Lawhern et al. (2018)**. *EEGNet: a compact convolutional neural network for EEG-based brain–computer interfaces*. J Neural Engineering 15(5).
- **Schirrmeister et al. (2017)**. *Deep learning with convolutional neural networks for EEG decoding and visualization*. Human Brain Mapping 38(11).
- **Roy et al. (2019)**. *Deep learning-based electroencephalography analysis: a systematic review*. J Neural Engineering 16(5).
- **Ingolfsson et al. (2020)**. *EEG-TCNet: An Accurate Temporal Convolutional Network for Embedded Motor-Imagery BMIs*. IEEE SMC.

### Datasets / benchmarks

- **NinaPro** (Atzori et al., múltiples papers 2014-2020). DB1-DB9, EMG multi-sujeto multi-gesto.
- **Capgmyo** (Geng et al.). EMG de alta densidad.

### Reviews

- **Rim, Sung & Min (2020)**. *Deep learning in physiological signal data: A survey*. Sensors 20(4).
- **Phinyomark & Scheme (2018)**. *EMG pattern recognition in the era of big data and deep learning*. Big Data and Cognitive Computing 2(3).

### Regularización por suavidad y supervisión de derivadas (ver §7.6)

- **Rudin, Osher & Fatemi (1992)**. *Nonlinear total variation based noise removal algorithms*. Physica D 60(1-4), 259-268.
  Formulación original del **Total Variation Denoising (TV-L1)** — base teórica del término `Σ |Δpred[t]|` que probamos en la versión legacy (§7.6.1). Diseñado para imágenes con regiones suaves y edges sharp; mal calzado para señales genuinamente suaves como velocidad de dedo.
- **Perona & Malik (1990)**. *Scale-Space and Edge Detection Using Anisotropic Diffusion*. IEEE TPAMI 12(7), 629-639.
  Introduce la **anisotropic diffusion** / **edge-preserving smoothness**, base conceptual del adaptive TV (§7.6.2) — peso de la penalidad varía con la magnitud local de la derivada. Propone dos funciones de conductancia: gaussiana `exp(-(s/K)²)` y Lorentziana `1/(1+(s/K)²)`.
- **Black, Sapiro, Marimont & Heeger (1998)**. *Robust anisotropic diffusion*. IEEE Transactions on Image Processing 7(3), 421-432.
  Reformula Perona-Malik desde la teoría de **M-estimators robustos**. Muestra que la Lorentziana corresponde a un estimador con cola más pesada (más robusto a outliers) que la gaussiana. Primer paper que aplica anisotropic diffusion a optical flow — el análogo en imágenes de "predecir velocidad por píxel".
- **Czarnecki, Osindero, Jaderberg, Świrszcz & Pascanu (2017)**. *Sobolev Training for Neural Networks*. NeurIPS. arXiv:1706.04859.
  Formalización del **entrenamiento con supervisión sobre las derivadas** del target (norma H¹ de Sobolev). Mostraron mejoras en generalización y data efficiency cuando la señal subyacente es suave — caso nuestro. Base teórica de la implementación actual `--accel-weight` (§7.6.3): el término cruzado `-2·Δpred·Δtarget` que aparece al expandir el cuadrado es exactamente supervisión sobre la derivada.

## 13. Próximos pasos

1. **Análisis por task** del modelo entrenado: separar MAE / R² del val por
   `task` para identificar dónde falla (sospecho que `noise_*` se predice
   mejor que `fast_*`).
2. **Sesiones 2–4**: replicar protocolo con mismo placement (`P01-A`) para
   estimar varianza intra-sujeto y mejorar el modelo con más data.
3. **Sesiones 5–9**: variar placement progresivamente. Re-entrenar y medir
   degradación de performance vs. placement.
4. **Sesión 10**: test final, no usada para training.
5. **Posibles mejoras al modelo**:
   - Scheduler de lr (cosine).
   - `hidden_channels` mayor (64) cuando haya más data.
   - Output multi-canal (velocidad + flag de reposo + flag de coactivación).
6. **Migración a ESP32**: cuando suceda, re-estimar `fs`, reentrenar o
   resamplear.

---

## 14. Setup 2-dedos (ESP32+ADS1256) — pipeline y gotchas

A partir de S04 el setup migra a ESP32-S3 + ADS1256 + 4 EXG pills BioAmp para grabar
2 dedos (índice y medio). El protocolo y el pipeline crecen para soportar 2 outputs.

### 14.1 Schema CSV y firmware

Frame binario v2 (31 B): `AA55 + t_micros[4] + 4×emg[4] + 4×joy[2] + crc[1]`.
CSV columns: `pc_timestamp, t_micros, emg0..emg3, joy_ext_ind, joy_flex_ind,
joy_ext_med, joy_flex_med, trial_id, task, session_id, subject_id, placement_id`.

EMG mapping: `emg0=ext_ind, emg1=flx_ind, emg2=ext_med, emg3=flx_med` (single-ended
vs AINCOM, PGA=1, DRATE 15 kSPS). Joysticks 4 ejes via ADC del ESP32 a 12 bits.

`t_micros` es `uint32` del ESP32 → **overflowea cada ~71.5 min**. El dataset
2-dedos detecta saltos negativos en `dt` y aplica `+2^32` desde ahí (unwrap).

### 14.2 Pipeline 2-dedos

- **`dataset/emg_dataset_2fingers.py`** (`EMGSession2FingersDataset`): replica
  `compute_emg_features` para 2 pares y concatena → 16 features (8 por dedo:
  `flx/ext × rms_norm, d_rms_norm, rms_slow_norm, d_rms_slow_norm`).
- **`train_2fingers.py`**: TCN con `in_channels=16, output_dim=2`. Loss = Huber
  sumado sobre los 2 outputs (cada dedo pesa igual). Target normalizado por dedo
  independiente con p99 de `|vel|` (luego harmonizado entre sesiones).
- **`visualize.py`**: detecta el modo por `output_dim` del checkpoint y arma
  layout 2 cajitas side-by-side + 2 paneles de velocidad. Checkbox "show target"
  para toggle del verde.

### 14.3 Bug MPS + `eval()` con esta arquitectura

**Síntoma**: train baja normal (data loss ~0.02), pero al entrar a `model.eval()`
(sin dropout) el head Conv1d 1×1 produce predicciones explosivas (magnitudes 1e15
a 1e23) en **uno** de los 2 outputs (índice o medio, no determinístico). Las
métricas R²/Pearson colapsan a 0 para ese dedo; las del otro dedo siguen siendo
válidas. `val_loss` queda igual EXACTAMENTE entre epochs (no varía con los pesos
entrenados), confirmando que la salida problemática es constante/NaN en eval.

**Confirmaciones**:
- Pasa con L1Loss, Huber delta < 1.0, y a veces con Huber delta = 1.0.
- Pasa con dropout 0.0, 0.05, 0.1, 0.15. NO depende del dropout.
- Pasa con `--device cpu` mucho menos frecuentemente (pero también puede pasar).
- NO se cura con `grad-clip` agresivo (es bug de eval, no de optimización).

**Workaround**: clipear las predicciones a [-5, +5] dentro de `train_epoch` y
`evaluate`:

```python
pred = model(x, horizon=...)
pred = torch.clamp(pred, -5.0, 5.0)
```

Justificación del 5: los targets normalizados están en [-1.8, +1.8] (max real
observado en S06+S07). ±5 es ~3× margen de seguridad. No corta predicciones
válidas, sólo atrapa explosiones numéricas.

### 14.4 Z-normalización robusta de features

Los features procesados (post `rms_norm` y `d_rms_norm`) tienen escalas muy
distintas:
- `rms_norm`: mean ≈ 1, std ≈ 0.7 (clip 20 toca a ~0.1% de samples).
- `d_rms_norm`: mean ≈ -0.2, std ≈ 12 (clip 20 toca **~28% de samples**).
- `rms_slow_norm`: std ≈ 0.5.
- `d_rms_slow_norm`: std ≈ 7.4 (sin outliers std ≈ 2.5 → colas 3× más pesadas
  que normal).

Esa asimetría hace que las derivadas dominen el gradient (escala 20× mayor).
Para corregir: **z-norm por canal con median + MAD** (robust a outliers):

```python
median = np.median(features, axis=1)
mad = np.median(|features − median|, axis=1) * 1.4826  # escala normal-equiv
features_norm = (features − median) / (mad + 1e-6)
```

Stats calculados sobre **train solamente** (no val, evita leak), aplicados a
train + val + inferencia online. Se guardan en `checkpoint["config"]` como
`feature_median` y `feature_mad` (16 floats cada uno). `visualize.py` los carga
del checkpoint y aplica antes de inferencia.

### 14.5 Outliers raw EMG y exclusiones

**`noise_wrist_grip`** está fuera del training para placements donde
`emg0` (ext_ind) está cerca de los músculos pronadores de la muñeca — la
contracción de la muñeca activa fuerte `emg0` (crosstalk), inflando los stats
y creando samples que el modelo intenta predecir como "movimiento del índice".
En placement D quedó claro: `noise_wrist_grip` aportaba 1500+ outliers a `emg0`.

**Artifacts en `rest_baseline`**: si el sujeto acomoda el brazo en los primeros
segundos, emg1/3 (flexores) pueden tener spikes que NO son señal muscular real.
En S07 observamos 899 outliers de `emg1` concentrados en los primeros 45 s.
Solución quirúrgica: borrar esos samples del CSV (preservando `.bak`).

**Identificación**: 3 patrones de outliers en EMG raw:
1. Crosstalk muscular (emg0 durante `noise_wrist_grip`, emg2 durante movimientos
   del medio). **Real, no se elimina**.
2. Artifact de electrodo en silencio (emg1 en `rest_baseline` S07). **Eliminable**.
3. Picos puntuales de impacto/contacto (1-2 samples cada uno). Mitigados por
   median+MAD norm.

### 14.6 Resultados parciales (en progreso)

Con S04+S05 (placements B + C):
- Baseline Huber+Sobolev: avg R² ≈ 0.62 (ep 28).
- + var matching λ=2: avg R² ≈ 0.62 (mejora std_ratio 0.74 → 0.97, R² igual).

Con S06+S07 (placement D) sin z-norm: plateau temprano avg R² ≈ 0.53.

Hipótesis: placement D introdujo más outliers en `emg0` (crosstalk pronadores)
y artifacts en `emg1` (despegue durante rest). Combinado con feature_clip=20
en derivadas (que ya saturaba ~28% antes), el modelo no aprende a distinguir
slow vs fast en el dedo correspondiente. Z-norm robusta + exclusiones + clamp
en pred son el approach actual para limpiar.

---

## Apéndice — comandos de referencia

```bash
# Captura de una sesión
python collect.py -s P01 -n 1 -p P01-A --notes "mcu=arduino_uno"

# Dry-run del stream serial (validar hardware antes)
python check_serial.py

# Training (default: split estratificado, sin WN para MPS)
python train.py --csv data/raw/P01/P01_S01_20260524_181522.csv \
                --epochs 20 --run-name tcn_s01_strat --no-weight-norm

# Análisis por task de un checkpoint
python analyze.py --checkpoint checkpoints/<run>/best.pt \
                  --csv data/raw/<subject>/<session>.csv

# Replay gráfico (offline)
python visualize.py --checkpoint checkpoints/<run>/best.pt \
                    --csv data/raw/<subject>/<session>.csv

# Test del modelo standalone (forward + causalidad)
python models/tcn.py
```

## Apéndice — archivos clave

```
robothand/
├── sketchs/arduino/arduino.ino    # firmware: emite 5 campos cada loop
├── collect.py                     # entry CLI para captura
├── check_serial.py                # dry-run del serial
├── train.py                       # training loop
├── analyze.py                     # análisis por task de un checkpoint
├── analyze_target_sigma.py        # sweep de sigma_ms del target (§10.8)
├── plot_compare.py                # plot comparativo de N checkpoints
├── visualize.py                   # replay gráfico offline
├── collection/                    # protocolos, runner, serial_io
│   ├── protocols.py               #   vocabulario de tasks + variantes
│   ├── runner.py                  #   ejecuta una sesión + escribe CSV/JSON
│   └── serial_io.py               #   SerialReader + parse_5field
├── dataset/                       # preprocesamiento + Dataset
│   ├── preprocessing.py           #   bandpass, RMS, EMA, smoothing, diffs
│   └── emg_dataset.py             #   EMGSessionDataset
├── models/
│   └── tcn.py                     #   Chomp1d, CausalConv1d, CausalConvBlock, EMGTCN
└── docs/
    └── methods.md                 #   este documento
```
