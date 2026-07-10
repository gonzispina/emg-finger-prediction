# Presentación: Pipeline EMG → Predicción de Velocidad de Dedos

Materia: Análisis de Señales  
Duración estimada: ~20 min + demo en vivo  
Estructura: 10 secciones, ~25 slides

---

## TEASER (apertura, antes del índice)

- **Video corto**: pantalla dividida — cámara del brazo con electrodos + visualize.py en tiempo real
- Duración: 30-60 segundos
- Objetivo: enganchar antes de entrar al contenido técnico
- Asset: grabar sesión con `python visualize.py --checkpoint checkpoints/tcn_s02s03_sig050_200ep/best.pt`

---

## SECCIÓN 1 — Motivación y objetivo (2 slides)

### Slide 1.1 — El proyecto
- Objetivo: mano robótica controlada por EMG propio
- División natural: (1) adquisición + procesamiento + predictor / (2) controlador + mecánica
- Diagrama de bloques del sistema completo
- **[PPT dibujado]**

### Slide 1.2 — Scope
- Este trabajo = Parte 1
- Resaltar la parte cubierta en el diagrama
- **[Foto: brazo con electrodos pegados]** — TODO: pasar del celular

---

## SECCIÓN 2 — Fundamentos fisiológicos (3 slides)

### Slide 2.1 — La señal EMG
- End plate potential (EPP) y su componente aleatoria (MEPP)
- Liberación de acetilcolina, períodos refractarios
- sEMG = suma espaciotemporal de MUAPs
- **[Diagrama: MUAP → suma → sEMG]** — imagen de literatura o PPT

### Slide 2.2 — Anatomía relevante
- Articulación MCP (metacarpofalángica): flexión/extensión
- Músculos involucrados: flexor digitorum superficialis + extensor digitorum (antebrazo)
- Por qué son captables con electrodos de superficie en el antebrazo
- **[Foto/diagrama anatómico]**

### Slide 2.3 — Requisitos de frecuencia de muestreo
- Espectro del EMG: energía entre 20–500 Hz, pico 50–150 Hz
- Nyquist: necesitás ≥1000 Hz para capturar hasta 500 Hz
- Setup v1 (Arduino ~605 Hz) → Nyquist ~300 Hz: captura banda principal
- Setup v2 (ESP32 ~638 Hz): más canales, mayor resolución ADC
- **[Plot: PSD de señal EMG mostrando distribución de energía]** — TODO: generar desde S02 o S03

---

## SECCIÓN 3 — Hardware v1: Arduino + EXG Pills (2 slides)

### Slide 3.1 — Setup inicial
- Arduino Uno: 6 canales ADC, 10 bits (~1024 niveles), ~605 Hz
- 2× EXG Pills (Upside Down Labs): amplificador diferencial para bioseñales
- Configuración crítica: vienen para EEG → reconfiguración de resistencias para EMG
  (menor ganancia, banda más alta)
- **[Foto: Arduino + EXG Pills + joysticks]** — TODO: pasar del celular

### Slide 3.2 — Primera limitación
- 6 canales ADC insuficientes para 2 dedos (mínimo 4: flx/ext × 2 dedos)
- Resolución 10 bits vs dinámica de la señal EMG
- Primer contacto con ruido eléctrico de red (50 Hz + armónicos)
- **[Plot: señal cruda Arduino mostrando ruido]** — `python plot_raw.py <csv_arduino>`

---

## SECCIÓN 4 — Exploración y análisis de señal (6 slides)

### Slide 4.1 — Ruido eléctrico: análisis frecuencial
- FFT de señal cruda: picos fuertes en 50 Hz y armónicos (100, 150, 200 Hz)
- Hipótesis: interferencia EM de la red eléctrica
- Experimento: desenchufar laptop + monitor
- Resultado: armónicos desaparecen, residuo en 50 Hz manejable
- **[Plot: FFT enchufado vs desenchufado]** — TODO: generar con `numpy.fft` sobre CSV con ruido
- Script a crear: `plots/plot_fft_noise.py`

### Slide 4.2 — Decisión: filtro pasa banda (no Notch)
- Opción descartada: Notch 50 Hz → elimina información fisiológica útil en 50 Hz
- Decisión: Butterworth orden 4, pasa banda 70 Hz – ~300 Hz
- Justificación: 50 Hz queda fuera de la banda, preserva pico de energía EMG (50–150 Hz)
- **Distinción clave para online vs offline:**
  - Offline (análisis, onset detection): `filtfilt` → zero-phase, no introduce delay pero NO es causal
  - Online (deployment): `sosfilt` (IIR causal, una pasada) → introduce phase delay propio, aceptable porque features y target comparten el mismo delay (entrenamos pares sincronizados)
- **[Plot: respuesta en frecuencia del filtro]** — `python plots/plot_filter_response.py` (ya creado)
- **[Plot: señal cruda vs filtrada]** — `python plot_raw.py <csv>` (ya muestra raw + filtered)

### Slide 4.3 — Onset detection: técnica y validación de causalidad

**Técnica de detección:**
1. Bandpass causal 70–300 Hz sobre señal EMG cruda
2. Envolvente: Butterworth LP orden 2 **bidireccional** (filtfilt), fc=1Hz para visualización, 2Hz para detección
   - Bidireccional = zero-phase → sin delay artificial → timestamps de onset precisos
   - Tradeoff: solo válida offline (requiere la señal completa)
3. **Umbral local**: para cada movimiento, baseline = mediana de los 300ms justo antes de la ventana de búsqueda + k×std (k=2.0) → robusto a deriva de la señal entre movimientos consecutivos
4. **Ventana de búsqueda**: 600ms antes del onset del joystick (extensión) / 500ms (flexión)
5. **Onset del joystick**: primer cruce sostenido del umbral de posición (k=5×std baseline, mínimo 20 unidades ADC, sostenido 35ms)
6. **Onset EMG**: último cruce sostenido del umbral EMG dentro de la ventana → el más cercano al movimiento
7. **Período refractario**: 200ms entre onsets consecutivos del joystick

- Datasets: S02 + S03, tareas fast_ext y fast_flex (168 onsets válidos de 195 detectados)
- Resultado clave: EMG precede al joystick consistentemente → causalidad confirmada
- **Distribución de delays por dirección (dos Pareto con shape invertido):**
  - Extensión (fast_ext): mediana=263ms, P95=567ms — cola larga, shape bajo
  - Flexión (fast_flex): mediana=250ms, P95=313ms — más concentrada, shape mayor
- **Hallazgo fisiológico**: la asimetría refleja el control motor diferencial
  - Flexión: músculo agonista directo → activación más consistente
  - Extensión: requiere inhibición del flexor + extensor compartido → timing más variable
- **[Plot: onset_overview.png]** — `python plots/plot_onset_analysis.py` ✓
- **[Plot: onset_histogram.png]** — ídem ✓

### Slide 4.4 — Del onset al Receptive Field
- Distribución de delays: masa principal 100–400ms, P95 extensión = 567ms
- El modelo necesita ver al menos ~400ms de historia para capturar la mayoría de los onsets
- RF = 243 samples × (1/605 Hz) ≈ 401ms → dilations [1, 3, 9, 27, 81]
- Trade-off: RF de 400ms cubre el 95% de la flexión y ~80% de la extensión
- **[Diagrama: ventana deslizante de 243 samples sobre la señal]** — PPT

### Slide 4.5 — Envolventes: selección del RMS
- Comparación: rectificado + media móvil / Butterworth LP / RMS
- RMS elegido: más estable, mejor SNR, interpretable como energía instantánea (raíz de la potencia media)
- Implementación: ventana rectangular de **40 ms**, cumsum para eficiencia O(1) por sample
- Dos escalas: RMS rápido 40 ms (captura onsets) + RMS lento 150 ms (tendencia de activación)
- **Normalización por EMA** (exponential moving average, τ=1s):
  `rms_norm = rms / (ema + ε)` → el output mide "veces el baseline reciente"
  Durante contracción aguda: >1 / Durante silencio: ~1 / Causal, implementable online
- **[Plot: señal filtrada + RMS_rápido + RMS_lento + envelope_LP superpuestos]**
  — `Onset Detection ADS1256.ipynb`, `preprocess_emg_rms()` vs `preprocess_emg_butter()` — TODO: exportar

### Slide 4.6 — Smoothing del target y selección de σ (hallazgo clave)
- Problema: joystick llega al tope eléctrico antes del mecánico → transitorios abruptos
- Solución inicial: kernel gaussiano 1D con σ=150 ms (no causal, solo offline)
- **Hallazgo**: σ=150 ms aplastaba los picos rápidos más que los lentos
  - `fast_flex` con σ=150: std=470 / `slow_flex` con σ=150: std=575 → **invertido**
  - El target le decía al modelo que los movimientos rápidos son de menor amplitud que los lentos
  - Este sesgo era el cap real del R², no el modelo ni las features
- Sweep σ ∈ {10, 30, 50, 75, 100, 150} ms:
  - σ=10: retiene ruido del joystick que EMG no puede predecir → plateau R²=0.38
  - σ=30: mejor en fast, peor en slow (ruido residual en movimientos suaves)
  - **σ=50 sweet spot**: R²=0.699 vs R²=0.616 del baseline σ=150 → **+0.083 R²**
  - σ≥75: empieza a aplastar picos rápidos nuevamente
- Pipeline: posición joystick → gaussiano (σ=50 ms) → derivada numérica → vel → norm p99
- **[Plot: target_sigma_sweep.png]** — YA EXISTE en `plots/target_sigma_sweep.png`
- **[Tabla: R² por task por σ]** — del methods.md §10.8, transcribir a PPT

---

## SECCIÓN 5 — Setup experimental y ground truth (3 slides)

### Slide 5.1 — Diseño del joystick dual
- Objetivo: capturar flexión/extensión de articulación MCP
- Setup: dos joysticks alineados verticalmente, dedo entre ambos
- Dedo empuja joystick superior (extensión) o inferior (flexión)
- **[Foto: setup de joysticks con dedo]** — TODO: pasar del celular
- **[Diagrama: unificación joy_top × (−1) + joy_bottom = señal unificada]** — PPT

### Slide 5.2 — Protocolo de recolección
- Tasks: extensión libre, flexión libre, hold, movimientos mixtos
- Exclusiones del training: transition, noise_wrist_grip, noise_other_fingers
- Sesiones S02–S07, variación de placement entre sesiones para robustez
- **[Foto/video: sesión de toma de datos]** — TODO: pasar del celular
- Tabla de sesiones: fecha, n_samples, tasks

### Slide 5.3 — Validación EMG→joystick
- Cross-correlación entre RMS extensor y joystick (extensión)
- Cross-correlación entre RMS flexor y joystick (flexión)
- Resultado: lag positivo confirma EMG precede al movimiento
- **[Plot: RMS extensor/flexor vs joystick por dedo]**
  — `Onset Detection ADS1256.ipynb`, plots "INDICE extensor" / "INDICE flexor" — TODO: exportar
- **[Plot: cross-correlación con lag]**
  — `Signal Visualization & Correlation Analysis.ipynb`, función `estimate_lag()` — TODO: exportar

---

## SECCIÓN 6 — Hardware v2: ESP32 + ADS1256 (2 slides)

### Slide 6.1 — Motivación del upgrade
- Problema principal: 6 canales ADC de 10 bits insuficientes para 2 dedos
- ADS1256: ADC externo de **24 bits** (~16.7M niveles vs 1024), hasta 8 canales diferenciales
- ESP32: dual core, 2 Mbps UART, permite round-robin 4 canales sin pérdida temporal
- Ajuste de ganancia PGA: flexores (mayor activación) vs extensores
- **[Foto: ESP32 + ADS1256 + breadboard montado]** — TODO: pasar del celular

### Slide 6.2 — Desafíos del bring-up
- SPI a alta velocidad: sincronización crítica para no perder muestras
- Round-robin 4 canales: tiempo de conversión ADS limita fs efectiva
- Overflow de timestamp uint32 para sesiones largas: detección y unwrap
- Resultado final: ~638 Hz por canal con 4 canales simultáneos
- **[Plot: señal ADS1256 vs Arduino — resolución visible]** — TODO: comparar plot_raw de ambas épocas

---

## SECCIÓN 7 — Pipeline de procesamiento completo (2 slides)

### Slide 7.1 — Pipeline de features
```
Señal cruda
  → Butterworth pasa banda (70–300 Hz)
  → RMS ventana rápida + RMS ventana lenta
  → Derivadas temporales de cada RMS
  → Clip outliers (±20 raw, ±10 post-norm)
  → 4 features × 2 canales (flx/ext) × N dedos
```
- **[Diagrama de flujo]** — PPT
- **[Plot: señal en cada etapa del pipeline superpuesta]** — TODO: script nuevo

### Slide 7.2 — Pipeline del target
```
Posición joystick cruda
  → Unificación dual (normalizar + ×(−1) + suma)
  → Suavizado gaussiano (σ=50 ms)
  → Derivada numérica → velocidad
  → Normalización p99 de |vel|
```
- **[Diagrama de flujo]** — PPT
- **[Plot: cada etapa del target superpuesta]**
  — `Onset Detection ADS1256.ipynb`, plot final de 2 dedos — TODO: exportar

---

## SECCIÓN 8 — Modelo TCN (3 slides)

### Slide 8.1 — Arquitectura
- Temporal Convolutional Network: convoluciones causales dilatadas
- Causal = solo ve el pasado → implementable en tiempo real
- Dilaciones [1, 3, 9, 27, 81] → RF = 243 samples (~400 ms)
- Input: 8–16 canales de features; Output: velocidad normalizada 1 o 2 dedos
- **[Diagrama: TCN con dilations mostrando RF]** — imagen estándar de literatura o PPT

### Slide 8.2 — Entrenamiento y regularización
- Split estratificado por task-block: evita leakage del RF entre train y val
- Loss: **Huber** con δ=1.0 (cuadrática en errores pequeños, lineal en picos)
- Problemas encontrados: WeightNorm + MPS = NaN (desactivado), bug MPS+eval mode → workaround CPU
- **Journey de regularización por suavidad** (relevante para análisis de señales):
  1. **TVD (descartado)**: `L = L_data + λ·Σ|Δpred[t]|` — colapsó el modelo. Razón: shuffle=False sesga gradientes; minimizar |Δpred| empuja a predicciones constantes
  2. **Adaptive TV / Perona-Malik (descartado teóricamente)**: peso variable con `1/(1+|Δtarget|²)`. Asimétrico: penaliza "target plano, pred salta" pero no "target salta, pred plana"
  3. **Sobolev H¹ (implementado)**: supervisar `Δpred` contra `Δtarget` — `L_accel = Σ(Δpred−Δtarget)²`. Simétrico, penaliza error en la derivada en ambas direcciones. Ref: Czarnecki et al., NeurIPS 2017
  - Resultado: Sobolev ayudó marginalmente (+0.009 R²). **El factor dominante fue el σ del target** (no la regularización)
- **[Plot: curvas train_loss / R² / Pearson vs epochs (3 paneles)]**
  — `python plots/plot_training_curve.py` — YA GENERADO: `plots/training_curve.png`

### Slide 8.3 — Resultados por task (análisis detallado)
- 1 dedo (índice, S02+S03): **R²=0.699, Pearson=0.837**
- 2 dedos simultáneos (S04–S07): **avg R²=0.629** (ind=0.599 / med=0.659)
- Tres grupos de comportamiento en val:
  1. **Movimientos voluntarios** (fast_flex R²=0.89, fast_ext=0.68, slow_*≈0.65): buen decoding
  2. **Holds y microcorrecciones** (R²≈0.30–0.47): MAE pequeño pero no captura varianza residual
     → causa: ambigüedad mecánica del joystick a resorte (F_músculo = F_resorte → vel=0)
  3. **Ruido/reposo** (R²→−∞ en rest_baseline): MAE pequeño (0.02), R² mal condicionado porque var(target)→0
     → el modelo se queda cerca de 0, que es la respuesta correcta
- **Oracle ensemble**: σ=50 (rápidos) + σ=150 (lentos) → R²=0.734 (+0.035 sobre champion)
  → confirma que el gap residual es de single-target training, no del modelo
- Comparación literatura: competitivo con 4 electrodos vs HD-EMG (8–150) de los papers
- **[Plot: r2_by_task.png]** — `python plots/plot_r2_by_task.py` ✓
- **[Plot: pred_vs_target_tasks.png]** — `python plots/plot_prediction.py` ✓
- **[Plot: pred_scatter.png]** — ídem ✓

---

## SECCIÓN 9 — Demo en vivo (1 slide → pantalla completa)

- `python visualize.py --checkpoint checkpoints/tcn_s02s03_sig050_200ep/best.pt --csv <sesion>`
- Pantalla dividida: cámara brazo + visualize.py
- Muestra: señal EMG cruda + features + predicción (azul) vs ground truth (verde)
- **[Slide de instrucciones por si algo falla]** — fallback screenshot

---

---

## SECCIÓN 10 — Papers clave (1 slide — backup o al final de TCN)

- **Bai, Kolter & Koltun (2018)** — An Empirical Evaluation of Generic Convolutional and Recurrent Networks. Base de la arquitectura TCN.
- **Hudgins, Parker & Scott (1993)** — A new strategy for multifunction myoelectric control. IEEE TBME. Primer feature set EMG (MAV, ZC, SSC, WL) — diseñado para clasificación, no regresión.
- **Englehart & Hudgins (2003)** — A robust, real-time control scheme for multifunction myoelectric control. IEEE TBME. Referencia para separación offline→online de features EMG.
- **Czarnecki et al. (2017)** — Sobolev Training for Neural Networks. NeurIPS. Base teórica de la regularización L_accel.
- **Hahne et al. (2014)** — Linear and nonlinear regression for simultaneous proportional myoelectric control. IEEE TNSRE. Baseline de regresión continua EMG con pocos electrodos.

---

## SECCIÓN 11 — Conclusiones y trabajo futuro (2 slides)

### Slide 10.1 — Conclusiones
- Análisis de onsets: pieza clave para validar causalidad, definir RF y seleccionar features
- Desenchufar la laptop: eliminó armónicos sin Notch, cambio crítico de bajo costo
- Migración ADS1256+ESP32: necesaria para escalar a 2 dedos sin sacrificar resolución temporal
- TCN causal con RF~400ms: predicción de velocidad en tiempo real viable con 4 electrodos

### Slide 10.2 — Trabajo futuro
- Controlador: red sobre la TCN para cerrar el loop (RL online, reward = −|error de posición|)
- Sensor flex en lugar de joystick: elimina ambigüedad EMG→vel durante holds
- Más sesiones y sujetos para generalización cross-session
- Construcción de la mano robótica (Parte 2 del proyecto original)
- Comparación con literatura: competitivo dado 4 electrodos vs HD-EMG (8–150 electrodos)

---

## TODO — Assets pendientes

### Plots a generar (scripts)
- [ ] FFT enchufado vs desenchufado → `plots/plot_fft_noise.py` ⚠️ REQUIERE nueva toma con laptop enchufada
- [x] Respuesta en frecuencia filtro Butterworth 70-300Hz → `plots/filter_response.png` ✓ GENERADO
- [ ] Histograma de delays EMG→joystick → exportar desde `Onset Detection.ipynb`
- [ ] Comparación envolventes (RMS_rápido vs RMS_lento vs Butter LP) → exportar desde `Onset Detection ADS1256.ipynb`
- [x] Sigma sweep → `plots/target_sigma_sweep.png` ✓ YA EXISTÍA
- [ ] Cross-correlación EMG vs joystick con lag → exportar desde `Signal Visualization & Correlation Analysis.ipynb`
- [x] Curvas de entrenamiento (train loss / R² / Pearson, 3 paneles) → `plots/training_curve.png` ✓ GENERADO
- [ ] R² por task → correr `python eval_by_task.py` y exportar
- [ ] PSD de señal EMG → `python plots/plot_emg_psd.py` (script creado, falta correr)

### Fotos/videos (desde celular)
- [ ] Brazo con electrodos pegados
- [ ] Arduino + EXG Pills + joysticks (setup v1)
- [ ] Setup joysticks con dedo (posición experimental)
- [ ] ESP32 + ADS1256 + breadboard (setup v2)
- [ ] Setup completo final
- [ ] Video sesión de toma de datos
- [ ] Teaser: brazo + pantalla visualize.py (grabar al final)

### Demo
- [ ] Verificar que `visualize.py` corre con el champion checkpoint
- [ ] Decidir si usar sesión grabada o captura en vivo
- [ ] Preparar fallback (screenshot/video pregrabado por si falla)
