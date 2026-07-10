"""
Definiciones de bloques y protocolos por sesion.

Vocabulario fijo de tareas (no agregar/renombrar entre sesiones; nuevos
labels solo via nuevas constantes para mantener consistencia historica).

Setup ESP32+ADS1256: 2 dedos instrumentados (indice y medio), cada uno
con su par flexor/extensor. Protocolo separa tareas mono-dedo con sufijo
*_ind / *_med y agrega combos para aprender independencia/coactivacion.
"""
from dataclasses import dataclass, field
from typing import List, Optional


TASK_REST = "rest_baseline"

# Mono-dedo: INDICE
TASK_SLOW_FLEX_IND = "slow_flex_ind"
TASK_SLOW_EXT_IND = "slow_ext_ind"
TASK_SLOW_CYCLES_IND = "slow_cycles_ind"
TASK_FAST_FLEX_IND = "fast_flex_ind"
TASK_FAST_EXT_IND = "fast_ext_ind"
TASK_HOLDS_IND = "holds_intermediate_ind"
TASK_MICRO_IND = "microcorrections_ind"
TASK_FREE_IND = "free_controlled_ind"

# Mono-dedo: MEDIO
TASK_SLOW_FLEX_MED = "slow_flex_med"
TASK_SLOW_EXT_MED = "slow_ext_med"
TASK_SLOW_CYCLES_MED = "slow_cycles_med"
TASK_FAST_FLEX_MED = "fast_flex_med"
TASK_FAST_EXT_MED = "fast_ext_med"
TASK_HOLDS_MED = "holds_intermediate_med"
TASK_MICRO_MED = "microcorrections_med"
TASK_FREE_MED = "free_controlled_med"

# Combos (dos dedos)
TASK_BOTH_CYCLES_INPHASE = "both_cycles_inphase"
TASK_BOTH_CYCLES_ANTIPHASE = "both_cycles_antiphase"
TASK_IND_MOVE_MED_HOLD = "ind_move_med_hold"
TASK_MED_MOVE_IND_HOLD = "med_move_ind_hold"
TASK_BOTH_FREE = "both_free"

# Distractores y comunes (aplican a ambos dedos)
TASK_NOISE_FINGERS = "noise_other_fingers"
TASK_NOISE_WRIST = "noise_wrist_grip"
TASK_TRANSITION = "transition"

TASKS = {
    TASK_REST,
    # Indice
    TASK_SLOW_FLEX_IND, TASK_SLOW_EXT_IND, TASK_SLOW_CYCLES_IND,
    TASK_FAST_FLEX_IND, TASK_FAST_EXT_IND,
    TASK_HOLDS_IND, TASK_MICRO_IND, TASK_FREE_IND,
    # Medio
    TASK_SLOW_FLEX_MED, TASK_SLOW_EXT_MED, TASK_SLOW_CYCLES_MED,
    TASK_FAST_FLEX_MED, TASK_FAST_EXT_MED,
    TASK_HOLDS_MED, TASK_MICRO_MED, TASK_FREE_MED,
    # Combos
    TASK_BOTH_CYCLES_INPHASE, TASK_BOTH_CYCLES_ANTIPHASE,
    TASK_IND_MOVE_MED_HOLD, TASK_MED_MOVE_IND_HOLD, TASK_BOTH_FREE,
    # Comunes
    TASK_NOISE_FINGERS, TASK_NOISE_WRIST, TASK_TRANSITION,
}


@dataclass
class Block:
    task: str
    duration: float
    instructions: str
    cue: Optional[str] = None

    def __post_init__(self):
        if self.task not in TASKS:
            raise ValueError(f"task '{self.task}' no esta en TASKS")
        if self.duration <= 0:
            raise ValueError("duration debe ser > 0")


def _mono_finger_blocks(finger: str) -> List[Block]:
    """Bloque mono-dedo (indice o medio). El otro dedo descansa apoyado."""
    f = finger.lower()
    assert f in ("ind", "med")
    label = {"ind": "INDICE", "med": "MEDIO"}[f]
    other = {"ind": "medio", "med": "indice"}[f]
    T = {
        "ind": (TASK_SLOW_FLEX_IND, TASK_SLOW_EXT_IND, TASK_SLOW_CYCLES_IND,
                TASK_FAST_FLEX_IND, TASK_FAST_EXT_IND,
                TASK_HOLDS_IND, TASK_MICRO_IND, TASK_FREE_IND),
        "med": (TASK_SLOW_FLEX_MED, TASK_SLOW_EXT_MED, TASK_SLOW_CYCLES_MED,
                TASK_FAST_FLEX_MED, TASK_FAST_EXT_MED,
                TASK_HOLDS_MED, TASK_MICRO_MED, TASK_FREE_MED),
    }[f]
    (T_SLOW_FLEX, T_SLOW_EXT, T_SLOW_CYCLES,
     T_FAST_FLEX, T_FAST_EXT,
     T_HOLDS, T_MICRO, T_FREE) = T

    return [
        Block(T_SLOW_FLEX, 75,
              f"Flexiona el {label} MUY lento (~3 s recorrido completo) y solta. "
              f"El {other} queda quieto, apoyado."),
        Block(T_SLOW_EXT, 75,
              f"Extiende el {label} MUY lento (~3 s) y solta. {other} quieto."),
        Block(T_SLOW_CYCLES, 100,
              f"Ciclos continuos flex<->ext lentos del {label} (~0.25 Hz). "
              f"Sin pausas. {other} quieto.",
              cue="Metronomo a 0.25 Hz si tenes."),
        Block(T_FAST_FLEX, 50,
              f"Flexion rapida del {label} y volve. ~2 s entre repeticiones. "
              f"{other} quieto."),
        Block(T_FAST_EXT, 50,
              f"Extension rapida del {label} y volve. ~2 s entre repeticiones. "
              f"{other} quieto."),
        Block(T_HOLDS, 90,
              f"Manten el {label} en 25 %, 50 %, 75 % del rango (~4 s c/u). "
              f"Rota entre las tres. {other} quieto.",
              cue="Marca visual de target si tenes."),
        Block(T_MICRO, 70,
              f"Microcorrecciones de 5-10 % del rango del {label} "
              f"alrededor de la mitad. {other} quieto."),
        Block(T_FREE, 50,
              f"{label} libre, mezcla velocidades y posiciones. "
              f"Cubri todo el rango. {other} quieto."),
    ]


def _combo_blocks() -> List[Block]:
    return [
        Block(TASK_BOTH_CYCLES_INPHASE, 100,
              "Ambos dedos en fase: flex juntos, ext juntos. Ciclos lentos.",
              cue="Metronomo a 0.25 Hz."),
        Block(TASK_BOTH_CYCLES_ANTIPHASE, 100,
              "Antifase: cuando el indice flexiona el medio extiende y "
              "viceversa. Ciclos lentos.",
              cue="Metronomo a 0.25 Hz."),
        Block(TASK_IND_MOVE_MED_HOLD, 80,
              "Mueve el INDICE libremente. Manten el MEDIO en hold ~50 %."),
        Block(TASK_MED_MOVE_IND_HOLD, 80,
              "Mueve el MEDIO libremente. Manten el INDICE en hold ~50 %."),
        Block(TASK_BOTH_FREE, 90,
              "Ambos dedos libres mezclando coactivacion, antifase e "
              "independencia. Cubri todo."),
    ]


def _distractor_blocks(short: bool = False) -> List[Block]:
    duration_fingers = 100 if not short else 180
    duration_wrist = 100 if not short else 180
    return [
        Block(TASK_NOISE_FINGERS, duration_fingers,
              "NO muevas ni indice ni medio. Mueve pulgar, anular, menique "
              "uno por vez."),
        Block(TASK_NOISE_WRIST, duration_wrist,
              "NO muevas ni indice ni medio. Flex/ext de muneca, pronacion, "
              "agarre con los otros dedos."),
    ]


def base_protocol() -> List[Block]:
    """Sesion estandar de entrenamiento (2 dedos). ~31 min totales."""
    return [
        Block(TASK_REST, 90,
              "Mano relajada sobre la mesa. No contraigas nada. "
              "Mira un punto fijo."),
        *_mono_finger_blocks("ind"),
        *_mono_finger_blocks("med"),
        *_combo_blocks(),
        *_distractor_blocks(short=False),
    ]


def _protocol_high_distractor() -> List[Block]:
    """Sesiones 7-8: bloques mono recortados y distractores extendidos."""
    short_mono_ind = [
        Block(TASK_SLOW_CYCLES_IND, 90, "Ciclos lentos del indice. Medio quieto."),
        Block(TASK_FAST_FLEX_IND, 50, "Flex rapida indice. Medio quieto."),
        Block(TASK_FAST_EXT_IND, 50, "Ext rapida indice. Medio quieto."),
        Block(TASK_HOLDS_IND, 60, "Holds 25/50/75 indice. Medio quieto."),
        Block(TASK_MICRO_IND, 50, "Microcorrecciones indice. Medio quieto."),
    ]
    short_mono_med = [
        Block(TASK_SLOW_CYCLES_MED, 90, "Ciclos lentos del medio. Indice quieto."),
        Block(TASK_FAST_FLEX_MED, 50, "Flex rapida medio. Indice quieto."),
        Block(TASK_FAST_EXT_MED, 50, "Ext rapida medio. Indice quieto."),
        Block(TASK_HOLDS_MED, 60, "Holds 25/50/75 medio. Indice quieto."),
        Block(TASK_MICRO_MED, 50, "Microcorrecciones medio. Indice quieto."),
    ]
    short_combos = [
        Block(TASK_BOTH_CYCLES_INPHASE, 70, "Ambos en fase."),
        Block(TASK_BOTH_CYCLES_ANTIPHASE, 70, "Ambos antifase."),
        Block(TASK_BOTH_FREE, 60, "Ambos libres."),
    ]
    return [
        Block(TASK_REST, 90, "Mano relajada."),
        *short_mono_ind,
        *short_mono_med,
        *short_combos,
        *_distractor_blocks(short=True),
    ]


def _protocol_test() -> List[Block]:
    """Sesion 10: test puro, NO entrenar."""
    return [
        Block(TASK_REST, 60, "Mano relajada."),
        Block(TASK_SLOW_CYCLES_IND, 120, "Ciclos lentos indice. Medio quieto."),
        Block(TASK_SLOW_CYCLES_MED, 120, "Ciclos lentos medio. Indice quieto."),
        Block(TASK_FAST_FLEX_IND, 60, "Flex rapida indice."),
        Block(TASK_FAST_EXT_IND, 60, "Ext rapida indice."),
        Block(TASK_FAST_FLEX_MED, 60, "Flex rapida medio."),
        Block(TASK_FAST_EXT_MED, 60, "Ext rapida medio."),
        Block(TASK_HOLDS_IND, 120, "Holds indice."),
        Block(TASK_HOLDS_MED, 120, "Holds medio."),
        Block(TASK_MICRO_IND, 80, "Microcorrecciones indice."),
        Block(TASK_MICRO_MED, 80, "Microcorrecciones medio."),
        Block(TASK_BOTH_CYCLES_INPHASE, 100, "Ambos en fase."),
        Block(TASK_BOTH_CYCLES_ANTIPHASE, 100, "Ambos antifase."),
        Block(TASK_IND_MOVE_MED_HOLD, 90, "Indice libre, medio en hold."),
        Block(TASK_MED_MOVE_IND_HOLD, 90, "Medio libre, indice en hold."),
        Block(TASK_BOTH_FREE, 180, "Ambos libres, todo el rango."),
        Block(TASK_NOISE_FINGERS, 100, "Distractores otros dedos."),
        Block(TASK_NOISE_WRIST, 100, "Distractores muneca/agarre."),
    ]


def mini_protocol() -> List[Block]:
    """Mini ~2 min para validar que el pipeline (firmware -> CSV) anda.
    NO usar para entrenar. Toca rest, mono indice, mono medio y un combo."""
    return [
        Block(TASK_REST, 15,
              "Mano relajada. Validacion: solo verificamos que el pipeline graba."),
        Block(TASK_SLOW_CYCLES_IND, 30,
              "Ciclos lentos del indice (~0.25 Hz). Medio quieto.",
              cue="cualquier ritmo lento sirve para esta prueba"),
        Block(TASK_SLOW_CYCLES_MED, 30,
              "Ciclos lentos del medio (~0.25 Hz). Indice quieto."),
        Block(TASK_BOTH_CYCLES_INPHASE, 40,
              "Ambos dedos en fase. Coordinacion basica."),
    ]


def protocol_1finger() -> List[Block]:
    """Protocolo solo-indice para Arduino (sin medio, sin combos, sin noise). ~12 min."""
    return [
        Block(TASK_REST, 90,
              "Mano relajada sobre la mesa. No contraigas nada."),
        Block(TASK_SLOW_FLEX_IND, 75,
              "Flexiona el INDICE MUY lento (~3 s recorrido completo) y solta."),
        Block(TASK_SLOW_EXT_IND, 100,
              "Extiende el INDICE MUY lento (~3 s) y solta."),
        Block(TASK_SLOW_CYCLES_IND, 100,
              "Ciclos continuos flex<->ext lentos del INDICE (~0.25 Hz). Sin pausas.",
              cue="Metronomo a 0.25 Hz si tenes."),
        Block(TASK_FAST_FLEX_IND, 50,
              "Flexion rapida del INDICE y volve. ~2 s entre repeticiones."),
        Block(TASK_FAST_EXT_IND, 100,
              "Extension rapida del INDICE y volve. ~2 s entre repeticiones."),
        Block(TASK_HOLDS_IND, 90,
              "Manten el INDICE en 25 %, 50 %, 75 % del rango (~4 s c/u). Rota entre las tres.",
              cue="Marca visual de target si tenes."),
        Block(TASK_MICRO_IND, 70,
              "Microcorrecciones de 5-10 % del rango del INDICE alrededor de la mitad."),
        Block(TASK_FREE_IND, 50,
              "INDICE libre, mezcla velocidades y posiciones. Cubri todo el rango."),
    ]


def protocol_free_eval() -> List[Block]:
    """Sesion de evaluacion: rest corto + free_controlled puro. ~115 s."""
    return [
        Block(TASK_REST, 15,
              "Mano relajada. Quedate quieto, esto calibra la zona muerta."),
        Block(TASK_FREE_IND, 100,
              "INDICE libre: mezcla velocidades lentas y rapidas, "
              "posiciones intermedias y extremos. Cubri todo el rango sin pausas largas."),
    ]


def protocol_for_session(session_number: int) -> List[Block]:
    # Sesion 10 = test puro (no entrenar). Resto usa protocolo base.
    if session_number == 10:
        return _protocol_test()
    return base_protocol()


def session_split(session_number: int) -> str:
    """Indica si la sesion va a train o test. Se guarda en metadata."""
    if session_number == 10:
        return "test"
    return "train"
