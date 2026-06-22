#!/bin/bash
# Bisecta qué feature combinada con qué dedo dispara el bug MPS+eval.
# Corre 2 epochs por combinación y reporta el val_loss del ep 2.
#
# Orden: ind solo, med solo, both. Dentro de cada finger, suma features
# acumulativamente (1, 1+2, 1+2+3, 1+2+3+4).

set -u
cd "$(dirname "$0")"
mkdir -p logs

FEATURES=("rms_norm" "d_rms_norm" "rms_slow_norm" "d_rms_slow_norm")
CSV_ARGS=(--csv data/raw/P01/P01_S04_20260618_162615.csv
          data/raw/P01/P01_S05_20260618_172229.csv)
COMMON_ARGS=(--epochs 2 --train-stride 8 --val-stride 8 --batch-size 256
             --hidden-channels 32 --dilations 1 3 9 27 81 --dropout 0.1
             --horizon 1 --accel-weight 0.0 --var-weight 0.0
             --loss huber --huber-delta 1.0
             --exclude-tasks transition noise_wrist_grip
             --feature-norm median_mad --clip-value 10.0             
             --no-weight-norm --device auto)

RESULTS_FILE="logs/bisect_results.txt"
: > "$RESULTS_FILE"
echo "finger    | n | features                                                | val_loss_ep2" >> "$RESULTS_FILE"
echo "----------+---+---------------------------------------------------------+-------------" >> "$RESULTS_FILE"

run_test() {
    local feature_fingers=$1
    shift
    local feats=("$@")
    local feats_str=$(IFS=,; echo "${feats[*]}")
    local n=${#feats[@]}
    # target_finger: si feature_fingers=both, predecimos solo ind (target=ind)
    local target_finger=$feature_fingers
    [ "$feature_fingers" = "both" ] && target_finger=ind
    local name="bisect_${feature_fingers}_n${n}_$(echo "${feats[*]}" | tr ' ' '-')"
    local log="logs/$name.log"

    echo ""
    echo "=== feature_fingers=$feature_fingers  target=$target_finger  features=[$feats_str] ==="
    python train_2fingers.py \
        "${CSV_ARGS[@]}" "${COMMON_ARGS[@]}" \
        --feature-fingers "$feature_fingers" \
        --target-finger "$target_finger" \
        --features "${feats[@]}" \
        --run-name "$name" 2>&1 | tee "$log" | tail -5

    # Parsear val_loss del epoch 2
    local val_loss=$(grep -oE "ep 2/2  val_loss [0-9.eE+-]+" "$log" | tail -1 | awk '{print $NF}')
    [ -z "$val_loss" ] && val_loss="N/A"
    printf "%-9s | %d | %-55s | %s\n" "$feature_fingers" "$n" "$feats_str" "$val_loss" >> "$RESULTS_FILE"
    echo "  -> val_loss=$val_loss"
}

# 1) Indice solo: features acumulativas
for i in 1 2 3 4; do
    run_test ind "${FEATURES[@]:0:$i}"
done

# 2) Medio solo: features acumulativas
for i in 1 2 3 4; do
    run_test med "${FEATURES[@]:0:$i}"
done

# 3) Ambos dedos: features acumulativas (target=ind)
for i in 1 2 3 4; do
    run_test both "${FEATURES[@]:0:$i}"
done

echo ""
echo "============================================================"
echo "RESUMEN (también en $RESULTS_FILE):"
echo "============================================================"
cat "$RESULTS_FILE"
