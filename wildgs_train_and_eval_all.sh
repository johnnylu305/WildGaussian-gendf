#!/usr/bin/env bash
set -u  # no -e, we handle errors explicitly
set -o pipefail


PARENT_DIR="${1%/}"

declare -a SUCCEEDED=()
declare -a FAILED=()

for subdir in "$PARENT_DIR"/*/; do
  [ -d "$subdir" ] || continue
  echo "==> Processing $subdir"

  # Run one scene; do NOT let a failure kill the batch.
  if ./wildgs_train_and_eval.sh "$subdir"; then
    echo "==> OK: $subdir"
    SUCCEEDED+=("$subdir")
  else
    echo "==> WARN: failed on $subdir (continuing)"
    FAILED+=("$subdir")
  fi

  echo "------------------------------------------------------------"
done

echo
echo "==================== Batch Summary ===================="
echo "Succeeded: ${#SUCCEEDED[@]}"
for s in "${SUCCEEDED[@]}"; do echo "  ✔ $s"; done
echo
echo "Failed: ${#FAILED[@]}"
for f in "${FAILED[@]}";   do echo "  ✖ $f"; done
echo "========================================================"

# Optional: return non-zero if any failures (uncomment next line if desired)
# [ "${#FAILED[@]}" -eq 0 ] || exit 1
