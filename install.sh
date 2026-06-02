#!/usr/bin/env bash
# Sets up frigate-inference alongside an existing Frigate installation.
# Run once after cloning, and again after pulling updates.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ENV_FILE="${SCRIPT_DIR}/.env"

# ── .env ──────────────────────────────────────────────────────────────────────
if [[ ! -f "${ENV_FILE}" ]]; then
    cp "${SCRIPT_DIR}/.env.example" "${ENV_FILE}"
    echo "Created ${ENV_FILE} — edit it before starting the container."
else
    echo "${ENV_FILE} already exists, skipping."
fi

# ── Load env ──────────────────────────────────────────────────────────────────
set -a; source "${ENV_FILE}"; set +a

FRIGATE_COMPOSE="${FRIGATE_COMPOSE:-/opt/frigate/compose.yaml}"
FRIGATE_CONFIG_DIR="${FRIGATE_CONFIG_DIR:-/opt/frigate/config}"

# ── CDI GPU pin ─────────────────────────────────────────────────────────────
# The inference container is pinned to one GPU via a CDI device name
# (compose.yaml: device_ids: [${INFERENCE_CDI_DEVICE}]). CDI keeps GPU access
# stable across `systemctl daemon-reload`, which can otherwise strip the device
# cgroup from the legacy --gpus/nvidia-runtime path and silently drop the GPU.
set_env_var() {  # set_env_var KEY VALUE — update or append KEY in .env
    local key="$1" val="$2"
    if grep -q "^${key}=" "${ENV_FILE}"; then
        sed -i.bak "s|^${key}=.*|${key}=${val}|" "${ENV_FILE}" && rm -f "${ENV_FILE}.bak"
    else
        printf '\n%s=%s\n' "${key}" "${val}" >> "${ENV_FILE}"
    fi
}
list_cdi() { nvidia-ctk cdi list 2>/dev/null | grep '^nvidia.com/gpu=' || true; }

if ! command -v nvidia-ctk >/dev/null 2>&1; then
    echo "nvidia-ctk not found — install the NVIDIA Container Toolkit to use CDI,"
    echo "then set INFERENCE_CDI_DEVICE in ${ENV_FILE} manually (see .env.example)."
else
    mapfile -t CDI_DEVICES < <(list_cdi)

    # Regenerate the spec if it's empty or the configured device is gone — a
    # driver upgrade rewrites the CDI spec (versioned lib paths) and can change
    # device names, so re-running install.sh after an upgrade refreshes it.
    regen=0
    if [[ ${#CDI_DEVICES[@]} -eq 0 ]]; then
        echo "No CDI devices found."; regen=1
    elif [[ -n "${INFERENCE_CDI_DEVICE:-}" ]] \
         && ! printf '%s\n' "${CDI_DEVICES[@]}" | grep -qxF "${INFERENCE_CDI_DEVICE}"; then
        echo "Configured INFERENCE_CDI_DEVICE='${INFERENCE_CDI_DEVICE}' is not in the CDI spec."
        regen=1
    fi
    if [[ ${regen} -eq 1 && -t 0 ]]; then
        read -rp "Generate/refresh the CDI spec now (sudo nvidia-ctk cdi generate)? [y/N] " a
        if [[ "${a:-}" =~ ^[Yy] ]]; then
            sudo nvidia-ctk cdi generate --output=/etc/cdi/nvidia.yaml
            sudo chmod 644 /etc/cdi/nvidia.yaml
            mapfile -t CDI_DEVICES < <(list_cdi)
        fi
    fi

    if [[ ${#CDI_DEVICES[@]} -eq 0 ]]; then
        echo "No CDI devices available — set INFERENCE_CDI_DEVICE in ${ENV_FILE} manually."
    elif [[ ! -t 0 ]]; then
        echo "Non-interactive shell — leaving INFERENCE_CDI_DEVICE=${INFERENCE_CDI_DEVICE:-<unset>} as-is."
    else
        keep=0
        if [[ -n "${INFERENCE_CDI_DEVICE:-}" ]] \
           && printf '%s\n' "${CDI_DEVICES[@]}" | grep -qxF "${INFERENCE_CDI_DEVICE}"; then
            read -rp "Inference GPU pinned to ${INFERENCE_CDI_DEVICE}. Keep it? [Y/n] " a
            [[ "${a:-}" =~ ^[Nn] ]] || keep=1
        fi
        if [[ ${keep} -eq 0 ]]; then
            echo "Select the GPU to pin the inference container to:"
            select dev in "${CDI_DEVICES[@]}"; do
                [[ -n "${dev:-}" ]] || { echo "Invalid selection."; continue; }
                set_env_var INFERENCE_CDI_DEVICE "${dev}"
                echo "Pinned INFERENCE_CDI_DEVICE=${dev}"
                break
            done
        fi
    fi
fi

# ── inference.yaml ────────────────────────────────────────────────────────────
SAMPLE="${SCRIPT_DIR}/config/inference.yaml"
TARGET="${FRIGATE_CONFIG_DIR}/inference.yaml"

if [[ ! -f "${TARGET}" ]]; then
    if [[ -d "${FRIGATE_CONFIG_DIR}" ]]; then
        cp "${SAMPLE}" "${TARGET}"
        echo "Copied inference.yaml to ${TARGET} — review and adjust as needed."
    else
        echo "Warning: ${FRIGATE_CONFIG_DIR} not found — copy config/inference.yaml there manually."
    fi
else
    echo "${TARGET} already exists, skipping."
fi

# ── Frigate compose integration ───────────────────────────────────────────────
if [[ -f "${FRIGATE_COMPOSE}" ]] && grep -q "${SCRIPT_DIR}/compose.yaml" "${FRIGATE_COMPOSE}" 2>/dev/null; then
    echo "Frigate compose already includes inference-engine, skipping."
else
    echo ""
    echo "Add the following include to ${FRIGATE_COMPOSE}:"
    echo ""
    echo "  include:"
    echo "    - path: ${SCRIPT_DIR}/compose.yaml"
    echo "      env_file: ${ENV_FILE}"
    echo ""
    echo "Also ensure the frigate service has:"
    echo "      volumes:"
    echo "        - zmq_ipc:/run/zmq"
    echo "      depends_on: [frigate-inference]"
fi

echo ""
echo "Done. Next steps:"
echo "  1. Review ${ENV_FILE}"
echo "  2. Build the image:  cd ${SCRIPT_DIR} && arch/sm_61/build.sh"
echo "  3. Start services:   cd \$(dirname ${FRIGATE_COMPOSE}) && docker compose up -d"
