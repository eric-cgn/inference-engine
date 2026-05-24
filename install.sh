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

# ── Load env to find FRIGATE_CONFIG_DIR ───────────────────────────────────────
set -a; source "${ENV_FILE}"; set +a
FRIGATE_CONFIG_DIR="${FRIGATE_CONFIG_DIR:-/opt/frigate/config}"

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
FRIGATE_COMPOSE="${FRIGATE_CONFIG_DIR%/config}/compose.yaml"
INCLUDE_LINE="    - path: ${SCRIPT_DIR}/compose.yaml"

if [[ -f "${FRIGATE_COMPOSE}" ]]; then
    if grep -q "${SCRIPT_DIR}/compose.yaml" "${FRIGATE_COMPOSE}" 2>/dev/null; then
        echo "Frigate compose already includes inference-engine, skipping."
    else
        echo ""
        echo "Add the following to your Frigate compose.yaml to include the inference service:"
        echo ""
        echo "  include:"
        echo "    - path: ${SCRIPT_DIR}/compose.yaml"
        echo "      env_file: ${ENV_FILE}"
        echo ""
        echo "Also add to the frigate service volumes:"
        echo "      - zmq_ipc:/run/zmq"
        echo ""
        echo "And add to the frigate service:"
        echo "      depends_on: [frigate-inference]"
    fi
else
    echo ""
    echo "Could not find Frigate compose at ${FRIGATE_COMPOSE}."
    echo "Add this to your Frigate compose.yaml manually:"
    echo ""
    echo "  include:"
    echo "    - path: ${SCRIPT_DIR}/compose.yaml"
    echo "      env_file: ${ENV_FILE}"
fi

echo ""
echo "Done. Next steps:"
echo "  1. Edit ${ENV_FILE}"
echo "  2. Build the image:  cd ${SCRIPT_DIR} && arch/sm_61/build.sh"
echo "  3. Start services:   cd <frigate-dir> && docker compose up -d"
