#!/bin/bash
# Create the modprobe files for v4l2loopback
set -euo pipefail

readonly LOAD_FILE="/etc/modules-load.d/v4l2loopback.conf"
readonly OPT_FILE="/etc/modprobe.d/chameleon.conf"
readonly DEFAULT_VIDEO_NR=10

# Check if running as root
check_root() {
    if [[ "${EUID}" -ne 0 ]]; then
        echo "Error: This script must be run as root" >&2
        exit 1
    fi
}

# Get the next available video device number
get_next_video_device() {
    local lastdev
    if command -v v4l2-ctl &> /dev/null; then
        lastdev=$(v4l2-ctl --list-devices 2>/dev/null | grep -Po "(?<=/dev/video)[0-9]+$" | sort -n | tail -n 1 || echo "")
        if [[ -n "${lastdev}" ]]; then
            echo "$((lastdev + 1))"
        else
            echo "${DEFAULT_VIDEO_NR}"
        fi
    else
        echo "${DEFAULT_VIDEO_NR}"
    fi
}

# Validate video device number
validate_video_nr() {
    local nr="$1"
    if ! [[ "${nr}" =~ ^[0-9]+$ ]]; then
        echo "Error: Video device number must be a positive integer" >&2
        exit 1
    fi
}

# Create modules load file
create_load_file() {
    if [[ -f "${LOAD_FILE}" ]]; then
        echo "File exists: ${LOAD_FILE}"
    else
        echo "v4l2loopback" > "${LOAD_FILE}"
        echo "Created: ${LOAD_FILE}"
    fi
}

# Create options file and load kernel module
create_options_file() {
    local video_nr="$1"

    if [[ -f "${OPT_FILE}" ]]; then
        echo "File exists: ${OPT_FILE}, no changes have been made"
    else
        echo "options v4l2loopback devices=1 video_nr=${video_nr} max_buffers=2 exclusive_caps=1 card_label=\"Virtual Webcam\"" > "${OPT_FILE}"
        echo "Created: ${OPT_FILE}"
        echo "Reloading kernel modules..."
        systemctl restart systemd-modules-load.service
        echo "Done"
    fi
}

# Main
main() {
    local video_nr

    check_root

    # Determine video device number
    if [[ "$#" -ne 0 ]]; then
        video_nr="$1"
        validate_video_nr "${video_nr}"
    else
        video_nr=$(get_next_video_device)
    fi

    echo "Creating virtual video device with nr. ${video_nr}"

    create_load_file
    create_options_file "${video_nr}"
}

main "$@"
