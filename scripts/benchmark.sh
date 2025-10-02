#!/bin/bash
# Benchmark script for Chameleon virtual webcam
set -euo pipefail

# ============================================================================
# Configuration
# ============================================================================

readonly SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
readonly DEFAULT_WIDTH=1920
readonly DEFAULT_HEIGHT=1080
readonly DEFAULT_FPS=60
readonly DEFAULT_TIMEOUT=180
readonly TEST_VIDEO_URL="https://google.github.io/mediapipe/images/selfie_segmentation_web.mp4"

# Device configuration
readonly WEBCAM_DEVICE="/dev/video99"
readonly VIRTUAL_DEVICE="/dev/video100"

# Available segmentation models
readonly -a MODELS=(
    "mediapipe"
    "yolo11n-seg"
    "fastsam-s"
)

# ============================================================================
# Utilities
# ============================================================================

print_error() {
    echo "Error: $*" >&2
}

print_info() {
    echo "==> $*"
}

check_command() {
    local cmd="$1"
    if ! command -v "${cmd}" &> /dev/null; then
        print_error "Required command '${cmd}' not found"
        return 1
    fi
    return 0
}

check_requirements() {
    local missing=0

    # Check for required commands
    for cmd in wget timeout v4l2-ctl uv; do
        if ! check_command "${cmd}"; then
            missing=1
        fi
    done

    # Check for ffmpeg or avconv
    if ! check_command ffmpeg && ! check_command avconv; then
        print_error "Either 'ffmpeg' or 'avconv' is required"
        missing=1
    fi

    if [[ "${missing}" -eq 1 ]]; then
        exit 1
    fi
}

get_ffmpeg_command() {
    if check_command ffmpeg; then
        echo "ffmpeg"
    elif check_command avconv; then
        echo "avconv"
    else
        print_error "No video encoder found"
        exit 1
    fi
}

# ============================================================================
# Cleanup Management
# ============================================================================

PIDS=()
TEMP_FILES=()

add_pid() {
    PIDS+=("$1")
}

add_temp_file() {
    TEMP_FILES+=("$1")
}

cleanup() {
    print_info "Cleaning up..."

    # Kill background processes
    for pid in "${PIDS[@]}"; do
        if kill -0 "${pid}" 2>/dev/null; then
            kill "${pid}" 2>/dev/null || true
        fi
    done
    wait 2>/dev/null || true

    # Remove temporary files
    for file in "${TEMP_FILES[@]}"; do
        rm -f "${file}"
    done

    # Reset v4l2loopback
    if command -v modprobe &> /dev/null; then
        sudo -- bash -c 'modprobe -r v4l2loopback 2>/dev/null || true; modprobe v4l2loopback 2>/dev/null || true' || true
    fi

    print_info "Cleanup complete"
}

trap cleanup EXIT INT TERM

# ============================================================================
# V4L2 Setup
# ============================================================================

setup_v4l2loopback() {
    print_info "Setting up v4l2loopback devices..."

    # Keep sudo alive
    sudo -v
    (
        while sleep 30; do
            sudo -v || exit
        done
    ) &
    add_pid "$!"

    # Setup v4l2loopback with two devices
    sudo -- bash -c 'modprobe -r v4l2loopback 2>/dev/null || true'
    sudo -- bash -c 'modprobe v4l2loopback devices=2 exclusive_caps=1,1 video_nr=99,100 max_buffers=2,2 card_label="Test Webcam,Test Virtual"'

    # Verify devices exist
    if [[ ! -c "${WEBCAM_DEVICE}" ]] || [[ ! -c "${VIRTUAL_DEVICE}" ]]; then
        print_error "Failed to create v4l2loopback devices"
        exit 1
    fi

    print_info "v4l2loopback devices created successfully"
}

# ============================================================================
# Video Feed Setup
# ============================================================================

setup_video_feed() {
    local width="$1"
    local height="$2"
    local fps="$3"
    local ffmpeg_cmd
    ffmpeg_cmd="$(get_ffmpeg_command)"

    print_info "Downloading test video..."
    local video_file
    video_file="$(mktemp --suffix=.mp4)"
    add_temp_file "${video_file}"

    if ! wget -qO "${video_file}" "${TEST_VIDEO_URL}"; then
        print_error "Failed to download test video"
        exit 1
    fi

    print_info "Starting video feed (${width}x${height}@${fps}fps)..."
    "${ffmpeg_cmd}" -hide_banner -loglevel quiet -hwaccel auto \
        -re -stream_loop -1 -i "${video_file}" \
        -vf "scale=${width}:${height}" -r "${fps}" \
        -f v4l2 -an -vcodec rawvideo -pix_fmt yuyv422 "${WEBCAM_DEVICE}" &
    add_pid "$!"

    # Wait for video feed to stabilize
    sleep 3
}

# ============================================================================
# Benchmarking
# ============================================================================

collect_system_info() {
    echo "==================================================================="
    echo "SYSTEM INFORMATION"
    echo "==================================================================="
    echo ""

    # OS information
    echo "Operating System:"
    uname -srvmo
    echo ""

    # CPU information
    echo "CPU:"
    sed -n 's/^model name\s*:\s*//p' /proc/cpuinfo | sort | uniq -c
    echo ""

    # GPU information (if available)
    if command -v nvidia-smi &> /dev/null; then
        echo "GPU:"
        nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader
        echo ""
    fi

    # Python environment
    echo "Python Environment:"
    uv run python3 -c '
import sys
import importlib.metadata

print(f"Python: {sys.version.split()[0]}")
print(f"Virtual environment: {hasattr(sys, 'real_prefix') or (hasattr(sys, 'base_prefix') and sys.base_prefix != sys.prefix)}")
print()

packages = [
    "opencv-python", "mediapipe", "pyfakewebcam", "numpy",
    "ultralytics", "onnxruntime", "onnxruntime-gpu"
]

for pkg in packages:
    try:
        version = importlib.metadata.version(pkg)
        print(f"{pkg}: {version}")
    except importlib.metadata.PackageNotFoundError:
        pass
'
    echo ""
}

parse_benchmark_args() {
    local model="$1"
    local extra_args="$2"

    # Base command with common settings
    local base_args="--no-ondemand"

    # Model-specific default arguments
    case "${model}" in
        "mediapipe")
            echo "${base_args} --model mediapipe ${extra_args}"
            ;;
        "yolo11n-seg")
            echo "${base_args} --model yolo11n-seg --use-clahe --adaptive-threshold --temporal-smoothing 0.3 --fast-blur ${extra_args}"
            ;;
        "fastsam-s")
            echo "${base_args} --model fastsam-s --use-clahe --adaptive-threshold --temporal-smoothing 0.3 ${extra_args}"
            ;;
        *)
            print_error "Unknown model: ${model}"
            exit 1
            ;;
    esac
}

run_benchmark() {
    local model="$1"
    local width="$2"
    local height="$3"
    local fps="$4"
    local timeout="$5"
    local extra_args="$6"

    print_info "Running benchmark for model: ${model}"

    local benchmark_args
    benchmark_args="$(parse_benchmark_args "${model}" "${extra_args}")"

    # shellcheck disable=SC2086
    PYTHONUNBUFFERED=yes timeout --foreground --preserve-status -s QUIT "${timeout}" \
        uv run chameleon run \
            -w "${WEBCAM_DEVICE}" \
            -v "${VIRTUAL_DEVICE}" \
            -W "${width}" \
            -H "${height}" \
            -F "${fps}" \
            ${benchmark_args} |
        stdbuf -oL tr '\r' '\n' |
        uv run python3 -c '
import sys
import statistics

fps_values = []
for line in sys.stdin:
    if line.startswith("FPS:"):
        try:
            fps = float(line.split(":")[-1].strip())
            fps_values.append(fps)
            print(f"{fps:.2f}", end="\r", flush=True, file=sys.stderr)
        except (ValueError, IndexError):
            pass

if fps_values:
    mean_fps = statistics.mean(fps_values)
    stdev_fps = statistics.stdev(fps_values) if len(fps_values) > 1 else 0.0
    min_fps = min(fps_values)
    max_fps = max(fps_values)

    print(f"\nResults: avg={mean_fps:.2f} fps, stdev={stdev_fps:.2f}, min={min_fps:.2f}, max={max_fps:.2f}")
else:
    print("No FPS data collected", file=sys.stderr)
'
}

# ============================================================================
# Main Benchmark Flow
# ============================================================================

run_all_benchmarks() {
    local width="$1"
    local height="$2"
    local fps="$3"
    local timeout="$4"
    local extra_args="$5"
    local git_head

    # Get git information
    if git rev-parse HEAD &> /dev/null; then
        git_head="$(git rev-parse HEAD)"
    else
        git_head="unknown"
    fi

    # Create output file
    local output_file="benchmark.${git_head}.txt"

    {
        echo "==================================================================="
        echo "BENCHMARK CONFIGURATION"
        echo "==================================================================="
        echo "Resolution: ${width}x${height} @ ${fps} fps"
        echo "Duration: ${timeout} seconds per test"
        echo "Git HEAD: ${git_head}"
        echo "Extra arguments: ${extra_args}"
        echo ""

        collect_system_info

        echo "==================================================================="
        echo "BENCHMARK RESULTS"
        echo "==================================================================="
        echo ""

        # Run benchmarks for each model
        for model in "${MODELS[@]}"; do
            echo "-------------------------------------------------------------------"
            echo "Model: ${model}"
            echo "-------------------------------------------------------------------"

            if run_benchmark "${model}" "${width}" "${height}" "${fps}" "${timeout}" "${extra_args}"; then
                echo "✓ Benchmark completed successfully"
            else
                echo "✗ Benchmark failed or was interrupted"
            fi
            echo ""
        done

        echo "==================================================================="
        echo "BENCHMARK COMPLETE"
        echo "==================================================================="
    } | tee "${output_file}"

    print_info "Benchmark results saved to: ${output_file}"
}

# ============================================================================
# Usage and Argument Parsing
# ============================================================================

usage() {
    cat <<EOF
Usage: $(basename "$0") [OPTIONS]

Benchmark Chameleon with different segmentation models and settings.

OPTIONS:
    -w, --width WIDTH       Video width (default: ${DEFAULT_WIDTH})
    -h, --height HEIGHT     Video height (default: ${DEFAULT_HEIGHT})
    -f, --fps FPS           Frame rate (default: ${DEFAULT_FPS})
    -t, --timeout SECONDS   Benchmark duration per test (default: ${DEFAULT_TIMEOUT})
    -m, --model MODEL       Test only specific model (mediapipe, yolo11n-seg, fastsam-s)
    -e, --extra ARGS        Extra arguments to pass to chameleon
    --help                  Show this help message

EXAMPLES:
    # Full benchmark at 1080p60
    $(basename "$0")

    # Quick benchmark at 720p30
    $(basename "$0") -w 1280 -h 720 -f 30 -t 60

    # Test only YOLO11n model
    $(basename "$0") -m yolo11n-seg

    # Custom background effects
    $(basename "$0") -e "--background=blur=50"

NOTES:
    - Requires root access for v4l2loopback module management
    - Creates temporary video devices /dev/video99 and /dev/video100
    - Results are saved to benchmark.<git-hash>.txt
EOF
}

main() {
    local width="${DEFAULT_WIDTH}"
    local height="${DEFAULT_HEIGHT}"
    local fps="${DEFAULT_FPS}"
    local timeout="${DEFAULT_TIMEOUT}"
    local extra_args="--background=no,blur=50"
    local single_model=""

    # Parse arguments
    while [[ $# -gt 0 ]]; do
        case "$1" in
            -w|--width)
                width="$2"
                shift 2
                ;;
            -h|--height)
                height="$2"
                shift 2
                ;;
            -f|--fps)
                fps="$2"
                shift 2
                ;;
            -t|--timeout)
                timeout="$2"
                shift 2
                ;;
            -m|--model)
                single_model="$2"
                shift 2
                ;;
            -e|--extra)
                extra_args="$2"
                shift 2
                ;;
            --help)
                usage
                exit 0
                ;;
            *)
                print_error "Unknown option: $1"
                usage
                exit 1
                ;;
        esac
    done

    # Validate model if specified
    if [[ -n "${single_model}" ]]; then
        local valid=0
        for model in "${MODELS[@]}"; do
            if [[ "${model}" == "${single_model}" ]]; then
                valid=1
                break
            fi
        done

        if [[ "${valid}" -eq 0 ]]; then
            print_error "Invalid model: ${single_model}"
            print_error "Valid models: ${MODELS[*]}"
            exit 1
        fi

        # Override MODELS array with single model
        MODELS=("${single_model}")
    fi

    # Check requirements
    check_requirements

    # Setup
    setup_v4l2loopback
    setup_video_feed "${width}" "${height}" "${fps}"

    # Run benchmarks
    run_all_benchmarks "${width}" "${height}" "${fps}" "${timeout}" "${extra_args}"
}

# ============================================================================
# Entry Point
# ============================================================================

if [[ "${BASH_SOURCE[0]}" == "${0}" ]]; then
    main "$@"
fi
