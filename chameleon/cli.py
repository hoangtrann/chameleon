"""Typer-based CLI interface for Linux Fake Background Webcam."""

import signal
import sys
from pathlib import Path
from typing import Annotated

import typer
from rich.console import Console

from chameleon.config import CameraConfig, Config, ProcessingConfig, SegmentationModel
from chameleon.core.pipeline import ProcessingPipeline
from chameleon.utils import create_filter_config, parse_filter_string

app = typer.Typer(
    name="chameleon",
    help="Chameleon - Virtual webcam with background effects",
    rich_markup_mode="rich",
    add_completion=False,
)
console = Console()


@app.command()
def run(
    config: Annotated[
        Path,
        typer.Option("-c", "--config", help="Config file path", exists=True),
    ] = None,
    webcam: Annotated[
        str,
        typer.Option("-w", "--webcam-path", help="Real webcam device path"),
    ] = "/dev/video0",
    output: Annotated[
        str,
        typer.Option("-v", "--v4l2loopback-path", help="Virtual camera device path"),
    ] = "/dev/video2",
    width: Annotated[int, typer.Option("-W", "--width", help="Real webcam width")] = 1280,
    height: Annotated[int, typer.Option("-H", "--height", help="Real webcam height")] = 720,
    fps: Annotated[int, typer.Option("-F", "--fps", help="Frame rate")] = 30,
    codec: Annotated[str, typer.Option("-C", "--codec", help="Video codec")] = "MJPG",
    selfie: Annotated[
        str,
        typer.Option("--selfie", help="Selfie effects (comma-separated)"),
    ] = "",
    background: Annotated[
        str,
        typer.Option("--background", help="Background effects (comma-separated)"),
    ] = "file=background.jpg",
    mask: Annotated[
        str,
        typer.Option("--mask", help="Mask effects (comma-separated)"),
    ] = "",
    no_ondemand: Annotated[
        bool,
        typer.Option("--no-ondemand", help="Disable on-demand processing"),
    ] = False,
    use_sigmoid: Annotated[
        bool,
        typer.Option("--use-sigmoid", help="Force mask to follow sigmoid distribution"),
    ] = False,
    threshold: Annotated[
        int,
        typer.Option("--threshold", help="Minimum percentage threshold for foreground (0-100)"),
    ] = 75,
    no_postprocess: Annotated[
        bool,
        typer.Option(
            "--no-postprocess", help="Disable postprocessing (masking dilation and blurring)"
        ),
    ] = False,
    select_model: Annotated[
        int,
        typer.Option("--select-model", help="Select MediaPipe model (0 or 1)"),
    ] = 1,
    model: Annotated[
        str,
        typer.Option("--model", help="Segmentation model (mediapipe, yolo11n-seg, fastsam-s)"),
    ] = "mediapipe",
    use_clahe: Annotated[
        bool,
        typer.Option("--use-clahe", help="Enable CLAHE for better low-light performance"),
    ] = False,
    adaptive_threshold: Annotated[
        bool,
        typer.Option(
            "--adaptive-threshold/--no-adaptive-threshold", help="Use Otsu's adaptive threshold"
        ),
    ] = True,
    temporal_smoothing: Annotated[
        float,
        typer.Option(
            "--temporal-smoothing",
            help="Temporal mask smoothing (0-1, 0.3 recommended)",
            min=0.0,
            max=1.0,
        ),
    ] = 0.0,
    fast_blur: Annotated[
        bool,
        typer.Option("--fast-blur/--no-fast-blur", help="Use optimized downscale-blur-upscale"),
    ] = True,
    dump: Annotated[
        bool,
        typer.Option("--dump", help="Dump filter configuration and exit"),
    ] = False,
):
    """Run the virtual webcam with background effects.

    Examples:
        # Basic usage with optimizations
        chameleon run --model yolo11n-seg --use-clahe --temporal-smoothing 0.3

        # Background blur with fast mode
        chameleon run --background=no,blur=50 --fast-blur

        # Custom effects
        chameleon run --selfie=blur=30,hologram
        chameleon run --background=file=mybg.jpg,cmap=viridis
        chameleon run --mask=foreground=logo.png,opacity=80

        # Performance tuning
        chameleon run --model yolo11n-seg --adaptive-threshold --temporal-smoothing 0.3
    """
    # Parse filter strings into FilterConfig objects
    selfie_filters = parse_filter_string(selfie)
    selfie_config = create_filter_config(selfie_filters, "selfie")

    background_filters = parse_filter_string(background)
    background_config = create_filter_config(background_filters, "background")

    mask_filters = parse_filter_string(mask)
    mask_config = create_filter_config(mask_filters, "mask")

    # Handle dump mode
    if dump:
        console.print("[bold]Filter Configuration:[/bold]")
        console.print(f"\n[cyan]Selfie:[/cyan]\n{selfie_config.model_dump_json(indent=2)}")
        console.print(f"\n[cyan]Background:[/cyan]\n{background_config.model_dump_json(indent=2)}")
        console.print(f"\n[cyan]Mask:[/cyan]\n{mask_config.model_dump_json(indent=2)}")
        return

    # Create camera configuration
    camera_config = CameraConfig(
        device=webcam,
        width=width,
        height=height,
        fps=fps,
        codec=codec,
    )

    # Parse model string to SegmentationModel enum
    try:
        model_enum = SegmentationModel(model)
    except ValueError:
        console.print(
            f"[red]Invalid model: {model}. Valid options: mediapipe, yolo11n-seg, fastsam-s[/red]"
        )
        raise typer.Exit(1) from None

    # Create processing configuration
    processing_config = ProcessingConfig(
        use_sigmoid=use_sigmoid,
        threshold=threshold,
        enable_postprocess=not no_postprocess,
        model=model_enum,
        model_selection=select_model,
        use_clahe=use_clahe,
        adaptive_threshold=adaptive_threshold,
        temporal_smoothing=temporal_smoothing,
        fast_blur=fast_blur,
    )

    # Create main application configuration
    app_config = Config(
        real_camera=camera_config,
        virtual_camera=output,
        processing=processing_config,
        selfie_filter=selfie_config,
        background_filter=background_config,
        mask_filter=mask_config,
        no_ondemand=no_ondemand,
    )

    # Create and setup pipeline
    console.print("[bold green]Starting Chameleon...[/bold green]")
    console.print(f"Real camera: {webcam}")
    console.print(f"Virtual camera: {output}")
    console.print(f"Resolution: {width}x{height} @ {fps}fps")

    try:
        pipeline = ProcessingPipeline(app_config)

        # Setup signal handlers
        def sigint_handler(sig, frame):
            pipeline.toggle_pause()

        def sigquit_handler(sig, frame):
            console.print("\n[yellow]Stopping...[/yellow]")
            pipeline.stop()
            sys.exit(0)

        signal.signal(signal.SIGINT, sigint_handler)
        signal.signal(signal.SIGQUIT, sigquit_handler)

        # Run pipeline
        with pipeline:
            pipeline.run()

    except KeyboardInterrupt:
        console.print("\n[yellow]Interrupted[/yellow]")
    except Exception as e:
        console.print(f"[bold red]Error:[/bold red] {e}")
        raise


@app.command()
def list_devices():
    """List available video devices and their capabilities."""
    console.print("[bold yellow]list-devices command not yet implemented[/bold yellow]")
    console.print("This will be implemented in Phase 2 of the refactoring.")
    console.print("\nFor now, you can use:")
    console.print("  v4l2-ctl --list-devices")


@app.command()
def benchmark(
    duration: Annotated[
        int,
        typer.Option("--duration", help="Benchmark duration in seconds"),
    ] = 30,
):
    """Benchmark performance with different settings."""
    console.print("[bold yellow]benchmark command not yet implemented[/bold yellow]")
    console.print("This will be implemented in Phase 2 of the refactoring.")


@app.command()
def download_model(
    model_name: Annotated[
        str,
        typer.Argument(help="Model to download (yolo11n-seg, fastsam-s, mediapipe)"),
    ],
):
    """Download and cache a segmentation model."""
    console.print("[bold yellow]download-model command not yet implemented[/bold yellow]")
    console.print("This will be implemented in Phase 5 of the refactoring.")


# Make the default command be 'run' when no subcommand is given
@app.callback(invoke_without_command=True)
def main(ctx: typer.Context):
    """Chameleon - Virtual webcam with background effects.

    Run 'chameleon run --help' for usage information.
    """
    if ctx.invoked_subcommand is None:
        # No subcommand provided, show help
        console.print(
            "[yellow]No command specified. Use 'chameleon run' to start the virtual webcam.[/yellow]"
        )
        console.print("\nAvailable commands:")
        console.print("  [cyan]run[/cyan]            - Run the virtual webcam")
        console.print("  [cyan]list-devices[/cyan]  - List available video devices")
        console.print("  [cyan]benchmark[/cyan]     - Benchmark performance")
        console.print("  [cyan]download-model[/cyan] - Download segmentation models")
        console.print("\nRun 'chameleon --help' for more information.")


if __name__ == "__main__":
    app()
