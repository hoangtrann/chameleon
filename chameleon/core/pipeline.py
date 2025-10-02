"""Main processing pipeline coordinator."""

import time

import numpy as np

from chameleon.config import Config
from chameleon.core.effects import apply_effects
from chameleon.input.camera import RealCamera
from chameleon.output.monitor import VirtualCameraMonitor
from chameleon.output.virtual import VirtualCameraOutput
from chameleon.processing.compositor import ImageCompositor
from chameleon.processing.segmentation import SegmentationEngine


class ProcessingPipeline:
    """Main processing pipeline coordinating all components."""

    def __init__(self, config: Config):
        """Initialize processing pipeline.

        Args:
            config: Application configuration
        """
        self.config = config
        self.running = False
        self.paused = False

        # Initialize components
        self.camera: RealCamera | None = None
        self.output: VirtualCameraOutput | None = None
        self.monitor: VirtualCameraMonitor | None = None
        self.segmentation: SegmentationEngine | None = None
        self.compositor: ImageCompositor | None = None
        self.frame_count = 0
        self.last_check_time = 0

    def setup(self):
        """Setup all pipeline components."""
        print("\n[Setup] Initializing pipeline components...")

        # Initialize real camera
        print("[Setup] Opening real camera...")
        self.camera = RealCamera(
            device=self.config.real_camera.device,
            width=self.config.real_camera.width,
            height=self.config.real_camera.height,
            fps=self.config.real_camera.fps,
            codec=self.config.real_camera.codec,
        )
        print("[Setup] Real camera ready")

        # Initialize virtual camera output
        print("[Setup] Opening virtual camera output...")
        self.output = VirtualCameraOutput(
            device=self.config.virtual_camera,
            width=self.config.real_camera.width,
            height=self.config.real_camera.height,
        )
        print("[Setup] Virtual camera output ready")

        # Initialize monitor for on-demand processing (unless disabled)
        if not self.config.no_ondemand:
            try:
                print("[Setup] Setting up camera monitor...")
                self.monitor = VirtualCameraMonitor(self.config.virtual_camera)
                print("[Setup] Camera monitor ready")
            except Exception as e:
                print(f"[Setup] Warning: Could not setup camera monitor: {e}")
                print("[Setup] On-demand processing disabled")
                self.monitor = None

        # Initialize segmentation engine with optimizations
        print("[Setup] Loading segmentation model (this may take a moment)...")
        self.segmentation = SegmentationEngine(
            model=self.config.processing.model,
            use_gpu=True,  # TODO: Make configurable based on hwaccel
            use_clahe=self.config.processing.use_clahe,
        )
        print("[Setup] Segmentation engine ready")

        # Initialize compositor with temporal smoothing
        self.compositor = ImageCompositor(
            width=self.config.real_camera.width,
            height=self.config.real_camera.height,
            background_config=self.config.background_filter,
            selfie_config=self.config.selfie_filter,
            mask_config=self.config.mask_filter,
            temporal_smoothing=self.config.processing.temporal_smoothing,
        )

        # Load background image if specified
        if self.config.background_filter.file:
            self.compositor.load_background(self.config.background_filter.file)

        # Load foreground image if specified
        if self.config.mask_filter.foreground_file:
            self.compositor.load_foreground(
                self.config.mask_filter.foreground_file,
                self.config.mask_filter.mask_file,
            )

    def process_frame(self) -> np.ndarray | None:
        """Process a single frame through the pipeline.

        Returns:
            Processed frame or None if failed
        """
        frame_start = time.perf_counter()

        # Read frame from camera
        t0 = time.perf_counter()
        frame = self.camera.read_frame()
        if frame is None:
            return None
        camera_time = (time.perf_counter() - t0) * 1000

        # Run segmentation
        t0 = time.perf_counter()
        mask = self.segmentation.segment(frame)
        segment_time = (time.perf_counter() - t0) * 1000

        # Process mask with adaptive threshold
        t0 = time.perf_counter()
        mask = self.compositor.process_mask(
            mask,
            threshold=self.config.processing.threshold / 100.0,
            use_sigmoid=self.config.processing.use_sigmoid,
            enable_postprocess=self.config.processing.enable_postprocess,
            adaptive_threshold=self.config.processing.adaptive_threshold,
        )
        mask_process_time = (time.perf_counter() - t0) * 1000

        # Apply selfie effects
        t0 = time.perf_counter()
        if (
            self.config.selfie_filter.blur
            or self.config.selfie_filter.hologram
            or self.config.selfie_filter.brightness
            or self.config.selfie_filter.opacity
        ):
            frame = apply_effects(frame, self.config.selfie_filter.model_dump())
        selfie_effects_time = (time.perf_counter() - t0) * 1000

        # Prepare background
        t0 = time.perf_counter()
        background = None
        if not self.config.background_filter.disabled:
            if self.config.background_filter.file:
                background = self.compositor.background_image
            elif self.config.background_filter.solid:
                background = self.compositor.create_solid_background(
                    self.config.background_filter.solid
                )

            # Apply background effects if background exists
            if background is not None and (
                self.config.background_filter.blur
                or self.config.background_filter.cmap
                or self.config.background_filter.brightness
            ):
                background = apply_effects(background, self.config.background_filter.model_dump())
        else:
            # Background disabled - use original camera frame with effects
            # This allows blur/effects on the real background instead of replacement
            if (
                self.config.background_filter.blur
                or self.config.background_filter.cmap
                or self.config.background_filter.brightness
            ):
                background = frame.copy()
                background = apply_effects(background, self.config.background_filter.model_dump())
        bg_effects_time = (time.perf_counter() - t0) * 1000

        # Composite layers
        t0 = time.perf_counter()
        output_frame = self.compositor.compose(frame, mask, background)
        composite_time = (time.perf_counter() - t0) * 1000

        total_time = (time.perf_counter() - frame_start) * 1000

        # Log timing every 30 frames (~1 second at 30fps)
        self.frame_count += 1
        if self.frame_count % 30 == 0:
            fps = 1000.0 / total_time if total_time > 0 else 0
            print(f"\n[Frame {self.frame_count}] Performance Breakdown (ms):")
            print(f"  Camera Read:      {camera_time:6.2f}ms")
            print(f"  Segmentation:     {segment_time:6.2f}ms  <-- Main bottleneck")
            print(f"  Mask Processing:  {mask_process_time:6.2f}ms")
            print(f"  Selfie Effects:   {selfie_effects_time:6.2f}ms")
            print(f"  Background FX:    {bg_effects_time:6.2f}ms")
            print(f"  Composition:      {composite_time:6.2f}ms")
            print("  ────────────────────────────")
            print(f"  TOTAL:            {total_time:6.2f}ms  ({fps:.1f} FPS)")

        return output_frame

    def run(self):
        """Run the main processing loop."""
        self.running = True
        target_frame_time = 1.0 / self.config.real_camera.fps

        print("\n" + "=" * 60)
        print("CHAMELEON - PERFORMANCE MONITOR")
        print("=" * 60)
        print("Configuration:")
        print(f"  Model:              {self.config.processing.model.value}")
        print(
            f"  Resolution:         {self.config.real_camera.width}x{self.config.real_camera.height}"
        )
        print(f"  Target FPS:         {self.config.real_camera.fps}")
        print(
            f"  CLAHE:              {'ENABLED' if self.config.processing.use_clahe else 'DISABLED'}"
        )
        print(
            f"  Adaptive Threshold: {'ENABLED' if self.config.processing.adaptive_threshold else 'DISABLED'}"
        )
        print(f"  Temporal Smoothing: {self.config.processing.temporal_smoothing}")
        print(
            f"  Fast Blur:          {'ENABLED' if self.config.processing.fast_blur else 'DISABLED'}"
        )
        print(
            f"  Background Blur:    {self.config.background_filter.blur if self.config.background_filter.blur else 'NONE'}"
        )
        print("=" * 60)
        print("Press CTRL-C to toggle pause/reload")
        print("Press CTRL-\\ to exit")

        # Check if on-demand mode is active
        if self.monitor:
            print("On-demand processing: enabled (pauses when no consumers)")
        else:
            print("On-demand processing: disabled")

        print("\nStarting pipeline... (performance stats every 30 frames)")
        print("=" * 60)

        while self.running:
            start_time = time.monotonic()

            # Check for consumers periodically (every 30 frames ~1 second at 30fps)
            should_process = True
            if self.monitor and not self.config.no_ondemand and self.frame_count % 30 == 0:
                # Simple check: if we can access the device, assume consumers exist
                # This is a simplified implementation
                should_process = True  # For now, always process
                # TODO: Implement proper consumer detection

            # Process and output frame
            if not self.paused and should_process:
                frame = self.process_frame()
                if frame is not None:
                    self.output.write_frame(frame)
            else:
                # Output black frame when paused or no consumers
                black_frame = np.zeros(
                    (
                        self.config.real_camera.height,
                        self.config.real_camera.width,
                        3,
                    ),
                    dtype=np.uint8,
                )
                self.output.write_frame(black_frame)
                # Slow down when paused/no consumers
                time.sleep(1.0)
                continue

            # Maintain target FPS
            elapsed = time.monotonic() - start_time
            sleep_time = max(0, target_frame_time - elapsed)
            time.sleep(sleep_time)

    def toggle_pause(self):
        """Toggle pause state and reload images."""
        self.paused = not self.paused
        if self.paused:
            print("\nPaused - Processing stopped")
        else:
            print("\nResumed - Reloading images")
            # Reload images
            if self.config.background_filter.file:
                self.compositor.load_background(self.config.background_filter.file)
            if self.config.mask_filter.foreground_file:
                self.compositor.load_foreground(
                    self.config.mask_filter.foreground_file,
                    self.config.mask_filter.mask_file,
                )

    def stop(self):
        """Stop the pipeline."""
        self.running = False

    def cleanup(self):
        """Cleanup resources."""
        if self.camera:
            self.camera.close()
        if self.output:
            self.output.close()
        if self.monitor:
            self.monitor.stop()
        if self.segmentation:
            self.segmentation.close()

    def __enter__(self):
        """Context manager entry."""
        self.setup()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.cleanup()
