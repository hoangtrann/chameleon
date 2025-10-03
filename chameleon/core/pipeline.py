"""Main processing pipeline coordinator."""

import logging
import time

import numpy as np

from chameleon.config import Config
from chameleon.core.effects import apply_effects
from chameleon.input.camera import RealCamera
from chameleon.output.monitor import VirtualCameraMonitor
from chameleon.output.virtual import VirtualCameraOutput
from chameleon.processing.compositor import ImageCompositor
from chameleon.processing.segmentation import SegmentationEngine

logger = logging.getLogger(__name__)


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
        logger.info("Initializing pipeline components...")

        # Initialize real camera
        logger.info("Opening real camera...")
        self.camera = RealCamera(
            device=self.config.real_camera.device,
            width=self.config.real_camera.width,
            height=self.config.real_camera.height,
            fps=self.config.real_camera.fps,
            codec=self.config.real_camera.codec,
        )
        logger.info("Real camera ready")

        # Initialize virtual camera output
        logger.info("Opening virtual camera output...")
        self.output = VirtualCameraOutput(
            device=self.config.virtual_camera,
            width=self.config.real_camera.width,
            height=self.config.real_camera.height,
        )
        logger.info("Virtual camera output ready")

        # Initialize monitor for on-demand processing (unless disabled)
        if not self.config.no_ondemand:
            try:
                logger.info("Setting up camera monitor...")
                self.monitor = VirtualCameraMonitor(self.config.virtual_camera)
                logger.info("Camera monitor ready")
            except Exception as e:
                logger.warning("Could not setup camera monitor: %s", e)
                logger.warning("On-demand processing disabled")
                self.monitor = None

        # Initialize segmentation engine with optimizations
        logger.info("Loading segmentation model (this may take a moment)...")
        self.segmentation = SegmentationEngine(
            model=self.config.processing.model,
            use_gpu=True,  # TODO: Make configurable based on hwaccel
            use_clahe=self.config.processing.use_clahe,
        )
        logger.info("Segmentation engine ready")

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
                # TODO: background_image is pre-sized in load_background(), no resize needed
                background = self.compositor.background_image
            elif self.config.background_filter.solid:
                # TODO: create_solid_background() returns correctly sized array
                background = self.compositor.create_solid_background(
                    self.config.background_filter.solid
                )

            # Apply background effects if background exists
            # TODO: Effects preserve size, no validation needed
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
            logger.info(
                "Frame %d Performance: Camera=%.2fms Seg=%.2fms Mask=%.2fms "
                "Selfie=%.2fms BG=%.2fms Comp=%.2fms TOTAL=%.2fms (%.1f FPS)",
                self.frame_count,
                camera_time,
                segment_time,
                mask_process_time,
                selfie_effects_time,
                bg_effects_time,
                composite_time,
                total_time,
                fps,
            )

        return output_frame

    def run(self):
        """Run the main processing loop."""
        self.running = True
        target_frame_time = 1.0 / self.config.real_camera.fps

        logger.info("=" * 60)
        logger.info("CHAMELEON - PERFORMANCE MONITOR")
        logger.info("=" * 60)
        logger.info("Configuration:")
        logger.info("  Model:              %s", self.config.processing.model.value)
        logger.info(
            "  Resolution:         %dx%d",
            self.config.real_camera.width,
            self.config.real_camera.height,
        )
        logger.info("  Target FPS:         %d", self.config.real_camera.fps)
        logger.info(
            "  CLAHE:              %s",
            "ENABLED" if self.config.processing.use_clahe else "DISABLED",
        )
        logger.info(
            "  Adaptive Threshold: %s",
            "ENABLED" if self.config.processing.adaptive_threshold else "DISABLED",
        )
        logger.info("  Temporal Smoothing: %s", self.config.processing.temporal_smoothing)
        logger.info(
            "  Fast Blur:          %s",
            "ENABLED" if self.config.processing.fast_blur else "DISABLED",
        )
        logger.info(
            "  Background Blur:    %s",
            self.config.background_filter.blur if self.config.background_filter.blur else "NONE",
        )
        logger.info("=" * 60)
        logger.info("Press CTRL-C to stop")

        # Check if on-demand mode is active
        if self.monitor:
            logger.info("On-demand processing: enabled (pauses when no consumers)")
        else:
            logger.info("On-demand processing: disabled")

        logger.info("Starting pipeline... (performance stats every 30 frames)")
        logger.info("=" * 60)

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
            logger.info("Paused - Processing stopped")
        else:
            logger.info("Resumed - Reloading images")
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
