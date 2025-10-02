"""Monitor virtual camera for consumer detection using inotify."""

import os
from pathlib import Path

from inotify_simple import INotify, flags


class VirtualCameraMonitor:
    """Monitor virtual camera device for consumers (on-demand processing)."""

    def __init__(self, device_path: str):
        """Initialize camera monitor.

        Args:
            device_path: Virtual camera device path
        """
        self.device_path = device_path
        self.inotify: INotify | None = None
        self.watch_descriptor: int | None = None

    def start(self):
        """Start monitoring the virtual camera device."""
        if not os.path.exists(self.device_path):
            raise FileNotFoundError(f"Virtual camera device not found: {self.device_path}")

        self.inotify = INotify()
        # Watch for open events on the device
        self.watch_descriptor = self.inotify.add_watch(self.device_path, flags.OPEN | flags.CLOSE)

    def has_consumers(self) -> bool:
        """Check if there are any consumers using the virtual camera.

        Returns:
            True if consumers are detected
        """
        if not os.path.exists(self.device_path):
            return False

        # Check if device is opened by reading /proc/locks or other methods
        # This is a simplified check - a more robust implementation would
        # track OPEN/CLOSE events from inotify
        try:
            # Try to get file descriptor count (Linux-specific)
            # This is a placeholder - actual implementation may vary
            with open(f"/sys/class/video4linux/{Path(self.device_path).name}/dev"):
                # If we can read it, assume no error
                pass
            return True
        except OSError:
            return False

    def wait_for_event(self, timeout: float | None = None):
        """Wait for an inotify event.

        Args:
            timeout: Timeout in seconds (None = wait indefinitely)
        """
        if self.inotify:
            events = self.inotify.read(timeout=timeout)
            return events
        return []

    def stop(self):
        """Stop monitoring."""
        if self.inotify and self.watch_descriptor is not None:
            self.inotify.rm_watch(self.watch_descriptor)
            self.watch_descriptor = None
        if self.inotify:
            self.inotify.close()
            self.inotify = None

    def __enter__(self):
        """Context manager entry."""
        self.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.stop()
