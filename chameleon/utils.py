"""Utility functions for backward compatibility and filter parsing."""

import sys

from cmapy import cmap_groups

from chameleon.config import FilterConfig

# Extract all colormap names from cmapy groups
colormaps = [cmap for group in cmap_groups for cmap in group["colormaps"]]


def parse_filter_string(filter_string: str) -> list[list]:
    """Process comma-separated filter arguments into a list of [name, *args] lists.

    This maintains backward compatibility with the legacy filter syntax.

    Args:
        filter_string: Comma-separated filter string (e.g., "blur=30,hologram")

    Returns:
        List of filter specifications
    """
    if not filter_string:
        return []

    filters = []
    i = 0

    while i < len(filter_string):
        # Find the next equals sign
        eq_pos = filter_string.find("=", i)

        # Check for simple flags at current position
        for flag in ["no", "hologram", "tile", "crop"]:
            if filter_string[i:].startswith(flag):
                # Check if it's followed by comma or end of string
                flag_end = i + len(flag)
                if flag_end >= len(filter_string) or filter_string[flag_end] == ",":
                    filters.append([flag])
                    i = flag_end
                    if i < len(filter_string) and filter_string[i] == ",":
                        i += 1
                    continue

        if eq_pos == -1 or eq_pos < i:
            # No more equals signs
            remaining = filter_string[i:].strip()
            if remaining and remaining not in ["no", "hologram", "tile", "crop"]:
                print(f"Warning: Unknown filter '{remaining}'", file=sys.stderr)
            break

        # Get the filter name
        name = filter_string[i:eq_pos].strip()
        if i > 0 and filter_string[i - 1] == ",":
            name = name[0:] if not name.startswith(",") else name[1:]

        # Find the value - need special handling for solid colors
        if name == "solid":
            # For solid colors, we need to find 3 comma-separated numbers
            value_start = eq_pos + 1
            # Skip whitespace
            while value_start < len(filter_string) and filter_string[value_start].isspace():
                value_start += 1

            # Count commas to find BGR values
            comma_count = 0
            value_end = value_start
            while value_end < len(filter_string) and comma_count < 2:
                if filter_string[value_end] == ",":
                    comma_count += 1
                value_end += 1

            # Find the end of the third number
            while value_end < len(filter_string) and (
                filter_string[value_end].isdigit() or filter_string[value_end].isspace()
            ):
                value_end += 1

            value = filter_string[value_start:value_end].strip()
            i = value_end
        else:
            # For other filters, find the next comma that's not inside a value
            value_start = eq_pos + 1
            next_comma = filter_string.find(",", value_start)

            # Check if there's another filter after this
            next_eq = filter_string.find("=", value_start)
            if next_eq != -1 and (next_comma == -1 or next_eq < next_comma):
                # There's another filter, find where this value ends
                j = next_eq - 1
                while j > value_start and filter_string[j].isspace():
                    j -= 1
                # Now backtrack to find the comma or start
                while j > value_start and filter_string[j] not in ",=":
                    j -= 1
                value_end = j if filter_string[j] == "," else j + 1
            elif next_comma != -1:
                value_end = next_comma
            else:
                value_end = len(filter_string)

            value = filter_string[value_start:value_end].strip()
            i = value_end

        # Skip comma if present
        if i < len(filter_string) and filter_string[i] == ",":
            i += 1

        # Process the filter
        if name == "no" or name == "hologram" or name == "tile" or name == "crop":
            filters.append([name])
        elif (
            name == "file"
            and value
            or name == "foreground"
            and value
            or name == "mask-file"
            and value
        ):
            filters.append([name, value])
        elif name == "opacity" and value:
            try:
                opacity_val = int(value)
                filters.append([name, opacity_val])
            except ValueError:
                print(f"Warning: Invalid opacity value '{value}'", file=sys.stderr)
        elif name == "brightness" and value:
            try:
                brightness_val = int(value)
                filters.append([name, brightness_val])
            except ValueError:
                print(f"Warning: Invalid brightness value '{value}'", file=sys.stderr)
        elif name == "cmap" and value:
            if value in colormaps:
                filters.append([name, value])
            else:
                print(f"Warning: Unknown colormap '{value}'", file=sys.stderr)
        elif name == "blur" and value:
            try:
                blur_val = int(value)
                filters.append([name, blur_val])
            except ValueError:
                print(f"Warning: Invalid blur value '{value}'", file=sys.stderr)
        elif name == "solid" and value:
            try:
                # Parse BGR color (e.g., "255,0,0")
                parts = [int(x.strip()) for x in value.split(",")]
                if len(parts) == 3:
                    filters.append([name, tuple(parts)])
                else:
                    print(
                        f"Warning: Invalid solid color '{value}' (expected B,G,R)", file=sys.stderr
                    )
            except ValueError:
                print(f"Warning: Invalid solid color '{value}'", file=sys.stderr)
        elif name == "mask-update-speed" and value:
            try:
                speed_val = float(value)
                # Convert from percentage (0-100) to decimal (0.0-1.0) if needed
                if speed_val > 1.0:
                    speed_val = min(max(speed_val, 0), 100) / 100.0
                filters.append([name, speed_val])
            except ValueError:
                print(f"Warning: Invalid mask-update-speed value '{value}'", file=sys.stderr)

    return filters


def create_filter_config(filter_list: list[list], component_type: str = "selfie") -> FilterConfig:
    """Convert parsed filter list to FilterConfig object.

    Args:
        filter_list: List of filter specifications from parse_filter_string
        component_type: 'selfie', 'background', or 'mask'

    Returns:
        FilterConfig object
    """
    # Set defaults based on component type
    defaults = {
        "selfie": {
            "disabled": False,
            "file": None,
            "hologram": False,
            "blur": None,
            "solid": None,
            "cmap": None,
            "tile": False,
            "crop": False,
            "mask_update_speed": None,
            "foreground_file": None,
            "mask_file": None,
            "opacity": None,
            "brightness": None,
        },
        "background": {
            "disabled": False,
            "file": "background.jpg",
            "hologram": False,
            "blur": None,
            "solid": None,
            "cmap": None,
            "tile": False,
            "crop": False,
            "mask_update_speed": 0.5,
            "foreground_file": None,
            "mask_file": None,
            "opacity": None,
            "brightness": None,
        },
        "mask": {
            "disabled": False,
            "file": "foreground-mask.png",
            "hologram": False,
            "blur": None,
            "solid": None,
            "cmap": None,
            "tile": False,
            "crop": False,
            "mask_update_speed": None,
            "foreground_file": "foreground.jpg",
            "mask_file": "foreground-mask.png",
            "opacity": None,
            "brightness": None,
        },
    }

    # Start with defaults for the component type
    config_dict = defaults.get(component_type, defaults["selfie"]).copy()

    # Apply filters from the list
    for filter_spec in filter_list:
        name = filter_spec[0]
        args = filter_spec[1:] if len(filter_spec) > 1 else []

        if name == "no":
            config_dict["disabled"] = True
            config_dict["file"] = None
            # For mask component, also clear foreground files
            if component_type == "mask":
                config_dict["foreground_file"] = None
                config_dict["mask_file"] = None
        elif name == "hologram":
            config_dict["hologram"] = True
        elif name == "tile":
            config_dict["tile"] = True
        elif name == "crop":
            config_dict["crop"] = True
        elif name == "file" and args:
            config_dict["file"] = args[0]
            config_dict["disabled"] = False
        elif name == "blur" and args:
            config_dict["blur"] = args[0]
        elif name == "solid" and args:
            config_dict["solid"] = args[0]
        elif name == "cmap" and args:
            config_dict["cmap"] = args[0]
        elif name == "mask-update-speed" and args:
            config_dict["mask_update_speed"] = args[0]
        elif name == "foreground" and args:
            config_dict["foreground_file"] = args[0]
        elif name == "mask-file" and args:
            config_dict["mask_file"] = args[0]
            config_dict["file"] = args[0]
        elif name == "opacity" and args:
            config_dict["opacity"] = args[0]
        elif name == "brightness" and args:
            config_dict["brightness"] = args[0]

    return FilterConfig(**config_dict)
