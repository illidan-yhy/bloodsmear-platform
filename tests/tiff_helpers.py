"""Small real TIFF fixtures, including signed 16-bit camera data Pillow saves as 32-bit."""
import struct


def grayscale16_tiff(*, byteorder="little", signed=False, low_values=False, orientation=1, photometric=1):
    endian = "<" if byteorder == "little" else ">"
    values = [0, 128, 200, 255] if low_values else ([0, 128, 1024, 32767] if signed else [0, 128, 4096, 65535])
    entries = [
        (256, 4, 4), (257, 4, 1), (258, 3, 16), (259, 3, 1),
        (262, 3, photometric), (273, 4, 146), (274, 3, orientation),
        (277, 3, 1), (278, 4, 1), (279, 4, 8), (339, 3, 2 if signed else 1),
    ]
    result = (b"II" if endian == "<" else b"MM") + struct.pack(endian + "HIH", 42, 8, len(entries))
    for tag, kind, value in entries:
        result += struct.pack(endian + "HHI", tag, kind, 1)
        result += struct.pack(endian + "HH", value, 0) if kind == 3 else struct.pack(endian + "I", value)
    return result + struct.pack(endian + "I", 0) + struct.pack(endian + ("4h" if signed else "4H"), *values)


def color16_tiff():
    # Three actual 16-bit channels; Pillow's existing reader takes their high byte.
    entries = [
        (256, 4, 1), (257, 4, 1), (258, 3, 3, 146), (259, 3, 1),
        (262, 3, 2), (273, 4, 152), (274, 3, 1),
        (277, 3, 3), (278, 4, 1), (279, 4, 6), (339, 3, 1),
    ]
    result = b"II" + struct.pack("<HIH", 42, 8, len(entries))
    for entry in entries:
        tag, kind, value = entry[:3]
        count = value if len(entry) == 4 else 1
        result += struct.pack("<HHI", tag, kind, count)
        result += struct.pack("<I", entry[3]) if len(entry) == 4 else (struct.pack("<HH", value, 0) if kind == 3 else struct.pack("<I", value))
    return result + struct.pack("<I3H3H", 0, 16, 16, 16, 2560, 32768, 61440)
