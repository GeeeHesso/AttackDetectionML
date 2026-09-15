"""Minimal pure-python TFRecord + tf.Event/Summary parser.

No dependency on the `tensorflow` or `tensorboard` packages (not installable
in this sandbox). Only extracts what we need: (step, tag, simple_value)
scalar summary entries. Skips CRC validation (we trust record framing).
"""
import struct
import os


def _read_records(path):
    with open(path, "rb") as f:
        data = f.read()
    pos = 0
    n = len(data)
    while pos < n:
        if pos + 12 > n:
            break
        length = struct.unpack_from("<Q", data, pos)[0]
        pos += 12  # 8-byte length + 4-byte length-crc
        if pos + length + 4 > n:
            break
        record = data[pos:pos + length]
        pos += length + 4  # data + 4-byte data-crc
        yield record


def _read_varint(buf, pos):
    result = 0
    shift = 0
    while True:
        b = buf[pos]
        pos += 1
        result |= (b & 0x7F) << shift
        if not (b & 0x80):
            break
        shift += 7
    return result, pos


def _iter_fields(buf):
    """Yield (field_number, wire_type, value_bytes_or_int) for a protobuf message."""
    pos = 0
    n = len(buf)
    while pos < n:
        tag, pos = _read_varint(buf, pos)
        field_no = tag >> 3
        wire_type = tag & 0x7
        if wire_type == 0:  # varint
            val, pos = _read_varint(buf, pos)
            yield field_no, wire_type, val
        elif wire_type == 1:  # 64-bit
            val = buf[pos:pos + 8]
            pos += 8
            yield field_no, wire_type, val
        elif wire_type == 2:  # length-delimited
            length, pos = _read_varint(buf, pos)
            val = buf[pos:pos + length]
            pos += length
            yield field_no, wire_type, val
        elif wire_type == 5:  # 32-bit
            val = buf[pos:pos + 4]
            pos += 4
            yield field_no, wire_type, val
        else:
            raise ValueError(f"Unsupported wire type {wire_type} at pos {pos}")


def parse_event_record(record):
    """Return (step, [(tag, value), ...]) for one Event protobuf record."""
    step = None
    scalars = []
    for field_no, wire_type, val in _iter_fields(record):
        if field_no == 2 and wire_type == 0:  # step
            step = val
        elif field_no == 5 and wire_type == 2:  # summary
            for sf_no, sf_wt, sf_val in _iter_fields(val):
                if sf_no == 1 and sf_wt == 2:  # Summary.value (repeated Value)
                    tag = None
                    simple_value = None
                    for vf_no, vf_wt, vf_val in _iter_fields(sf_val):
                        if vf_no == 1 and vf_wt == 2:  # tag
                            tag = vf_val.decode("utf-8", errors="replace")
                        elif vf_no == 2 and vf_wt == 5:  # simple_value (float32)
                            simple_value = struct.unpack("<f", vf_val)[0]
                    if tag is not None and simple_value is not None:
                        scalars.append((tag, simple_value))
    return step, scalars


def read_scalars(event_file_path):
    """Yield (step, tag, value) tuples from one events.out.tfevents.* file."""
    for record in _read_records(event_file_path):
        step, scalars = parse_event_record(record)
        if step is None:
            continue
        for tag, value in scalars:
            yield step, tag, value


def find_event_file(run_dir):
    for fname in os.listdir(run_dir):
        if "tfevents" in fname:
            return os.path.join(run_dir, fname)
    return None


if __name__ == "__main__":
    import sys
    p = find_event_file(sys.argv[1]) if len(sys.argv) > 1 and os.path.isdir(sys.argv[1]) else sys.argv[1]
    tags = {}
    count = 0
    for step, tag, value in read_scalars(p):
        count += 1
        tags.setdefault(tag, 0)
        tags[tag] += 1
    print(f"{count} scalar points read from {p}")
    for t, c in sorted(tags.items()):
        print(f"  {t}: {c} points")
