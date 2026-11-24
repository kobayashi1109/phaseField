#!/usr/bin/env python3
"""Compare two VTK XML .vtu files without numpy or vtk.

Supports ASCII, inline base64, appended base64/raw, uncompressed or
vtkZLibDataCompressor compressed VTK XML arrays. Prints errors for PointData u.
Python 3 standard library only. Does not modify the VTU files.
"""

import base64
import binascii
import math
from pathlib import Path
import re
import struct
import sys
import xml.etree.ElementTree as ET
import zlib

TYPES = {
    "Float64": "d", "Float32": "f", "Int64": "q", "UInt64": "Q",
    "Int32": "i", "UInt32": "I", "Int16": "h", "UInt16": "H",
    "Int8": "b", "UInt8": "B",
}


def parse_document(path):
    raw = Path(path).read_bytes()
    # Raw appended VTK binary can make XML invalid. Temporarily discard the
    # appended payload during XML parsing; retain the payload separately.
    pos = raw.find(b"<AppendedData")
    appended_data = None
    appended_encoding = None
    if pos >= 0:
        after_tag = raw.find(b">", pos)
        if after_tag < 0:
            raise ValueError("Malformed <AppendedData> tag")
        underscore = raw.find(b"_", after_tag + 1)
        close = raw.rfind(b"</AppendedData>")
        if underscore < 0 or close < underscore:
            raise ValueError("Malformed appended data section")
        doc_for_xml = (raw[:underscore + 1] +
                       raw[close:])
        root = ET.fromstring(doc_for_xml)
        app = root.find("AppendedData")
        if app is None:
            raise ValueError("Missing AppendedData node")
        appended_encoding = app.attrib.get("encoding", "base64").lower()
        appended_data = raw[underscore + 1:close]
    else:
        root = ET.fromstring(raw)

    if root.tag != "VTKFile" or root.attrib.get("type") != "UnstructuredGrid":
        raise ValueError("Expected VTK XML UnstructuredGrid (.vtu)")
    piece = root.find("./UnstructuredGrid/Piece")
    if piece is None:
        raise ValueError("No UnstructuredGrid/Piece found")

    byte_order = root.attrib.get("byte_order", "LittleEndian")
    endian = "<" if byte_order == "LittleEndian" else ">" if byte_order == "BigEndian" else None
    if endian is None:
        raise ValueError("Unsupported byte order " + byte_order)
    header_type = root.attrib.get("header_type", "UInt32")
    if header_type not in ("UInt32", "UInt64"):
        raise ValueError("Unsupported header type " + header_type)
    header_code = "I" if header_type == "UInt32" else "Q"
    compressor = root.attrib.get("compressor", "")
    if compressor not in ("", "vtkZLibDataCompressor"):
        raise ValueError("Unsupported compressor " + compressor)
    if appended_data is not None and appended_encoding not in ("raw", "base64"):
        raise ValueError("Unsupported appended encoding " + appended_encoding)

    offsets = sorted({int(a.attrib["offset"]) for a in piece.iter("DataArray")
                      if a.attrib.get("format") == "appended"})

    def payload_bytes(a):
        encoding = a.attrib.get("format", "ascii").lower()
        if encoding == "binary":
            chunk = re.sub(rb"\s+", b"", (a.text or "").encode("ascii"))
            return decode_base64_binary(chunk, endian, header_code, compressor)
        if encoding == "appended":
            if appended_data is None:
                raise ValueError("Array requests appended data, but none exists")
            start = int(a.attrib["offset"])
            larger = [o for o in offsets if o > start]
            end = min(larger) if larger else len(appended_data)
            chunk = appended_data[start:end]
            if appended_encoding == "raw":
                return unpack_vtk_binary(chunk, endian, header_code, compressor)
            chunk = re.sub(rb"\s+", b"", chunk)
            return decode_base64_binary(chunk, endian, header_code, compressor)
        raise ValueError("Cannot decode array format " + encoding)

    def read(a):
        if a is None:
            raise ValueError("Missing required VTK DataArray")
        vtk_type = a.attrib.get("type", "")
        code = TYPES.get(vtk_type)
        if code is None:
            raise ValueError("Unsupported VTK numeric type " + vtk_type)
        if a.attrib.get("format", "ascii").lower() == "ascii":
            conv = float if code in ("d", "f") else int
            result = [conv(v) for v in (a.text or "").split()]
        else:
            payload = payload_bytes(a)
            item_size = struct.calcsize(code)
            if len(payload) % item_size:
                raise ValueError("Decoded byte count is not a multiple of item size")
            result = struct.unpack(endian + str(len(payload)//item_size) + code, payload)
        return result

    def find_a(xpath):
        return piece.find(xpath)

    point_field = find_a("PointData/DataArray[@Name='u']")
    if point_field is None:
        names = [a.attrib.get("Name", "(unnamed)") for a in piece.findall("PointData/DataArray")]
        raise ValueError("PointData field 'u' not found. Available: " + str(names))

    coords = read(find_a("Points/DataArray"))
    connectivity = read(find_a("Cells/DataArray[@Name='connectivity']"))
    cell_offsets = read(find_a("Cells/DataArray[@Name='offsets']"))
    cell_types = read(find_a("Cells/DataArray[@Name='types']"))
    u = read(point_field)

    if len(u) != int(piece.attrib["NumberOfPoints"]):
        raise ValueError("Expected one scalar u for each point")
    return {
        "points": int(piece.attrib["NumberOfPoints"]),
        "cells": int(piece.attrib["NumberOfCells"]),
        "coords": coords,
        "connectivity": connectivity,
        "cell_offsets": cell_offsets,
        "cell_types": cell_types,
        "u": u,
        "format": point_field.attrib.get("format", "ascii"),
        "compressor": compressor or "none",
    }


def unpack_vtk_binary(data, endian, header_code, compressor):
    w = struct.calcsize(header_code)

    def word(idx):
        st = idx*w
        if len(data) < st+w:
            raise ValueError("Truncated VTK binary array header")
        return struct.unpack_from(endian+header_code, data, st)[0]

    if not compressor:
        size = word(0)
        if len(data) < w+size:
            raise ValueError("VTK binary payload is truncated")
        return data[w:w+size]

    blocks = word(0)
    block_size = word(1)
    last_size = word(2)
    if blocks > 1000000:
        raise ValueError("Unreasonable number of zlib blocks")
    header_bytes = w*(3+blocks)
    if len(data) < header_bytes:
        raise ValueError("Truncated compressed VTK header")
    sizes = struct.unpack_from(endian+str(blocks)+header_code, data, 3*w) if blocks else ()
    cursor = header_bytes
    out = bytearray()
    for size in sizes:
        comp = data[cursor:cursor+size]
        if len(comp) != size:
            raise ValueError("Truncated compressed VTK block")
        out.extend(zlib.decompress(comp))
        cursor += size
    expected = (blocks-1)*block_size + (last_size or block_size) if blocks else 0
    if len(out) != expected:
        raise ValueError(f"Decompressed {len(out)} bytes, expected {expected}")
    return bytes(out)


def decode_base64_binary(chunk, endian, header_code, compressor):
    # Most VTK exports use one base64 string for an uncompressed array and
    # two adjacent base64 strings (header and data separately) when compressed.
    # Try one stream first; when padding terminates after just the header,
    # decode the header and body individually.
    try:
        whole = base64.b64decode(chunk)
        return unpack_vtk_binary(whole, endian, header_code, compressor)
    except (ValueError, zlib.error, struct.error, binascii.Error):
        if not compressor:
            raise
        w = struct.calcsize(header_code)
        head3 = base64.b64decode(chunk[:4*math.ceil(3*w/3)])
        blocks = struct.unpack_from(endian+header_code, head3, 0)[0]
        if blocks > 1000000:
            raise ValueError("Unreasonable compressed block count")
        head_size = (3+blocks)*w
        encoded_head_size = 4*math.ceil(head_size/3)
        head = base64.b64decode(chunk[:encoded_head_size])
        body = base64.b64decode(chunk[encoded_head_size:])
        return unpack_vtk_binary(head+body, endian, header_code, compressor)


def main():
    if len(sys.argv) != 3:
        print("Usage: python3 compare_vtu_stdlib.py CPU.vtu GPU.vtu")
        return 2
    cpu = parse_document(sys.argv[1])
    gpu = parse_document(sys.argv[2])
    print(f"CPU: {cpu['points']} points, {cpu['cells']} cells, "
          f"{cpu['format']}, {cpu['compressor']}")
    print(f"GPU: {gpu['points']} points, {gpu['cells']} cells, "
          f"{gpu['format']}, {gpu['compressor']}")
    for label in ("points", "cells", "coords", "connectivity", "cell_offsets", "cell_types"):
        if cpu[label] != gpu[label]:
            print(f"FAIL: mesh differs at {label}")
            return 1
    print("PASS: mesh coordinates/connectivity/cell types agree")

    u_cpu, u_gpu = cpu["u"], gpu["u"]
    if len(u_cpu) != len(u_gpu):
        print("FAIL: u array lengths differ")
        return 1
    err2 = math.fsum((float(x)-float(y))**2 for x,y in zip(u_cpu,u_gpu))
    ref2 = math.fsum(float(x)**2 for x in u_cpu)
    gpu2 = math.fsum(float(x)**2 for x in u_gpu)
    rel = math.sqrt(err2/max(ref2,1.e-300))
    ma = max((abs(float(x)-float(y)) for x,y in zip(u_cpu,u_gpu)),default=0.0)
    print(f"u number of values : {len(u_cpu)}")
    print(f"CPU u L2 norm      : {math.sqrt(ref2):.16e}")
    print(f"GPU u L2 norm      : {math.sqrt(gpu2):.16e}")
    print(f"Relative L2 error  : {rel:.16e}")
    print(f"Max abs error      : {ma:.16e}")
    ok = rel < 1.e-12
    print("PASS: CPU and GPU VTU solutions agree" if ok else "FAIL: VTU solutions differ")
    return 0 if ok else 1


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception as exc:
        print(f"ERROR: {type(exc).__name__}: {exc}", file=sys.stderr)
        sys.exit(2)
