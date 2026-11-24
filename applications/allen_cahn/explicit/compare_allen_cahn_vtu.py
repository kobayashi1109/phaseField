#!/usr/bin/env python3
"""Compare inline XML VTU files from CPU/GPU Allen-Cahn runs (stdlib only)."""
import argparse
import array
import base64
import math
import re
import struct
import sys
import xml.etree.ElementTree as ET
import zlib
from pathlib import Path

TYPES = {'Float32': 'f', 'Float64': 'd', 'Int8': 'b', 'UInt8': 'B',
         'Int16': 'h', 'UInt16': 'H', 'Int32': 'i', 'UInt32': 'I',
         'Int64': 'q', 'UInt64': 'Q'}


def decode_b64(text):
    # VTK may encode the compression header and the payload separately.
    s = ''.join(text.split())
    return b''.join(base64.b64decode(chunk, validate=True)
                    for chunk in re.findall(r'[^=]+={0,2}', s))


def read_array(root, el):
    typ = el.attrib['type']
    code = TYPES[typ]
    vals = array.array(code)
    fmt = el.attrib.get('format', 'ascii')
    if fmt == 'ascii':
        convert = float if code in ('f', 'd') else int
        vals.extend(convert(x) for x in (el.text or '').split())
        return vals
    if fmt != 'binary':
        raise ValueError('Unsupported DataArray format: ' + fmt +
                         ' (this comparator supports inline binary/ascii VTU)')
    raw = decode_b64(el.text or '')
    endian = '<' if root.attrib.get('byte_order', 'LittleEndian') == 'LittleEndian' else '>'
    hcode = {'UInt32': 'I', 'UInt64': 'Q'}[root.attrib.get('header_type', 'UInt32')]
    hs = struct.calcsize(hcode)
    compressor = root.attrib.get('compressor', '')
    if compressor:
        if compressor != 'vtkZLibDataCompressor':
            raise ValueError('Unsupported compressor: ' + compressor)
        nb, bs, last = struct.unpack_from(endian + '3' + hcode, raw)
        start = (3 + nb) * hs
        if start > len(raw):
            raise ValueError('Truncated VTK compression header')
        sizes = struct.unpack_from(endian + str(nb) + hcode, raw, 3 * hs)
        blocks = []
        for i, length in enumerate(sizes):
            if start + length > len(raw):
                raise ValueError('Truncated VTK compressed block')
            block = zlib.decompress(raw[start:start + length])
            expected = bs if i != nb - 1 else (last or bs)
            if len(block) != expected:
                raise ValueError('Unexpected decompressed block size')
            blocks.append(block)
            start += length
        if start != len(raw):
            raise ValueError('Unexpected trailing VTK compressed bytes')
        data = b''.join(blocks)
    else:
        length, = struct.unpack_from(endian + hcode, raw)
        if len(raw) != hs + length:
            raise ValueError('VTK binary byte count mismatch')
        data = raw[hs:]
    vals.frombytes(data)
    if (endian == '<') != (sys.byteorder == 'little'):
        vals.byteswap()
    return vals


def read_vtu(path):
    root = ET.parse(path).getroot()
    pieces = root.findall('./UnstructuredGrid/Piece')
    if len(pieces) != 1:
        raise ValueError(f'{path}: expected one Piece, got {len(pieces)}')
    piece = pieces[0]
    points = read_array(root, piece.find('./Points/DataArray'))
    cells = {e.attrib['Name']: read_array(root, e) for e in piece.findall('./Cells/DataArray')}
    fields = {e.attrib['Name']: (int(e.attrib.get('NumberOfComponents', 1)), read_array(root, e))
              for e in piece.findall('./PointData/DataArray') if 'Name' in e.attrib}
    return (int(piece.attrib['NumberOfPoints']), int(piece.attrib['NumberOfCells']), points, cells, fields)


def compare(cpu_path, gpu_path, reltol, abstol):
    a, b = read_vtu(cpu_path), read_vtu(gpu_path)
    if a[:4] != b[:4]:
        raise ValueError(f'{cpu_path.name}: mesh differs (point order, coordinates or cells)')
    print(f'\n===== {cpu_path.name}: {a[0]} points, {a[1]} cells =====')
    print('PASS: mesh agrees')
    good = True
    for name in ('n', 'mg_n', 'f_tot'):
        ac, x = a[4][name]
        bc, y = b[4][name]
        if ac != 1 or bc != 1 or len(x) != a[0] or len(x) != len(y):
            raise ValueError(f'{name}: field size/component mismatch')
        if not all(math.isfinite(v) for v in x) or not all(math.isfinite(v) for v in y):
            raise ValueError(f'{name}: non-finite value detected')
        norm = math.sqrt(math.fsum(v*v for v in x))
        err = math.sqrt(math.fsum((u-v)**2 for u,v in zip(x,y)))
        maxerr = max((abs(u-v) for u,v in zip(x,y)), default=0.0)
        rel = err/norm if norm else (0.0 if err == 0 else math.inf)
        ok = rel <= reltol or (norm == 0 and maxerr <= abstol)
        good &= ok
        print(f'{name:6s} relative L2 = {rel:.12e}, max abs = {maxerr:.12e}  {"PASS" if ok else "CHECK"}')
    return good


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('run_directory', type=Path, help='directory containing cpu/solutions and gpu/solutions')
    p.add_argument('--rtol', type=float, default=1e-7, help='relative L2 comparison threshold (VTU output precision)')
    p.add_argument('--atol', type=float, default=1e-12, help='max absolute threshold for zero-norm reference fields')
    args = p.parse_args()
    cpu = args.run_directory / 'cpu' / 'solutions'
    gpu = args.run_directory / 'gpu' / 'solutions'
    a = {p.name:p for p in cpu.glob('solution_*.vtu')}
    b = {p.name:p for p in gpu.glob('solution_*.vtu')}
    if not a or a.keys() != b.keys():
        raise ValueError(f'Output filenames do not match: CPU={sorted(a)}, GPU={sorted(b)}')
    good = True
    for name in sorted(a):
        good = compare(a[name], b[name], args.rtol, args.atol) and good
    print(f'\n{"PASS" if good else "CHECK"}: {len(a)} matching output times; relative L2 threshold={args.rtol:g}')
    return 0 if good else 1


if __name__ == '__main__':
    try:
        sys.exit(main())
    except (ValueError, KeyError, OSError, ET.ParseError, struct.error, zlib.error) as exc:
        print('ERROR:', exc, file=sys.stderr)
        sys.exit(2)
