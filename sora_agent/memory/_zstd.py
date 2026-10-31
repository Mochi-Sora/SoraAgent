"""Small zstd adapter: stdlib on 3.14+, the optional package before that."""

from __future__ import annotations

from pathlib import Path

try:
    from compression import zstd as _stdlib_zstd  # type: ignore[import-not-found]
except ImportError:
    _stdlib_zstd = None

try:
    import zstandard as _zstandard  # type: ignore[import-not-found]  # pyright: ignore[reportMissingImports]
except ImportError:
    _zstandard = None


class CompressionUnavailable(RuntimeError):
    pass


def available() -> bool:
    return _stdlib_zstd is not None or _zstandard is not None


def compress_bytes(data: bytes, level: int = 9) -> bytes:
    if _stdlib_zstd is not None:
        compressor = _stdlib_zstd.ZstdCompressor(level=level)
        return compressor.compress(data) + compressor.flush()
    if _zstandard is not None:
        return _zstandard.ZstdCompressor(
            level=level, write_checksum=True, write_content_size=True
        ).compress(data)
    raise CompressionUnavailable(
        "zstd support requires Python 3.14+ or the optional 'zstandard' package"
    )


def decompress_bytes(data: bytes) -> bytes:
    if _stdlib_zstd is not None:
        return _stdlib_zstd.ZstdDecompressor().decompress(data)
    if _zstandard is not None:
        return _zstandard.ZstdDecompressor().decompress(data)
    raise CompressionUnavailable(
        "zstd support requires Python 3.14+ or the optional 'zstandard' package"
    )


def compress_file(source: Path, destination: Path, level: int = 9) -> None:
    data = Path(source).read_bytes()
    Path(destination).write_bytes(compress_bytes(data, level=level))


def decompress_file(source: Path, destination: Path) -> None:
    data = Path(source).read_bytes()
    Path(destination).write_bytes(decompress_bytes(data))
