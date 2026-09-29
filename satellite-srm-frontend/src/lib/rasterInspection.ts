/**
 * Client-side Satellite Raster Inspection & Validation Utility
 * Inspects GeoTIFF, TIFF, PNG, and JPEG files directly in the browser.
 * Extracts dimensions, band count, CRS, pixel scale, and flags corrupted rasters.
 */

export interface RasterMetadata {
  filename: string;
  fileSize: number;
  fileType: string;
  dimensions: string;
  width: number;
  height: number;
  bandCount: number;
  bandsDescription: string;
  crs: string;
  spatialResolution: string;
  bitsPerSample?: string;
}

export interface RasterInspectionResult {
  isValid: boolean;
  error?: string;
  metadata?: RasterMetadata;
}

const SUPPORTED_EXTENSIONS = ['tif', 'tiff', 'geotiff'];
const REQUIRED_BAND_COUNT = 4;
const MAX_FILE_SIZE_MB = 500;

/**
 * Parses binary TIFF / GeoTIFF headers to extract real raster structure.
 */
function parseTiffHeader(buffer: ArrayBuffer, filename: string, fileSize: number): RasterInspectionResult {
  if (buffer.byteLength < 8) {
    return {
      isValid: false,
      error: 'Corrupted or truncated file: TIFF file size is too small to contain a valid header (<8 bytes).',
    };
  }

  const view = new DataView(buffer);
  const byte0 = view.getUint8(0);
  const byte1 = view.getUint8(1);

  let littleEndian = false;
  if (byte0 === 0x49 && byte1 === 0x49) {
    littleEndian = true; // "II" (Intel little-endian)
  } else if (byte0 === 0x4d && byte1 === 0x4d) {
    littleEndian = false; // "MM" (Motorola big-endian)
  } else {
    return {
      isValid: false,
      error: 'Invalid TIFF magic number: File is not a valid TIFF/GeoTIFF raster.',
    };
  }

  const magic = view.getUint16(2, littleEndian);
  if (magic !== 42) {
    return {
      isValid: false,
      error: `Invalid TIFF identifier (${magic} != 42): Corrupted raster header structure.`,
    };
  }

  const firstIfdOffset = view.getUint32(4, littleEndian);
  if (firstIfdOffset < 8 || firstIfdOffset + 2 > buffer.byteLength) {
    return {
      isValid: false,
      error: 'Corrupted TIFF: Image File Directory (IFD) offset points outside the file boundary.',
    };
  }

  let width = 0;
  let height = 0;
  let samplesPerPixel = 1;
  let bitsPerSample = 8;
  let epsgCode: number | null = null;
  let pixelScaleX: number | null = null;

  try {
    const numEntries = view.getUint16(firstIfdOffset, littleEndian);
    let offset = firstIfdOffset + 2;

    for (let i = 0; i < numEntries; i++) {
      if (offset + 12 > buffer.byteLength) break;

      const tag = view.getUint16(offset, littleEndian);
      const type = view.getUint16(offset + 2, littleEndian);
      const count = view.getUint32(offset + 4, littleEndian);
      const valOffset = offset + 8;

      const readVal = (): number => {
        if (type === 3) return view.getUint16(valOffset, littleEndian); // SHORT
        if (type === 4) return view.getUint32(valOffset, littleEndian); // LONG
        return view.getUint16(valOffset, littleEndian);
      };

      if (tag === 256) {
        // ImageWidth
        width = readVal();
      } else if (tag === 257) {
        // ImageLength (Height)
        height = readVal();
      } else if (tag === 258) {
        // BitsPerSample
        bitsPerSample = readVal();
      } else if (tag === 277) {
        // SamplesPerPixel (Band count)
        samplesPerPixel = readVal();
      } else if (tag === 33550) {
        // ModelPixelScaleTag (DOUBLE, 3 values: ScaleX, ScaleY, ScaleZ)
        const dataOffset = view.getUint32(valOffset, littleEndian);
        if (dataOffset + 8 <= buffer.byteLength) {
          pixelScaleX = view.getFloat64(dataOffset, littleEndian);
        }
      } else if (tag === 34735) {
        // GeoKeyDirectoryTag (GeoTIFF Keys)
        const dataOffset = count <= 2 ? valOffset : view.getUint32(valOffset, littleEndian);
        if (dataOffset + count * 2 <= buffer.byteLength) {
          const numKeys = view.getUint16(dataOffset + 6, littleEndian);
          for (let k = 0; k < numKeys; k++) {
            const kOffset = dataOffset + 8 + k * 8;
            if (kOffset + 8 > buffer.byteLength) break;
            const keyId = view.getUint16(kOffset, littleEndian);
            const val = view.getUint16(kOffset + 6, littleEndian);
            if (keyId === 3072) {
              // ProjectedCSTypeGeoKey
              epsgCode = val;
            } else if (keyId === 2048 && !epsgCode) {
              // GeographicTypeGeoKey
              epsgCode = val;
            }
          }
        }
      }

      offset += 12;
    }
  } catch (err) {
    return {
      isValid: false,
      error: `Failed to decode TIFF IFD entries: ${err instanceof Error ? err.message : 'Corrupted tags'}`,
    };
  }

  if (width <= 0 || height <= 0) {
    return {
      isValid: false,
      error: 'Invalid input: Raster dimensions are invalid. Width and height must be positive non-zero values.',
    };
  }

  // ── Strict 4-band enforcement ────────────────────────────────────
  if (samplesPerPixel !== REQUIRED_BAND_COUNT) {
    if (samplesPerPixel === 3) {
      return {
        isValid: false,
        error:
          `Invalid input: GeoTIFF must contain 4 spectral bands (B02, B03, B04, B08). ` +
          `This file has 3 bands (RGB). Expected Sentinel-2 band order: B02, B03, B04, B08.`,
      };
    }
    if (samplesPerPixel === 1) {
      return {
        isValid: false,
        error:
          `Invalid input: GeoTIFF must contain 4 spectral bands. ` +
          `This file is single-band (panchromatic). ` +
          `Please upload a Sentinel-2 L2A multispectral GeoTIFF with bands B02, B03, B04, B08.`,
      };
    }
    return {
      isValid: false,
      error:
        `Invalid input: GeoTIFF must contain exactly 4 spectral bands (B02, B03, B04, B08). ` +
        `This file has ${samplesPerPixel} bands. ` +
        `Raster dimensions or spectral configuration are incompatible.`,
    };
  }

  // ── CRS validation ───────────────────────────────────────────────
  // Warn if no geospatial projection tag found, but still accept the file
  // (some valid GeoTIFFs store CRS in sidecar .prj files)
  let crsStr: string;
  let crsWarning = false;
  if (epsgCode) {
    const suffix =
      epsgCode === 32644 ? '(WGS 84 / UTM Zone 44N)'
      : epsgCode === 4326 ? '(WGS 84 Geographic)'
      : epsgCode >= 32601 && epsgCode <= 32660 ? `(WGS 84 / UTM Zone ${epsgCode - 32600}N)`
      : epsgCode >= 32701 && epsgCode <= 32760 ? `(WGS 84 / UTM Zone ${epsgCode - 32700}S)`
      : '';
    crsStr = `EPSG:${epsgCode} ${suffix}`.trim();
  } else if (filename.toLowerCase().includes('s2') || filename.toLowerCase().includes('sentinel')) {
    crsStr = 'EPSG:32644 (Sentinel-2 L2A UTM – inferred from filename)';
    crsWarning = true;
  } else {
    crsStr = 'No CRS embedded — geospatial metadata may be missing';
    crsWarning = true;
  }

  // ── Spatial resolution ───────────────────────────────────────────
  let resStr: string;
  if (pixelScaleX && pixelScaleX > 0 && pixelScaleX < 100000) {
    // ModelPixelScaleTag stores scale in the CRS native units (metres for UTM)
    resStr = `${pixelScaleX.toFixed(2)} m GSD (from GeoTIFF ModelPixelScale)`;
  } else {
    resStr = '10.0 m GSD (Sentinel-2 native — scale not embedded)';
  }

  return {
    isValid: !crsWarning || !!epsgCode,  // Reject only if CRS is completely absent and filename gives no hint
    error: crsWarning && !epsgCode && !filename.toLowerCase().includes('s2') && !filename.toLowerCase().includes('sentinel')
      ? 'Invalid input: Geospatial metadata is missing. The GeoTIFF has no embedded CRS (EPSG code). Please use a properly georeferenced Sentinel-2 GeoTIFF.'
      : undefined,
    metadata: {
      filename,
      fileSize,
      fileType: 'GeoTIFF (4-Band Multispectral)',
      dimensions: `${width} \u00d7 ${height} px`,
      width,
      height,
      bandCount: samplesPerPixel,
      bandsDescription: '4 Bands: B02 (Blue), B03 (Green), B04 (Red), B08 (NIR)',
      crs: crsStr,
      spatialResolution: resStr,
      bitsPerSample: `${bitsPerSample}-bit per channel`,
    },
  };
}

/**
 * Parses standard PNG / JPEG files via an in-memory HTML Image.
 */
function parseStandardImage(file: File): Promise<RasterInspectionResult> {
  return new Promise((resolve) => {
    const objectUrl = URL.createObjectURL(file);
    const img = new Image();

    img.onload = () => {
      URL.revokeObjectURL(objectUrl);
      const width = img.naturalWidth;
      const height = img.naturalHeight;

      if (width <= 0 || height <= 0) {
        resolve({
          isValid: false,
          error: 'Invalid image dimensions: Raster has zero dimensions or could not be decoded.',
        });
        return;
      }

      const isPng = file.type.includes('png') || file.name.toLowerCase().endsWith('.png');
      resolve({
        isValid: true,
        metadata: {
          filename: file.name,
          fileSize: file.size,
          fileType: isPng ? 'PNG Satellite Tile' : 'JPEG Satellite Tile',
          dimensions: `${width} × ${height} px`,
          width,
          height,
          bandCount: 3,
          bandsDescription: '3 Bands (RGB Display Channels)',
          crs: 'Standard Raster (Unprojected Graphic)',
          spatialResolution: '10.0m GSD Simulated (RGB)',
          bitsPerSample: '8-bit sRGB',
        },
      });
    };

    img.onerror = () => {
      URL.revokeObjectURL(objectUrl);
      resolve({
        isValid: false,
        error: `Could not decode "${file.name}": Image data is corrupted or unsupported.`,
      });
    };

    img.src = objectUrl;
  });
}

/**
 * Comprehensive client-side inspection and validation entry point.
 * Only accepts GeoTIFF files with exactly 4 spectral bands (B02, B03, B04, B08).
 */
export async function inspectAndValidateRaster(file: File): Promise<RasterInspectionResult> {
  // 1. Basic sanity checks
  if (!file) {
    return { isValid: false, error: 'No file provided.' };
  }

  const ext = file.name.toLowerCase().split('.').pop() || '';

  // Reject non-TIFF files immediately with clear message
  if (!SUPPORTED_EXTENSIONS.includes(ext)) {
    if (['png', 'jpg', 'jpeg', 'webp', 'bmp'].includes(ext)) {
      return {
        isValid: false,
        error:
          `Invalid input: TerraSR requires a 4-band Sentinel-2 GeoTIFF (.tif, .tiff). ` +
          `Standard images (PNG, JPEG) do not contain spectral bands or geospatial metadata.`,
      };
    }
    return {
      isValid: false,
      error: `Invalid input: Unsupported file type ".${ext}". Upload a 4-band Sentinel-2 GeoTIFF (.tif or .tiff).`,
    };
  }

  if (file.size <= 0) {
    return {
      isValid: false,
      error: `File "${file.name}" is completely empty (0 bytes).`,
    };
  }

  if (file.size > MAX_FILE_SIZE_MB * 1024 * 1024) {
    return {
      isValid: false,
      error: `File "${file.name}" exceeds maximum allowed upload size (${MAX_FILE_SIZE_MB} MB).`,
    };
  }

  // 2. Parse TIFF header (always — only TIFFs reach this point)
  try {
    const sliceSize = Math.min(file.size, 128 * 1024);
    const buffer = await file.slice(0, sliceSize).arrayBuffer();
    return parseTiffHeader(buffer, file.name, file.size);
  } catch (err) {
    return {
      isValid: false,
      error: `Failed to read TIFF header: ${err instanceof Error ? err.message : 'Unknown I/O error'}`,
    };
  }
}
