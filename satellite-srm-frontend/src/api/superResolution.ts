import { apiClient, resolveApiUrl } from './client';
import { ProcessingJob } from '../types/satellite';

export interface StartReconstructionParams {
  file?: File;
  files?: File[];
  referenceFile?: File;
  model?: string;
  enableUncertainty?: boolean;
  scaleFactor?: number;
}

export interface SuperResolutionResponse {
  job_id: string;
  job_ids?: string[];
  count?: number;
  status: string;
  job?: ProcessingJob;
}

export async function startSingleSuperResolution(
  file: File,
  options: {
    model?: string;
    enableUncertainty?: boolean;
    scaleFactor?: number;
    referenceFile?: File;
  } = {}
): Promise<SuperResolutionResponse> {
  const formData = new FormData();
  formData.append('file', file, file.name);

  if (options.model) {
    formData.append('model', options.model);
  }
  formData.append('enable_uncertainty', String(options.enableUncertainty ?? true));
  if (options.scaleFactor) {
    formData.append('scale_factor', String(options.scaleFactor));
  }
  if (options.referenceFile) {
    formData.append('reference_file', options.referenceFile, options.referenceFile.name);
  }

  return apiClient<SuperResolutionResponse>(
    '/api/v1/super-resolution',
    {
      method: 'POST',
      body: formData,
    }
  );
}

export async function startSuperResolution(
  params: StartReconstructionParams
): Promise<SuperResolutionResponse> {
  const primaryFile = params.file || (params.files && params.files[0]);
  if (!primaryFile) {
    throw new Error('No satellite image file provided for super-resolution.');
  }

  return startSingleSuperResolution(primaryFile, {
    model: params.model,
    enableUncertainty: params.enableUncertainty,
    scaleFactor: params.scaleFactor,
    referenceFile: params.referenceFile,
  });
}

export async function checkBackendHealth(): Promise<boolean> {
  try {
    const url = resolveApiUrl('/api/v1/health');
    const res = await fetch(url);
    if (!res.ok) return false;
    
    const contentType = res.headers.get('content-type');
    if (!contentType || !contentType.includes('application/json')) {
      return false; // Vite SPA fallback returns text/html
    }
    
    const data = await res.json();
    return data && data.status === 'healthy';
  } catch {
    return false;
  }
}

export async function validateBandsApi(
  bands: { b02?: File | null; b03?: File | null; b04?: File | null; b08?: File | null }
): Promise<import('../types/satellite').FourBandValidationResult> {
  const formData = new FormData();
  if (bands.b02) formData.append('b02', bands.b02, bands.b02.name);
  if (bands.b03) formData.append('b03', bands.b03, bands.b03.name);
  if (bands.b04) formData.append('b04', bands.b04, bands.b04.name);
  if (bands.b08) formData.append('b08', bands.b08, bands.b08.name);

  return apiClient<import('../types/satellite').FourBandValidationResult>(
    '/api/v1/validate-bands',
    {
      method: 'POST',
      body: formData,
    }
  );
}

export async function startFourBandSuperResolution(
  bands: { b02: File; b03: File; b04: File; b08: File },
  options: {
    model?: string;
    enableUncertainty?: boolean;
    scaleFactor?: number;
  } = {}
): Promise<SuperResolutionResponse> {
  const formData = new FormData();
  formData.append('b02', bands.b02, bands.b02.name);
  formData.append('b03', bands.b03, bands.b03.name);
  formData.append('b04', bands.b04, bands.b04.name);
  formData.append('b08', bands.b08, bands.b08.name);

  if (options.model) {
    formData.append('model', options.model);
  }
  formData.append('enable_uncertainty', String(options.enableUncertainty ?? true));
  if (options.scaleFactor) {
    formData.append('scale_factor', String(options.scaleFactor));
  }

  return apiClient<SuperResolutionResponse>(
    '/api/v1/super-resolution',
    {
      method: 'POST',
      body: formData,
    }
  );
}

