import React, { useEffect, useState } from 'react';
import { useNavigate } from 'react-router-dom';
import {
  Zap, Upload, FileCheck, RefreshCw, CheckCircle2,
  AlertCircle, AlertTriangle, RotateCcw, Settings2,
  ArrowLeft, ShieldAlert
} from 'lucide-react';
import { FourBandUploader } from '../components/enhance/FourBandUploader';
import { BandMetadataPanel } from '../components/enhance/BandMetadataPanel';
import { ModelSettings, ModelSettingsValues } from '../components/enhance/ModelSettings';
import { ProcessingProgress } from '../components/enhance/ProcessingProgress';
import { ResultsViewer } from '../components/enhance/ResultsViewer';
import { ValidationScannerModal } from '../components/enhance/ValidationScannerModal';
import { useSrmStore, DEMO_JOB } from '../store/useSrmStore';
import {
  validateBandsApi,
  startFourBandSuperResolution,
  checkBackendHealth
} from '../api/superResolution';
import { getJobStatus } from '../api/jobs';
import { ProcessingJob } from '../types/satellite';

type Step = 'upload' | 'metadata' | 'processing' | 'results';

const STEP_LABELS: { id: Step; label: string; icon: React.ReactNode }[] = [
  { id: 'upload',     label: '1. Upload Bands',       icon: <Upload size={14} />       },
  { id: 'metadata',   label: '2. Validate & Metadata', icon: <FileCheck size={14} />    },
  { id: 'processing', label: '3. SwinIR Processing',   icon: <RefreshCw size={14} />    },
  { id: 'results',    label: '4. Analysis Results',    icon: <CheckCircle2 size={14} /> },
];

const DEFAULT_SETTINGS: ModelSettingsValues = {
  model:               'SwinIR-SRM',
  scaleFactor:         3,
  outputFormat:        'GeoTIFF (Georeferenced)',
  enableUncertainty:   true,
  spectralConsistency: true,
  artifactReduction:   true,
};

export const EnhancePage: React.FC = () => {
  const navigate = useNavigate();
  const {
    fourBands,
    validationResult,
    setValidationResult,
    activeJob,
    setActiveJob,
    activeJobs,
    setActiveJobs,
    setSelectedJobId,
    setMetadata,
    resetAll,
  } = useSrmStore();

  const [step, setStep] = useState<Step>('upload');
  const [settings, setSettings] = useState<ModelSettingsValues>(DEFAULT_SETTINGS);
  const [backendAvailable, setBackendAvailable] = useState<boolean | null>(null);
  const [isCheckingBackend, setIsCheckingBackend] = useState(false);

  const [isValidating, setIsValidating] = useState(false);
  const [validationError, setValidationError] = useState<string | null>(null);
  const [validationPhase, setValidationPhase] = useState<'idle' | 'scanning' | 'success' | 'failed'>('idle');
  const [isValidationModalOpen, setIsValidationModalOpen] = useState(false);

  const [isSubmitting, setIsSubmitting] = useState(false);
  const [inferenceError, setInferenceError] = useState<string | null>(null);

  // Destructive action confirmation modal
  const [isConfirmModalOpen, setIsConfirmModalOpen] = useState(false);
  const [pendingConfirmAction, setPendingConfirmAction] = useState<(() => void) | null>(null);

  // Check backend health on mount
  const checkHealth = async () => {
    setIsCheckingBackend(true);
    const ok = await checkBackendHealth();
    setBackendAvailable(ok);
    setIsCheckingBackend(false);
  };

  useEffect(() => {
    checkHealth();
  }, []);

  // If activeJob already completed, default to results view
  useEffect(() => {
    if (activeJob?.status === 'completed' && step === 'upload') {
      setStep('results');
    }
  }, []); // eslint-disable-line

  // ── Step 1: Validate 4 Bands ──
  const handleValidate = async () => {
    if (!fourBands.b02 || !fourBands.b03 || !fourBands.b04 || !fourBands.b08) {
      setValidationError('All four Sentinel-2 bands (B02, B03, B04, B08) must be provided.');
      return;
    }

    setValidationError(null);
    setValidationPhase('scanning');
    setIsValidationModalOpen(true);
    setIsValidating(true);

    try {
      const result = await validateBandsApi(fourBands);
      if (result.valid) {
        setValidationResult(result);
        if (result.metadata?.common) {
          const c = result.metadata.common;
          setMetadata({
            filename: 'Sentinel2_4Band_Composite.tif',
            fileSize: (fourBands.b02.size + fourBands.b03.size + fourBands.b04.size + fourBands.b08.size),
            format: 'GeoTIFF (4-Band Aligned)',
            width: c.width,
            height: c.height,
            bands: ['B02 (Blue)', 'B03 (Green)', 'B04 (Red)', 'B08 (NIR)'],
            nativeResolution: c.nativeResolution,
            targetResolution: c.targetResolution,
            crs: c.crs,
            bounds: c.bounds,
            center: [17.4106, 78.4776],
            sensor: c.sensor,
            acquisitionDate: new Date().toISOString()
          });
        }
        // REAL validation succeeded: display short celebration animation then transition
        setValidationPhase('success');
        await new Promise((res) => setTimeout(res, 1400));
        setIsValidationModalOpen(false);
        setValidationPhase('idle');
        setStep('metadata');
      } else {
        const errorMsg = result.error || result.message || 'Spatial validation failed.';
        setValidationError(errorMsg);
        setValidationPhase('failed');
      }
    } catch (err: unknown) {
      const msg = err instanceof Error ? err.message : 'Validation failed';
      setValidationError(msg);
      setValidationPhase('failed');
    } finally {
      setIsValidating(false);
    }
  };

  // ── Step 2: Run Super Resolution Inference ──
  const handleRunFourBandSR = async () => {
    if (!fourBands.b02 || !fourBands.b03 || !fourBands.b04 || !fourBands.b08) {
      setValidationError('Missing one or more required bands.');
      setStep('upload');
      return;
    }

    setInferenceError(null);
    setIsSubmitting(true);
    setStep('processing');

    try {
      const resp = await startFourBandSuperResolution(
        {
          b02: fourBands.b02,
          b03: fourBands.b03,
          b04: fourBands.b04,
          b08: fourBands.b08,
        },
        {
          model: settings.model,
          enableUncertainty: settings.enableUncertainty,
          scaleFactor: settings.scaleFactor,
        }
      );

      const jobId = resp.job_id;
      if (!jobId) throw new Error('Inference server did not return a job ID.');

      // Poll until done
      let isDone = false;
      let consecutiveErrors = 0;
      const MAX_ERRORS = 10;

      while (!isDone) {
        await new Promise((res) => setTimeout(res, 1200));

        try {
          const updatedJob = await getJobStatus(jobId);
          consecutiveErrors = 0;
          setActiveJob(updatedJob);
          setSelectedJobId(updatedJob.jobId);

          if (updatedJob.status === 'completed') {
            isDone = true;
            setActiveJobs([updatedJob]);
            setStep('results');
          } else if (updatedJob.status === 'failed') {
            isDone = true;
            setInferenceError(
              updatedJob.error ||
              updatedJob.message ||
              'Super-resolution processing failed in neural inference pipeline.'
            );
          }
        } catch (pollErr) {
          consecutiveErrors++;
          if (consecutiveErrors >= MAX_ERRORS) {
            isDone = true;
            setInferenceError('Connection to inference server was lost during processing.');
          }
        }
      }
    } catch (err: unknown) {
      const msg = err instanceof Error ? err.message : 'Failed to start super-resolution.';
      setInferenceError(msg);
    } finally {
      setIsSubmitting(false);
    }
  };

  // ── Accidental Result Loss Protection ──
  const requestConfirmation = (action: () => void) => {
    if (activeJob?.status === 'completed' || step === 'results') {
      setPendingConfirmAction(() => action);
      setIsConfirmModalOpen(true);
    } else {
      action();
    }
  };

  const handleConfirmAction = () => {
    if (pendingConfirmAction) {
      pendingConfirmAction();
    }
    setIsConfirmModalOpen(false);
    setPendingConfirmAction(null);
  };

  const handleCancelConfirm = () => {
    setIsConfirmModalOpen(false);
    setPendingConfirmAction(null);
  };

  const executeReset = () => {
    resetAll();
    setValidationError(null);
    setInferenceError(null);
    setIsSubmitting(false);
    setValidationPhase('idle');
    setIsValidationModalOpen(false);
    setStep('upload');
  };

  const executeNewAnalysis = () => {
    executeReset();
  };

  return (
    <div className="min-h-screen bg-gradient-to-b from-slate-50 via-blue-50/20 to-white dark:from-transparent dark:via-transparent dark:to-transparent dark:bg-transparent py-10 px-4 sm:px-6">
      <div className="max-w-[1120px] mx-auto">
        {/* ── Header ── */}
        <div className="mb-10 text-center sm:text-left">
          <div className="flex items-center gap-1.5 px-3 py-1 text-[11px] uppercase tracking-widest font-bold rounded-full bg-cyan-500/10 border border-cyan-500/20 text-cyan-700 dark:bg-cyan-500/15 dark:text-cyan-400 mb-4 w-fit sm:mx-0 mx-auto">
            <Zap size={11} />
            4-band Sentinel-2 super-resolution
          </div>
          <h1 className="text-3xl sm:text-4xl font-black text-primary tracking-tight mb-3">
            Enhance multispectral imagery
          </h1>
          <p className="text-sm sm:text-base text-secondary max-w-xl leading-relaxed sm:mx-0 mx-auto">
            Upload separate 10m Sentinel-2 bands (B02, B03, B04, B08), validate spatial compatibility, and reconstruct sub-4m maps using SwinIR.
          </p>
        </div>

        {/* ── Backend Offline Warning ── */}
        {backendAvailable === false && (
          <div className="mb-8 p-5 rounded-2xl bg-amber-500/20 border border-amber-500/30 text-amber-800 anim-fade-in">
            <div className="flex items-start gap-3.5">
              <AlertTriangle size={20} className="text-amber-600 shrink-0 mt-0.5" />
              <div className="space-y-2 flex-1">
                <h4 className="font-bold text-primary text-sm">TerraSR backend is unavailable</h4>
                <p className="text-xs text-amber-700 dark:text-amber-400/90 leading-relaxed">
                  Please make sure the TerraSR FastAPI backend is running and try again.<br />
                  Backend endpoint: <code className="bg-white/50 dark:bg-slate-900/50 px-1.5 py-0.5 rounded text-amber-800 dark:text-amber-500 font-mono border border-amber-500/20">{import.meta.env.VITE_API_BASE_URL || 'http://localhost:8000'}</code>
                </p>
                <button
                  onClick={checkHealth}
                  disabled={isCheckingBackend}
                  className="inline-flex items-center gap-1.5 px-3 py-1.5 rounded-lg bg-amber-500/20 hover:bg-amber-500/30 border border-amber-500/40 text-amber-800 dark:text-amber-500 text-xs font-semibold transition-all disabled:opacity-50"
                >
                  <RefreshCw size={13} className={isCheckingBackend ? 'animate-spin' : ''} />
                  <span>Retry Connection</span>
                </button>
              </div>
            </div>
          </div>
        )}

        {/* ── 4-Stage Step Indicator ── */}
        <div className="flex items-center justify-between gap-0 mb-10 max-w-3xl mx-auto p-4 sm:px-8 rounded-2xl bg-white/60 dark:bg-slate-950/25 backdrop-blur-md border border-slate-200/60 dark:border-white/10 shadow-sm">
          {STEP_LABELS.map(({ id, label, icon }, i) => {
            const isCurrent = step === id;
            const isCompleted =
              (step === 'metadata' && id === 'upload') ||
              (step === 'processing' && (id === 'upload' || id === 'metadata')) ||
              (step === 'results' && id !== 'results');

            return (
              <React.Fragment key={id}>
                <div className="flex flex-col items-center gap-2 relative z-10">
                  <div
                    className={`w-9 h-9 rounded-xl flex items-center justify-center border transition-all duration-200 ${
                      isCurrent
                        ? 'bg-cyan-500/10 border-cyan-500/30 text-cyan-600 dark:text-cyan-400 dark:shadow-[0_0_15px_rgba(6,182,212,0.15)]'
                        : isCompleted
                        ? 'bg-emerald-500/10 border-emerald-500/20 text-emerald-600 dark:text-emerald-400'
                        : 'bg-white/50 dark:bg-slate-900/40 border-slate-300/50 dark:border-white/10 text-muted-foreground'
                    }`}
                  >
                    {isCompleted ? <CheckCircle2 size={15} /> : icon}
                  </div>
                  <span
                    className={`text-[11px] font-semibold tracking-tight ${
                      isCurrent
                        ? 'text-cyan-700 dark:text-cyan-400'
                        : isCompleted
                        ? 'text-emerald-600 dark:text-emerald-400'
                        : 'text-secondary'
                    }`}
                  >
                    {label}
                  </span>
                </div>
                {i < STEP_LABELS.length - 1 && (
                  <div
                    className={`flex-1 h-[2px] mx-1 sm:mx-3 mb-5 rounded-full transition-colors duration-200 ${
                      isCompleted
                        ? 'bg-emerald-500/30 dark:bg-emerald-500/40'
                        : 'bg-slate-300 dark:bg-white/10'
                    }`}
                  />
                )}
              </React.Fragment>
            );
          })}
        </div>

        {/* ── STAGE 1: UPLOAD (Four-Band Ingestion) ── */}
        {step === 'upload' && (
          <div className="grid lg:grid-cols-5 gap-6 anim-fade-in">
            <div className="lg:col-span-3 space-y-5">
              <FourBandUploader
                onValidate={handleValidate}
                isValidating={isValidating}
                validationError={validationError}
                disabled={isSubmitting}
              />
            </div>

            {/* Model Settings sidebar */}
            <div className="lg:col-span-2">
              <div className="bg-white/75 dark:bg-slate-950/65 backdrop-blur-xl border border-slate-900/10 dark:border-white/15 shadow-sm rounded-2xl p-6 h-full space-y-5">
                <div className="flex items-center gap-2 border-b border-slate-900/10 dark:border-white/15 pb-4">
                  <div className="w-7 h-7 rounded-lg bg-cyan-500/10 border border-cyan-400/30 flex items-center justify-center">
                    <Settings2 size={13} className="text-cyan-700 dark:text-cyan-400" />
                  </div>
                  <h2 className="text-base font-bold text-primary">Super-Resolution Settings</h2>
                </div>

                <ModelSettings
                  values={settings}
                  onChange={setSettings}
                  disabled={isSubmitting}
                />
              </div>
            </div>
          </div>
        )}

        {/* ── STAGE 2: METADATA & VALIDATION RESULTS ── */}
        {step === 'metadata' && validationResult && (
          <div className="anim-fade-in max-w-5xl mx-auto space-y-6">
            <BandMetadataPanel
              validationResult={validationResult}
              onBack={() => setStep('upload')}
              onRunInference={handleRunFourBandSR}
              isRunning={isSubmitting}
            />
          </div>
        )}

        {/* ── STAGE 3: SWINIR PROCESSING ── */}
        {step === 'processing' && (
          <div className="max-w-3xl mx-auto anim-fade-in space-y-6">
            {inferenceError ? (
              <div className="glass rounded-2xl p-6 border border-red-500/30 bg-red-500/[0.03] space-y-4">
                <div className="flex items-start gap-3">
                  <AlertCircle size={22} className="text-red-600 shrink-0 mt-0.5" />
                  <div>
                    <h3 className="text-base font-bold text-red-600">
                      Super-resolution processing failed
                    </h3>
                    <p className="text-xs text-secondary mt-1 leading-relaxed">
                      The backend returned an error while processing the imagery.
                      Please check the input files and backend logs, then try again.
                    </p>
                    <div className="mt-3 p-3 rounded-lg glass-panel border border-red-500/20 font-mono text-xs text-red-700">
                      {inferenceError}
                    </div>
                  </div>
                </div>

                <div className="flex items-center gap-3 pt-2 border-t border-theme">
                  <button
                    type="button"
                    onClick={handleRunFourBandSR}
                    className="inline-flex items-center gap-2 px-4 py-2 rounded-xl bg-accent/20 hover:bg-accent/30 border border-cyan-200 text-cyan-700 text-xs font-semibold transition-all"
                  >
                    <RotateCcw size={13} />
                    <span>Try Again</span>
                  </button>
                  <button
                    type="button"
                    onClick={() => setStep('upload')}
                    className="inline-flex items-center gap-2 px-4 py-2 rounded-xl glass-panel hover:glass-panel border border-theme text-secondary text-xs font-semibold transition-all"
                  >
                    <ArrowLeft size={13} />
                    <span>Back to Upload</span>
                  </button>
                </div>
              </div>
            ) : (
              <div className="glass rounded-2xl p-6 space-y-5 bg-white border border-slate-200/60 shadow-[0_8px_30px_rgb(0,0,0,0.04)] dark:bg-transparent dark:shadow-none dark:border-theme">
                <div className="flex items-center justify-between border-b border-theme pb-4">
                  <div className="flex items-center gap-2.5">
                    <RefreshCw size={16} className="text-accent animate-spin" />
                    <div>
                      <h3 className="text-sm font-bold text-primary">
                        Processing Multispectral Imagery
                      </h3>
                      <p className="text-[11px] text-secondary font-mono">
                        Preparing B02 / B03 / B04 / B08 · 4-Channel SwinIR
                      </p>
                    </div>
                  </div>
                  <span className="text-xs font-mono px-2.5 py-1 rounded-full bg-accent/20 border border-cyan-200 text-cyan-700 font-semibold">
                    {settings.model}
                  </span>
                </div>

                {activeJob ? (
                  <ProcessingProgress job={activeJob} />
                ) : (
                  <div className="py-12 text-center text-secondary text-xs font-mono">
                    Initializing SwinIR inference pipeline...
                  </div>
                )}

                <div className="pt-3 border-t border-theme flex items-center justify-between text-xs">
                  <button
                    type="button"
                    onClick={() => setStep('metadata')}
                    className="inline-flex items-center gap-1.5 text-secondary hover:text-primary transition-colors"
                  >
                    <ArrowLeft size={13} />
                    <span>Back to Metadata</span>
                  </button>
                  <span className="text-muted-foreground font-mono text-[11px]">
                    Non-destructive · Process runs in background
                  </span>
                </div>
              </div>
            )}
          </div>
        )}

        {/* ── STAGE 4: RESULTS ── */}
        {step === 'results' && activeJob && (
          <div className="anim-fade-in space-y-6">
            {/* Top Navigation Strip */}
            <div className="glass rounded-2xl p-4 flex items-center justify-between gap-4 border border-slate-200/60 bg-white shadow-[0_8px_30px_rgb(0,0,0,0.04)] dark:bg-transparent dark:shadow-none dark:border-theme">
              <div className="flex items-center gap-3">
                <button
                  type="button"
                  onClick={() => setStep('metadata')}
                  className="inline-flex items-center gap-1.5 px-3 py-1.5 rounded-lg glass-panel hover:glass-panel border border-theme text-xs text-secondary transition-all"
                >
                  <ArrowLeft size={13} />
                  <span>Back to Metadata</span>
                </button>
                <span className="text-xs font-mono text-muted-foreground">
                  Mission: <span className="text-secondary font-bold">{activeJob.jobId}</span>
                </span>
              </div>

              <div className="flex items-center gap-3">
                <button
                  type="button"
                  onClick={() => requestConfirmation(executeNewAnalysis)}
                  className="inline-flex items-center gap-2 px-4 py-2 rounded-xl bg-accent/20 hover:bg-accent/25 border border-cyan-400/30 text-cyan-700 text-xs font-bold transition-all shadow-[0_0_15px_rgba(0,212,255,0.15)]"
                >
                  <Upload size={13} />
                  <span>New Analysis</span>
                </button>
              </div>
            </div>

            {/* Results Viewer */}
            <ResultsViewer
              job={activeJob}
              onNewEnhancement={() => requestConfirmation(executeNewAnalysis)}
            />
          </div>
        )}

        {/* ── Accidental Loss Confirmation Modal ── */}
        {isConfirmModalOpen && (
          <div className="fixed inset-0 z-50 flex items-center justify-center p-4 glass-panel backdrop-blur-sm anim-fade-in">
            <div className="glass rounded-2xl p-6 max-w-md w-full border border-slate-200/60 bg-white shadow-[0_8px_30px_rgb(0,0,0,0.08)] dark:bg-transparent dark:border-amber-500/30 dark:shadow-sm space-y-4">
              <div className="flex items-center gap-3 text-amber-600">
                <div className="w-9 h-9 rounded-xl bg-amber-500/15 border border-amber-500/30 flex items-center justify-center shrink-0">
                  <AlertCircle size={20} />
                </div>
                <h3 className="text-base font-bold text-primary">
                  Active Analysis in Progress
                </h3>
              </div>

              <p className="text-xs text-secondary leading-relaxed">
                You have an active analysis and generated results. Starting a new analysis will clear the current analysis state.
              </p>

              <div className="flex items-center justify-end gap-3 pt-2">
                <button
                  type="button"
                  onClick={handleCancelConfirm}
                  className="px-4 py-2 rounded-xl glass-panel hover:glass-panel border border-theme text-xs font-semibold text-secondary transition-all"
                >
                  Cancel
                </button>
                <button
                  type="button"
                  onClick={handleConfirmAction}
                  className="px-4 py-2 rounded-xl bg-red-500/20 hover:bg-red-500/30 border border-red-500/40 text-xs font-bold text-red-700 transition-all shadow-[0_0_15px_rgba(239,68,68,0.2)]"
                >
                  Start New Analysis
                </button>
              </div>
            </div>
          </div>
        )}

        {/* ── Satellite Imagery Validation Scanner Modal (Animation 2) ── */}
        <ValidationScannerModal
          isOpen={isValidationModalOpen}
          phase={validationPhase}
          validationResult={validationResult}
          error={validationError}
          onClose={() => {
            setIsValidationModalOpen(false);
            setValidationPhase('idle');
          }}
        />
      </div>
    </div>
  );
};

export default EnhancePage;
