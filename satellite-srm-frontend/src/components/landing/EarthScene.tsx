import React, { Suspense, useState, useEffect } from 'react';
import { Canvas } from '@react-three/fiber';
import { OrbitControls } from '@react-three/drei';
import { RealisticEarth } from '../scene/RealisticEarth';
import { Satellite } from '../scene/Satellite';
import { StarField } from '../scene/StarField';
import { DeepSpaceElements } from '../scene/DeepSpaceElements';
import { ErrorBoundary } from '../scene/ErrorBoundary';
import { Satellite as SatelliteIcon } from 'lucide-react';
import * as THREE from 'three';
import { useThree, useFrame } from '@react-three/fiber';

// ── INTRO ZOOM ANIMATION ──
// Camera starts at (60,30,90) via the Canvas camera prop — no jump, no flicker.
// Immediately starts the smooth lerp toward Earth.
const IntroAnimation: React.FC<{ onFinished: () => void }> = ({ onFinished }) => {
  const { camera } = useThree();

  useFrame(() => {
    // Zoom in smoother from a less extreme distance
    const targetPos = new THREE.Vector3(-1.2, 0.5, 4.0);
    camera.position.lerp(targetPos, 0.015);
    camera.lookAt(0, 0, 0);

    if (camera.position.distanceTo(targetPos) < 0.5) {
      onFinished();
    }
  });
  return null;
};

interface EarthSceneProps {
  interactive?: boolean;
  enableZoom?: boolean;
  autoRotateSpeed?: number;
  className?: string;
  showHint?: boolean;
  // Satellite configurations
  showSatellite?: boolean;
  satelliteOrbitRadius?: number;
  satelliteOrbitSpeed?: number;
  satelliteOrbitInclination?: number;
  showScanBeam?: boolean;
}

// ── ERROR-SAFE FALLBACK VISUAL ──
export const EarthVisualFallback: React.FC<{ message?: string }> = ({
  message = 'Earth Observation Simulation Active',
}) => (
  <div className="w-full h-full min-h-[380px] flex flex-col items-center justify-center p-6 text-center select-none relative overflow-hidden rounded-3xl bg-gradient-to-b from-srm-surface via-srm-base to-srm-elevated border border-cyan-500/25 backdrop-blur-xl shadow-[0_0_50px_rgba(0,212,255,0.08)]">
    <div className="absolute w-72 h-72 rounded-full border border-cyan-200 animate-ping opacity-20 pointer-events-none" />
    <div className="absolute w-96 h-96 rounded-full border border-blue-500/10 pointer-events-none" />

    {/* Center Earth Graphic Orb */}
    <div className="relative w-48 h-48 sm:w-60 sm:h-60 rounded-full bg-gradient-to-tr from-srm-surface via-srm-base to-srm-elevated border-2 border-cyan-200 shadow-[0_0_60px_rgba(0,212,255,0.35)] flex items-center justify-center mb-6 overflow-hidden">
      <div className="absolute inset-0 bg-[radial-gradient(#00e5ff18_1px,transparent_1px)] [background-size:12px_12px] opacity-70" />
      <div className="absolute inset-0 rounded-full shadow-[inset_0_0_30px_rgba(0,212,255,0.6)]" />
      <div className="absolute inset-0 rounded-full border border-cyan-200 border-dashed animate-spin-slow pointer-events-none" />

      {/* Orbiting Satellite Graphic */}
      <div className="relative z-10 flex flex-col items-center">
        <div className="w-14 h-14 rounded-2xl bg-accent/20 border border-cyan-200 flex items-center justify-center text-cyan-700 shadow-[0_0_20px_rgba(0,212,255,0.4)] mb-2">
          <SatelliteIcon size={28} className="animate-pulse" />
        </div>
        <span className="font-mono text-[11px] font-bold text-cyan-700 uppercase tracking-wider">
          TerraSR-1 Satellite
        </span>
      </div>
    </div>

    <div className="relative z-10 flex items-center gap-2 px-3 py-1 rounded-full bg-accent/20 border border-cyan-400/30 text-xs font-mono text-cyan-700 mb-2">
      <span className="w-2 h-2 rounded-full bg-cyan-400 animate-pulse" />
      <span>{message}</span>
    </div>

    <p className="relative z-10 text-xs text-secondary max-w-xs leading-relaxed font-mono">
      AI multispectral super-resolution & orbital observation active.
    </p>
  </div>
);

export const EarthScene: React.FC<EarthSceneProps> = ({
  interactive = true,
  enableZoom = true,
  autoRotateSpeed = 0.5,
  className = 'w-full h-[520px] lg:h-[620px]',
  showHint = true,
  // Satellite configurations
  showSatellite = true,
  satelliteOrbitRadius = 1.75,
  satelliteOrbitSpeed = 0.32,
  satelliteOrbitInclination = 45,
  showScanBeam = true,
}) => {
  const [hasWebGL, setHasWebGL] = useState<boolean>(true);
  const [isInteracting, setIsInteracting] = useState<boolean>(false);
  const [introDone, setIntroDone] = useState<boolean>(false);

  useEffect(() => {
    try {
      const canvas = document.createElement('canvas');
      const gl =
        canvas.getContext('webgl') ||
        canvas.getContext('experimental-webgl');
      if (!gl) {
        setHasWebGL(false);
      }
    } catch {
      setHasWebGL(false);
    }
  }, []);

  if (!hasWebGL) {
    return (
      <div className={`relative ${className}`}>
        <EarthVisualFallback message="WebGL Mode (Fallback Display)" />
      </div>
    );
  }

  return (
    <div className={`relative ${className} select-none`}>
      {/* Subtle outer ambient space glow */}
      <div className="absolute inset-0 rounded-full bg-accent/[0.04] blur-3xl pointer-events-none" />

      {/* ErrorBoundary wraps WebGL Canvas */}
      <ErrorBoundary fallback={<EarthVisualFallback />}>
        <Canvas
          camera={{ position: [8, 4, 12], fov: 42 }}
          gl={{
            antialias: true,
            alpha: true,
            powerPreference: 'high-performance',
          }}
          dpr={[1, 1.5]}
          className="w-full h-full cursor-grab active:cursor-grabbing"
          onPointerDown={() => setIsInteracting(true)}
          onPointerUp={() => setIsInteracting(false)}
        >
          {/* Deep Space Background Stars with subtle gentle movement */}
          <StarField count={2200} speed={0.6} />

          {/* Meteoroids and Distant Planets */}
          <DeepSpaceElements />

          {/* ── ENHANCED CINEMATIC ILLUMINATION (matching reference) ── */}
          {/* 1. Low ambient to allow metallic highlights to pop */}
          <ambientLight color="#c8d8e8" intensity={0.7} />

          {/* 2. Key Sun — upper right, warm white, main source */}
          <directionalLight
            position={[6.0, 5.0, 4.0]}
            intensity={3.2}
            color="#fff5e6"
            castShadow
          />

          {/* 3. Soft frontal fill — prevents harsh shadows on Earth */}
          <directionalLight
            position={[1, 1.0, 6]}
            intensity={0.6}
            color="#f0f9ff"
          />

          {/* 4. Atmospheric blue rim from lower-left (Earth shine) */}
          <directionalLight
            position={[-5, -2, -4]}
            intensity={0.9}
            color="#60a5fa"
          />

          {/* 5. Soft under-fill for bottom hemisphere */}
          <directionalLight
            position={[0, -4, 2]}
            intensity={0.3}
            color="#dbeafe"
          />

          <Suspense fallback={null}>
            <RealisticEarth />
            {showSatellite && (
              <Satellite
                orbitRadius={1.82}
                orbitSpeed={0.18}
                orbitInclination={35}
                showOrbitPath={true}
                showScanBeam={showScanBeam}
              />
            )}
          </Suspense>

          {!introDone && <IntroAnimation onFinished={() => setIntroDone(true)} />}

          {introDone && interactive && (
            <OrbitControls
              enableRotate={true}
              autoRotate={!isInteracting}
              autoRotateSpeed={autoRotateSpeed}
              enableZoom={enableZoom}
              minDistance={2.2}
              maxDistance={250.0}
              enablePan={false}
              enableDamping={true}
              dampingFactor={0.06}
              rotateSpeed={0.8}
            />
          )}
        </Canvas>
      </ErrorBoundary>

      {/* Interactive Control Pill Hint */}
      {showHint && (
        <div className="absolute bottom-4 left-1/2 -translate-x-1/2 z-20 flex items-center gap-2.5 px-3.5 py-1.5 rounded-full glass-panel border border-theme text-xs text-secondary pointer-events-none shadow-lg shadow-black/40">
          <span className="w-2 h-2 rounded-full bg-cyan-400 animate-pulse" />
          <span className="font-mono text-[11px] tracking-wide text-cyan-700/90 uppercase">
            3D Satellite Observation
          </span>
          <span className="text-muted-foreground">•</span>
          <span className="text-secondary text-[11px]">Drag to rotate</span>
          {enableZoom && (
            <>
              <span className="text-muted-foreground">•</span>
              <span className="text-secondary text-[11px]">Scroll to zoom</span>
            </>
          )}
        </div>
      )}

      {/* Real-Time Telemetry Badge */}
      <div className="absolute top-4 right-4 z-20 hidden sm:flex items-center gap-2 px-3 py-1.5 rounded-xl bg-surface/85 backdrop-blur-md border border-cyan-500/25 text-xs text-secondary shadow-md">
        <SatelliteIcon size={14} className="text-accent animate-pulse" />
        <span className="font-mono text-[11px] text-secondary">TerraSR-1:</span>
        <span className="font-mono text-[11px] text-cyan-700 font-semibold">Active Scanning</span>
      </div>
    </div>
  );
};
