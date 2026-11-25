import React from 'react';
import {
  Check,
  ChevronDown,
  ChevronUp,
  Download,
  Eye,
  Grid3X3,
  Link2,
  Orbit,
  Bone,
  Waves,
  Sun,
  Video,
  Zap,
} from 'lucide-react';
import type { AnimationMode, CrocControlParams, EnvironmentMode } from '../crocodile/kinematics';

export interface ControlDockProps {
  params: CrocControlParams;
  resolvedMode: Exclude<AnimationMode, 'autoDemo'>;
  wireframe: boolean;
  showSkeleton: boolean;
  autoRotate: boolean;
  followCamera: boolean;
  collapsed: boolean;
  exporting: boolean;
  exportMessage: string | null;
  shareCopied: boolean;
  onModeChange: (mode: AnimationMode) => void;
  onTriggerBite: () => void;
  onTriggerBlink: () => void;
  onParamChange: <K extends keyof CrocControlParams>(key: K, value: CrocControlParams[K]) => void;
  onEnvironmentChange: (env: EnvironmentMode) => void;
  onToggleWireframe: () => void;
  onToggleSkeleton: () => void;
  onToggleAutoRotate: () => void;
  onToggleFollowCamera: () => void;
  onToggleCollapsed: () => void;
  onShareLink: () => void;
  onExportGLB: () => void;
}

const ANIMATION_MODES: Array<{ id: AnimationMode; label: string }> = [
  { id: 'idle', label: 'Idle' },
  { id: 'walk', label: 'Walk' },
  { id: 'highWalk', label: 'High Walk' },
  { id: 'swim', label: 'Swim' },
  { id: 'autoDemo', label: 'Auto-Demo' },
];

export const ControlDock: React.FC<ControlDockProps> = ({
  params,
  resolvedMode,
  wireframe,
  showSkeleton,
  autoRotate,
  followCamera,
  collapsed,
  exporting,
  exportMessage,
  shareCopied,
  onModeChange,
  onTriggerBite,
  onTriggerBlink,
  onParamChange,
  onEnvironmentChange,
  onToggleWireframe,
  onToggleSkeleton,
  onToggleAutoRotate,
  onToggleFollowCamera,
  onToggleCollapsed,
  onShareLink,
  onExportGLB,
}) => {
  return (
    <div
      className="pointer-events-none fixed inset-x-0 bottom-0 z-20 flex flex-col items-center px-3 pb-4 sm:px-5 sm:pb-5"
      data-testid="control-dock-wrapper"
    >
      {/* Top-edge compact disclosure toggle & quiet keyboard hint */}
      <div className="pointer-events-auto mb-2 flex items-center gap-3">
        <button
          type="button"
          onClick={onToggleCollapsed}
          aria-expanded={!collapsed}
          className="inline-flex items-center gap-1.5 rounded-lg border border-white/15 bg-[#161815]/92 px-3 py-1.5 text-xs font-medium text-stone-200 shadow-md transition-colors hover:border-amber-400/40 hover:text-white focus-visible:outline-2 focus-visible:outline-offset-2 focus-visible:outline-amber-400 whitespace-nowrap shrink-0"
        >
          {collapsed ? <ChevronUp className="h-3.5 w-3.5" /> : <ChevronDown className="h-3.5 w-3.5" />}
          <span>{collapsed ? 'Show Controls' : 'Hide Controls'}</span>
        </button>

        {!collapsed && (
          <div className="hidden items-center gap-2 rounded-lg border border-white/10 bg-[#161815]/84 px-3 py-1.5 text-[11px] text-stone-300 md:flex">
            <span>WASD Drive</span>
            <span aria-hidden="true" className="text-stone-500">·</span>
            <span>Shift Sprint</span>
            <span aria-hidden="true" className="text-stone-500">·</span>
            <span>Space Bite</span>
            <span aria-hidden="true" className="text-stone-500">·</span>
            <span>B Blink</span>
            <span aria-hidden="true" className="text-stone-500">·</span>
            <span>H Toggle HUD</span>
          </div>
        )}

        {exportMessage && (
          <div
            role="status"
            className="rounded-lg border border-amber-400/35 bg-[#1d201a]/95 px-3 py-1.5 text-xs font-medium text-amber-200 shadow-md whitespace-nowrap"
          >
            {exportMessage}
          </div>
        )}
      </div>

      {/* Collapsible Unified Bottom Dock (solid dark rgba surface — zero backdrop-filter over WebGL) */}
      {!collapsed && (
        <section
          aria-label="Crocodile Controls"
          className="pointer-events-auto w-full max-w-[1040px] rounded-xl border border-white/14 bg-[#151714]/94 p-3.5 text-stone-100 shadow-2xl sm:p-4"
        >
          {/* Row 1: Animation Modes, Bite Trigger, Environment Switch, Export .GLB */}
          <div className="flex flex-wrap items-center justify-between gap-2.5 border-b border-white/10 pb-3">
            {/* Animation Mode Segmented Control */}
            <div className="flex flex-wrap items-center gap-1.5" role="group" aria-label="Animation State">
              <div className="flex items-center gap-1 rounded-lg bg-black/35 p-1">
                {ANIMATION_MODES.map((m) => {
                  const active = params.mode === m.id;
                  return (
                    <button
                      key={m.id}
                      type="button"
                      onClick={() => onModeChange(m.id)}
                      aria-pressed={active}
                      className={`rounded-md px-2.5 py-1.5 text-xs font-medium transition-colors whitespace-nowrap shrink-0 focus-visible:outline-2 focus-visible:outline-amber-400 ${
                        active
                          ? 'bg-[#c9ae63] text-[#141612] shadow-sm'
                          : 'text-stone-300 hover:bg-white/8 hover:text-white'
                      }`}
                    >
                      {m.label}
                    </button>
                  );
                })}
              </div>

              {/* Dedicated One-Shot Bite Trigger */}
              <button
                type="button"
                onClick={onTriggerBite}
                className="inline-flex items-center gap-1.5 rounded-lg border border-amber-500/40 bg-amber-600/20 px-3 py-1.5 text-xs font-semibold text-amber-200 transition-colors hover:bg-amber-600/35 hover:text-amber-100 whitespace-nowrap shrink-0 focus-visible:outline-2 focus-visible:outline-amber-400"
              >
                <Zap className="h-3.5 w-3.5" />
                <span>Bite</span>
              </button>

              {/* On-Demand Eye Blink Trigger */}
              <button
                type="button"
                onClick={onTriggerBlink}
                className="inline-flex items-center gap-1.5 rounded-lg border border-white/15 bg-white/8 px-2.5 py-1.5 text-xs font-medium text-stone-200 transition-colors hover:bg-white/15 hover:text-white whitespace-nowrap shrink-0 focus-visible:outline-2 focus-visible:outline-amber-400"
              >
                <Eye className="h-3.5 w-3.5" />
                <span>Blink</span>
              </button>
            </div>

            {/* Right group: Environment Toggle + Export .GLB CTA */}
            <div className="flex flex-wrap items-center gap-2">
              <div className="flex items-center gap-1 rounded-lg bg-black/35 p-1" role="group" aria-label="Environment">
                <button
                  type="button"
                  onClick={() => onEnvironmentChange('studio')}
                  aria-pressed={params.environment === 'studio'}
                  className={`inline-flex items-center gap-1.5 rounded-md px-2.5 py-1.5 text-xs font-medium transition-colors whitespace-nowrap shrink-0 ${
                    params.environment === 'studio'
                      ? 'bg-stone-200 text-stone-900'
                      : 'text-stone-300 hover:bg-white/8 hover:text-white'
                  }`}
                >
                  <Sun className="h-3.5 w-3.5" />
                  <span>Studio</span>
                </button>
                <button
                  type="button"
                  onClick={() => onEnvironmentChange('water')}
                  aria-pressed={params.environment === 'water'}
                  className={`inline-flex items-center gap-1.5 rounded-md px-2.5 py-1.5 text-xs font-medium transition-colors whitespace-nowrap shrink-0 ${
                    params.environment === 'water'
                      ? 'bg-teal-600 text-white'
                      : 'text-stone-300 hover:bg-white/8 hover:text-white'
                  }`}
                >
                  <Waves className="h-3.5 w-3.5" />
                  <span>Riverbank</span>
                </button>
              </div>

              <button
                type="button"
                onClick={onShareLink}
                className={`inline-flex items-center gap-1.5 rounded-lg border px-3 py-1.5 text-xs font-medium transition-colors whitespace-nowrap shrink-0 focus-visible:outline-2 focus-visible:outline-amber-400 ${
                  shareCopied
                    ? 'border-emerald-400/50 bg-emerald-500/20 text-emerald-200'
                    : 'border-white/15 bg-white/8 text-stone-100 hover:bg-white/15 hover:text-white'
                }`}
              >
                {shareCopied ? <Check className="h-3.5 w-3.5" /> : <Link2 className="h-3.5 w-3.5" />}
                <span>{shareCopied ? 'Link Copied' : 'Share Link'}</span>
              </button>

              <button
                type="button"
                onClick={onExportGLB}
                disabled={exporting}
                className="inline-flex items-center gap-1.5 rounded-lg bg-stone-100 px-3.5 py-1.5 text-xs font-semibold text-stone-950 shadow-sm transition-colors hover:bg-white disabled:opacity-60 whitespace-nowrap shrink-0 focus-visible:outline-2 focus-visible:outline-amber-400"
              >
                <Download className="h-3.5 w-3.5" />
                <span>{exporting ? 'Exporting...' : 'Download .GLB'}</span>
              </button>
            </div>
          </div>

          {/* Row 2: 4 Parameter Sliders + View Toggles */}
          <div className="mt-3 flex flex-col gap-3 lg:flex-row lg:items-center lg:justify-between">
            {/* Parameter Sliders */}
            <div className="grid flex-1 grid-cols-2 gap-x-5 gap-y-2.5 sm:grid-cols-4">
              <label className="flex flex-col gap-1">
                <div className="flex items-center justify-between text-[11px] text-stone-300">
                  <span>Walk Speed</span>
                  <span className="font-mono-readout text-stone-200">{params.walkSpeed.toFixed(1)}x</span>
                </div>
                <input
                  type="range"
                  min={0.3}
                  max={2.4}
                  step={0.1}
                  value={params.walkSpeed}
                  onChange={(e) => onParamChange('walkSpeed', parseFloat(e.target.value))}
                  aria-label="Walk Speed"
                  className="croc-slider w-full"
                />
              </label>

              <label className="flex flex-col gap-1">
                <div className="flex items-center justify-between text-[11px] text-stone-300">
                  <span>Jaw Open Angle</span>
                  <span className="font-mono-readout text-stone-200">{Math.round(params.manualJawAngleDeg)}°</span>
                </div>
                <input
                  type="range"
                  min={0}
                  max={45}
                  step={1}
                  value={params.manualJawAngleDeg}
                  onChange={(e) => onParamChange('manualJawAngleDeg', parseFloat(e.target.value))}
                  aria-label="Jaw Open Angle"
                  className="croc-slider w-full"
                />
              </label>

              <label className="flex flex-col gap-1">
                <div className="flex items-center justify-between text-[11px] text-stone-300">
                  <span>Tail Sway</span>
                  <span className="font-mono-readout text-stone-200">{Math.round(params.tailSwayIntensity * 100)}%</span>
                </div>
                <input
                  type="range"
                  min={0}
                  max={2.0}
                  step={0.05}
                  value={params.tailSwayIntensity}
                  onChange={(e) => onParamChange('tailSwayIntensity', parseFloat(e.target.value))}
                  aria-label="Tail Sway Intensity"
                  className="croc-slider w-full"
                />
              </label>

              <label className="flex flex-col gap-1">
                <div className="flex items-center justify-between text-[11px] text-stone-300">
                  <span>Wetness</span>
                  <span className="font-mono-readout text-stone-200">{Math.round(params.wetness * 100)}%</span>
                </div>
                <input
                  type="range"
                  min={0}
                  max={1}
                  step={0.02}
                  value={params.wetness}
                  onChange={(e) => onParamChange('wetness', parseFloat(e.target.value))}
                  aria-label="Scale Roughness and Wetness"
                  className="croc-slider w-full"
                />
              </label>
            </div>

            {/* View Inspection Toggles */}
            <div className="flex flex-wrap items-center gap-1.5 border-t border-white/10 pt-2.5 lg:border-t-0 lg:border-l lg:pt-0 lg:pl-4">
              <button
                type="button"
                onClick={onToggleWireframe}
                aria-pressed={wireframe}
                className={`inline-flex items-center gap-1.5 rounded-md border px-2.5 py-1.5 text-xs font-medium transition-colors whitespace-nowrap shrink-0 ${
                  wireframe
                    ? 'border-amber-400/60 bg-amber-500/20 text-amber-200'
                    : 'border-white/10 bg-white/5 text-stone-300 hover:bg-white/10 hover:text-white'
                }`}
              >
                <Grid3X3 className="h-3.5 w-3.5" />
                <span>Wireframe</span>
              </button>

              <button
                type="button"
                onClick={onToggleSkeleton}
                aria-pressed={showSkeleton}
                className={`inline-flex items-center gap-1.5 rounded-md border px-2.5 py-1.5 text-xs font-medium transition-colors whitespace-nowrap shrink-0 ${
                  showSkeleton
                    ? 'border-sky-400/60 bg-sky-500/20 text-sky-200'
                    : 'border-white/10 bg-white/5 text-stone-300 hover:bg-white/10 hover:text-white'
                }`}
              >
                <Bone className="h-3.5 w-3.5" />
                <span>Skeleton</span>
              </button>

              <button
                type="button"
                onClick={onToggleAutoRotate}
                aria-pressed={autoRotate}
                className={`inline-flex items-center gap-1.5 rounded-md border px-2.5 py-1.5 text-xs font-medium transition-colors whitespace-nowrap shrink-0 ${
                  autoRotate
                    ? 'border-amber-400/60 bg-amber-500/20 text-amber-200'
                    : 'border-white/10 bg-white/5 text-stone-300 hover:bg-white/10 hover:text-white'
                }`}
              >
                <Orbit className="h-3.5 w-3.5" />
                <span>Turntable</span>
              </button>

              <button
                type="button"
                onClick={onToggleFollowCamera}
                aria-pressed={followCamera}
                className={`inline-flex items-center gap-1.5 rounded-md border px-2.5 py-1.5 text-xs font-medium transition-colors whitespace-nowrap shrink-0 ${
                  followCamera
                    ? 'border-emerald-400/60 bg-emerald-500/20 text-emerald-200'
                    : 'border-white/10 bg-white/5 text-stone-300 hover:bg-white/10 hover:text-white'
                }`}
              >
                <Video className="h-3.5 w-3.5" />
                <span>Follow Cam</span>
              </button>

              <span className="sr-only">Active state: {resolvedMode}</span>
              <Eye className="sr-only" aria-hidden="true" />
            </div>
          </div>
        </section>
      )}
    </div>
  );
};
