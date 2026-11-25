import type { AnimationMode, CrocControlParams, EnvironmentMode } from './kinematics';

export type Vec3Tuple = [number, number, number];

export interface SharedViewState {
  params: CrocControlParams;
  wireframe: boolean;
  showSkeleton: boolean;
  autoRotate: boolean;
  followCamera: boolean;
  collapsed: boolean;
  cameraPosition: Vec3Tuple;
  cameraTarget: Vec3Tuple;
}

/**
 * Default elevated high-angle overhead Studio showcase camera ("up view")
 * framing the full armored back, predatory head, sprawling legs, and sweeping tail.
 */
export const DEFAULT_CAMERA_POS: Vec3Tuple = [3.75, 3.25, 2.75];
export const DEFAULT_CAMERA_TARGET: Vec3Tuple = [0.0, 0.22, -0.05];

export const DEFAULT_VIEW_STATE: SharedViewState = {
  params: {
    mode: 'idle',
    walkSpeed: 1.3,
    manualJawAngleDeg: 0,
    tailSwayIntensity: 1.75,
    wetness: 0.14,
    environment: 'studio',
  },
  wireframe: false,
  showSkeleton: false,
  autoRotate: false,
  followCamera: false,
  collapsed: false,
  cameraPosition: [...DEFAULT_CAMERA_POS],
  cameraTarget: [...DEFAULT_CAMERA_TARGET],
};

const VALID_MODES = new Set<AnimationMode>(['idle', 'walk', 'highWalk', 'swim', 'autoDemo']);
const VALID_ENVS = new Set<EnvironmentMode>(['studio', 'water']);

function clamp(v: number, min: number, max: number): number {
  return Math.max(min, Math.min(max, v));
}

function parseFiniteFloat(raw: string | null): number | null {
  if (raw === null || raw.trim() === '') return null;
  const n = Number(raw);
  return Number.isFinite(n) ? n : null;
}

function parseBoolFlag(raw: string | null, fallback: boolean): boolean {
  if (raw === '1' || raw === 'true') return true;
  if (raw === '0' || raw === 'false') return false;
  return fallback;
}

function parseVec3(raw: string | null, fallback: Vec3Tuple, minVal: number, maxVal: number): Vec3Tuple {
  if (!raw) return [...fallback];
  const parts = raw.split(',');
  if (parts.length !== 3) return [...fallback];
  const nums = parts.map((p) => Number(p));
  if (nums.some((n) => !Number.isFinite(n))) return [...fallback];
  return [
    clamp(nums[0], minVal, maxVal),
    clamp(nums[1], minVal, maxVal),
    clamp(nums[2], minVal, maxVal),
  ];
}

function formatVec3(vec: Vec3Tuple): string {
  return vec.map((n) => Number(n.toFixed(2))).join(',');
}

/**
 * Parses and sanitizes URL query parameters into a complete, safe viewer state.
 * Missing or invalid values fall back cleanly to DEFAULT_VIEW_STATE.
 */
export function parseViewStateFromSearch(search: string): SharedViewState {
  const sp = new URLSearchParams(search.startsWith('?') ? search.slice(1) : search);
  const d = DEFAULT_VIEW_STATE;

  const rawMode = sp.get('mode') as AnimationMode | null;
  const mode: AnimationMode = rawMode && VALID_MODES.has(rawMode) ? rawMode : d.params.mode;

  const rawEnv = sp.get('env') as EnvironmentMode | null;
  const environment: EnvironmentMode =
    rawEnv && VALID_ENVS.has(rawEnv) ? rawEnv : d.params.environment;

  const parsedSpeed = parseFiniteFloat(sp.get('speed'));
  const walkSpeed = parsedSpeed !== null ? clamp(parsedSpeed, 0.3, 2.4) : d.params.walkSpeed;

  const parsedJaw = parseFiniteFloat(sp.get('jaw'));
  const manualJawAngleDeg =
    parsedJaw !== null ? clamp(parsedJaw, 0, 45) : d.params.manualJawAngleDeg;

  const parsedSway = parseFiniteFloat(sp.get('sway'));
  const tailSwayIntensity =
    parsedSway !== null ? clamp(parsedSway, 0, 2.0) : d.params.tailSwayIntensity;

  const parsedWet = parseFiniteFloat(sp.get('wet'));
  const wetness = parsedWet !== null ? clamp(parsedWet, 0, 1.0) : d.params.wetness;

  const cameraPosition = parseVec3(sp.get('cam'), d.cameraPosition, -25, 25);
  // Keep camera above floor plane so shared links never trap user underground
  cameraPosition[1] = Math.max(0.25, cameraPosition[1]);

  const cameraTarget = parseVec3(sp.get('tgt'), d.cameraTarget, -25, 25);
  cameraTarget[1] = clamp(cameraTarget[1], -2, 8);

  return {
    params: {
      mode,
      walkSpeed,
      manualJawAngleDeg,
      tailSwayIntensity,
      wetness,
      environment,
    },
    wireframe: parseBoolFlag(sp.get('wire'), d.wireframe),
    showSkeleton: parseBoolFlag(sp.get('skel'), d.showSkeleton),
    autoRotate: parseBoolFlag(sp.get('turn'), d.autoRotate),
    followCamera: parseBoolFlag(sp.get('follow'), d.followCamera),
    collapsed: parseBoolFlag(sp.get('hud') === '0' ? '1' : sp.get('hud') === '1' ? '0' : null, d.collapsed),
    cameraPosition,
    cameraTarget,
  };
}

/**
 * Serializes the live crocodile pose, controls, and camera angle into a compact shareable URL.
 */
export function buildShareUrl(baseHref: string, state: SharedViewState): string {
  const url = new URL(baseHref);
  const sp = new URLSearchParams();

  sp.set('mode', state.params.mode);
  sp.set('env', state.params.environment);
  sp.set('speed', state.params.walkSpeed.toFixed(1));
  if (Math.round(state.params.manualJawAngleDeg) > 0) {
    sp.set('jaw', String(Math.round(state.params.manualJawAngleDeg)));
  }
  sp.set('sway', state.params.tailSwayIntensity.toFixed(2));
  sp.set('wet', state.params.wetness.toFixed(2));

  if (state.wireframe) sp.set('wire', '1');
  if (state.showSkeleton) sp.set('skel', '1');
  if (state.autoRotate) sp.set('turn', '1');
  if (state.followCamera) sp.set('follow', '1');
  if (state.collapsed) sp.set('hud', '0');

  sp.set('cam', formatVec3(state.cameraPosition));
  sp.set('tgt', formatVec3(state.cameraTarget));

  url.search = sp.toString();
  return url.toString();
}
