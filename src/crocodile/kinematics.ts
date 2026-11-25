/**
 * Pure Reptile Locomotion, State Machine & 3-Joint Sprawling Limb IK Engine.
 * Zero DOM/WebGL dependencies so all biomechanics can be deterministically unit-tested.
 */

export type AnimationMode = 'idle' | 'walk' | 'highWalk' | 'swim' | 'bite' | 'autoDemo';
export type EnvironmentMode = 'studio' | 'water';
export type LimbId = 'FL' | 'FR' | 'BL' | 'BR';

export interface InputKeys {
  forward: boolean;
  backward: boolean;
  left: boolean;
  right: boolean;
  sprint: boolean;
}

export interface CrocControlParams {
  mode: AnimationMode;
  walkSpeed: number; // 0.3 to 2.5
  manualJawAngleDeg: number; // 0 to 45
  tailSwayIntensity: number; // 0 to 2.0
  wetness: number; // 0 (dry matte) to 1 (wet glossy)
  environment: EnvironmentMode;
}

export interface LimbJointAngles {
  /** Hip/shoulder yaw (swing fore/aft around Y) */
  hipYaw: number;
  /** Hip/shoulder abduction/elevation (outward/downward around Z) */
  hipRoll: number;
  /** Hip/shoulder pitch (around X) */
  hipPitch: number;
  /** Elbow/knee flexion bend */
  kneeBend: number;
  /** Elbow/knee inward yaw */
  kneeYaw: number;
  /** Ankle/wrist pitch to keep foot flat against ground */
  anklePitch: number;
  /** Ankle/wrist roll to keep foot flat against ground */
  ankleRoll: number;
  /** Foot ground-contact factor (1 = planted flat on Y=0, 0 = lifted in swing) */
  groundContact: number;
  /** World-relative foot target height above ground */
  footLiftY: number;
}

export interface CrocPose {
  /** Pelvis world position & heading */
  posX: number;
  posZ: number;
  heading: number;
  /** Pelvis vertical elevation above Y=0 */
  pelvisY: number;
  /** Pelvis roll (sway side to side during walk) and pitch */
  pelvisRoll: number;
  pelvisPitch: number;
  /** Forward speed (m/s) */
  currentSpeed: number;
  /** Spine segment lateral yaw angles: [midTorso, chest, neck1, neck2, neck3, head] */
  anteriorYaw: [number, number, number, number, number, number];
  /** Spine segment pitch angles: [midTorso, chest, neck1, neck2, neck3, head] */
  anteriorPitch: [number, number, number, number, number, number];
  /** 11 tail segment yaw angles (pelvis -> tip) */
  tailYaw: number[];
  /** 11 tail segment pitch angles (subtle vertical wave / ground clearance) */
  tailPitch: number[];
  /** Lower jaw opening angle in radians (0 = closed, up to ~0.785 rad = 45 deg) */
  jawAngleRad: number;
  /** Ribcage breathing expansion scale (1.0 +/- 0.025) */
  breathScale: number;
  /** Throat gular pulsing offset */
  throatPulse: number;
  /** Eyelid blink closure (0 = open, 1 = closed) */
  blinkAmount: number;
  /** Limb tuck factor (0 = planted/walking, 1 = tucked against flanks for swimming) */
  limbTuck: number;
  /** Recoil trauma from bite snap (0..1) for subtle camera shake */
  biteRecoil: number;
  /** Individual 3-joint limb angles */
  limbs: Record<LimbId, LimbJointAngles>;
  /** Effective locomotion mode currently active (resolved from autoDemo / WASD) */
  resolvedMode: Exclude<AnimationMode, 'autoDemo'>;
}

export interface KinematicsState {
  elapsedTime: number;
  gaitPhase: number;
  posX: number;
  posZ: number;
  heading: number;
  currentSpeed: number;
  currentTurnRate: number;
  /** Smooth cross-fade weights */
  wIdle: number;
  wWalk: number;
  wHighWalk: number;
  wSwim: number;
  /** Bite one-shot timer (< 0 means inactive, 0..1.35s active) */
  biteTimer: number;
  /** Previous mode to return to after one-shot bite */
  preBiteMode: AnimationMode;
  /** Periodic blink timer */
  nextBlinkTime: number;
  blinkTimer: number;
  /** Auto-demo cycle timer */
  demoTimer: number;
}

export const TAIL_SEGMENT_COUNT = 11;

export function createInitialKinematicsState(): KinematicsState {
  return {
    elapsedTime: 0,
    gaitPhase: 0,
    posX: 0,
    posZ: 0,
    heading: 0,
    currentSpeed: 0,
    currentTurnRate: 0,
    wIdle: 1,
    wWalk: 0,
    wHighWalk: 0,
    wSwim: 0,
    biteTimer: -1,
    preBiteMode: 'idle',
    nextBlinkTime: 3.2,
    blinkTimer: -1,
    demoTimer: 0,
  };
}

export function triggerBiteAction(state: KinematicsState): void {
  state.biteTimer = 0;
}

export function triggerBlinkAction(state: KinematicsState): void {
  state.blinkTimer = 0;
}

function damp(current: number, target: number, lambda: number, dt: number): number {
  return current + (target - current) * (1 - Math.exp(-lambda * dt));
}

function clamp(v: number, min: number, max: number): number {
  return Math.max(min, Math.min(max, v));
}

/**
 * Diagonal tetrapod gait phase offsets:
 * FL (0) + BR (0) move in unison; FR (PI) + BL (PI) alternate in opposite phase.
 */
export const LIMB_PHASE_OFFSET: Record<LimbId, number> = {
  FL: 0,
  BR: 0,
  FR: Math.PI,
  BL: Math.PI,
};

/**
 * Computes analytical 3-joint sprawling reptilian IK for a single limb.
 * Guarantees foot plants flat at Y=0 during stance phase and lifts in a smooth arc during swing.
 */
export function solveReptileLimbIK(params: {
  limbId: LimbId;
  gaitPhase: number;
  locomotionWeight: number;
  highWalkFactor: number;
  swimTuck: number;
  pelvisY: number;
  steeringBias: number;
}): LimbJointAngles {
  const { limbId, gaitPhase, locomotionWeight, highWalkFactor, swimTuck, pelvisY, steeringBias } = params;
  const isLeft = limbId === 'FL' || limbId === 'BL';
  const isFront = limbId === 'FL' || limbId === 'FR';
  const sideSign = isLeft ? 1 : -1;

  // Normalize limb phase into [0, 1)
  const rawPhase = ((gaitPhase + LIMB_PHASE_OFFSET[limbId]) % (Math.PI * 2) + Math.PI * 2) % (Math.PI * 2);
  const u = rawPhase / (Math.PI * 2);

  // Duty factor: stance occupies ~60% of cycle (longer during belly crawl, slightly snappier in high walk)
  const stanceFrac = 0.62 - 0.07 * highWalkFactor;

  let strideForeAft = 0; // -1 (retracted/back) to +1 (protracted/forward)
  let lift = 0; // 0 (planted on ground) to 1 (peak swing)
  let groundContact = 1;

  if (locomotionWeight > 0.001) {
    if (u < stanceFrac) {
      // Stance phase: foot locked on ground, moving linearly backward relative to torso (+1 -> -1)
      const s = u / stanceFrac;
      strideForeAft = 1 - 2 * s;
      lift = 0;
      groundContact = 1;
    } else {
      // Swing phase: foot lifts in smooth sine arc and advances forward (-1 -> +1)
      const s = (u - stanceFrac) / (1 - stanceFrac);
      strideForeAft = -1 + 2 * (0.5 - 0.5 * Math.cos(s * Math.PI));
      lift = Math.sin(s * Math.PI);
      groundContact = 0;
    }
  } else {
    // Reference alert resting stance: slight natural asymmetry matching the reference photo!
    // Reference shows front-left stepped slightly forward, front-right braced, hind legs planted wide.
    if (limbId === 'FL') strideForeAft = 0.24;
    else if (limbId === 'FR') strideForeAft = -0.14;
    else if (limbId === 'BL') strideForeAft = 0.10;
    else strideForeAft = -0.18;
    lift = 0;
    groundContact = 1;
  }

  // Blend resting stride with dynamic walking stride
  const restStride =
    limbId === 'FL' ? 0.24 : limbId === 'FR' ? -0.14 : limbId === 'BL' ? 0.1 : -0.18;
  const effectiveStride =
    restStride * (1 - locomotionWeight) + strideForeAft * locomotionWeight;
  const effectiveLift = lift * locomotionWeight;

  // Steering adjusts outer vs inner stride slightly
  const steerScale = 1 + sideSign * steeringBias * 0.22;
  const strideAngle = effectiveStride * (isFront ? 0.48 : 0.44) * steerScale;

  // Analytical sprawling reptile posture:
  // Upper arm/thigh extends OUTWARD (+sideSign*X) and slightly downward; forearm/shin drops to flat foot on Y=0!
  const socketHeight = clamp(pelvisY + (isFront ? -0.062 : -0.048), 0.20, 0.68);
  const footLiftY = effectiveLift * (isFront ? 0.14 : 0.12) * (1 - swimTuck);
  const footPadH = 0.046;

  // Segment lengths matching sleek predatory crocodileBuilder.ts
  const upperLen = isFront ? 0.28 : 0.32;
  const lowerLen = isFront ? 0.26 : 0.29;

  // Forearm/shin tilts slightly outward from vertical (~0.16 rad in high-walk, ~0.34 rad in belly crawl)
  const forearmOutTilt = (isFront ? 0.16 : 0.14) + (1 - highWalkFactor) * 0.18;
  const lowerVertDrop = lowerLen * Math.cos(forearmOutTilt);

  // Exact upper arm angle from vertical (sprawlOut) so upperLen*cos(sprawlOut) + lowerVertDrop + footPadH = socketHeight - footLiftY
  const neededUpperDrop = socketHeight - footLiftY - footPadH - lowerVertDrop;
  const cosSprawl = clamp(neededUpperDrop / upperLen, -0.35, 0.88);
  const sprawlOutAngle = Math.acos(cosSprawl); // ~1.22 rad (70 deg outward) in alert stance, ~1.62 rad in belly crawl!

  // Knee bends back inward so forearm ends up tilted slightly outward at forearmOutTilt
  const kneeReturnAngle = Math.max(0.25, sprawlOutAngle - forearmOutTilt + effectiveLift * 0.32);

  // Right-hand rotation around +Z: for Left (+1), +angle swings -Y OUTWARD toward +X; knee -angle bends it back down!
  let hipYaw = strideAngle * sideSign;
  let hipRoll = sideSign * sprawlOutAngle;
  let hipPitch = effectiveStride * (isFront ? -0.24 : -0.28) - effectiveLift * 0.26;
  let kneeBend = -sideSign * kneeReturnAngle;
  let kneeYaw = sideSign * (isFront ? -0.14 : 0.12);

  // Ankle keeps foot parallel to Y=0 ground during stance
  let anklePitch = -hipPitch * 0.72 + effectiveLift * 0.24;
  let ankleRoll = -sideSign * forearmOutTilt;

  // Swimming mode tuck: smoothly fold limbs tightly back against flanks!
  if (swimTuck > 0.001) {
    const tuckHipYaw = sideSign * (isFront ? -1.15 : -1.24);
    const tuckHipRoll = sideSign * 0.32;
    const tuckHipPitch = 0.92;
    const tuckKneeBend = -sideSign * 0.35;
    const tuckKneeYaw = sideSign * -0.12;
    const tuckAnklePitch = 0.65;
    const tuckAnkleRoll = 0;

    hipYaw = hipYaw * (1 - swimTuck) + tuckHipYaw * swimTuck;
    hipRoll = hipRoll * (1 - swimTuck) + tuckHipRoll * swimTuck;
    hipPitch = hipPitch * (1 - swimTuck) + tuckHipPitch * swimTuck;
    kneeBend = kneeBend * (1 - swimTuck) + tuckKneeBend * swimTuck;
    kneeYaw = kneeYaw * (1 - swimTuck) + tuckKneeYaw * swimTuck;
    anklePitch = anklePitch * (1 - swimTuck) + tuckAnklePitch * swimTuck;
    ankleRoll = ankleRoll * (1 - swimTuck) + tuckAnkleRoll * swimTuck;
    groundContact = groundContact * (1 - swimTuck);
  }

  return {
    hipYaw,
    hipRoll,
    hipPitch,
    kneeBend,
    kneeYaw,
    anklePitch,
    ankleRoll,
    groundContact,
    footLiftY,
  };
}

/**
 * Steps the full procedural reptile kinematics state machine forward by dt seconds.
 */
export function stepKinematics(
  state: KinematicsState,
  params: CrocControlParams,
  keys: InputKeys,
  dtRaw: number
): CrocPose {
  const dt = clamp(dtRaw, 0.0001, 0.14);
  state.elapsedTime += dt;
  const t = state.elapsedTime;

  // Determine effective locomotion target mode (handles Auto-Demo + WASD override)
  const isDriving = keys.forward || keys.backward || keys.left || keys.right;
  let targetMode: Exclude<AnimationMode, 'autoDemo'>;

  if (params.mode === 'autoDemo') {
    state.demoTimer += dt;
    const cycle = state.demoTimer % 22;
    if (cycle < 4.5) targetMode = 'idle';
    else if (cycle < 9.5) targetMode = 'walk';
    else if (cycle < 14.5) targetMode = 'highWalk';
    else if (cycle < 19.5) targetMode = 'swim';
    else {
      targetMode = 'highWalk';
      if (state.biteTimer < 0 && cycle > 19.6 && cycle < 20.1) {
        state.biteTimer = 0;
      }
    }
  } else if (params.mode === 'bite') {
    if (state.biteTimer < 0) {
      state.biteTimer = 0;
    }
    targetMode = 'idle';
  } else {
    targetMode = params.mode;
  }

  // Interactive WASD driving overrides idle to walk/highWalk (or swim in water)
  if (isDriving) {
    if (params.environment === 'water' || targetMode === 'swim') {
      targetMode = 'swim';
    } else if (keys.sprint) {
      targetMode = 'highWalk';
    } else if (targetMode === 'idle') {
      targetMode = 'walk';
    }
  }

  // Smooth exponential cross-fade weights between animation regimes
  const fadeRate = 4.2;
  state.wIdle = damp(state.wIdle, targetMode === 'idle' ? 1 : 0, fadeRate, dt);
  state.wWalk = damp(state.wWalk, targetMode === 'walk' ? 1 : 0, fadeRate, dt);
  state.wHighWalk = damp(state.wHighWalk, targetMode === 'highWalk' ? 1 : 0, fadeRate, dt);
  state.wSwim = damp(state.wSwim, targetMode === 'swim' ? 1 : 0, fadeRate, dt);

  const wSum = state.wIdle + state.wWalk + state.wHighWalk + state.wSwim + 1e-6;
  const wIdle = state.wIdle / wSum;
  const wWalk = state.wWalk / wSum;
  const wHighWalk = state.wHighWalk / wSum;
  const wSwim = state.wSwim / wSum;

  const groundLocoWeight = clamp(wWalk + wHighWalk, 0, 1);
  const highWalkFactor = clamp(wHighWalk + wIdle * 0.88, 0, 1); // Reference idle is semi-erect alert stance!
  const limbTuck = wSwim;

  // Steering & forward speed calculation
  let desiredSpeed = 0;
  let desiredTurn = 0;

  if (isDriving) {
    const dir = (keys.forward ? 1 : 0) - (keys.backward ? 1 : 0);
    const turnInput = (keys.left ? 1 : 0) - (keys.right ? 1 : 0);
    const speedBase =
      wSwim > 0.5 ? 1.85 : keys.sprint || wHighWalk > 0.5 ? 2.15 : 1.05;
    desiredSpeed = dir * speedBase * params.walkSpeed;
    // Allow turning even slightly in place
    const turnAuthority = dir !== 0 ? 1.0 : 0.72;
    desiredTurn = turnInput * 1.15 * turnAuthority * (dir < 0 ? -1 : 1);
    if (dir === 0 && turnInput !== 0) {
      desiredSpeed = 0.25 * params.walkSpeed;
    }
  } else {
    // In-place animation treadmill speed (drives gait phase without drifting off turntable)
    desiredSpeed = 0;
    desiredTurn = 0;
  }

  state.currentSpeed = damp(state.currentSpeed, desiredSpeed, 5.5, dt);
  state.currentTurnRate = damp(state.currentTurnRate, desiredTurn, 6.0, dt);

  // Integrate world position and heading when driving
  state.heading += state.currentTurnRate * dt;
  state.posX += Math.sin(state.heading) * state.currentSpeed * dt;
  state.posZ += Math.cos(state.heading) * state.currentSpeed * dt;

  // Keep within a generous walkable radius of 18m so user never wanders into fog void
  const distSq = state.posX * state.posX + state.posZ * state.posZ;
  if (distSq > 18 * 18) {
    const dist = Math.sqrt(distSq);
    state.posX = (state.posX / dist) * 18;
    state.posZ = (state.posZ / dist) * 18;
  }

  // Advance gait/swim phase
  const strideDirection = state.currentSpeed < -0.05 ? -1 : 1;
  const speedScale = isDriving
    ? Math.max(0.55, Math.abs(state.currentSpeed) / 1.1)
    : params.walkSpeed;
  const gaitFreq =
    wWalk * 2.9 +
    wHighWalk * 4.5 +
    wSwim * 5.2 +
    wIdle * 0.4;
  state.gaitPhase += dt * gaitFreq * speedScale * strideDirection;

  // Bite one-shot envelope (duration = 1.35 seconds)
  let biteJawAdd = 0;
  let biteNeckLungePitch = 0;
  let biteHeadLift = 0;
  let biteRecoil = 0;

  if (state.biteTimer >= 0) {
    state.biteTimer += dt;
    const bt = state.biteTimer;
    if (bt < 0.28) {
      // Phase 1: Explosive anticipation gape (0 -> 42 deg) & slight head raise
      const p = bt / 0.28;
      const ease = 1 - Math.pow(1 - p, 3);
      biteJawAdd = ease * (42 * Math.PI) / 180;
      biteHeadLift = ease * -0.18;
      biteNeckLungePitch = ease * -0.08;
    } else if (bt < 0.40) {
      // Phase 2: Violent high-velocity jaw snap shut + forward lunge
      const p = (bt - 0.28) / 0.12;
      const snap = Math.pow(p, 2.2);
      biteJawAdd = (1 - snap) * (42 * Math.PI) / 180;
      biteHeadLift = (1 - snap) * -0.18 + snap * 0.06;
      biteNeckLungePitch = (1 - snap) * -0.08 + snap * 0.08;
    } else if (bt < 0.72) {
      // Phase 3: Impact recoil & jaw shudder
      const p = (bt - 0.40) / 0.32;
      const decay = Math.exp(-p * 6.5);
      biteRecoil = decay;
      biteJawAdd = Math.abs(Math.sin(p * Math.PI * 3)) * 0.03 * decay;
      biteHeadLift = 0.06 * (1 - p) - Math.sin(p * Math.PI * 2) * 0.04 * decay;
      biteNeckLungePitch = 0.08 * (1 - p);
    } else if (bt < 1.35) {
      // Phase 4: Smooth settle back to active pose
      const p = (bt - 0.72) / 0.63;
      const smooth = 1 - p;
      biteHeadLift = 0.01 * smooth;
    } else {
      state.biteTimer = -1;
    }
  }

  // Periodic natural eyelid + nictitating membrane blink (realistic snappy close, smooth open)
  let blinkAmount = 0;
  if (state.blinkTimer >= 0) {
    state.blinkTimer += dt;
    const dur = 0.34;
    if (state.blinkTimer < dur) {
      const p = state.blinkTimer / dur;
      // Snappy close (0..0.38) -> brief hold -> smooth open (0.38..1.0)
      blinkAmount = p < 0.38 ? Math.sin((p / 0.38) * Math.PI * 0.5) : Math.cos(((p - 0.38) / 0.62) * Math.PI * 0.5);
    } else {
      state.blinkTimer = -1;
      state.nextBlinkTime = t + 2.2 + (Math.sin(t * 7.13) * 0.5 + 0.5) * 2.4;
    }
  } else if (t >= state.nextBlinkTime) {
    state.blinkTimer = 0;
  }

  // Breathing & gular throat pulse
  const breathCycle = Math.sin(t * 1.45);
  const breathScale = 1.0 + breathCycle * 0.025 * (1 - wSwim * 0.4);
  const throatPulse = Math.sin(t * 2.2 + 0.5) * 0.016;

  // Pelvis elevation above Y=0:
  // Heavy barrel torso hangs ~0.34m below spine centerline, so pelvisY=0.52 gives ~0.16m realistic ground clearance matching reference!
  const strideBounce =
    Math.abs(Math.sin(state.gaitPhase)) * (wWalk * 0.02 + wHighWalk * 0.045);
  const swimBob = Math.sin(t * 2.1) * 0.015 * wSwim;
  const pelvisY =
    wIdle * 0.42 +
    wWalk * 0.33 +
    wHighWalk * 0.49 +
    wSwim * 0.35 +
    strideBounce +
    swimBob -
    biteRecoil * 0.025;

  // Pelvis lateral roll & forward pitch
  const pelvisRoll =
    Math.sin(state.gaitPhase) * (wWalk * 0.055 + wHighWalk * 0.075) +
    Math.sin(state.gaitPhase * 0.9) * wSwim * 0.04 -
    state.currentTurnRate * 0.05;
  const pelvisPitch =
    wHighWalk * 0.025 + wIdle * 0.012 + biteNeckLungePitch * 0.35;

  // Steering curvature: head leads into the turn, neck & torso curve progressively
  const steer = clamp(state.currentTurnRate, -1.2, 1.2);
  const swayMult = params.tailSwayIntensity;

  // Lateral S-wave along the anterior spine [midTorso, chest, neck1, neck2, neck3, head]
  const walkSpineAmp = (wWalk * 0.14 + wHighWalk * 0.095) * swayMult;
  const swimSpineAmp = wSwim * 0.16 * swayMult;
  const idleHeadScan = Math.sin(t * 0.42) * 0.045 * wIdle;

  const antMidTorsoYaw =
    Math.sin(state.gaitPhase + 0.3) * (walkSpineAmp * 0.55 + swimSpineAmp * 0.55) +
    steer * 0.09;
  const antChestYaw =
    Math.sin(state.gaitPhase + 0.85) * (walkSpineAmp * 0.72 + swimSpineAmp * 0.65) +
    steer * 0.12;
  const antNeck1Yaw =
    -Math.sin(state.gaitPhase + 0.95) * (walkSpineAmp * 0.45) +
    Math.sin(state.gaitPhase + 1.35) * (swimSpineAmp * 0.55) +
    steer * 0.14 +
    idleHeadScan * 0.25;
  const antNeck2Yaw =
    -Math.sin(state.gaitPhase + 1.15) * (walkSpineAmp * 0.4) +
    Math.sin(state.gaitPhase + 1.75) * (swimSpineAmp * 0.45) +
    steer * 0.15 +
    idleHeadScan * 0.35;
  const antNeck3Yaw =
    -Math.sin(state.gaitPhase + 1.35) * (walkSpineAmp * 0.3) +
    Math.sin(state.gaitPhase + 2.05) * (swimSpineAmp * 0.3) +
    steer * 0.14 +
    idleHeadScan * 0.4;
  // Head stabilizes gaze forward while walking/swimming, leads during turns
  const antHeadYaw =
    -Math.sin(state.gaitPhase + 1.55) * (walkSpineAmp * 0.25) -
    Math.sin(state.gaitPhase + 1.8) * (swimSpineAmp * 0.28) +
    steer * 0.18 +
    idleHeadScan * 0.45;

  const anteriorYaw: [number, number, number, number, number, number] = [
    antMidTorsoYaw,
    antChestYaw,
    antNeck1Yaw,
    antNeck2Yaw,
    antNeck3Yaw,
    antHeadYaw,
  ];

  // Subtle vertical posture curve along spine + bite head lunge
  const anteriorPitch: [number, number, number, number, number, number] = [
    -0.015 * wIdle - 0.01 * wHighWalk,
    -0.018 * wIdle - 0.012 * wHighWalk,
    -0.025 * wIdle + biteNeckLungePitch * 0.4,
    -0.012 * wIdle + biteNeckLungePitch * 0.35,
    0.01 * wIdle + biteNeckLungePitch * 0.25,
    0.02 * wIdle + 0.04 * wHighWalk + biteHeadLift,
  ];

  // Tail 11-segment travelling sinusoidal wave + iconic graceful idle curl matching reference!
  // In reference photo, tail sweeps gently backward and curls smoothly toward the crocodile's left (+Y yaw).
  const tailYaw: number[] = [];
  const tailPitch: number[] = [];

  for (let i = 0; i < TAIL_SEGMENT_COUNT; i++) {
    const frac = i / (TAIL_SEGMENT_COUNT - 1); // 0 at base -> 1 at tip
    const ampEnvelope = 0.38 + 0.85 * Math.pow(frac, 1.15);

    // Idle pose: natural reference sweeping curve toward viewer foreground-right (+X) + lazy tip drift
    const refIdleCurl = -(0.068 + 0.165 * Math.pow(frac, 1.3)) * wIdle;
    const idleDrift =
      Math.sin(t * 0.72 - i * 0.38) * 0.032 * ampEnvelope * swayMult * wIdle;

    // Walk / High Walk counter-balance wave
    const walkTailWave =
      Math.sin(state.gaitPhase - i * 0.46) *
      (wWalk * 0.115 + wHighWalk * 0.09) *
      ampEnvelope *
      swayMult;

    // Swim: powerful fluid horizontal S-wave propulsion propagating posteriorly!
    const swimTailWave =
      Math.sin(state.gaitPhase - i * 0.56) *
      wSwim *
      0.245 *
      ampEnvelope *
      swayMult;

    // Curved steering follow-through: tail lags opposite to turn
    const steerFollow = -steer * 0.085 * (0.5 + 0.5 * frac);

    tailYaw.push(refIdleCurl + idleDrift + walkTailWave + swimTailWave + steerFollow);

    // Tail pitch: arcs smoothly down from elevated pelvis toward floor, then levels out at the tip
    // Ensures tail tip rests naturally near ground in idle/walk without sinking below Y=0
    let basePitch = 0;
    if (i < 4) {
      basePitch = (wIdle * 0.075 + wHighWalk * 0.068 + wWalk * 0.032) * (1 - i * 0.12);
    } else if (i < 8) {
      basePitch = -(wIdle * 0.038 + wHighWalk * 0.032 + wWalk * 0.012);
    } else {
      basePitch = -(wIdle * 0.02 + wHighWalk * 0.015);
    }
    const swimPitch = Math.sin(t * 2.2 - i * 0.4) * 0.012 * wSwim;
    tailPitch.push(basePitch * (1 - wSwim * 0.85) + swimPitch);
  }

  // Final jaw angle combines manual slider + bite action (clamped cleanly to 0..45 degrees)
  const manualJawRad = (clamp(params.manualJawAngleDeg, 0, 45) * Math.PI) / 180;
  const jawAngleRad = clamp(Math.max(manualJawRad, biteJawAdd), 0, (45 * Math.PI) / 180);

  // Solve 3-joint sprawling IK for all 4 limbs
  const limbs: Record<LimbId, LimbJointAngles> = {
    FL: solveReptileLimbIK({
      limbId: 'FL',
      gaitPhase: state.gaitPhase,
      locomotionWeight: groundLocoWeight,
      highWalkFactor,
      swimTuck: limbTuck,
      pelvisY,
      steeringBias: steer,
    }),
    FR: solveReptileLimbIK({
      limbId: 'FR',
      gaitPhase: state.gaitPhase,
      locomotionWeight: groundLocoWeight,
      highWalkFactor,
      swimTuck: limbTuck,
      pelvisY,
      steeringBias: steer,
    }),
    BL: solveReptileLimbIK({
      limbId: 'BL',
      gaitPhase: state.gaitPhase,
      locomotionWeight: groundLocoWeight,
      highWalkFactor,
      swimTuck: limbTuck,
      pelvisY,
      steeringBias: steer,
    }),
    BR: solveReptileLimbIK({
      limbId: 'BR',
      gaitPhase: state.gaitPhase,
      locomotionWeight: groundLocoWeight,
      highWalkFactor,
      swimTuck: limbTuck,
      pelvisY,
      steeringBias: steer,
    }),
  };

  return {
    posX: state.posX,
    posZ: state.posZ,
    heading: state.heading,
    pelvisY,
    pelvisRoll,
    pelvisPitch,
    currentSpeed: state.currentSpeed,
    anteriorYaw,
    anteriorPitch,
    tailYaw,
    tailPitch,
    jawAngleRad,
    breathScale,
    throatPulse,
    blinkAmount,
    limbTuck,
    biteRecoil,
    limbs,
    resolvedMode: targetMode,
  };
}
