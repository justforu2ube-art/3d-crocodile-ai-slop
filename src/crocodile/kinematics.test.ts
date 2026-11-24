import test from 'node:test';
import assert from 'node:assert/strict';
import {
  createInitialKinematicsState,
  stepKinematics,
  triggerBiteAction,
  solveReptileLimbIK,
  TAIL_SEGMENT_COUNT,
  type CrocControlParams,
  type InputKeys,
} from './kinematics';
import {
  buildLoftedGeometry,
  buildOsteodermScuteGeometry,
  buildTailFinScuteGeometry,
  buildConicalToothGeometry,
  buildCurvedClawGeometry,
  buildReptileToeGeometry,
  validateBufferGeometry,
} from './geometryUtils';

const neutralKeys: InputKeys = {
  forward: false,
  backward: false,
  left: false,
  right: false,
  sprint: false,
};

const baseParams: CrocControlParams = {
  mode: 'idle',
  walkSpeed: 1.0,
  manualJawAngleDeg: 0,
  tailSwayIntensity: 1.0,
  wetness: 0.2,
  environment: 'studio',
};

test('idle state maintains alert posture and finite transforms without input', () => {
  const state = createInitialKinematicsState();
  let pose = stepKinematics(state, baseParams, neutralKeys, 1 / 60);
  for (let i = 0; i < 180; i++) {
    pose = stepKinematics(state, baseParams, neutralKeys, 1 / 60);
  }
  assert.equal(pose.resolvedMode, 'idle');
  assert.ok(pose.pelvisY > 0.45 && pose.pelvisY < 0.62, `Expected alert pelvisY, got ${pose.pelvisY}`);
  assert.equal(pose.tailYaw.length, TAIL_SEGMENT_COUNT);
  assert.equal(pose.tailPitch.length, TAIL_SEGMENT_COUNT);
  assert.ok(pose.tailYaw.every((v) => Number.isFinite(v)));
  assert.ok(pose.anteriorYaw.every((v) => Number.isFinite(v)));
  assert.equal(pose.limbTuck, 0);
});

test('diagonal tetrapod gait pairs FL+BR in phase and FR+BL in opposite phase', () => {
  const fl = solveReptileLimbIK({
    limbId: 'FL',
    gaitPhase: 0.4,
    locomotionWeight: 1,
    highWalkFactor: 0,
    swimTuck: 0,
    pelvisY: 0.34,
    steeringBias: 0,
  });
  const br = solveReptileLimbIK({
    limbId: 'BR',
    gaitPhase: 0.4,
    locomotionWeight: 1,
    highWalkFactor: 0,
    swimTuck: 0,
    pelvisY: 0.34,
    steeringBias: 0,
  });
  const fr = solveReptileLimbIK({
    limbId: 'FR',
    gaitPhase: 0.4,
    locomotionWeight: 1,
    highWalkFactor: 0,
    swimTuck: 0,
    pelvisY: 0.34,
    steeringBias: 0,
  });

  // At phase 0.4 rad (~6% of cycle), FL and BR are in stance (groundContact = 1, lift = 0)
  assert.equal(fl.groundContact, 1);
  assert.equal(br.groundContact, 1);
  assert.equal(fl.footLiftY, 0);
  assert.equal(br.footLiftY, 0);

  // At phase 0.4 + PI (~56% of cycle), FR should be near end of stance or entering swing; check at 1.2 rad (~19% vs 69%)
  const flMidStance = solveReptileLimbIK({
    limbId: 'FL',
    gaitPhase: 1.2,
    locomotionWeight: 1,
    highWalkFactor: 0,
    swimTuck: 0,
    pelvisY: 0.34,
    steeringBias: 0,
  });
  const frSwing = solveReptileLimbIK({
    limbId: 'FR',
    gaitPhase: 1.2,
    locomotionWeight: 1,
    highWalkFactor: 0,
    swimTuck: 0,
    pelvisY: 0.34,
    steeringBias: 0,
  });
  assert.equal(flMidStance.groundContact, 1);
  assert.equal(frSwing.groundContact, 0);
  assert.ok(frSwing.footLiftY > 0.02, 'Swing foot should lift cleanly off ground');
  assert.ok(Number.isFinite(fr.hipYaw));
});

test('swim mode smoothly tucks limbs to limbTuck -> 1.0 and amplifies tail S-wave', () => {
  const state = createInitialKinematicsState();
  const swimParams: CrocControlParams = { ...baseParams, mode: 'swim' };
  let maxTailTipYaw = 0;
  let pose = stepKinematics(state, swimParams, neutralKeys, 0.02);
  for (let i = 0; i < 160; i++) {
    pose = stepKinematics(state, swimParams, neutralKeys, 0.02);
    maxTailTipYaw = Math.max(maxTailTipYaw, Math.abs(pose.tailYaw[TAIL_SEGMENT_COUNT - 1]));
  }
  assert.ok(pose.limbTuck > 0.98, `Expected limbTuck near 1, got ${pose.limbTuck}`);
  assert.ok(maxTailTipYaw > 0.22, `Expected strong swim tail wave, got ${maxTailTipYaw}`);
});

test('bite trigger opens jaw to ~42 deg then violently snaps shut with recoil', () => {
  const state = createInitialKinematicsState();
  triggerBiteAction(state);
  let peakJawDeg = 0;
  let peakRecoil = 0;

  for (let i = 0; i < 90; i++) {
    const pose = stepKinematics(state, baseParams, neutralKeys, 0.02);
    peakJawDeg = Math.max(peakJawDeg, (pose.jawAngleRad * 180) / Math.PI);
    peakRecoil = Math.max(peakRecoil, pose.biteRecoil);
  }

  assert.ok(peakJawDeg >= 38 && peakJawDeg <= 45, `Expected peak bite jaw ~42 deg, got ${peakJawDeg}`);
  assert.ok(peakRecoil > 0.5, `Expected noticeable bite recoil, got ${peakRecoil}`);
  assert.equal(state.biteTimer, -1, 'Bite timer should settle cleanly back to inactive');
});

test('WASD forward + turn drives crocodile along curved heading with spine steering', () => {
  const state = createInitialKinematicsState();
  const driveKeys: InputKeys = {
    forward: true,
    backward: false,
    left: true,
    right: false,
    sprint: true,
  };
  let pose = stepKinematics(state, baseParams, driveKeys, 0.02);
  for (let i = 0; i < 60; i++) {
    pose = stepKinematics(state, baseParams, driveKeys, 0.02);
  }
  assert.equal(pose.resolvedMode, 'highWalk');
  assert.ok(pose.heading > 0.5, `Expected positive left heading, got ${pose.heading}`);
  assert.ok(Math.hypot(pose.posX, pose.posZ) > 1.2, 'Crocodile should move forward when holding W');
  assert.ok(pose.anteriorYaw[5] > 0.05, 'Head should lead into the left turn');
});

test('all procedural BufferGeometry builders produce valid meshes with zero NaNs', () => {
  const loftForward = buildLoftedGeometry([
    { z: 0, yOffset: 0, rx: 0.25, ry: 0.16, uvV: 0.4 },
    { z: 0.4, yOffset: 0.0, rx: 0.30, ry: 0.19, uvV: 0.5 },
    { z: 0.8, yOffset: 0.0, rx: 0.22, ry: 0.15, uvV: 0.6 },
  ], { bumpAmp: 0 });
  const loftBackward = buildLoftedGeometry([
    { z: 0.0, yOffset: 0, rx: 0.25, ry: 0.18, uvV: 0.4 },
    { z: -0.4, yOffset: 0, rx: 0.20, ry: 0.15, uvV: 0.2 },
    { z: -0.8, yOffset: 0, rx: 0.12, ry: 0.10, uvV: 0.0 },
  ], { bumpAmp: 0 });
  const scute = buildOsteodermScuteGeometry(0.14, 0.09, 0.045);
  const fin = buildTailFinScuteGeometry(0.16, 0.04, 0.11);
  const tooth = buildConicalToothGeometry(0.014, 0.055);
  const claw = buildCurvedClawGeometry(0.015, 0.065);
  const toe = buildReptileToeGeometry(0.14, 0.028, 0.016);

  // Verify outward-facing radial normals on mid-ring vertices for both forward (+Z) and backward (-Z) lofts!
  for (const [label, g] of [['loftForward', loftForward], ['loftBackward', loftBackward]] as const) {
    const pos = g.attributes.position;
    const norm = g.attributes.normal;
    // Middle ring starts at radialSegments + 1 = 25
    let outwardCount = 0;
    for (let i = 25; i < 50; i++) {
      const dot = pos.getX(i) * norm.getX(i) + pos.getY(i) * norm.getY(i);
      if (dot > 0) outwardCount++;
    }
    assert.equal(outwardCount, 25, `${label} must have 100% outward-facing radial normals`);
  }

  for (const [name, geo] of [
    ['loftForward', loftForward],
    ['loftBackward', loftBackward],
    ['scute', scute],
    ['fin', fin],
    ['tooth', tooth],
    ['claw', claw],
    ['toe', toe],
  ] as const) {
    const res = validateBufferGeometry(geo);
    assert.equal(res.valid, true, `${name} invalid: ${res.error}`);
    assert.ok(res.triangleCount > 8, `${name} should have real sculpted geometry`);
    geo.dispose();
  }
});
