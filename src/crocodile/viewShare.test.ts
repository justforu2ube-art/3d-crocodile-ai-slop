import test from 'node:test';
import assert from 'node:assert/strict';
import {
  DEFAULT_CAMERA_POS,
  DEFAULT_CAMERA_TARGET,
  DEFAULT_VIEW_STATE,
  buildShareUrl,
  parseViewStateFromSearch,
  type SharedViewState,
} from './viewShare';

test('default landing state uses elevated overhead Studio view with 175% tail sway and 1.3x walk speed', () => {
  const state = parseViewStateFromSearch('');
  assert.equal(state.params.environment, 'studio');
  assert.equal(state.params.mode, 'idle');
  assert.equal(state.params.walkSpeed, 1.3);
  assert.equal(state.params.tailSwayIntensity, 1.75);
  assert.equal(state.collapsed, false);
  assert.deepEqual(state.cameraPosition, DEFAULT_CAMERA_POS);
  assert.deepEqual(state.cameraTarget, DEFAULT_CAMERA_TARGET);
  // Elevated overhead camera must sit well above the crocodile spine
  assert.ok(state.cameraPosition[1] >= 3.0, 'Default camera should be an elevated high-angle view');
});

test('buildShareUrl and parseViewStateFromSearch round-trip pose, toggles, and camera coordinates', () => {
  const custom: SharedViewState = {
    params: {
      mode: 'swim',
      walkSpeed: 1.9,
      manualJawAngleDeg: 34,
      tailSwayIntensity: 1.95,
      wetness: 0.72,
      environment: 'water',
    },
    wireframe: true,
    showSkeleton: true,
    autoRotate: true,
    followCamera: true,
    collapsed: true,
    cameraPosition: [2.45, 1.85, -3.1],
    cameraTarget: [0.15, 0.4, 0.25],
  };

  const url = buildShareUrl('https://example.github.io/nile-croc/?old=1', custom);
  const parsedUrl = new URL(url);
  assert.equal(parsedUrl.pathname, '/nile-croc/');

  const restored = parseViewStateFromSearch(parsedUrl.search);
  assert.deepEqual(restored, custom);
});

test('parseViewStateFromSearch safely clamps out-of-range values and ignores malformed inputs', () => {
  const restored = parseViewStateFromSearch(
    '?mode=invalidMode&env=mars&speed=99&jaw=-10&sway=5&wet=abc&cam=bad,1,2&tgt=0,-50,0'
  );
  assert.equal(restored.params.mode, DEFAULT_VIEW_STATE.params.mode);
  assert.equal(restored.params.environment, DEFAULT_VIEW_STATE.params.environment);
  assert.equal(restored.params.walkSpeed, 2.4);
  assert.equal(restored.params.manualJawAngleDeg, 0);
  assert.equal(restored.params.tailSwayIntensity, 2.0);
  assert.equal(restored.params.wetness, DEFAULT_VIEW_STATE.params.wetness);
  assert.deepEqual(restored.cameraPosition, DEFAULT_CAMERA_POS);
  assert.equal(restored.cameraTarget[1], -2);
});
