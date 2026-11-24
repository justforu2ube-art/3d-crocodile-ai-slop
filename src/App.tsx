import { useEffect, useRef, useState, useCallback } from 'react';
import * as THREE from 'three';
import { OrbitControls } from 'three/addons/controls/OrbitControls.js';
import { RoomEnvironment } from 'three/addons/environments/RoomEnvironment.js';
import {
  createInitialKinematicsState,
  stepKinematics,
  triggerBiteAction,
  type AnimationMode,
  type CrocControlParams,
  type EnvironmentMode,
  type InputKeys,
} from './crocodile/kinematics';
import { createCrocodileTextures } from './crocodile/textures';
import { buildNileCrocodile, type CrocodileRig } from './crocodile/crocodileBuilder';
import { createSceneEnvironments, type SceneEnvironmentRig } from './crocodile/environment';
import { exportCrocodileToGLB } from './crocodile/glbExporter';
import { ControlDock } from './components/ControlDock';

function detectSoftwareWebGL(): boolean {
  try {
    const probe = document.createElement('canvas');
    const gl = probe.getContext('webgl2') || probe.getContext('webgl');
    if (!gl) return true;
    const ext = gl.getExtension('WEBGL_debug_renderer_info');
    const param = ext ? ext.UNMASKED_RENDERER_WEBGL : gl.RENDERER;
    const gpu = String(gl.getParameter(param) || '');
    gl.getExtension('WEBGL_lose_context')?.loseContext();
    return /SwiftShader|llvmpipe|softpipe|Software|Basic Render/i.test(gpu);
  } catch {
    return true;
  }
}

export default function App() {
  const canvasMountRef = useRef<HTMLDivElement | null>(null);
  const crocRigRef = useRef<CrocodileRig | null>(null);
  const envRigRef = useRef<SceneEnvironmentRig | null>(null);
  const sceneRef = useRef<THREE.Scene | null>(null);
  const controlsRef = useRef<OrbitControls | null>(null);
  const kinStateRef = useRef(createInitialKinematicsState());

  // React UI state
  const [params, setParams] = useState<CrocControlParams>({
    mode: 'idle',
    walkSpeed: 1.0,
    manualJawAngleDeg: 0,
    tailSwayIntensity: 1.0,
    wetness: 0.14,
    environment: 'studio',
  });
  const [resolvedMode, setResolvedMode] = useState<Exclude<AnimationMode, 'autoDemo'>>('idle');
  const [wireframe, setWireframe] = useState(false);
  const [showSkeleton, setShowSkeleton] = useState(false);
  const [autoRotate, setAutoRotate] = useState(false);
  const [followCamera, setFollowCamera] = useState(false);
  const [collapsed, setCollapsed] = useState(false);
  const [exporting, setExporting] = useState(false);
  const [exportMessage, setExportMessage] = useState<string | null>(null);
  const [webglError, setWebglError] = useState<string | null>(null);

  // Mutable refs mirrored from React state so the WebGL render loop never re-mounts or recreates scene
  const paramsRef = useRef(params);
  paramsRef.current = params;
  const autoRotateRef = useRef(autoRotate);
  autoRotateRef.current = autoRotate;
  const followCameraRef = useRef(followCamera);
  followCameraRef.current = followCamera;

  const keysRef = useRef<InputKeys>({
    forward: false,
    backward: false,
    left: false,
    right: false,
    sprint: false,
  });

  // Sync environment mode cleanly
  useEffect(() => {
    if (envRigRef.current && sceneRef.current) {
      envRigRef.current.setMode(params.environment, sceneRef.current);
    }
  }, [params.environment]);

  // Sync wireframe toggle
  useEffect(() => {
    crocRigRef.current?.setWireframe(wireframe);
  }, [wireframe]);

  // Sync skeleton debug toggle
  useEffect(() => {
    crocRigRef.current?.setSkeletonVisible(showSkeleton);
  }, [showSkeleton]);

  // Keyboard controls (WASD/Arrows drive, Space bite, Shift sprint, H toggle HUD)
  useEffect(() => {
    const onKeyDown = (e: KeyboardEvent) => {
      if (e.target instanceof HTMLInputElement || e.target instanceof HTMLTextAreaElement) {
        return;
      }
      const code = e.code;
      if (code === 'KeyW' || code === 'ArrowUp') keysRef.current.forward = true;
      else if (code === 'KeyS' || code === 'ArrowDown') keysRef.current.backward = true;
      else if (code === 'KeyA' || code === 'ArrowLeft') keysRef.current.left = true;
      else if (code === 'KeyD' || code === 'ArrowRight') keysRef.current.right = true;
      else if (code === 'ShiftLeft' || code === 'ShiftRight') keysRef.current.sprint = true;
      else if (code === 'Space') {
        e.preventDefault();
        triggerBiteAction(kinStateRef.current);
      } else if (code === 'KeyH') {
        setCollapsed((c) => !c);
      }
    };

    const onKeyUp = (e: KeyboardEvent) => {
      const code = e.code;
      if (code === 'KeyW' || code === 'ArrowUp') keysRef.current.forward = false;
      else if (code === 'KeyS' || code === 'ArrowDown') keysRef.current.backward = false;
      else if (code === 'KeyA' || code === 'ArrowLeft') keysRef.current.left = false;
      else if (code === 'KeyD' || code === 'ArrowRight') keysRef.current.right = false;
      else if (code === 'ShiftLeft' || code === 'ShiftRight') keysRef.current.sprint = false;
    };

    const onBlur = () => {
      keysRef.current = { forward: false, backward: false, left: false, right: false, sprint: false };
    };

    window.addEventListener('keydown', onKeyDown);
    window.addEventListener('keyup', onKeyUp);
    window.addEventListener('blur', onBlur);
    return () => {
      window.removeEventListener('keydown', onKeyDown);
      window.removeEventListener('keyup', onKeyUp);
      window.removeEventListener('blur', onBlur);
    };
  }, []);

  // Mount Three.js WebGL scene ONCE (never tears down on React UI state updates)
  useEffect(() => {
    const container = canvasMountRef.current;
    if (!container) return;

    const isSoftwareGl = detectSoftwareWebGL();
    let renderer: THREE.WebGLRenderer;
    try {
      renderer = new THREE.WebGLRenderer({
        antialias: !isSoftwareGl,
        powerPreference: 'high-performance',
      });
    } catch (err) {
      setWebglError(err instanceof Error ? err.message : 'Unable to initialize WebGL renderer.');
      return;
    }

    renderer.outputColorSpace = THREE.SRGBColorSpace;
    renderer.toneMapping = THREE.ACESFilmicToneMapping;
    renderer.toneMappingExposure = 1.06;
    renderer.setPixelRatio(isSoftwareGl ? 1 : Math.min(window.devicePixelRatio, 2));
    renderer.shadowMap.enabled = !isSoftwareGl;
    renderer.shadowMap.type = THREE.PCFSoftShadowMap;
    renderer.setSize(container.clientWidth || window.innerWidth, container.clientHeight || window.innerHeight);

    container.appendChild(renderer.domElement);

    const handleContextLost = (e: Event) => {
      e.preventDefault();
      setWebglError('WebGL rendering context was lost. Reloading or switching tabs will restore it.');
    };
    const handleContextRestored = () => {
      setWebglError(null);
    };
    renderer.domElement.addEventListener('webglcontextlost', handleContextLost);
    renderer.domElement.addEventListener('webglcontextrestored', handleContextRestored);

    // Scene & Studio Environment Map (PMREM RoomEnvironment for physical specular reflections)
    const scene = new THREE.Scene();
    sceneRef.current = scene;

    const pmremGenerator = new THREE.PMREMGenerator(renderer);
    const roomEnv = new RoomEnvironment();
    const envTarget = pmremGenerator.fromScene(roomEnv, 0.04);
    scene.environment = envTarget.texture;
    scene.environmentIntensity = 0.68;
    roomEnv.dispose();
    pmremGenerator.dispose();

    // Camera framed to showcase the full crocodile (snout tip to curving serrated tail) matching the reference studio angle!
    const width = container.clientWidth || window.innerWidth;
    const height = container.clientHeight || window.innerHeight;
    const camera = new THREE.PerspectiveCamera(34, width / height, 0.08, 120);
    camera.position.set(4.35, 1.52, 3.35);

    const controls = new OrbitControls(camera, renderer.domElement);
    controls.target.set(0.05, 0.34, 0.16);
    controls.enableDamping = !isSoftwareGl;
    controls.dampingFactor = 0.06;
    controls.maxPolarAngle = Math.PI * 0.492;
    controls.minDistance = 1.1;
    controls.maxDistance = 16;
    controls.autoRotateSpeed = 1.1;
    controls.update();
    controlsRef.current = controls;

    // Build Procedural 1024px PBR Textures, Nile Crocodile Rig & Environments
    const maxAniso = renderer.capabilities.getMaxAnisotropy();
    const textures = createCrocodileTextures(maxAniso);
    const crocRig = buildNileCrocodile(textures);
    crocRigRef.current = crocRig;
    scene.add(crocRig.root);

    const envRig = createSceneEnvironments(scene, textures, isSoftwareGl, maxAniso);
    envRigRef.current = envRig;

    // Apply initial hero pose immediately before first render
    const initialPose = stepKinematics(kinStateRef.current, paramsRef.current, keysRef.current, 0.016);
    crocRig.applyPose(initialPose, paramsRef.current.wetness, paramsRef.current.environment === 'studio');

    // Pre-warm shaders on GPU (skip on software WebGL)
    if (!isSoftwareGl) {
      renderer.compile(scene, camera);
    }

    // Resize observer
    const onResize = () => {
      const w = container.clientWidth || window.innerWidth;
      const h = container.clientHeight || window.innerHeight;
      camera.aspect = w / h;
      camera.updateProjectionMatrix();
      renderer.setSize(w, h);
    };
    window.addEventListener('resize', onResize);

    // Pre-allocated scratch vectors for Follow Camera & Bite Recoil (zero per-frame GC)
    const followDesiredPos = new THREE.Vector3();
    const followDesiredTarget = new THREE.Vector3();
    let lastTime = performance.now();
    let hudSyncAccum = 0;
    let lastReportedMode: Exclude<AnimationMode, 'autoDemo'> = 'idle';
    let disposed = false;
    let frameId = 0;

    // Software-WebGL GPU fence pacing (never starves headless Chromium / software rendering)
    const afterDraw = (cb: () => void) => {
      const gl = renderer.getContext() as WebGL2RenderingContext;
      const sync = gl.fenceSync ? gl.fenceSync(gl.SYNC_GPU_COMMANDS_COMPLETE, 0) : null;
      gl.flush();
      const poll = () => {
        if (disposed) {
          if (sync) gl.deleteSync(sync);
          return;
        }
        const status = sync ? gl.getSyncParameter(sync, gl.SYNC_STATUS) : gl.SIGNALED;
        if (status === gl.UNSIGNALED) {
          frameId = window.setTimeout(poll, 5);
        } else {
          if (sync) gl.deleteSync(sync);
          frameId = window.setTimeout(cb, 80);
        }
      };
      frameId = window.setTimeout(poll, 5);
    };

    const tick = () => {
      if (disposed) return;

      const now = performance.now();
      const dt = Math.min(isSoftwareGl ? 0.14 : 0.06, Math.max(0.001, (now - lastTime) / 1000));
      lastTime = now;

      const p = paramsRef.current;
      const pose = stepKinematics(kinStateRef.current, p, keysRef.current, dt);

      // Update crocodile joints, foot ground-lock, and skin wetness
      crocRig.applyPose(pose, p.wetness, p.environment === 'studio');

      // Update lighting shadow frustum & riverbank animated water plane
      envRig.update(kinStateRef.current.elapsedTime, pose.posX, pose.posZ, pose.limbTuck);

      // OrbitControls auto-rotate & Follow Camera tracking
      controls.autoRotate = autoRotateRef.current && !followCameraRef.current && !isSoftwareGl;

      if (followCameraRef.current) {
        const behindDist = 4.6;
        const heightOffset = 1.65;
        const sinH = Math.sin(pose.heading);
        const cosH = Math.cos(pose.heading);
        followDesiredPos.set(
          pose.posX - sinH * behindDist + cosH * 0.9,
          heightOffset,
          pose.posZ - cosH * behindDist - sinH * 0.9
        );
        followDesiredTarget.set(
          pose.posX + sinH * 0.45,
          pose.pelvisY + 0.05,
          pose.posZ + cosH * 0.45
        );
        const camAlpha = 1 - Math.exp(-4.5 * dt);
        camera.position.lerp(followDesiredPos, camAlpha);
        controls.target.lerp(followDesiredTarget, camAlpha);
      } else if (Math.abs(pose.currentSpeed) > 0.05) {
        // Gently track crocodile center when driving even if Follow Cam is off
        const trackAlpha = 1 - Math.exp(-3.5 * dt);
        controls.target.x += (pose.posX - controls.target.x) * trackAlpha;
        controls.target.z += (pose.posZ - controls.target.z) * trackAlpha;
      }

      // Subtle bite recoil camera impulse
      if (pose.biteRecoil > 0.02) {
        const shake = pose.biteRecoil * pose.biteRecoil * 0.028;
        camera.position.y += Math.sin(kinStateRef.current.elapsedTime * 55) * shake;
      }

      controls.update();
      renderer.render(scene, camera);

      // Throttled HUD state sync (<= 5 Hz, only when resolved mode actually changes)
      hudSyncAccum += dt;
      if (hudSyncAccum >= 0.2) {
        hudSyncAccum = 0;
        if (pose.resolvedMode !== lastReportedMode) {
          lastReportedMode = pose.resolvedMode;
          setResolvedMode(pose.resolvedMode);
        }
      }

      if (isSoftwareGl) {
        afterDraw(tick);
      } else {
        frameId = requestAnimationFrame(tick);
      }
    };

    tick();

    return () => {
      disposed = true;
      cancelAnimationFrame(frameId);
      clearTimeout(frameId);
      window.removeEventListener('resize', onResize);
      renderer.domElement.removeEventListener('webglcontextlost', handleContextLost);
      renderer.domElement.removeEventListener('webglcontextrestored', handleContextRestored);
      controls.dispose();
      crocRig.dispose();
      envRig.dispose();
      textures.dispose();
      envTarget.dispose();
      renderer.dispose();
      if (renderer.domElement.parentNode === container) {
        container.removeChild(renderer.domElement);
      }
    };
  }, []);

  const handleModeChange = useCallback((mode: AnimationMode) => {
    setParams((prev) => ({ ...prev, mode }));
  }, []);

  const handleTriggerBite = useCallback(() => {
    triggerBiteAction(kinStateRef.current);
  }, []);

  const handleParamChange = useCallback(
    <K extends keyof CrocControlParams>(key: K, value: CrocControlParams[K]) => {
      setParams((prev) => ({ ...prev, [key]: value }));
    },
    []
  );

  const handleEnvironmentChange = useCallback((environment: EnvironmentMode) => {
    setParams((prev) => ({
      ...prev,
      environment,
      // Automatically bump wetness slightly in water mode for realistic wet hide sheen
      wetness: environment === 'water' ? Math.max(prev.wetness, 0.58) : prev.wetness,
    }));
  }, []);

  const handleExportGLB = useCallback(async () => {
    if (!crocRigRef.current || exporting) return;
    setExporting(true);
    setExportMessage(null);
    try {
      const result = await exportCrocodileToGLB(crocRigRef.current.modelRoot);
      const mb = (result.byteLength / (1024 * 1024)).toFixed(2);
      setExportMessage(`Exported ${result.filename} (${mb} MB, ${result.meshCount} meshes)`);
      window.setTimeout(() => setExportMessage(null), 5000);
    } catch (err) {
      setExportMessage(err instanceof Error ? `Export failed: ${err.message}` : 'Export failed');
      window.setTimeout(() => setExportMessage(null), 5000);
    } finally {
      setExporting(false);
    }
  }, [exporting]);

  const isLightStudio = params.environment === 'studio';

  return (
    <main
      className="relative h-screen w-screen overflow-hidden select-none"
      data-state={resolvedMode}
      data-env={params.environment}
      data-jaw={Math.round(params.manualJawAngleDeg)}
      data-wireframe={wireframe ? 'on' : 'off'}
      data-skeleton={showSkeleton ? 'on' : 'off'}
    >
      {/* Dedicated Isolated WebGL Canvas Container */}
      <div
        ref={canvasMountRef}
        data-testid="crocodile-canvas"
        className="h-full w-full cursor-grab active:cursor-grabbing"
      />

      {/* Graceful WebGL Fallback Notice */}
      {webglError && (
        <div className="fixed inset-0 z-30 flex items-center justify-center bg-[#cfd1d3] p-6">
          <div className="max-w-md rounded-xl border border-stone-300 bg-white p-6 text-stone-900 shadow-lg">
            <h2 className="text-lg font-semibold">WebGL Viewport Unavailable</h2>
            <p className="mt-2 text-sm text-stone-600">{webglError}</p>
          </div>
        </div>
      )}

      {/* Minimal Top-Left Specimen Title (Zero Occlusion of Central 3D Viewport) */}
      <header className="pointer-events-none fixed top-4 left-5 z-10 sm:top-5 sm:left-6">
        <h1
          className={`font-serif-specimen text-2xl font-semibold tracking-tight sm:text-3xl ${
            isLightStudio ? 'text-stone-900' : 'text-stone-50 drop-shadow-sm'
          }`}
        >
          Nile Crocodile
        </h1>
        <p
          className={`mt-0.5 text-xs tracking-wide ${
            isLightStudio ? 'text-stone-600' : 'text-stone-200/90'
          }`}
        >
          <span className="italic">Crocodylus niloticus</span>
          <span aria-hidden="true" className="mx-1.5">·</span>
          <span>Procedural Anatomy &amp; Kinematics</span>
        </p>
      </header>

      {/* Collapsible Unified Bottom Control Dock */}
      <ControlDock
        params={params}
        resolvedMode={resolvedMode}
        wireframe={wireframe}
        showSkeleton={showSkeleton}
        autoRotate={autoRotate}
        followCamera={followCamera}
        collapsed={collapsed}
        exporting={exporting}
        exportMessage={exportMessage}
        onModeChange={handleModeChange}
        onTriggerBite={handleTriggerBite}
        onParamChange={handleParamChange}
        onEnvironmentChange={handleEnvironmentChange}
        onToggleWireframe={() => setWireframe((w) => !w)}
        onToggleSkeleton={() => setShowSkeleton((s) => !s)}
        onToggleAutoRotate={() => setAutoRotate((r) => !r)}
        onToggleFollowCamera={() => setFollowCamera((f) => !f)}
        onToggleCollapsed={() => setCollapsed((c) => !c)}
        onExportGLB={handleExportGLB}
      />
    </main>
  );
}
