import * as THREE from 'three';
import type { EnvironmentMode } from './kinematics';
import type { CrocodileTextureSet } from './textures';

export interface SceneEnvironmentRig {
  studioGroup: THREE.Group;
  waterGroup: THREE.Group;
  keyLight: THREE.DirectionalLight;
  fillHemi: THREE.HemisphereLight;
  rimLight: THREE.DirectionalLight;
  setMode: (mode: EnvironmentMode, scene: THREE.Scene) => void;
  update: (elapsed: number, crocX: number, crocZ: number, swimWeight: number) => void;
  dispose: () => void;
}

/** Generates a procedural sandy-silt Nile riverbed texture on an offscreen canvas */
function createRiverbedTexture(maxAnisotropy: number): THREE.CanvasTexture {
  const size = 512;
  const canvas = document.createElement('canvas');
  canvas.width = size;
  canvas.height = size;
  const ctx = canvas.getContext('2d')!;
  const img = ctx.createImageData(size, size);
  for (let y = 0; y < size; y++) {
    for (let x = 0; x < size; x++) {
      const n1 = Math.sin(x * 0.06 + Math.cos(y * 0.05) * 2.0) * 0.5 + 0.5;
      const n2 = Math.sin(x * 0.19 - y * 0.17) * 0.5 + 0.5;
      const speck = (Math.sin(x * 127.1 + y * 311.7) * 43758.5453) % 1;
      const s = speck < 0 ? speck + 1 : speck;
      const mixV = n1 * 0.5 + n2 * 0.3 + s * 0.2;
      const idx = (y * size + x) * 4;
      img.data[idx] = Math.round(74 + mixV * 36);
      img.data[idx + 1] = Math.round(68 + mixV * 32);
      img.data[idx + 2] = Math.round(52 + mixV * 22);
      img.data[idx + 3] = 255;
    }
  }
  ctx.putImageData(img, 0, 0);
  const tex = new THREE.CanvasTexture(canvas);
  tex.colorSpace = THREE.SRGBColorSpace;
  tex.wrapS = THREE.RepeatWrapping;
  tex.wrapT = THREE.RepeatWrapping;
  tex.repeat.set(16, 16);
  tex.anisotropy = Math.max(1, Math.min(16, maxAnisotropy));
  return tex;
}

/**
 * Creates the balanced 3-point lighting rig + toggleable Clean Studio Turntable
 * (matching the attached light-grey seamless studio reference!) and Shallow Riverbank Water environment.
 */
export function createSceneEnvironments(
  scene: THREE.Scene,
  textures: CrocodileTextureSet,
  isSoftwareGl: boolean,
  maxAnisotropy: number
): SceneEnvironmentRig {
  // --- THREE-POINT PHYSICAL LIGHTING RIG ---
  // Warm directional key sunlight with tight PCFSoftShadowMap frustum
  const keyLight = new THREE.DirectionalLight(0xfff7ea, 2.35);
  keyLight.position.set(5.5, 9.5, 6.0);
  keyLight.castShadow = !isSoftwareGl;
  keyLight.shadow.mapSize.set(2048, 2048);
  keyLight.shadow.camera.near = 1;
  keyLight.shadow.camera.far = 28;
  const shadowExtent = 5.5;
  keyLight.shadow.camera.left = -shadowExtent;
  keyLight.shadow.camera.right = shadowExtent;
  keyLight.shadow.camera.top = shadowExtent;
  keyLight.shadow.camera.bottom = -shadowExtent;
  keyLight.shadow.bias = -0.0004;
  keyLight.shadow.normalBias = 0.025;
  keyLight.shadow.radius = 3.5;
  scene.add(keyLight);
  scene.add(keyLight.target);

  // Cool sky / warm ground bounce hemisphere fill (zero flat ambient wash)
  const fillHemi = new THREE.HemisphereLight(0xdde8f2, 0x9b968a, 1.05);
  scene.add(fillHemi);

  // Crisp subtle back-right rim light to carve dorsal osteoderms & tail fin silhouettes
  const rimLight = new THREE.DirectionalLight(0xd9e8ff, 0.95);
  rimLight.position.set(-6.5, 4.2, -5.5);
  scene.add(rimLight);

  // =========================================================================
  // 1. CLEAN STUDIO TURNTABLE (Light Seamless Cyclorama matching reference!)
  // =========================================================================
  const studioGroup = new THREE.Group();
  studioGroup.name = 'StudioEnvironment';

  // Studio matte cyclorama floor calibrated for ACES Filmic to match soft neutral reference grey
  const studioFloorGeo = new THREE.CircleGeometry(52, 64);
  studioFloorGeo.rotateX(-Math.PI / 2);
  const studioFloorMat = new THREE.MeshStandardMaterial({
    color: new THREE.Color('#a4a7aa'),
    roughness: 0.96,
    metalness: 0.0,
  });
  const studioFloor = new THREE.Mesh(studioFloorGeo, studioFloorMat);
  studioFloor.receiveShadow = true;
  studioGroup.add(studioFloor);

  // Subtle architectural turntable ring decal on floor (offset +0.003 to avoid Z-fighting)
  const ringGeo = new THREE.RingGeometry(2.95, 2.97, 96);
  ringGeo.rotateX(-Math.PI / 2);
  const ringMat = new THREE.MeshBasicMaterial({
    color: 0x9ca0a4,
    transparent: true,
    opacity: 0.30,
    polygonOffset: true,
    polygonOffsetFactor: -1,
    polygonOffsetUnits: -1,
  });
  const studioRing = new THREE.Mesh(ringGeo, ringMat);
  studioRing.position.y = 0.003;
  studioGroup.add(studioRing);

  scene.add(studioGroup);

  // =========================================================================
  // 2. SHALLOW RIVERBANK WATER ENVIRONMENT
  // =========================================================================
  const waterGroup = new THREE.Group();
  waterGroup.name = 'RiverbankWaterEnvironment';
  waterGroup.visible = false;

  const riverbedTex = createRiverbedTexture(maxAnisotropy);
  const riverbedGeo = new THREE.PlaneGeometry(76, 76, 48, 48);
  riverbedGeo.rotateX(-Math.PI / 2);
  // Gentle natural undulating riverbank terrain: shallow channel in center, rising sandbars around edges
  const bedPos = riverbedGeo.attributes.position as THREE.BufferAttribute;
  for (let i = 0; i < bedPos.count; i++) {
    const x = bedPos.getX(i);
    const z = bedPos.getZ(i);
    const dist = Math.hypot(x, z);
    const bankRise = Math.max(0, (dist - 14) * 0.075);
    const ripples = Math.sin(x * 0.35) * Math.cos(z * 0.35) * 0.05;
    bedPos.setY(i, -0.22 + bankRise + ripples);
  }
  riverbedGeo.computeVertexNormals();

  const riverbedMat = new THREE.MeshStandardMaterial({
    color: 0xffffff,
    map: riverbedTex,
    roughness: 0.94,
    metalness: 0.02,
  });
  const riverbedMesh = new THREE.Mesh(riverbedGeo, riverbedMat);
  riverbedMesh.receiveShadow = true;
  waterGroup.add(riverbedMesh);

  // Instanced smooth river stones & boulders along perimeter riverbank
  const rockGeo = new THREE.DodecahedronGeometry(0.45, 2);
  const rockPos = rockGeo.attributes.position as THREE.BufferAttribute;
  for (let i = 0; i < rockPos.count; i++) {
    const vx = rockPos.getX(i);
    const vy = rockPos.getY(i);
    const vz = rockPos.getZ(i);
    const d = 0.88 + Math.sin(vx * 5 + vz * 4) * 0.14;
    rockPos.setXYZ(i, vx * d, vy * d * 0.58, vz * d);
  }
  rockGeo.computeVertexNormals();
  const rockMat = new THREE.MeshStandardMaterial({
    color: new THREE.Color('#6a675c'),
    roughness: 0.85,
    metalness: 0.04,
  });
  const rockCount = 44;
  const rockInstances = new THREE.InstancedMesh(rockGeo, rockMat, rockCount);
  rockInstances.castShadow = !isSoftwareGl;
  rockInstances.receiveShadow = true;
  const dummy = new THREE.Object3D();
  const rockColor = new THREE.Color();
  for (let i = 0; i < rockCount; i++) {
    const angle = (i / rockCount) * Math.PI * 2 + Math.sin(i * 3.7) * 0.25;
    const radius = 11.5 + (Math.sin(i * 9.1) * 0.5 + 0.5) * 14.0;
    const sx = 0.7 + ((i * 37) % 100) / 70;
    const sy = 0.5 + ((i * 53) % 100) / 110;
    const sz = 0.7 + ((i * 71) % 100) / 75;
    dummy.position.set(Math.cos(angle) * radius, -0.08 + Math.max(0, (radius - 14) * 0.06), Math.sin(angle) * radius);
    dummy.rotation.set(0, angle * 2.1, 0);
    dummy.scale.set(sx, sy, sz);
    dummy.updateMatrix();
    rockInstances.setMatrixAt(i, dummy.matrix);
    const shade = 0.82 + ((i * 29) % 100) / 280;
    rockColor.setRGB(shade * 0.46, shade * 0.44, shade * 0.39);
    rockInstances.setColorAt(i, rockColor);
  }
  rockInstances.instanceMatrix.needsUpdate = true;
  if (rockInstances.instanceColor) rockInstances.instanceColor.needsUpdate = true;
  rockInstances.computeBoundingSphere();
  waterGroup.add(rockInstances);

  // Instanced shoreline papyrus/reeds framing the outer riverbank
  const reedGeo = new THREE.CylinderGeometry(0.012, 0.032, 1.85, 6);
  reedGeo.translate(0, 0.92, 0);
  const reedMat = new THREE.MeshStandardMaterial({
    color: new THREE.Color('#5a6b42'),
    roughness: 0.78,
    metalness: 0.0,
  });
  const reedCount = 96;
  const reedInstances = new THREE.InstancedMesh(reedGeo, reedMat, reedCount);
  const reedColor = new THREE.Color();
  for (let i = 0; i < reedCount; i++) {
    const cluster = Math.floor(i / 6);
    const cAngle = (cluster / 16) * Math.PI * 2 + 0.3;
    const cRad = 15.5 + (cluster % 3) * 2.4;
    const jitterX = Math.sin(i * 12.7) * 0.75;
    const jitterZ = Math.cos(i * 17.3) * 0.75;
    dummy.position.set(
      Math.cos(cAngle) * cRad + jitterX,
      -0.1,
      Math.sin(cAngle) * cRad + jitterZ
    );
    dummy.rotation.set(Math.sin(i * 4.1) * 0.14, i * 1.1, Math.cos(i * 3.3) * 0.14);
    const hScale = 0.72 + ((i * 43) % 100) / 145;
    dummy.scale.set(1, hScale, 1);
    dummy.updateMatrix();
    reedInstances.setMatrixAt(i, dummy.matrix);
    reedColor.setRGB(0.32 + (i % 7) * 0.015, 0.42 + (i % 5) * 0.02, 0.23 + (i % 4) * 0.01);
    reedInstances.setColorAt(i, reedColor);
  }
  reedInstances.instanceMatrix.needsUpdate = true;
  if (reedInstances.instanceColor) reedInstances.instanceColor.needsUpdate = true;
  reedInstances.computeBoundingSphere();
  waterGroup.add(reedInstances);

  // Animated procedural shallow water plane using PBR MeshPhysicalMaterial + vertex wave shader via onBeforeCompile!
  const waterUniforms = {
    uTime: { value: 0 },
    uCrocPos: { value: new THREE.Vector2(0, 0) },
  };
  const waterGeo = new THREE.PlaneGeometry(72, 72, 64, 64);
  waterGeo.rotateX(-Math.PI / 2);

  const waterMat = new THREE.MeshPhysicalMaterial({
    color: new THREE.Color('#2d4a44'),
    roughness: 0.14,
    metalness: 0.08,
    clearcoat: 0.9,
    clearcoatRoughness: 0.1,
    normalMap: textures.waterNormal,
    normalScale: new THREE.Vector2(0.45, 0.45),
    transparent: true,
    opacity: 0.74,
    reflectivity: 0.9,
  });

  waterMat.onBeforeCompile = (shader) => {
    shader.uniforms.uTime = waterUniforms.uTime;
    shader.uniforms.uCrocPos = waterUniforms.uCrocPos;
    shader.vertexShader = shader.vertexShader
      .replace(
        '#include <common>',
        `#include <common>
        uniform float uTime;
        uniform vec2 uCrocPos;
        varying float vWakeRing;`
      )
      .replace(
        '#include <begin_vertex>',
        `#include <begin_vertex>
        float distCroc = length(transformed.xz - uCrocPos);
        float swell = sin(transformed.x * 0.85 + uTime * 1.4) * cos(transformed.z * 0.72 - uTime * 1.1) * 0.016;
        float wake = sin(distCroc * 5.2 - uTime * 4.8) * exp(-distCroc * 0.36) * 0.026;
        transformed.y += swell + wake;
        vWakeRing = clamp(wake * 16.0, 0.0, 1.0);`
      );
    shader.fragmentShader = shader.fragmentShader
      .replace(
        '#include <common>',
        `#include <common>
        varying float vWakeRing;`
      )
      .replace(
        '#include <color_fragment>',
        `#include <color_fragment>
        diffuseColor.rgb = mix(diffuseColor.rgb, vec3(0.76, 0.85, 0.83), vWakeRing * 0.28);`
      );
  };

  const waterPlane = new THREE.Mesh(waterGeo, waterMat);
  waterPlane.position.y = 0.24;
  waterPlane.receiveShadow = true;
  waterGroup.add(waterPlane);

  scene.add(waterGroup);

  // Colors for seamless fog-horizon matching (matches reference studio cyclorama!)
  const studioBg = new THREE.Color('#c8cbce');
  const waterBg = new THREE.Color('#8ca6a8');

  const setMode = (mode: EnvironmentMode, targetScene: THREE.Scene): void => {
    if (mode === 'studio') {
      studioGroup.visible = true;
      waterGroup.visible = false;
      targetScene.background = studioBg;
      targetScene.fog = new THREE.Fog(studioBg.getHex(), 11, 34);
      keyLight.color.setHex(0xfff7ea);
      keyLight.intensity = 2.05;
      fillHemi.color.setHex(0xdce6f0);
      fillHemi.groundColor.setHex(0x8e8a80);
      fillHemi.intensity = 0.95;
      rimLight.color.setHex(0xd9e8ff);
      rimLight.intensity = 0.85;
    } else {
      studioGroup.visible = false;
      waterGroup.visible = true;
      targetScene.background = waterBg;
      targetScene.fog = new THREE.FogExp2(waterBg.getHex(), 0.024);
      keyLight.color.setHex(0xfff0d0);
      keyLight.intensity = 2.65;
      fillHemi.color.setHex(0x9fc6d8);
      fillHemi.groundColor.setHex(0x4a4534);
      fillHemi.intensity = 0.95;
      rimLight.color.setHex(0xffe6ba);
      rimLight.intensity = 0.85;
    }
  };

  // Initialize default Studio mode matching reference
  setMode('studio', scene);

  const update = (elapsed: number, crocX: number, crocZ: number, swimWeight: number): void => {
    // Keep shadow frustum centered over the crocodile while driving
    keyLight.target.position.set(crocX, 0.4, crocZ);
    keyLight.position.set(crocX + 5.5, 9.5, crocZ + 6.0);

    if (waterGroup.visible) {
      waterUniforms.uTime.value = elapsed;
      waterUniforms.uCrocPos.value.set(crocX, crocZ);
      // Scroll water normal map smoothly for lively riverbank ripples
      textures.waterNormal.offset.set(elapsed * 0.028, elapsed * 0.019);
      // Rise shallow water level smoothly when swimming so crocodile sits half-submerged like a real Nile croc!
      const targetWaterY = 0.16 + swimWeight * 0.28;
      waterPlane.position.y += (targetWaterY - waterPlane.position.y) * 0.08;
    }
  };

  const dispose = (): void => {
    riverbedTex.dispose();
    studioGroup.traverse((o) => {
      if (o instanceof THREE.Mesh) {
        o.geometry.dispose();
        (o.material as THREE.Material).dispose();
      }
    });
    waterGroup.traverse((o) => {
      if (o instanceof THREE.Mesh) {
        o.geometry.dispose();
        (o.material as THREE.Material).dispose();
      }
    });
  };

  return {
    studioGroup,
    waterGroup,
    keyLight,
    fillHemi,
    rimLight,
    setMode,
    update,
    dispose,
  };
}
