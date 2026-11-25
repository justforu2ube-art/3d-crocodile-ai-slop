import * as THREE from 'three';
import {
  buildLoftedGeometry,
  buildOsteodermScuteGeometry,
  buildTailFinScuteGeometry,
  buildConicalToothGeometry,
  buildCurvedClawGeometry,
  buildReptileToeGeometry,
  type LoftRingSpec,
} from './geometryUtils';
import type { CrocodileTextureSet } from './textures';
import { TAIL_SEGMENT_COUNT, type CrocPose, type LimbId } from './kinematics';

export interface LimbRig {
  id: LimbId;
  sideSign: number; // +1 left (+X), -1 right (-X)
  isFront: boolean;
  hipPivot: THREE.Group;
  kneePivot: THREE.Group;
  anklePivot: THREE.Group;
  footSoleMarker: THREE.Object3D;
  footShadow: THREE.Mesh;
}

export interface EyeBlinkRig {
  upperLid: THREE.Mesh;
  nictitating: THREE.Mesh;
  sideSign: number;
}

export interface CrocodileRig {
  /** Top-level root placed at world (posX, 0, posZ) and rotated by heading */
  root: THREE.Group;
  /** Exportable anatomical model group (excludes debug skeleton & studio contact shadows) */
  modelRoot: THREE.Group;
  /** Articulated pelvis anchor at (0, pelvisY, 0) */
  pelvisRoot: THREE.Group;
  midTorsoPivot: THREE.Group;
  chestPivot: THREE.Group;
  neckPivots: [THREE.Group, THREE.Group, THREE.Group];
  headPivot: THREE.Group;
  jawPivot: THREE.Group;
  tailPivots: THREE.Group[];
  eyeRigs: EyeBlinkRig[];
  ribcageMesh: THREE.Mesh;
  throatMesh: THREE.Mesh;
  limbs: Record<LimbId, LimbRig>;
  torsoShadow: THREE.Mesh;
  /** Toggleable debug skeleton group */
  skeletonDebugGroup: THREE.Group;
  skeletonBonesLine: THREE.LineSegments;
  skeletonJointNodes: THREE.Object3D[];
  /** Materials exposed for wetness & wireframe updates */
  skinMaterials: THREE.MeshStandardMaterial[];
  allCrocMaterials: THREE.Material[];
  applyPose: (pose: CrocPose, wetness: number, showShadows: boolean) => void;
  setWireframe: (enabled: boolean) => void;
  setSkeletonVisible: (visible: boolean) => void;
  dispose: () => void;
}

// Reusable scratch vectors (zero per-frame GC allocations)
const _worldPos = new THREE.Vector3();
const _worldPosB = new THREE.Vector3();
const _localPos = new THREE.Vector3();
const _worldQuat = new THREE.Quaternion();
const _invParentQuat = new THREE.Quaternion();
const _flatQuat = new THREE.Quaternion();
const _euler = new THREE.Euler(0, 0, 0, 'YXZ');

/**
 * Helper to create a smooth anatomical torso/neck segment spanning along +Z.
 */
function createSpineSegmentMesh(
  rings: LoftRingSpec[],
  material: THREE.Material,
  radialSegments = 30,
  bumpAmp = 0.0068
): THREE.Mesh {
  const geo = buildLoftedGeometry(rings, {
    radialSegments,
    capStart: true,
    capEnd: true,
    bumpAmp,
    bumpFreq: 18,
    bakeVertexAO: true,
  });
  const mesh = new THREE.Mesh(geo, material);
  mesh.castShadow = true;
  mesh.receiveShadow = true;
  return mesh;
}

/**
 * Builds a heavy, stout, sculpted Nile Crocodile upper skull & festooned snout
 * with broad muscular cheeks, prominent supraorbital brow ridges, recessed orbital sockets,
 * wavy maxillary lip festooning, and a chunky raised tip nasal boss.
 */
function buildUpperSkullGeometry(): THREE.BufferGeometry {
  const rings: LoftRingSpec[] = [
    // Heavy occipital collar blending seamlessly into thick muscular neck3
    { z: -0.11, yOffset: 0.015, rx: 0.238, ry: 0.155, flatTop: 0.28, flatBottom: 0.56, lowerBulge: 0.16, uvV: 0.80 },
    // Massive post-orbital cranial table & heavy pterygoideus jaw cheeks
    { z: -0.01, yOffset: 0.025, rx: 0.278, ry: 0.168, flatTop: 0.36, flatBottom: 0.68, lowerBulge: 0.24, uvV: 0.83 },
    { z: 0.08, yOffset: 0.030, rx: 0.265, ry: 0.162, flatTop: 0.40, flatBottom: 0.74, lowerBulge: 0.22, uvV: 0.855 },
    // Supraorbital brow ridges & sculpted recessed orbital eye sockets!
    {
      z: 0.165,
      yOffset: 0.028,
      rx: 0.224,
      ry: 0.146,
      flatTop: 0.34,
      flatBottom: 0.78,
      lowerBulge: 0.16,
      uvV: 0.878,
      deform: (theta, x, y, z) => {
        const cosT = Math.cos(theta);
        const absSin = Math.abs(Math.sin(theta));
        // Prominent bony supraorbital brow crest arching directly over the eye sockets
        const browPeak = Math.exp(-Math.pow((absSin - 0.52) * 4.5, 2)) * Math.max(0, cosT);
        // Interorbital valley between left & right brow ridges
        const centerDip = Math.exp(-Math.pow(absSin * 4.2, 2)) * Math.max(0, cosT);
        // Sculpted orbital recess on upper-lateral flank so eyeball & eyelids sit cleanly exposed!
        const socketRecess =
          Math.exp(-Math.pow((absSin - 0.78) * 5.2, 2)) *
          Math.exp(-Math.pow((cosT - 0.56) * 4.8, 2));
        return [
          x - Math.sign(x) * socketRecess * 0.032,
          y + browPeak * 0.048 - centerDip * 0.016,
          z,
        ];
      },
    },
    // Anterior orbital slope descending into sturdy, thick snout bridge
    {
      z: 0.26,
      yOffset: 0.018,
      rx: 0.194,
      ry: 0.120,
      flatTop: 0.28,
      flatBottom: 0.82,
      lowerBulge: 0.12,
      uvV: 0.90,
      deform: (theta, x, y, z) => {
        const cosT = Math.cos(theta);
        const absSin = Math.abs(Math.sin(theta));
        const lacrimalRidge = Math.exp(-Math.pow((absSin - 0.48) * 4.2, 2)) * Math.max(0, cosT);
        return [x, y + lacrimalRidge * 0.022, z];
      },
    },
    // Mid-snout festooned maxillary wave (broad heavy Nile croc muzzle!)
    {
      z: 0.39,
      yOffset: 0.012,
      rx: 0.172,
      ry: 0.104,
      flatTop: 0.24,
      flatBottom: 0.84,
      lowerBulge: 0.12,
      uvV: 0.928,
      deform: (theta, x, y, z) => {
        const cosT = Math.cos(theta);
        const lipWave = Math.exp(-Math.pow((cosT + 0.25) * 3.0, 2));
        return [x, y - lipWave * 0.015, z];
      },
    },
    // Snout constriction notch accommodating the exposed 4th lower mandibular tooth
    {
      z: 0.53,
      yOffset: 0.016,
      rx: 0.138,
      ry: 0.088,
      flatTop: 0.20,
      flatBottom: 0.82,
      lowerBulge: 0.06,
      uvV: 0.955,
      deform: (theta, x, y, z) => {
        const cosT = Math.cos(theta);
        const notch = Math.exp(-Math.pow((cosT + 0.25) * 3.2, 2));
        return [x * (1 - notch * 0.07), y + notch * 0.015, z];
      },
    },
    // Chunky premaxillary flared snout tip & prominent raised dorsal nasal boss!
    {
      z: 0.66,
      yOffset: 0.018,
      rx: 0.146,
      ry: 0.086,
      flatTop: 0.12,
      flatBottom: 0.80,
      lowerBulge: 0.11,
      uvV: 0.978,
      deform: (theta, x, y, z) => {
        const cosT = Math.cos(theta);
        const absSin = Math.abs(Math.sin(theta));
        const nasalBoss = Math.exp(-Math.pow(absSin * 2.2, 2)) * Math.max(0, cosT);
        return [x, y + nasalBoss * 0.036, z];
      },
    },
    // Thick rounded anterior snout tip
    { z: 0.74, yOffset: 0.012, rx: 0.106, ry: 0.064, flatTop: 0.14, flatBottom: 0.72, lowerBulge: 0.06, uvV: 0.992 },
    { z: 0.78, yOffset: 0.008, rx: 0.042, ry: 0.032, flatTop: 0.10, flatBottom: 0.50, lowerBulge: 0, uvV: 1.0 },
  ];

  return buildLoftedGeometry(rings, {
    radialSegments: 32,
    capStart: true,
    capEnd: true,
    bumpAmp: 0.0058,
    bumpFreq: 26,
    bakeVertexAO: true,
  });
}

/**
 * Builds the thick, powerful articulated lower jaw (mandible) with heavy muscular rear cheeks.
 */
function buildLowerJawGeometry(): THREE.BufferGeometry {
  const rings: LoftRingSpec[] = [
    { z: -0.08, yOffset: -0.036, rx: 0.232, ry: 0.108, flatTop: 0.70, flatBottom: 0.28, lowerBulge: 0.18, uvV: 0.82 },
    { z: 0.04, yOffset: -0.038, rx: 0.252, ry: 0.112, flatTop: 0.76, flatBottom: 0.32, lowerBulge: 0.20, uvV: 0.845 },
    { z: 0.16, yOffset: -0.032, rx: 0.208, ry: 0.095, flatTop: 0.78, flatBottom: 0.36, lowerBulge: 0.14, uvV: 0.875 },
    { z: 0.31, yOffset: -0.022, rx: 0.162, ry: 0.078, flatTop: 0.78, flatBottom: 0.38, lowerBulge: 0.09, uvV: 0.91 },
    { z: 0.46, yOffset: -0.015, rx: 0.136, ry: 0.066, flatTop: 0.76, flatBottom: 0.40, lowerBulge: 0.07, uvV: 0.945 },
    { z: 0.59, yOffset: -0.010, rx: 0.118, ry: 0.056, flatTop: 0.74, flatBottom: 0.38, lowerBulge: 0.05, uvV: 0.972 },
    { z: 0.68, yOffset: -0.007, rx: 0.084, ry: 0.042, flatTop: 0.70, flatBottom: 0.34, lowerBulge: 0.03, uvV: 0.99 },
    { z: 0.72, yOffset: -0.004, rx: 0.032, ry: 0.022, flatTop: 0.50, flatBottom: 0.25, lowerBulge: 0, uvV: 1.0 },
  ];
  return buildLoftedGeometry(rings, {
    radialSegments: 26,
    capStart: true,
    capEnd: true,
    bumpAmp: 0.0042,
    bumpFreq: 28,
    bakeVertexAO: true,
  });
}

/**
 * Builds the fleshy interior tongue mesh inside the lower jaw.
 */
function buildTongueGeometry(): THREE.BufferGeometry {
  const rings: LoftRingSpec[] = [
    { z: 0.06, yOffset: 0.008, rx: 0.105, ry: 0.026, flatTop: 0.45, flatBottom: 0.6, lowerBulge: 0, uvV: 0.2 },
    { z: 0.22, yOffset: 0.007, rx: 0.086, ry: 0.022, flatTop: 0.45, flatBottom: 0.6, lowerBulge: 0, uvV: 0.5 },
    { z: 0.42, yOffset: 0.006, rx: 0.062, ry: 0.016, flatTop: 0.40, flatBottom: 0.55, lowerBulge: 0, uvV: 0.8 },
    { z: 0.56, yOffset: 0.004, rx: 0.028, ry: 0.010, flatTop: 0.30, flatBottom: 0.5, lowerBulge: 0, uvV: 1.0 },
  ];
  return buildLoftedGeometry(rings, { radialSegments: 14, capStart: true, capEnd: true, bumpAmp: 0.001, bumpFreq: 30 });
}

/**
 * Builds the pale fleshy upper palate lining tucked cleanly inside the roof of the upper skull.
 */
function buildUpperPalateGeometry(): THREE.BufferGeometry {
  const rings: LoftRingSpec[] = [
    { z: 0.06, yOffset: -0.006, rx: 0.132, ry: 0.018, flatTop: 0.5, flatBottom: 0.6, lowerBulge: 0, uvV: 0.2 },
    { z: 0.25, yOffset: -0.003, rx: 0.104, ry: 0.015, flatTop: 0.5, flatBottom: 0.6, lowerBulge: 0, uvV: 0.5 },
    { z: 0.48, yOffset: -0.001, rx: 0.072, ry: 0.012, flatTop: 0.5, flatBottom: 0.6, lowerBulge: 0, uvV: 0.8 },
    { z: 0.66, yOffset: 0.001, rx: 0.040, ry: 0.009, flatTop: 0.5, flatBottom: 0.6, lowerBulge: 0, uvV: 1.0 },
  ];
  return buildLoftedGeometry(rings, { radialSegments: 14, capStart: true, capEnd: true, bumpAmp: 0.001, bumpFreq: 20 });
}

/**
 * Constructs a prominent, anatomically authentic multi-layered crocodile eye:
 * - Luminous golden-amber iris with razor-sharp vertical slit pupil facing cleanly outward
 * - Wet glossy cornea specular dome
 * - Sculpted almond-shaped scaly supraorbital brow hood & lower cheek eyelid fold
 * - Dual-layer animated blink: outer scaly upper eyelid + translucent pale nictitating membrane!
 */
function createReptileEye(
  sideSign: number,
  irisMat: THREE.Material,
  corneaMat: THREE.Material,
  nictitatingMat: THREE.Material,
  eyelidMat: THREE.Material,
  eyeRigsOut: EyeBlinkRig[]
): THREE.Group {
  const eyeGroup = new THREE.Group();
  eyeGroup.name = sideSign > 0 ? 'Eye_L' : 'Eye_R';

  // Inner eyeball: in THREE.SphereGeometry, UV (0.5, 0.5) faces +X at rotation.y = 0.
  // Orient outward laterally (+X for left, -X for right) and slightly forward (+Z) so the golden iris & vertical slit gleam clearly!
  const eyeballGeo = new THREE.SphereGeometry(0.0325, 24, 18);
  const eyeball = new THREE.Mesh(eyeballGeo, irisMat);
  eyeball.rotation.y = sideSign > 0 ? -0.18 : Math.PI + 0.18;
  eyeGroup.add(eyeball);

  // Translucent milky-teal nictitating membrane (crocodile third eyelid that sweeps horizontally across cornea!)
  // phiStart=0.60*PI, phiLength=0.80*PI centers the dome at phi=PI (+X).
  const nictGeo = new THREE.SphereGeometry(
    0.0342,
    18,
    14,
    Math.PI * 0.60,
    Math.PI * 0.80,
    Math.PI * 0.14,
    Math.PI * 0.72
  );
  const nictitating = new THREE.Mesh(nictGeo, nictitatingMat);
  nictitating.rotation.y = sideSign > 0 ? 1.25 : Math.PI - 1.25;
  nictitating.visible = false;
  nictitating.renderOrder = 2;
  eyeGroup.add(nictitating);

  // Glossy wet cornea outer dome (crisp studio catchlight)
  const corneaGeo = new THREE.SphereGeometry(0.0352, 20, 16);
  const cornea = new THREE.Mesh(corneaGeo, corneaMat);
  cornea.renderOrder = 3;
  eyeGroup.add(cornea);

  // Sleek sculpted scaly supraorbital brow crest arching snugly over the top of the eye (matching reference!)
  const upperHoodGeo = new THREE.SphereGeometry(0.043, 20, 12, 0, Math.PI * 2, 0, Math.PI * 0.40);
  const upperHood = new THREE.Mesh(upperHoodGeo, eyelidMat);
  upperHood.scale.set(0.92, 0.68, 1.52);
  upperHood.position.set(-sideSign * 0.013, 0.011, 0.0);
  upperHood.rotation.z = sideSign * 0.22; // Tilted slightly inward into skull brow ridge
  upperHood.castShadow = true;
  upperHood.receiveShadow = true;
  eyeGroup.add(upperHood);

  // Lower scaly cheek eyelid fold framing the bottom of the golden iris in a sharp almond opening
  const lowerRimGeo = new THREE.SphereGeometry(0.041, 20, 10, 0, Math.PI * 2, Math.PI * 0.68, Math.PI * 0.32);
  const lowerRim = new THREE.Mesh(lowerRimGeo, eyelidMat);
  lowerRim.scale.set(0.94, 0.68, 1.42);
  lowerRim.position.set(-sideSign * 0.010, -0.007, 0.0);
  lowerRim.rotation.z = sideSign * -0.16;
  eyeGroup.add(lowerRim);

  // Animated scaly upper eyelid:
  // Right-hand rule around +Z: for Left (+1), +angle tilts cap INWARD (open/tucked under brow); -angle sweeps cap DOWNWARD over eye (closed)!
  const blinkGeo = new THREE.SphereGeometry(0.0372, 18, 12, 0, Math.PI * 2, 0, Math.PI * 0.54);
  const upperLid = new THREE.Mesh(blinkGeo, eyelidMat);
  upperLid.scale.set(1.0, 0.98, 1.08);
  upperLid.rotation.z = sideSign * 0.88; // Open: cleanly tucked inside brow hood!
  eyeGroup.add(upperLid);

  eyeRigsOut.push({ upperLid, nictitating, sideSign });
  return eyeGroup;
}

/**
 * Constructs a meaty, heavily armored sprawling reptilian limb (3-joint hierarchy):
 * Hip/Shoulder -> Elbow/Knee -> Ankle/Foot + splayed toes (5 front, 4 webbed hind) + heavy curved claws.
 */
function createReptileLimb(params: {
  id: LimbId;
  sideSign: number;
  isFront: boolean;
  limbMat: THREE.Material;
  clawMat: THREE.Material;
  webMat: THREE.Material;
  shadowMat: THREE.Material;
}): LimbRig {
  const { id, sideSign, isFront, limbMat, clawMat, webMat, shadowMat } = params;

  const hipPivot = new THREE.Group();
  hipPivot.name = `Joint_Hip_${id}`;

  const upperLen = isFront ? 0.28 : 0.32;
  const lowerLen = isFront ? 0.26 : 0.29;

  // 1. SLEEK MUSCULAR UPPER LIMB (Humerus / Thigh) — strong & athletic, not bloated
  const upperBaseR = isFront ? 0.132 : 0.164;
  const upperMidR = isFront ? 0.112 : 0.142;
  const upperDistalR = isFront ? 0.090 : 0.106;
  const upperRings: LoftRingSpec[] = [
    { z: -0.06, yOffset: 0, rx: upperBaseR * 0.94, ry: upperBaseR * 1.06, flatTop: 0.08, flatBottom: 0.08, lowerBulge: 0.10, uvV: 0.0 },
    { z: 0.03, yOffset: 0.008, rx: upperBaseR * 1.06, ry: upperBaseR * 1.12, flatTop: 0.08, flatBottom: 0.08, lowerBulge: 0.10, uvV: 0.14 },
    { z: upperLen * 0.45, yOffset: 0.005, rx: upperMidR, ry: upperMidR * 1.06, flatTop: 0.10, flatBottom: 0.10, lowerBulge: 0.08, uvV: 0.34 },
    { z: upperLen * 0.85, yOffset: 0, rx: upperDistalR * 1.04, ry: upperDistalR * 1.04, flatTop: 0.10, flatBottom: 0.10, lowerBulge: 0.04, uvV: 0.48 },
    { z: upperLen + 0.032, yOffset: 0, rx: upperDistalR * 0.86, ry: upperDistalR * 0.86, flatTop: 0.08, flatBottom: 0.08, lowerBulge: 0, uvV: 0.54 },
  ];
  const upperGeo = buildLoftedGeometry(upperRings, { radialSegments: 20, bumpAmp: 0.0045, bumpFreq: 24 });
  upperGeo.rotateX(Math.PI / 2); // +Z -> -Y
  const upperMesh = new THREE.Mesh(upperGeo, limbMat);
  upperMesh.castShadow = true;
  upperMesh.receiveShadow = true;
  hipPivot.add(upperMesh);

  // 2. ELBOW / KNEE PIVOT + DARK-SCALED ARMORED JOINT CAP (rotated so outer dark limb scales face outward!)
  const kneePivot = new THREE.Group();
  kneePivot.name = `Joint_Knee_${id}`;
  kneePivot.position.set(0, -upperLen, 0);
  hipPivot.add(kneePivot);

  const jointCap = new THREE.Mesh(
    new THREE.SphereGeometry(upperDistalR * 0.96, 14, 12),
    limbMat
  );
  // Rotate sphere so dark outer limb texture (u=0 / u=1) wraps outward rather than pale inner-thigh patch (u=0.5)
  jointCap.rotation.y = sideSign > 0 ? Math.PI : 0;
  jointCap.castShadow = true;
  jointCap.receiveShadow = true;
  kneePivot.add(jointCap);

  // Muscular armored forearm / calf
  const lowerProxR = isFront ? 0.090 : 0.106;
  const lowerMidR = isFront ? 0.078 : 0.092;
  const lowerAnkleR = isFront ? 0.066 : 0.078;
  const lowerRings: LoftRingSpec[] = [
    { z: -0.032, yOffset: 0, rx: lowerProxR * 0.90, ry: lowerProxR * 0.92, flatTop: 0.10, flatBottom: 0.10, lowerBulge: 0.06, uvV: 0.50 },
    { z: 0.038, yOffset: 0.005, rx: lowerProxR * 1.04, ry: lowerProxR * 1.05, flatTop: 0.10, flatBottom: 0.10, lowerBulge: 0.08, uvV: 0.60 },
    { z: lowerLen * 0.48, yOffset: 0.004, rx: lowerMidR, ry: lowerMidR * 1.04, flatTop: 0.12, flatBottom: 0.12, lowerBulge: 0.06, uvV: 0.76 },
    { z: lowerLen * 0.86, yOffset: 0, rx: lowerAnkleR * 1.05, ry: lowerAnkleR * 0.96, flatTop: 0.14, flatBottom: 0.20, lowerBulge: 0.04, uvV: 0.90 },
    { z: lowerLen + 0.022, yOffset: 0, rx: lowerAnkleR * 0.92, ry: lowerAnkleR * 0.82, flatTop: 0.14, flatBottom: 0.25, lowerBulge: 0, uvV: 0.95 },
  ];
  const lowerGeo = buildLoftedGeometry(lowerRings, { radialSegments: 18, bumpAmp: 0.0042, bumpFreq: 26 });
  lowerGeo.rotateX(Math.PI / 2);
  const lowerMesh = new THREE.Mesh(lowerGeo, limbMat);
  lowerMesh.castShadow = true;
  lowerMesh.receiveShadow = true;
  kneePivot.add(lowerMesh);

  // 3. ANKLE / WRIST & BROAD PLANTED SPLAYED FOOT
  const anklePivot = new THREE.Group();
  anklePivot.name = `Joint_Ankle_${id}`;
  anklePivot.position.set(0, -lowerLen, 0);
  kneePivot.add(anklePivot);

  const padWidth = isFront ? 0.100 : 0.124;
  const padLen = isFront ? 0.142 : 0.175;
  const padRings: LoftRingSpec[] = [
    { z: -0.065, yOffset: -0.014, rx: padWidth * 0.70, ry: 0.046, flatTop: 0.2, flatBottom: 0.74, lowerBulge: 0.06, uvV: 0.85 },
    { z: 0.020, yOffset: -0.020, rx: padWidth * 0.98, ry: 0.038, flatTop: 0.25, flatBottom: 0.84, lowerBulge: 0.06, uvV: 0.92 },
    { z: padLen * 0.78, yOffset: -0.026, rx: padWidth * 1.08, ry: 0.028, flatTop: 0.28, flatBottom: 0.86, lowerBulge: 0.04, uvV: 0.97 },
    { z: padLen, yOffset: -0.028, rx: padWidth * 0.78, ry: 0.020, flatTop: 0.22, flatBottom: 0.82, lowerBulge: 0, uvV: 1.0 },
  ];
  const padGeo = buildLoftedGeometry(padRings, { radialSegments: 16, bumpAmp: 0.0028, bumpFreq: 32 });
  const padMesh = new THREE.Mesh(padGeo, limbMat);
  padMesh.castShadow = true;
  padMesh.receiveShadow = true;
  anklePivot.add(padMesh);

  // Front feet: 5 distinct splayed clawed toes; Hind feet: 4 larger webbed clawed toes
  const toeSpecs = isFront
    ? [
        { x: -sideSign * 0.076, z: 0.066, yaw: -sideSign * 0.74, len: 0.108, r: 0.020, claw: 0.050 },
        { x: -sideSign * 0.040, z: 0.100, yaw: -sideSign * 0.35, len: 0.142, r: 0.022, claw: 0.058 },
        { x: 0.0, z: 0.112, yaw: sideSign * 0.02, len: 0.156, r: 0.023, claw: 0.060 },
        { x: sideSign * 0.042, z: 0.100, yaw: sideSign * 0.39, len: 0.136, r: 0.021, claw: 0.052 },
        { x: sideSign * 0.074, z: 0.066, yaw: sideSign * 0.78, len: 0.104, r: 0.019, claw: 0.044 },
      ]
    : [
        { x: -sideSign * 0.076, z: 0.108, yaw: -sideSign * 0.58, len: 0.152, r: 0.026, claw: 0.068 },
        { x: -sideSign * 0.024, z: 0.136, yaw: -sideSign * 0.18, len: 0.188, r: 0.028, claw: 0.074 },
        { x: sideSign * 0.030, z: 0.134, yaw: sideSign * 0.22, len: 0.182, r: 0.027, claw: 0.068 },
        { x: sideSign * 0.080, z: 0.105, yaw: sideSign * 0.62, len: 0.144, r: 0.024, claw: 0.054 },
      ];

  const toeTipPositions: THREE.Vector3[] = [];

  for (const ts of toeSpecs) {
    const toeGroup = new THREE.Group();
    toeGroup.position.set(ts.x, -0.028, ts.z);
    toeGroup.rotation.y = ts.yaw;
    anklePivot.add(toeGroup);

    const toeGeo = buildReptileToeGeometry(ts.len, ts.r, ts.r * 0.60);
    const toeMesh = new THREE.Mesh(toeGeo, limbMat);
    toeMesh.castShadow = true;
    toeMesh.receiveShadow = true;
    toeGroup.add(toeMesh);

    const clawGeo = buildCurvedClawGeometry(ts.r * 0.56, ts.claw);
    const clawMesh = new THREE.Mesh(clawGeo, clawMat);
    clawMesh.position.set(0, -0.002, ts.len - 0.006);
    clawMesh.castShadow = true;
    toeGroup.add(clawMesh);

    toeTipPositions.push(
      new THREE.Vector3(
        ts.x + Math.sin(ts.yaw) * ts.len * 0.70,
        -0.036,
        ts.z + Math.cos(ts.yaw) * ts.len * 0.70
      )
    );
  }

  // Interdigital webbing membranes on hind feet
  if (!isFront && toeTipPositions.length === 4) {
    for (let i = 0; i < 3; i++) {
      const a = toeSpecs[i];
      const b = toeSpecs[i + 1];
      const tipA = toeTipPositions[i];
      const tipB = toeTipPositions[i + 1];
      const webGeo = new THREE.BufferGeometry();
      const verts = new Float32Array([
        a.x, -0.034, a.z,
        tipA.x, tipA.y, tipA.z,
        tipB.x, tipB.y, tipB.z,
        b.x, -0.034, b.z,
      ]);
      webGeo.setAttribute('position', new THREE.BufferAttribute(verts, 3));
      webGeo.setAttribute('uv', new THREE.BufferAttribute(new Float32Array([0, 0, 0, 1, 1, 1, 1, 0]), 2));
      webGeo.setIndex([0, 1, 2, 0, 2, 3, 0, 2, 1, 0, 3, 2]);
      webGeo.computeVertexNormals();
      const webMesh = new THREE.Mesh(webGeo, webMat);
      webMesh.receiveShadow = true;
      anklePivot.add(webMesh);
    }
  }

  const footSoleMarker = new THREE.Object3D();
  footSoleMarker.position.set(0, -0.048, padLen * 0.55);
  anklePivot.add(footSoleMarker);

  const footShadowGeo = new THREE.PlaneGeometry(isFront ? 0.62 : 0.70, isFront ? 0.66 : 0.74);
  footShadowGeo.rotateX(-Math.PI / 2);
  const footShadow = new THREE.Mesh(footShadowGeo, shadowMat.clone());
  footShadow.position.y = 0.008;
  footShadow.renderOrder = 1;

  return {
    id,
    sideSign,
    isFront,
    hipPivot,
    kneePivot,
    anklePivot,
    footSoleMarker,
    footShadow,
  };
}

/**
 * Helper to mount rows of keeled dorsal osteoderms and lateral flank tubercles onto a spine segment.
 */
function addDorsalOsteoderms(
  parent: THREE.Group,
  specs: Array<{
    x: number;
    y: number;
    z: number;
    length: number;
    width: number;
    height: number;
    roll?: number;
    pitch?: number;
    sweep?: number;
  }>,
  scuteMat: THREE.Material
): void {
  for (const s of specs) {
    const geo = buildOsteodermScuteGeometry(s.length, s.width, s.height, s.sweep ?? 0.24);
    const mesh = new THREE.Mesh(geo, scuteMat);
    mesh.position.set(s.x, s.y, s.z);
    if (s.roll) mesh.rotation.z = s.roll;
    if (s.pitch) mesh.rotation.x = s.pitch;
    mesh.castShadow = true;
    mesh.receiveShadow = true;
    parent.add(mesh);
  }
}

/**
 * Builds the complete, heavy, muscular, anatomically articulated Nile Crocodile (Crocodylus niloticus).
 */
export function buildNileCrocodile(textures: CrocodileTextureSet): CrocodileRig {
  const root = new THREE.Group();
  root.name = 'NileCrocodileRoot';

  const modelRoot = new THREE.Group();
  modelRoot.name = 'NileCrocodileModel';
  root.add(modelRoot);

  // --- HIGH-DEFINITION PROCEDURAL PBR MATERIALS ---
  const bodySkinMat = new THREE.MeshStandardMaterial({
    color: 0xffffff,
    map: textures.bodyAlbedo,
    normalMap: textures.bodyNormal,
    normalScale: new THREE.Vector2(2.15, 2.15),
    roughnessMap: textures.bodyRoughness,
    roughness: 0.92,
    aoMap: textures.bodyAO,
    aoMapIntensity: 0.85,
    metalness: 0.02,
    vertexColors: true,
  });

  // Raised bony osteoderm armor plates: dark weathered olive-charcoal with satin worn ridge highlights
  const scuteMat = new THREE.MeshStandardMaterial({
    color: new THREE.Color('#5a5e50'),
    map: textures.bodyAlbedo,
    normalMap: textures.bodyNormal,
    normalScale: new THREE.Vector2(1.75, 1.75),
    roughnessMap: textures.bodyRoughness,
    roughness: 0.58,
    metalness: 0.04,
  });

  const limbSkinMat = new THREE.MeshStandardMaterial({
    color: 0xffffff,
    map: textures.limbAlbedo,
    normalMap: textures.limbNormal,
    normalScale: new THREE.Vector2(1.95, 1.95),
    roughnessMap: textures.limbRoughness,
    roughness: 0.90,
    metalness: 0.02,
  });

  const webMat = new THREE.MeshStandardMaterial({
    color: new THREE.Color('#3f4338'),
    roughness: 0.82,
    metalness: 0.02,
    side: THREE.DoubleSide,
  });

  const mouthFleshMat = new THREE.MeshStandardMaterial({
    color: new THREE.Color('#d4948a'),
    roughness: 0.40,
    metalness: 0.01,
  });

  const tongueMat = new THREE.MeshStandardMaterial({
    color: new THREE.Color('#c87f76'),
    roughness: 0.34,
    metalness: 0.01,
  });

  const toothMat = new THREE.MeshPhysicalMaterial({
    color: new THREE.Color('#dfd7c4'),
    roughness: 0.24,
    metalness: 0.02,
    clearcoat: 0.35,
    clearcoatRoughness: 0.24,
  });

  const clawMat = new THREE.MeshStandardMaterial({
    color: new THREE.Color('#21201d'),
    roughness: 0.32,
    metalness: 0.06,
  });

  // Dedicated weathered head-skin material WITHOUT vertexColors for eyelids, brow hoods, and nostril rims
  // (SphereGeometry has no vertex color attribute, so vertexColors: true would multiply it to black!)
  const eyelidSkinMat = new THREE.MeshStandardMaterial({
    color: new THREE.Color('#666a5b'),
    normalMap: textures.limbNormal,
    normalScale: new THREE.Vector2(1.4, 1.4),
    roughnessMap: textures.limbRoughness,
    roughness: 0.86,
    metalness: 0.02,
  });

  // Slightly warm-emissive reptile iris so golden-amber slit-pupil eye gleams vividly under the brow ridge!
  const eyeIrisMat = new THREE.MeshStandardMaterial({
    color: 0xffffff,
    map: textures.eyeIris,
    emissive: new THREE.Color('#6e581c'),
    emissiveMap: textures.eyeIris,
    emissiveIntensity: 0.52,
    roughness: 0.18,
    metalness: 0.02,
  });

  // Authentic crocodile translucent milky-teal nictitating membrane (third eyelid)
  const nictitatingMat = new THREE.MeshPhysicalMaterial({
    color: new THREE.Color('#b6beb8'),
    roughness: 0.25,
    metalness: 0.0,
    transparent: true,
    opacity: 0.72,
    depthWrite: false,
  });

  const corneaMat = new THREE.MeshPhysicalMaterial({
    color: 0xffffff,
    roughness: 0.03,
    metalness: 0.0,
    clearcoat: 1.0,
    clearcoatRoughness: 0.03,
    transparent: true,
    opacity: 0.22,
    depthWrite: false,
  });

  const nostrilInteriorMat = new THREE.MeshStandardMaterial({
    color: new THREE.Color('#161714'),
    roughness: 0.72,
    metalness: 0.0,
  });

  const contactShadowMat = new THREE.MeshBasicMaterial({
    map: textures.contactShadow,
    transparent: true,
    opacity: 0.62,
    depthWrite: false,
  });

  const skinMaterials = [bodySkinMat, scuteMat, limbSkinMat, eyelidSkinMat];
  const allCrocMaterials: THREE.Material[] = [
    bodySkinMat,
    scuteMat,
    limbSkinMat,
    eyelidSkinMat,
    webMat,
    mouthFleshMat,
    tongueMat,
    toothMat,
    clawMat,
    eyeIrisMat,
    nictitatingMat,
    corneaMat,
    nostrilInteriorMat,
  ];

  // =========================================================================
  // 1. HEAVY MUSCULAR BARREL SPINE CHAIN (Pelvis -> MidTorso -> Chest -> Neck x3 -> Head)
  // =========================================================================

  // --- PELVIS ROOT (Sleek Muscular Hips Flowing into Long Tail) ---
  const pelvisRoot = new THREE.Group();
  pelvisRoot.name = 'Joint_Pelvis';
  pelvisRoot.position.set(0, 0.42, 0);
  modelRoot.add(pelvisRoot);

  const pelvisMesh = createSpineSegmentMesh(
    [
      { z: -0.24, yOffset: -0.018, rx: 0.270, ry: 0.222, flatTop: 0.26, flatBottom: 0.38, lowerBulge: 0.14, uvV: 0.435 },
      { z: -0.06, yOffset: -0.024, rx: 0.308, ry: 0.248, flatTop: 0.28, flatBottom: 0.40, lowerBulge: 0.17, uvV: 0.465 },
      { z: 0.10, yOffset: -0.028, rx: 0.332, ry: 0.266, flatTop: 0.28, flatBottom: 0.40, lowerBulge: 0.18, uvV: 0.495 },
      { z: 0.28, yOffset: -0.030, rx: 0.342, ry: 0.274, flatTop: 0.28, flatBottom: 0.40, lowerBulge: 0.19, uvV: 0.525 },
    ],
    bodySkinMat
  );
  pelvisMesh.name = 'PelvisMesh';
  pelvisRoot.add(pelvisMesh);

  const pelvisScutes = [];
  for (const z of [-0.14, 0.01, 0.15]) {
    pelvisScutes.push(
      { x: -0.068, y: 0.174, z, length: 0.140, width: 0.094, height: 0.052, roll: 0.12 },
      { x: 0.068, y: 0.174, z, length: 0.140, width: 0.094, height: 0.052, roll: -0.12 },
      { x: -0.165, y: 0.148, z: z - 0.01, length: 0.118, width: 0.080, height: 0.042, roll: 0.38 },
      { x: 0.165, y: 0.148, z: z - 0.01, length: 0.118, width: 0.080, height: 0.042, roll: -0.38 },
      { x: -0.252, y: 0.092, z: z + 0.01, length: 0.090, width: 0.066, height: 0.032, roll: 0.72 },
      { x: 0.252, y: 0.092, z: z + 0.01, length: 0.090, width: 0.066, height: 0.032, roll: -0.72 }
    );
  }
  addDorsalOsteoderms(pelvisRoot, pelvisScutes, scuteMat);

  // --- MID-TORSO (Athletic Predatory Barrel — Sleek from Above, Deep & Muscular from the Side!) ---
  const midTorsoPivot = new THREE.Group();
  midTorsoPivot.name = 'Joint_MidTorso';
  midTorsoPivot.position.set(0, 0.010, 0.23);
  pelvisRoot.add(midTorsoPivot);

  const ribcageMesh = createSpineSegmentMesh(
    [
      { z: -0.09, yOffset: -0.032, rx: 0.342, ry: 0.274, flatTop: 0.28, flatBottom: 0.40, lowerBulge: 0.19, uvV: 0.52 },
      { z: 0.12, yOffset: -0.038, rx: 0.376, ry: 0.296, flatTop: 0.30, flatBottom: 0.42, lowerBulge: 0.21, uvV: 0.56 },
      { z: 0.32, yOffset: -0.036, rx: 0.374, ry: 0.294, flatTop: 0.30, flatBottom: 0.42, lowerBulge: 0.20, uvV: 0.60 },
      { z: 0.49, yOffset: -0.030, rx: 0.354, ry: 0.276, flatTop: 0.30, flatBottom: 0.42, lowerBulge: 0.18, uvV: 0.635 },
    ],
    bodySkinMat
  );
  ribcageMesh.name = 'MidTorsoMesh';
  midTorsoPivot.add(ribcageMesh);

  // 6-row dorsal osteoderm armor + sculpted lateral flank tubercles seated cleanly on sleek torso
  const midScutes = [];
  for (const z of [0.02, 0.16, 0.30, 0.43]) {
    midScutes.push(
      { x: -0.070, y: 0.182, z, length: 0.144, width: 0.096, height: 0.054, roll: 0.10 },
      { x: 0.070, y: 0.182, z, length: 0.144, width: 0.096, height: 0.054, roll: -0.10 },
      { x: -0.172, y: 0.162, z: z - 0.01, length: 0.128, width: 0.086, height: 0.045, roll: 0.30 },
      { x: 0.172, y: 0.162, z: z - 0.01, length: 0.128, width: 0.086, height: 0.045, roll: -0.30 },
      { x: -0.268, y: 0.118, z: z + 0.02, length: 0.104, width: 0.074, height: 0.035, roll: 0.62 },
      { x: 0.268, y: 0.118, z: z + 0.02, length: 0.104, width: 0.074, height: 0.035, roll: -0.62 },
      // Lateral flank osteoderm bosses
      { x: -0.352, y: 0.030, z: z - 0.03, length: 0.084, width: 0.064, height: 0.030, roll: 1.08 },
      { x: 0.352, y: 0.030, z: z - 0.03, length: 0.084, width: 0.064, height: 0.030, roll: -1.08 },
      { x: -0.376, y: -0.048, z: z + 0.04, length: 0.072, width: 0.056, height: 0.024, roll: 1.38 },
      { x: 0.376, y: -0.048, z: z + 0.04, length: 0.072, width: 0.056, height: 0.024, roll: -1.38 }
    );
  }
  addDorsalOsteoderms(midTorsoPivot, midScutes, scuteMat);

  // --- CHEST / MUSCULAR SHOULDERS ---
  const chestPivot = new THREE.Group();
  chestPivot.name = 'Joint_Chest';
  chestPivot.position.set(0, 0.006, 0.44);
  midTorsoPivot.add(chestPivot);

  const chestMesh = createSpineSegmentMesh(
    [
      { z: -0.09, yOffset: -0.030, rx: 0.352, ry: 0.274, flatTop: 0.30, flatBottom: 0.42, lowerBulge: 0.18, uvV: 0.63 },
      { z: 0.10, yOffset: -0.028, rx: 0.342, ry: 0.262, flatTop: 0.30, flatBottom: 0.42, lowerBulge: 0.18, uvV: 0.665 },
      { z: 0.28, yOffset: -0.024, rx: 0.312, ry: 0.236, flatTop: 0.28, flatBottom: 0.42, lowerBulge: 0.16, uvV: 0.70 },
      { z: 0.45, yOffset: -0.018, rx: 0.278, ry: 0.208, flatTop: 0.28, flatBottom: 0.42, lowerBulge: 0.14, uvV: 0.728 },
    ],
    bodySkinMat
  );
  chestMesh.name = 'ChestMesh';
  chestPivot.add(chestMesh);

  const chestScutes = [];
  for (const z of [0.06, 0.20, 0.34]) {
    const taper = 1 - (z / 0.45) * 0.20;
    chestScutes.push(
      { x: -0.064 * taper, y: 0.168 * taper + 0.006, z, length: 0.132, width: 0.092, height: 0.052, roll: 0.11 },
      { x: 0.064 * taper, y: 0.168 * taper + 0.006, z, length: 0.132, width: 0.092, height: 0.052, roll: -0.11 },
      { x: -0.158 * taper, y: 0.146 * taper, z: z - 0.01, length: 0.118, width: 0.082, height: 0.042, roll: 0.34 },
      { x: 0.158 * taper, y: 0.146 * taper, z: z - 0.01, length: 0.118, width: 0.082, height: 0.042, roll: -0.34 },
      { x: -0.246 * taper, y: 0.098 * taper, z: z + 0.01, length: 0.094, width: 0.068, height: 0.032, roll: 0.66 },
      { x: 0.246 * taper, y: 0.098 * taper, z: z + 0.01, length: 0.094, width: 0.068, height: 0.032, roll: -0.66 }
    );
  }
  addDorsalOsteoderms(chestPivot, chestScutes, scuteMat);

  // --- MUSCULAR NECK (x3 articulated) TAPERING CLEANLY INTO LARGE PREDATORY SKULL ---
  const neck1Pivot = new THREE.Group();
  neck1Pivot.name = 'Joint_Neck_1';
  neck1Pivot.position.set(0, -0.004, 0.38);
  chestPivot.add(neck1Pivot);

  const neck1Mesh = createSpineSegmentMesh(
    [
      { z: -0.07, yOffset: -0.016, rx: 0.280, ry: 0.208, flatTop: 0.26, flatBottom: 0.40, lowerBulge: 0.14, uvV: 0.725 },
      { z: 0.08, yOffset: -0.012, rx: 0.264, ry: 0.194, flatTop: 0.26, flatBottom: 0.40, lowerBulge: 0.14, uvV: 0.752 },
      { z: 0.21, yOffset: -0.008, rx: 0.254, ry: 0.184, flatTop: 0.26, flatBottom: 0.40, lowerBulge: 0.14, uvV: 0.774 },
    ],
    bodySkinMat,
    24
  );
  neck1Pivot.add(neck1Mesh);

  addDorsalOsteoderms(
    neck1Pivot,
    [
      { x: -0.062, y: 0.142, z: 0.05, length: 0.128, width: 0.094, height: 0.058, roll: 0.10 },
      { x: 0.062, y: 0.142, z: 0.05, length: 0.128, width: 0.094, height: 0.058, roll: -0.10 },
      { x: -0.152, y: 0.114, z: 0.08, length: 0.102, width: 0.074, height: 0.044, roll: 0.38 },
      { x: 0.152, y: 0.114, z: 0.08, length: 0.102, width: 0.074, height: 0.044, roll: -0.38 },
    ],
    scuteMat
  );

  const neck2Pivot = new THREE.Group();
  neck2Pivot.name = 'Joint_Neck_2';
  neck2Pivot.position.set(0, 0.006, 0.16);
  neck1Pivot.add(neck2Pivot);

  const neck2Mesh = createSpineSegmentMesh(
    [
      { z: -0.06, yOffset: -0.008, rx: 0.256, ry: 0.184, flatTop: 0.26, flatBottom: 0.40, lowerBulge: 0.16, uvV: 0.768 },
      { z: 0.08, yOffset: -0.002, rx: 0.250, ry: 0.178, flatTop: 0.26, flatBottom: 0.40, lowerBulge: 0.16, uvV: 0.790 },
      { z: 0.19, yOffset: 0.002, rx: 0.246, ry: 0.174, flatTop: 0.26, flatBottom: 0.42, lowerBulge: 0.16, uvV: 0.808 },
    ],
    bodySkinMat,
    24
  );
  neck2Pivot.add(neck2Mesh);
  const throatMesh = neck2Mesh;

  addDorsalOsteoderms(
    neck2Pivot,
    [
      { x: -0.058, y: 0.138, z: 0.05, length: 0.122, width: 0.088, height: 0.054, roll: 0.10 },
      { x: 0.058, y: 0.138, z: 0.05, length: 0.122, width: 0.088, height: 0.054, roll: -0.10 },
    ],
    scuteMat
  );

  const neck3Pivot = new THREE.Group();
  neck3Pivot.name = 'Joint_Neck_3';
  neck3Pivot.position.set(0, 0.008, 0.15);
  neck2Pivot.add(neck3Pivot);

  const neck3Mesh = createSpineSegmentMesh(
    [
      { z: -0.06, yOffset: -0.002, rx: 0.248, ry: 0.174, flatTop: 0.28, flatBottom: 0.44, lowerBulge: 0.16, uvV: 0.802 },
      { z: 0.06, yOffset: 0.004, rx: 0.246, ry: 0.168, flatTop: 0.28, flatBottom: 0.46, lowerBulge: 0.16, uvV: 0.818 },
      { z: 0.16, yOffset: 0.008, rx: 0.244, ry: 0.162, flatTop: 0.30, flatBottom: 0.48, lowerBulge: 0.16, uvV: 0.830 },
    ],
    bodySkinMat,
    24
  );
  neck3Pivot.add(neck3Mesh);

  // Post-occipital scutes directly behind skull
  addDorsalOsteoderms(
    neck3Pivot,
    [
      { x: -0.052, y: 0.132, z: 0.05, length: 0.080, width: 0.062, height: 0.040, roll: 0.08 },
      { x: 0.052, y: 0.132, z: 0.05, length: 0.080, width: 0.062, height: 0.040, roll: -0.08 },
      { x: -0.130, y: 0.116, z: 0.04, length: 0.070, width: 0.054, height: 0.034, roll: 0.28 },
      { x: 0.130, y: 0.116, z: 0.04, length: 0.070, width: 0.054, height: 0.034, roll: -0.28 },
    ],
    scuteMat
  );

  // =========================================================================
  // 2. LARGE PREDATORY SKULL, ELONGATED SNOUT, PROMINENT BLINKING EYES, JAW & 34 TEETH
  // =========================================================================
  const headPivot = new THREE.Group();
  headPivot.name = 'Joint_Head';
  headPivot.position.set(0, 0.018, 0.13);
  // Scaled to authentic predatory Nile Crocodile head-to-body ratio (~19% of total length!)
  headPivot.scale.set(1.06, 1.04, 1.26);
  neck3Pivot.add(headPivot);

  const upperSkullMesh = new THREE.Mesh(buildUpperSkullGeometry(), bodySkinMat);
  upperSkullMesh.name = 'UpperSkullSnout';
  upperSkullMesh.castShadow = true;
  upperSkullMesh.receiveShadow = true;
  headPivot.add(upperSkullMesh);

  const upperPalateMesh = new THREE.Mesh(buildUpperPalateGeometry(), mouthFleshMat);
  upperPalateMesh.name = 'UpperPalate';
  upperPalateMesh.receiveShadow = true;
  headPivot.add(upperPalateMesh);

  // Twin sculpted crescent nostrils atop the chunky raised nasal boss at snout tip
  for (const side of [-1, 1]) {
    const nostrilRim = new THREE.Mesh(
      buildOsteodermScuteGeometry(0.062, 0.048, 0.020, 0.0),
      eyelidSkinMat
    );
    nostrilRim.position.set(side * 0.042, 0.092, 0.665);
    nostrilRim.rotation.y = side * 0.22;
    headPivot.add(nostrilRim);

    const nostrilHole = new THREE.Mesh(
      new THREE.CylinderGeometry(0.015, 0.012, 0.018, 10),
      nostrilInteriorMat
    );
    nostrilHole.position.set(side * 0.043, 0.102, 0.668);
    nostrilHole.scale.set(1.25, 1.0, 0.65);
    nostrilHole.rotation.y = side * 0.28;
    headPivot.add(nostrilHole);
  }

  // PROMINENT MULTI-LAYERED BLINKING REPTILE EYES!
  // Nestled snugly inside the sculpted orbital socket beneath the scaly supraorbital brow crest!
  const eyeRigs: EyeBlinkRig[] = [];
  for (const side of [-1, 1]) {
    const eye = createReptileEye(side, eyeIrisMat, corneaMat, nictitatingMat, eyelidSkinMat, eyeRigs);
    eye.position.set(side * 0.145, 0.106, 0.172);
    headPivot.add(eye);
  }

  // --- INDEPENDENT ARTICULATED HEAVY LOWER JAW ---
  const jawPivot = new THREE.Group();
  jawPivot.name = 'Joint_LowerJaw';
  jawPivot.position.set(0, -0.026, 0.035);
  headPivot.add(jawPivot);

  const lowerJawMesh = new THREE.Mesh(buildLowerJawGeometry(), bodySkinMat);
  lowerJawMesh.name = 'LowerJawMandible';
  lowerJawMesh.castShadow = true;
  lowerJawMesh.receiveShadow = true;
  jawPivot.add(lowerJawMesh);

  const tongueMesh = new THREE.Mesh(buildTongueGeometry(), tongueMat);
  tongueMesh.name = 'TongueMesh';
  tongueMesh.receiveShadow = true;
  jawPivot.add(tongueMesh);

  // --- 34 INDIVIDUAL CONICAL INTERLOCKING TEETH (18 Upper + 16 Lower) ---
  const upperTeethSpecs: Array<{ z: number; x: number; y: number; len: number; r: number; cantOut: number }> = [
    { z: 0.735, x: 0.044, y: -0.012, len: 0.044, r: 0.011, cantOut: 0.14 },
    { z: 0.698, x: 0.086, y: -0.012, len: 0.052, r: 0.012, cantOut: 0.18 },
    { z: 0.650, x: 0.122, y: -0.014, len: 0.062, r: 0.014, cantOut: 0.22 },
    { z: 0.538, x: 0.136, y: -0.018, len: 0.066, r: 0.015, cantOut: 0.20 },
    { z: 0.474, x: 0.150, y: -0.022, len: 0.054, r: 0.013, cantOut: 0.16 },
    { z: 0.412, x: 0.162, y: -0.024, len: 0.048, r: 0.012, cantOut: 0.14 },
    { z: 0.350, x: 0.172, y: -0.024, len: 0.044, r: 0.011, cantOut: 0.12 },
    { z: 0.288, x: 0.184, y: -0.022, len: 0.038, r: 0.0105, cantOut: 0.10 },
    { z: 0.224, x: 0.194, y: -0.020, len: 0.032, r: 0.010, cantOut: 0.08 },
  ];

  for (const side of [-1, 1]) {
    for (const ts of upperTeethSpecs) {
      const tGeo = buildConicalToothGeometry(ts.r, ts.len, 0.22);
      const tooth = new THREE.Mesh(tGeo, toothMat);
      tooth.position.set(side * ts.x, ts.y, ts.z);
      tooth.rotation.x = Math.PI;
      tooth.rotation.z = side * ts.cantOut;
      tooth.castShadow = true;
      headPivot.add(tooth);
    }
  }

  const lowerTeethSpecs: Array<{ z: number; x: number; y: number; len: number; r: number; cantOut: number }> = [
    { z: 0.660, x: 0.042, y: 0.004, len: 0.044, r: 0.0105, cantOut: 0.12 },
    { z: 0.614, x: 0.084, y: 0.005, len: 0.040, r: 0.010, cantOut: 0.12 },
    // Prominent exposed 4th mandibular tooth protruding outside the upper snout notch!
    { z: 0.550, x: 0.126, y: 0.006, len: 0.072, r: 0.0155, cantOut: 0.24 },
    { z: 0.460, x: 0.130, y: 0.004, len: 0.040, r: 0.0105, cantOut: 0.10 },
    { z: 0.396, x: 0.142, y: 0.002, len: 0.042, r: 0.011, cantOut: 0.10 },
    { z: 0.332, x: 0.152, y: 0.000, len: 0.044, r: 0.0115, cantOut: 0.10 },
    { z: 0.268, x: 0.164, y: -0.002, len: 0.038, r: 0.0105, cantOut: 0.08 },
    { z: 0.205, x: 0.174, y: -0.004, len: 0.032, r: 0.010, cantOut: 0.06 },
  ];

  for (const side of [-1, 1]) {
    for (const ts of lowerTeethSpecs) {
      const tGeo = buildConicalToothGeometry(ts.r, ts.len, 0.18);
      const tooth = new THREE.Mesh(tGeo, toothMat);
      tooth.position.set(side * ts.x, ts.y, ts.z);
      tooth.rotation.z = -side * ts.cantOut;
      tooth.castShadow = true;
      jawPivot.add(tooth);
    }
  }

  // =========================================================================
  // 3. LONG POWERFUL 11-SEGMENT TAIL (~50% TOTAL LENGTH) + DOUBLE-TO-SINGLE SERRATED FIN CREST
  // =========================================================================
  const tailPivots: THREE.Group[] = [];
  let tailParent: THREE.Group = pelvisRoot;

  // Authentic ~2.60m sweeping muscular crocodile tail!
  const tailSegLengths = [0.30, 0.29, 0.28, 0.27, 0.26, 0.25, 0.24, 0.22, 0.20, 0.18, 0.16];

  for (let i = 0; i < TAIL_SEGMENT_COUNT; i++) {
    const frac0 = i / TAIL_SEGMENT_COUNT;
    const frac1 = (i + 1) / TAIL_SEGMENT_COUNT;
    const segLen = tailSegLengths[i];

    const pivot = new THREE.Group();
    pivot.name = `Joint_Tail_${i}`;
    if (i === 0) {
      pivot.position.set(0, -0.016, -0.18);
    } else {
      pivot.position.set(0, -0.004, -tailSegLengths[i - 1] * 0.94);
    }
    tailParent.add(pivot);
    tailPivots.push(pivot);

    // Muscular tail base blending smoothly from pelvis into a long, laterally compressed swimming blade!
    const rxAt = (f: number) => 0.268 * Math.pow(1 - f, 1.32) + 0.012;
    const ryAt = (f: number) => 0.222 * Math.pow(1 - f, 0.94) + 0.016;
    const uvAt = (f: number) => 0.44 * (1 - f);

    const overlapStart = 0.055 * (1 - frac0 * 0.65);
    const isLast = i === TAIL_SEGMENT_COUNT - 1;
    const rings: LoftRingSpec[] = [
      {
        z: overlapStart,
        yOffset: 0,
        rx: rxAt(Math.max(0, frac0 - 0.02)),
        ry: ryAt(Math.max(0, frac0 - 0.02)),
        flatTop: 0.20 * (1 - frac0),
        flatBottom: 0.34 * (1 - frac0 * 0.6),
        dorsalRidge: 0.014 * frac0,
        lowerBulge: 0.12 * (1 - frac0),
        uvV: uvAt(Math.max(0, frac0 - 0.015)),
      },
      {
        z: -segLen * 0.48,
        yOffset: -0.004,
        rx: rxAt((frac0 + frac1) * 0.5),
        ry: ryAt((frac0 + frac1) * 0.5),
        flatTop: 0.16 * (1 - (frac0 + frac1) * 0.5),
        flatBottom: 0.30 * (1 - (frac0 + frac1) * 0.3),
        dorsalRidge: 0.018 * ((frac0 + frac1) * 0.5),
        lowerBulge: 0.09 * (1 - (frac0 + frac1) * 0.5),
        uvV: uvAt((frac0 + frac1) * 0.5),
      },
      {
        z: -segLen * (isLast ? 1.05 : 1.02),
        yOffset: -0.006,
        rx: isLast ? 0.009 : rxAt(frac1),
        ry: isLast ? 0.016 : ryAt(frac1),
        flatTop: 0.14 * (1 - frac1),
        flatBottom: 0.24 * (1 - frac1 * 0.5),
        dorsalRidge: isLast ? 0 : 0.020 * frac1,
        lowerBulge: 0.05 * (1 - frac1),
        uvV: uvAt(frac1),
      },
    ];

    const segGeo = buildLoftedGeometry(rings, {
      radialSegments: Math.max(14, Math.round(24 - frac0 * 8)),
      capStart: true,
      capEnd: true,
      bumpAmp: 0.005 * (1 - frac0 * 0.6),
      bumpFreq: 20,
      bakeVertexAO: true,
    });
    const segMesh = new THREE.Mesh(segGeo, bodySkinMat);
    segMesh.name = `TailSegment_${i}`;
    segMesh.castShadow = true;
    segMesh.receiveShadow = true;
    pivot.add(segMesh);

    // DRAMATIC DOUBLE-TO-SINGLE SERRATED CAUDAL FIN CRESTS!
    const scuteCount = i < 7 ? 2 : i < 10 ? 2 : 1;
    for (let k = 0; k < scuteCount; k++) {
      const subT = scuteCount === 1 ? 0.45 : (k + 0.45) / scuteCount;
      const fLocal = frac0 + (frac1 - frac0) * subT;
      const zPos = -segLen * subT;
      const topY = ryAt(fLocal) * (1 - 0.15 * (1 - fLocal)) - 0.010;

      if (i <= 5) {
        const converge = i / 5.4;
        const xSpread = 0.112 * (1 - converge * 0.80);
        const finHeight = 0.068 + Math.sin(fLocal * Math.PI * 1.15) * 0.062;
        const finLen = segLen * 0.58;
        const finThick = 0.032 * (1 - fLocal * 0.35);

        for (const side of [-1, 1]) {
          const finGeo = buildTailFinScuteGeometry(finLen, finThick, finHeight, side * (0.26 - converge * 0.18));
          const finMesh = new THREE.Mesh(finGeo, scuteMat);
          finMesh.position.set(side * xSpread, topY, zPos);
          finMesh.castShadow = true;
          finMesh.receiveShadow = true;
          pivot.add(finMesh);
        }
      } else {
        const tailTipFade = 1 - Math.pow((fLocal - 0.52) / 0.48, 1.50);
        const finHeight = Math.max(0.028, 0.125 * Math.max(0.18, tailTipFade));
        const finLen = segLen * 0.60;
        const finThick = 0.028 * Math.max(0.35, tailTipFade);
        const finGeo = buildTailFinScuteGeometry(finLen, finThick, finHeight, 0);
        const finMesh = new THREE.Mesh(finGeo, scuteMat);
        finMesh.position.set(0, topY, zPos);
        finMesh.castShadow = true;
        finMesh.receiveShadow = true;
        pivot.add(finMesh);
      }
    }

    tailParent = pivot;
  }

  // =========================================================================
  // 4. FOUR MUSCULAR SPRAWLING LIMBS (Anchored snugly to sleek torso)
  // =========================================================================
  const fl = createReptileLimb({
    id: 'FL',
    sideSign: 1,
    isFront: true,
    limbMat: limbSkinMat,
    clawMat,
    webMat,
    shadowMat: contactShadowMat,
  });
  fl.hipPivot.position.set(0.268, -0.062, 0.21);
  chestPivot.add(fl.hipPivot);

  const fr = createReptileLimb({
    id: 'FR',
    sideSign: -1,
    isFront: true,
    limbMat: limbSkinMat,
    clawMat,
    webMat,
    shadowMat: contactShadowMat,
  });
  fr.hipPivot.position.set(-0.268, -0.062, 0.21);
  chestPivot.add(fr.hipPivot);

  const bl = createReptileLimb({
    id: 'BL',
    sideSign: 1,
    isFront: false,
    limbMat: limbSkinMat,
    clawMat,
    webMat,
    shadowMat: contactShadowMat,
  });
  bl.hipPivot.position.set(0.242, -0.048, 0.01);
  pelvisRoot.add(bl.hipPivot);

  const br = createReptileLimb({
    id: 'BR',
    sideSign: -1,
    isFront: false,
    limbMat: limbSkinMat,
    clawMat,
    webMat,
    shadowMat: contactShadowMat,
  });
  br.hipPivot.position.set(-0.242, -0.048, 0.01);
  pelvisRoot.add(br.hipPivot);

  const limbs: Record<LimbId, LimbRig> = { FL: fl, FR: fr, BL: bl, BR: br };

  const torsoShadowGeo = new THREE.PlaneGeometry(1.75, 4.15);
  torsoShadowGeo.rotateX(-Math.PI / 2);
  const torsoShadow = new THREE.Mesh(torsoShadowGeo, contactShadowMat);
  torsoShadow.position.set(0, 0.006, 0.15);
  torsoShadow.renderOrder = 1;
  root.add(torsoShadow);
  root.add(fl.footShadow, fr.footShadow, bl.footShadow, br.footShadow);

  // =========================================================================
  // 5. SKELETON / JOINTS DEBUG VISUALIZER
  // =========================================================================
  const skeletonDebugGroup = new THREE.Group();
  skeletonDebugGroup.name = 'SkeletonDebugGroup';
  skeletonDebugGroup.visible = false;
  root.add(skeletonDebugGroup);

  const bonePairs: Array<[THREE.Object3D, THREE.Object3D]> = [
    [pelvisRoot, midTorsoPivot],
    [midTorsoPivot, chestPivot],
    [chestPivot, neck1Pivot],
    [neck1Pivot, neck2Pivot],
    [neck2Pivot, neck3Pivot],
    [neck3Pivot, headPivot],
    [headPivot, jawPivot],
    [pelvisRoot, tailPivots[0]],
    ...tailPivots.slice(0, -1).map((tp, idx): [THREE.Object3D, THREE.Object3D] => [tp, tailPivots[idx + 1]]),
    [chestPivot, fl.hipPivot],
    [fl.hipPivot, fl.kneePivot],
    [fl.kneePivot, fl.anklePivot],
    [chestPivot, fr.hipPivot],
    [fr.hipPivot, fr.kneePivot],
    [fr.kneePivot, fr.anklePivot],
    [pelvisRoot, bl.hipPivot],
    [bl.hipPivot, bl.kneePivot],
    [bl.kneePivot, bl.anklePivot],
    [pelvisRoot, br.hipPivot],
    [br.hipPivot, br.kneePivot],
    [br.kneePivot, br.anklePivot],
  ];

  const jointNodesSet = new Set<THREE.Object3D>();
  for (const [a, b] of bonePairs) {
    jointNodesSet.add(a);
    jointNodesSet.add(b);
  }
  const skeletonJointNodes = Array.from(jointNodesSet);

  const jointGeo = new THREE.SphereGeometry(0.036, 10, 8);
  const jointMat = new THREE.MeshBasicMaterial({
    color: 0xf59e0b,
    depthTest: false,
    transparent: true,
    opacity: 0.92,
  });
  const jointSpheres: THREE.Mesh[] = skeletonJointNodes.map(() => {
    const m = new THREE.Mesh(jointGeo, jointMat);
    m.renderOrder = 999;
    skeletonDebugGroup.add(m);
    return m;
  });

  const boneLinePositions = new Float32Array(bonePairs.length * 2 * 3);
  const boneLineGeo = new THREE.BufferGeometry();
  boneLineGeo.setAttribute('position', new THREE.BufferAttribute(boneLinePositions, 3));
  const boneLineMat = new THREE.LineBasicMaterial({
    color: 0x38bdf8,
    depthTest: false,
    transparent: true,
    opacity: 0.9,
  });
  const skeletonBonesLine = new THREE.LineSegments(boneLineGeo, boneLineMat);
  skeletonBonesLine.renderOrder = 998;
  skeletonBonesLine.frustumCulled = false;
  skeletonDebugGroup.add(skeletonBonesLine);

  // =========================================================================
  // 6. REAL-TIME POSE APPLICATION, DUAL-LAYER BLINK & Y=0 FOOT GROUND-LOCK
  // =========================================================================
  const applyPose = (pose: CrocPose, wetness: number, showShadows: boolean): void => {
    root.position.set(pose.posX, 0, pose.posZ);
    root.rotation.y = pose.heading;

    pelvisRoot.position.y = pose.pelvisY;
    pelvisRoot.rotation.set(pose.pelvisPitch, 0, pose.pelvisRoll, 'YXZ');

    midTorsoPivot.rotation.set(pose.anteriorPitch[0], pose.anteriorYaw[0], 0, 'YXZ');
    chestPivot.rotation.set(pose.anteriorPitch[1], pose.anteriorYaw[1], 0, 'YXZ');
    neck1Pivot.rotation.set(pose.anteriorPitch[2], pose.anteriorYaw[2], 0, 'YXZ');
    neck2Pivot.rotation.set(pose.anteriorPitch[3], pose.anteriorYaw[3], 0, 'YXZ');
    neck3Pivot.rotation.set(pose.anteriorPitch[4], pose.anteriorYaw[4], 0, 'YXZ');
    headPivot.rotation.set(pose.anteriorPitch[5], pose.anteriorYaw[5], 0, 'YXZ');

    ribcageMesh.scale.set(pose.breathScale, 1 + (pose.breathScale - 1) * 0.78, 1);
    throatMesh.scale.set(1, 1 + pose.throatPulse, 1);

    jawPivot.rotation.x = pose.jawAngleRad;

    // Dual-layer authentic crocodile eye blink:
    // 1) Outer scaly upper eyelid drops smoothly downward over the golden iris (right-hand rule around +Z: +open -> -closed for +X!)
    // 2) Translucent milky-teal nictitating membrane sweeps rapidly across cornea ahead of lid!
    for (const rig of eyeRigs) {
      const s = rig.sideSign;
      // Open: +0.88*s (tucked cleanly inside brow ridge); Closed: -0.46*s (fully covering iris!)
      rig.upperLid.rotation.z = s * (0.88 - pose.blinkAmount * 1.34);
      const nictSweep = Math.min(1, pose.blinkAmount * 1.35);
      rig.nictitating.rotation.y =
        s > 0
          ? 1.25 - nictSweep * 1.43
          : Math.PI - 1.25 + nictSweep * 1.43;
      rig.nictitating.visible = pose.blinkAmount > 0.04;
    }

    for (let i = 0; i < tailPivots.length; i++) {
      tailPivots[i].rotation.set(pose.tailPitch[i], pose.tailYaw[i], 0, 'YXZ');
    }

    const limbList = [fl, fr, bl, br];
    for (const limb of limbList) {
      const a = pose.limbs[limb.id];
      limb.hipPivot.rotation.set(a.hipPitch, a.hipYaw, a.hipRoll, 'YXZ');
      limb.kneePivot.rotation.set(
        limb.isFront ? -0.14 : 0.18,
        a.kneeYaw,
        a.kneeBend,
        'YXZ'
      );
      limb.anklePivot.rotation.set(a.anklePitch, 0, a.ankleRoll, 'YXZ');
    }

    root.updateMatrixWorld(true);

    for (const limb of limbList) {
      const a = pose.limbs[limb.id];
      if (pose.limbTuck < 0.85) {
        const targetY = a.footLiftY;
        for (let pass = 0; pass < 2; pass++) {
          limb.footSoleMarker.getWorldPosition(_worldPos);
          const yError = _worldPos.y - targetY;
          if (Math.abs(yError) <= 0.002) break;
          const corr = THREE.MathUtils.clamp(-yError * 2.1, -0.55, 0.55) * (1 - pose.limbTuck);
          limb.hipPivot.rotation.z += limb.sideSign * corr;
          limb.kneePivot.rotation.z -= limb.sideSign * corr;
          limb.hipPivot.updateMatrixWorld(true);
        }

        const flatWeight = (1 - pose.limbTuck) * (0.55 + 0.45 * a.groundContact);
        if (flatWeight > 0.05) {
          limb.kneePivot.getWorldQuaternion(_worldQuat);
          _invParentQuat.copy(_worldQuat).invert();
          const toeOut = limb.sideSign * (limb.isFront ? 0.24 : 0.30);
          _euler.set(0, pose.heading + toeOut, 0, 'YXZ');
          _flatQuat.setFromEuler(_euler);
          _flatQuat.premultiply(_invParentQuat);
          limb.anklePivot.quaternion.slerp(_flatQuat, flatWeight);
          limb.anklePivot.updateMatrixWorld(true);

          limb.footSoleMarker.getWorldPosition(_worldPos);
          const finalErr = _worldPos.y - targetY;
          if (Math.abs(finalErr) > 0.002) {
            const c2 = THREE.MathUtils.clamp(-finalErr * 1.8, -0.25, 0.25) * (1 - pose.limbTuck);
            limb.hipPivot.rotation.z += limb.sideSign * c2;
            limb.kneePivot.rotation.z -= limb.sideSign * c2;
            limb.hipPivot.updateMatrixWorld(true);
          }
        }
      }

      limb.footShadow.visible = showShadows;
      if (showShadows) {
        limb.footSoleMarker.getWorldPosition(_worldPos);
        root.worldToLocal(_localPos.copy(_worldPos));
        limb.footShadow.position.set(_localPos.x, 0.008, _localPos.z);
        const heightFade = THREE.MathUtils.clamp(1 - Math.max(0, _worldPos.y) * 3.8, 0.08, 1);
        (limb.footShadow.material as THREE.MeshBasicMaterial).opacity =
          0.56 * heightFade * (1 - pose.limbTuck * 0.85);
      }
    }

    torsoShadow.visible = showShadows;
    if (showShadows) {
      const liftFade = THREE.MathUtils.clamp(1.18 - pose.pelvisY * 0.72, 0.25, 0.68);
      contactShadowMat.opacity = liftFade * (1 - pose.limbTuck * 0.65);
    }

    const dryRough = 0.94 - wetness * 0.56;
    const scuteRough = 0.62 - wetness * 0.40;
    bodySkinMat.roughness = dryRough;
    limbSkinMat.roughness = dryRough;
    scuteMat.roughness = scuteRough;

    if (skeletonDebugGroup.visible) {
      root.updateMatrixWorld(true);
      for (let i = 0; i < skeletonJointNodes.length; i++) {
        skeletonJointNodes[i].getWorldPosition(_worldPos);
        root.worldToLocal(_localPos.copy(_worldPos));
        jointSpheres[i].position.copy(_localPos);
      }
      const posAttr = boneLineGeo.attributes.position as THREE.BufferAttribute;
      for (let i = 0; i < bonePairs.length; i++) {
        bonePairs[i][0].getWorldPosition(_worldPos);
        bonePairs[i][1].getWorldPosition(_worldPosB);
        root.worldToLocal(_worldPos);
        root.worldToLocal(_worldPosB);
        posAttr.setXYZ(i * 2, _worldPos.x, _worldPos.y, _worldPos.z);
        posAttr.setXYZ(i * 2 + 1, _worldPosB.x, _worldPosB.y, _worldPosB.z);
      }
      posAttr.needsUpdate = true;
    }
  };

  const setWireframe = (enabled: boolean): void => {
    for (const mat of allCrocMaterials) {
      if ('wireframe' in mat) {
        (mat as THREE.MeshStandardMaterial).wireframe = enabled;
      }
    }
  };

  const setSkeletonVisible = (visible: boolean): void => {
    skeletonDebugGroup.visible = visible;
  };

  const dispose = (): void => {
    root.traverse((obj) => {
      if (obj instanceof THREE.Mesh || obj instanceof THREE.LineSegments) {
        obj.geometry.dispose();
        if (Array.isArray(obj.material)) {
          obj.material.forEach((m) => m.dispose());
        } else {
          obj.material.dispose();
        }
      }
    });
  };

  return {
    root,
    modelRoot,
    pelvisRoot,
    midTorsoPivot,
    chestPivot,
    neckPivots: [neck1Pivot, neck2Pivot, neck3Pivot],
    headPivot,
    jawPivot,
    tailPivots,
    eyeRigs,
    ribcageMesh,
    throatMesh,
    limbs,
    torsoShadow,
    skeletonDebugGroup,
    skeletonBonesLine,
    skeletonJointNodes,
    skinMaterials,
    allCrocMaterials,
    applyPose,
    setWireframe,
    setSkeletonVisible,
    dispose,
  };
}
