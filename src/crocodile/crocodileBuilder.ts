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
  eyelids: THREE.Mesh[];
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
 * Helper to create a smooth anatomical torso/neck segment spanning along +Z from zStart to zEnd.
 */
function createSpineSegmentMesh(
  rings: LoftRingSpec[],
  material: THREE.Material,
  radialSegments = 26,
  bumpAmp = 0.0055
): THREE.Mesh {
  const geo = buildLoftedGeometry(rings, {
    radialSegments,
    capStart: true,
    capEnd: true,
    bumpAmp,
    bumpFreq: 16,
    bakeVertexAO: true,
  });
  const mesh = new THREE.Mesh(geo, material);
  mesh.castShadow = true;
  mesh.receiveShadow = true;
  return mesh;
}

/**
 * Builds sculpted upper skull & festooned V/U-shaped snout with raised supraorbital brow ridges,
 * post-orbital cheek muscle mass, recessed eye sockets, and raised tip nasal boss.
 */
function buildUpperSkullGeometry(): THREE.BufferGeometry {
  // Local Z runs from -0.09 (occipital neck overlap) to +0.76 (rounded snout tip)
  const rings: LoftRingSpec[] = [
    // Occipital collar overlapping neck3
    { z: -0.10, yOffset: 0.015, rx: 0.185, ry: 0.115, flatTop: 0.30, flatBottom: 0.55, lowerBulge: 0.12, uvV: 0.80 },
    // Heavy post-orbital cheek / pterygoid jaw muscle flare (gives crocodile its broad triangular head!)
    { z: -0.01, yOffset: 0.022, rx: 0.228, ry: 0.128, flatTop: 0.38, flatBottom: 0.65, lowerBulge: 0.20, uvV: 0.83 },
    { z: 0.08, yOffset: 0.026, rx: 0.222, ry: 0.126, flatTop: 0.42, flatBottom: 0.72, lowerBulge: 0.18, uvV: 0.855 },
    // Supraorbital brow ridges & eye sockets!
    {
      z: 0.16,
      yOffset: 0.024,
      rx: 0.192,
      ry: 0.115,
      flatTop: 0.35,
      flatBottom: 0.78,
      lowerBulge: 0.12,
      uvV: 0.878,
      deform: (theta, x, y, z) => {
        const cosT = Math.cos(theta);
        const absSin = Math.abs(Math.sin(theta));
        // Prominent supraorbital brow domes at +/-40 deg off dorsal midline
        const browPeak = Math.exp(-Math.pow((absSin - 0.54) * 4.5, 2)) * Math.max(0, cosT);
        // Subtle recessed trough between the two eye domes (interorbital valley)
        const centerDip = Math.exp(-Math.pow(absSin * 4.2, 2)) * Math.max(0, cosT);
        // Recessed eye socket lateral indentation just below brow
        const socketDip = Math.exp(-Math.pow((absSin - 0.82) * 5.0, 2)) * Math.max(0, cosT + 0.15);
        return [x - Math.sign(x) * socketDip * 0.016, y + browPeak * 0.044 - centerDip * 0.014, z];
      },
    },
    // Anterior orbital rim sloping cleanly down into the elongated snout bridge
    {
      z: 0.25,
      yOffset: 0.012,
      rx: 0.168,
      ry: 0.094,
      flatTop: 0.30,
      flatBottom: 0.82,
      lowerBulge: 0.08,
      uvV: 0.90,
      deform: (theta, x, y, z) => {
        const cosT = Math.cos(theta);
        const absSin = Math.abs(Math.sin(theta));
        const lacrimalRidge = Math.exp(-Math.pow((absSin - 0.52) * 4.2, 2)) * Math.max(0, cosT);
        return [x, y + lacrimalRidge * 0.020, z];
      },
    },
    // Mid-snout festooned maxillary wave (broadens slightly over large 5th maxillary tooth)
    {
      z: 0.38,
      yOffset: 0.008,
      rx: 0.148,
      ry: 0.080,
      flatTop: 0.25,
      flatBottom: 0.84,
      lowerBulge: 0.10,
      uvV: 0.928,
      deform: (theta, x, y, z) => {
        const cosT = Math.cos(theta);
        // Slight festoon downward dip along upper lip edge
        const lipZone = Math.exp(-Math.pow((cosT + 0.25) * 3.0, 2));
        return [x, y - lipZone * 0.012, z];
      },
    },
    // Snout constriction notch (accommodates the iconic exposed 4th lower mandibular tooth!)
    {
      z: 0.52,
      yOffset: 0.012,
      rx: 0.118,
      ry: 0.068,
      flatTop: 0.22,
      flatBottom: 0.82,
      lowerBulge: 0.04,
      uvV: 0.955,
      deform: (theta, x, y, z) => {
        const cosT = Math.cos(theta);
        // Upward scalloped notch on lip margin so 4th lower tooth sits visibly outside!
        const notchZone = Math.exp(-Math.pow((cosT + 0.25) * 3.2, 2));
        return [x * (1 - notchZone * 0.06), y + notchZone * 0.012, z];
      },
    },
    // Premaxillary flared snout tip & raised dorsal nasal boss!
    {
      z: 0.65,
      yOffset: 0.014,
      rx: 0.124,
      ry: 0.066,
      flatTop: 0.14,
      flatBottom: 0.80,
      lowerBulge: 0.09,
      uvV: 0.978,
      deform: (theta, x, y, z) => {
        const cosT = Math.cos(theta);
        const absSin = Math.abs(Math.sin(theta));
        // Raised nasal boss dome right on snout tip
        const nasalBoss = Math.exp(-Math.pow(absSin * 2.4, 2)) * Math.max(0, cosT);
        return [x, y + nasalBoss * 0.028, z];
      },
    },
    // Rounded anterior snout rim
    { z: 0.73, yOffset: 0.008, rx: 0.088, ry: 0.048, flatTop: 0.15, flatBottom: 0.72, lowerBulge: 0.05, uvV: 0.992 },
    { z: 0.768, yOffset: 0.005, rx: 0.032, ry: 0.022, flatTop: 0.10, flatBottom: 0.50, lowerBulge: 0, uvV: 1.0 },
  ];

  return buildLoftedGeometry(rings, {
    radialSegments: 28,
    capStart: true,
    capEnd: true,
    bumpAmp: 0.0045,
    bumpFreq: 24,
    bakeVertexAO: true,
  });
}

/**
 * Builds the independent articulated lower jaw (mandible) hinged at jawPivot (local z=0).
 */
function buildLowerJawGeometry(): THREE.BufferGeometry {
  const rings: LoftRingSpec[] = [
    // Posterior articular jaw hinge & heavy pterygoideus throat transition
    { z: -0.07, yOffset: -0.025, rx: 0.185, ry: 0.075, flatTop: 0.72, flatBottom: 0.30, lowerBulge: 0.14, uvV: 0.82 },
    { z: 0.04, yOffset: -0.026, rx: 0.205, ry: 0.078, flatTop: 0.78, flatBottom: 0.34, lowerBulge: 0.16, uvV: 0.845 },
    { z: 0.16, yOffset: -0.022, rx: 0.172, ry: 0.068, flatTop: 0.80, flatBottom: 0.38, lowerBulge: 0.10, uvV: 0.875 },
    { z: 0.31, yOffset: -0.014, rx: 0.136, ry: 0.056, flatTop: 0.80, flatBottom: 0.40, lowerBulge: 0.06, uvV: 0.91 },
    { z: 0.46, yOffset: -0.008, rx: 0.114, ry: 0.048, flatTop: 0.78, flatBottom: 0.42, lowerBulge: 0.05, uvV: 0.945 },
    { z: 0.58, yOffset: -0.005, rx: 0.100, ry: 0.042, flatTop: 0.76, flatBottom: 0.40, lowerBulge: 0.04, uvV: 0.972 },
    { z: 0.67, yOffset: -0.004, rx: 0.070, ry: 0.032, flatTop: 0.72, flatBottom: 0.35, lowerBulge: 0.02, uvV: 0.99 },
    { z: 0.705, yOffset: -0.003, rx: 0.025, ry: 0.015, flatTop: 0.50, flatBottom: 0.25, lowerBulge: 0, uvV: 1.0 },
  ];
  return buildLoftedGeometry(rings, {
    radialSegments: 24,
    capStart: true,
    capEnd: true,
    bumpAmp: 0.0032,
    bumpFreq: 26,
    bakeVertexAO: true,
  });
}

/**
 * Builds the fleshy interior tongue mesh inside the lower jaw.
 */
function buildTongueGeometry(): THREE.BufferGeometry {
  const rings: LoftRingSpec[] = [
    { z: 0.06, yOffset: 0.006, rx: 0.088, ry: 0.022, flatTop: 0.45, flatBottom: 0.6, lowerBulge: 0, uvV: 0.2 },
    { z: 0.22, yOffset: 0.005, rx: 0.072, ry: 0.018, flatTop: 0.45, flatBottom: 0.6, lowerBulge: 0, uvV: 0.5 },
    { z: 0.42, yOffset: 0.004, rx: 0.052, ry: 0.014, flatTop: 0.40, flatBottom: 0.55, lowerBulge: 0, uvV: 0.8 },
    { z: 0.55, yOffset: 0.003, rx: 0.024, ry: 0.009, flatTop: 0.30, flatBottom: 0.5, lowerBulge: 0, uvV: 1.0 },
  ];
  return buildLoftedGeometry(rings, { radialSegments: 14, capStart: true, capEnd: true, bumpAmp: 0.001, bumpFreq: 30 });
}

/**
 * Builds the pale fleshy upper palate lining tucked cleanly inside the roof of the upper skull.
 */
function buildUpperPalateGeometry(): THREE.BufferGeometry {
  const rings: LoftRingSpec[] = [
    { z: 0.06, yOffset: -0.004, rx: 0.112, ry: 0.016, flatTop: 0.5, flatBottom: 0.6, lowerBulge: 0, uvV: 0.2 },
    { z: 0.25, yOffset: -0.002, rx: 0.090, ry: 0.014, flatTop: 0.5, flatBottom: 0.6, lowerBulge: 0, uvV: 0.5 },
    { z: 0.48, yOffset: 0.000, rx: 0.062, ry: 0.011, flatTop: 0.5, flatBottom: 0.6, lowerBulge: 0, uvV: 0.8 },
    { z: 0.65, yOffset: 0.002, rx: 0.036, ry: 0.008, flatTop: 0.5, flatBottom: 0.6, lowerBulge: 0, uvV: 1.0 },
  ];
  return buildLoftedGeometry(rings, { radialSegments: 14, capStart: true, capEnd: true, bumpAmp: 0.001, bumpFreq: 20 });
}

/**
 * Constructs a multi-layered realistic reptile eye (golden-amber vertical-slit iris,
 * glossy wet cornea dome, sculpted scaly eyelid rims, and animated blink lid).
 */
function createReptileEye(
  sideSign: number,
  irisMat: THREE.Material,
  corneaMat: THREE.Material,
  skinMat: THREE.Material,
  eyelidsOut: THREE.Mesh[]
): THREE.Group {
  const eyeGroup = new THREE.Group();
  eyeGroup.name = sideSign > 0 ? 'Eye_L' : 'Eye_R';

  // Inner eyeball: in THREE.SphereGeometry, UV (0.5, 0.5) faces +X at rotation.y = 0!
  // Point iris cleanly outward along sideSign*X and slightly forward (+Z).
  const eyeballGeo = new THREE.SphereGeometry(0.027, 20, 16);
  const eyeball = new THREE.Mesh(eyeballGeo, irisMat);
  eyeball.rotation.y = sideSign > 0 ? -0.22 : Math.PI + 0.22;
  eyeGroup.add(eyeball);

  // Glossy wet cornea outer dome (clearcoat specular highlight)
  const corneaGeo = new THREE.SphereGeometry(0.029, 20, 16);
  const cornea = new THREE.Mesh(corneaGeo, corneaMat);
  cornea.renderOrder = 2;
  eyeGroup.add(cornea);

  // Sculpted upper supraorbital eyelid hood blended tightly into cranial brow
  const upperHoodGeo = new THREE.SphereGeometry(0.034, 18, 12, 0, Math.PI * 2, 0, Math.PI * 0.44);
  const upperHood = new THREE.Mesh(upperHoodGeo, skinMat);
  upperHood.scale.set(1.06, 0.84, 1.32);
  upperHood.position.set(-sideSign * 0.005, 0.004, 0);
  upperHood.rotation.z = sideSign * -0.14;
  upperHood.castShadow = true;
  eyeGroup.add(upperHood);

  // Lower scaly eyelid rim
  const lowerRimGeo = new THREE.SphereGeometry(0.033, 18, 10, 0, Math.PI * 2, Math.PI * 0.65, Math.PI * 0.35);
  const lowerRim = new THREE.Mesh(lowerRimGeo, skinMat);
  lowerRim.scale.set(1.04, 0.85, 1.24);
  lowerRim.rotation.z = sideSign * 0.10;
  eyeGroup.add(lowerRim);

  // Animated nictitating / upper blink eyelid shell
  const blinkGeo = new THREE.SphereGeometry(0.0305, 16, 10, 0, Math.PI * 2, 0, Math.PI * 0.50);
  const blinkLid = new THREE.Mesh(blinkGeo, skinMat);
  blinkLid.scale.set(1.02, 1.0, 1.08);
  blinkLid.rotation.z = sideSign * -0.78;
  blinkLid.userData.sideSign = sideSign;
  eyeGroup.add(blinkLid);
  eyelidsOut.push(blinkLid);

  return eyeGroup;
}

/**
 * Constructs a single sprawling articulated reptilian limb (3-joint hierarchy):
 * Hip/Shoulder -> Elbow/Knee -> Ankle/Foot + splayed toes (5 front, 4 webbed hind) + curved claws.
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

  const upperLen = isFront ? 0.32 : 0.36;
  const lowerLen = isFront ? 0.32 : 0.35;

  // 1. UPPER LIMB (Humerus / Femur): extends along local -Y inside hipPivot, muscular flare at shoulder/thigh!
  // We loft along +Z then rotateX(Math.PI/2) so +Z maps cleanly onto -Y (downward from joint).
  const upperBaseR = isFront ? 0.115 : 0.145;
  const upperMidR = isFront ? 0.102 : 0.132;
  const upperDistalR = isFront ? 0.082 : 0.098;
  const upperRings: LoftRingSpec[] = [
    { z: -0.05, yOffset: 0, rx: upperBaseR * 0.92, ry: upperBaseR * 1.05, flatTop: 0.1, flatBottom: 0.1, lowerBulge: 0.08, uvV: 0.0 },
    { z: 0.03, yOffset: 0.008, rx: upperBaseR * 1.04, ry: upperBaseR * 1.12, flatTop: 0.1, flatBottom: 0.1, lowerBulge: 0.10, uvV: 0.14 },
    { z: upperLen * 0.45, yOffset: 0.005, rx: upperMidR, ry: upperMidR * 1.06, flatTop: 0.12, flatBottom: 0.12, lowerBulge: 0.08, uvV: 0.34 },
    { z: upperLen * 0.85, yOffset: 0, rx: upperDistalR * 1.04, ry: upperDistalR * 1.02, flatTop: 0.12, flatBottom: 0.12, lowerBulge: 0.04, uvV: 0.48 },
    { z: upperLen + 0.04, yOffset: 0, rx: upperDistalR * 0.85, ry: upperDistalR * 0.85, flatTop: 0.08, flatBottom: 0.08, lowerBulge: 0, uvV: 0.54 },
  ];
  const upperGeo = buildLoftedGeometry(upperRings, { radialSegments: 18, bumpAmp: 0.004, bumpFreq: 24 });
  upperGeo.rotateX(Math.PI / 2); // +Z -> -Y
  const upperMesh = new THREE.Mesh(upperGeo, limbMat);
  upperMesh.castShadow = true;
  upperMesh.receiveShadow = true;
  hipPivot.add(upperMesh);

  // 2. ELBOW / KNEE PIVOT
  const kneePivot = new THREE.Group();
  kneePivot.name = `Joint_Knee_${id}`;
  kneePivot.position.set(0, -upperLen, 0);
  hipPivot.add(kneePivot);

  // Lower limb (Forearm / Shin): thick muscular upper calf tapering into sturdy reptilian wrist/ankle
  const lowerProxR = isFront ? 0.084 : 0.100;
  const lowerMidR = isFront ? 0.072 : 0.086;
  const lowerAnkleR = isFront ? 0.058 : 0.070;
  const lowerRings: LoftRingSpec[] = [
    { z: -0.04, yOffset: 0, rx: lowerProxR * 0.90, ry: lowerProxR * 0.92, flatTop: 0.1, flatBottom: 0.1, lowerBulge: 0.05, uvV: 0.50 },
    { z: 0.04, yOffset: 0.006, rx: lowerProxR * 1.04, ry: lowerProxR * 1.06, flatTop: 0.1, flatBottom: 0.1, lowerBulge: 0.08, uvV: 0.60 },
    { z: lowerLen * 0.48, yOffset: 0.003, rx: lowerMidR, ry: lowerMidR * 1.04, flatTop: 0.12, flatBottom: 0.12, lowerBulge: 0.06, uvV: 0.76 },
    { z: lowerLen * 0.86, yOffset: 0, rx: lowerAnkleR * 1.05, ry: lowerAnkleR * 0.95, flatTop: 0.15, flatBottom: 0.20, lowerBulge: 0.03, uvV: 0.90 },
    { z: lowerLen + 0.025, yOffset: 0, rx: lowerAnkleR * 0.92, ry: lowerAnkleR * 0.82, flatTop: 0.15, flatBottom: 0.25, lowerBulge: 0, uvV: 0.95 },
  ];
  const lowerGeo = buildLoftedGeometry(lowerRings, { radialSegments: 16, bumpAmp: 0.0035, bumpFreq: 26 });
  lowerGeo.rotateX(Math.PI / 2);
  const lowerMesh = new THREE.Mesh(lowerGeo, limbMat);
  lowerMesh.castShadow = true;
  lowerMesh.receiveShadow = true;
  kneePivot.add(lowerMesh);

  // 3. ANKLE / WRIST & PLANTED SPLAYED FOOT
  const anklePivot = new THREE.Group();
  anklePivot.name = `Joint_Ankle_${id}`;
  anklePivot.position.set(0, -lowerLen, 0);
  kneePivot.add(anklePivot);

  // Broad flat scaly foot pad (metatarsus / metacarpus)
  const padWidth = isFront ? 0.082 : 0.102;
  const padLen = isFront ? 0.125 : 0.155;
  const padRings: LoftRingSpec[] = [
    { z: -0.055, yOffset: -0.012, rx: padWidth * 0.68, ry: 0.040, flatTop: 0.2, flatBottom: 0.72, lowerBulge: 0.05, uvV: 0.85 },
    { z: 0.015, yOffset: -0.018, rx: padWidth * 0.95, ry: 0.034, flatTop: 0.25, flatBottom: 0.82, lowerBulge: 0.06, uvV: 0.92 },
    { z: padLen * 0.78, yOffset: -0.024, rx: padWidth * 1.05, ry: 0.025, flatTop: 0.28, flatBottom: 0.85, lowerBulge: 0.04, uvV: 0.97 },
    { z: padLen, yOffset: -0.026, rx: padWidth * 0.76, ry: 0.018, flatTop: 0.22, flatBottom: 0.80, lowerBulge: 0, uvV: 1.0 },
  ];
  const padGeo = buildLoftedGeometry(padRings, { radialSegments: 14, bumpAmp: 0.0022, bumpFreq: 32 });
  const padMesh = new THREE.Mesh(padGeo, limbMat);
  padMesh.castShadow = true;
  padMesh.receiveShadow = true;
  anklePivot.add(padMesh);

  // Splayed toes with curved dark keratin claws:
  // Front feet: 5 distinct splayed toes; Hind feet: 4 larger webbed toes!
  const toeSpecs = isFront
    ? [
        // [xOffset, zOffset, splayAngleRad, toeLength, baseR, clawLen]
        { x: -sideSign * 0.064, z: 0.058, yaw: -sideSign * 0.72, len: 0.095, r: 0.017, claw: 0.044 },
        { x: -sideSign * 0.034, z: 0.088, yaw: -sideSign * 0.34, len: 0.128, r: 0.019, claw: 0.052 },
        { x: 0.0, z: 0.098, yaw: sideSign * 0.02, len: 0.140, r: 0.020, claw: 0.054 },
        { x: sideSign * 0.036, z: 0.088, yaw: sideSign * 0.38, len: 0.122, r: 0.018, claw: 0.046 },
        { x: sideSign * 0.062, z: 0.058, yaw: sideSign * 0.78, len: 0.092, r: 0.016, claw: 0.038 },
      ]
    : [
        // 4 heavier hind toes with webbing between them
        { x: -sideSign * 0.062, z: 0.095, yaw: -sideSign * 0.56, len: 0.135, r: 0.022, claw: 0.062 },
        { x: -sideSign * 0.020, z: 0.120, yaw: -sideSign * 0.18, len: 0.168, r: 0.024, claw: 0.068 },
        { x: sideSign * 0.026, z: 0.118, yaw: sideSign * 0.22, len: 0.162, r: 0.023, claw: 0.062 },
        { x: sideSign * 0.066, z: 0.092, yaw: sideSign * 0.62, len: 0.128, r: 0.020, claw: 0.048 },
      ];

  const toeTipPositions: THREE.Vector3[] = [];

  for (const ts of toeSpecs) {
    const toeGroup = new THREE.Group();
    toeGroup.position.set(ts.x, -0.026, ts.z);
    toeGroup.rotation.y = ts.yaw;
    anklePivot.add(toeGroup);

    const toeGeo = buildReptileToeGeometry(ts.len, ts.r, ts.r * 0.58);
    const toeMesh = new THREE.Mesh(toeGeo, limbMat);
    toeMesh.castShadow = true;
    toeMesh.receiveShadow = true;
    toeGroup.add(toeMesh);

    const clawGeo = buildCurvedClawGeometry(ts.r * 0.56, ts.claw);
    const clawMesh = new THREE.Mesh(clawGeo, clawMat);
    clawMesh.position.set(0, -0.002, ts.len - 0.005);
    clawMesh.castShadow = true;
    toeGroup.add(clawMesh);

    // Record mid-toe position for hind-foot webbing membranes
    toeTipPositions.push(
      new THREE.Vector3(
        ts.x + Math.sin(ts.yaw) * ts.len * 0.68,
        -0.034,
        ts.z + Math.cos(ts.yaw) * ts.len * 0.68
      )
    );
  }

  // Interdigital webbing membranes on hind feet (between adjacent toes)
  if (!isFront && toeTipPositions.length === 4) {
    for (let i = 0; i < 3; i++) {
      const a = toeSpecs[i];
      const b = toeSpecs[i + 1];
      const tipA = toeTipPositions[i];
      const tipB = toeTipPositions[i + 1];
      const webGeo = new THREE.BufferGeometry();
      const verts = new Float32Array([
        a.x, -0.032, a.z,
        tipA.x, tipA.y, tipA.z,
        tipB.x, tipB.y, tipB.z,
        b.x, -0.032, b.z,
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

  // Foot sole ground-contact marker (exact bottom of foot pad)
  const footSoleMarker = new THREE.Object3D();
  footSoleMarker.position.set(0, -0.045, padLen * 0.55);
  anklePivot.add(footSoleMarker);

  // Per-foot soft radial studio contact shadow disc
  const footShadowGeo = new THREE.PlaneGeometry(isFront ? 0.50 : 0.58, isFront ? 0.54 : 0.62);
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
 * Builds the complete, anatomically articulated procedural Nile Crocodile (Crocodylus niloticus).
 */
export function buildNileCrocodile(textures: CrocodileTextureSet): CrocodileRig {
  const root = new THREE.Group();
  root.name = 'NileCrocodileRoot';

  const modelRoot = new THREE.Group();
  modelRoot.name = 'NileCrocodileModel';
  root.add(modelRoot);

  // --- PROCEDURAL PBR MATERIALS ---
  const bodySkinMat = new THREE.MeshStandardMaterial({
    color: 0xffffff,
    map: textures.bodyAlbedo,
    normalMap: textures.bodyNormal,
    normalScale: new THREE.Vector2(1.45, 1.45),
    roughnessMap: textures.bodyRoughness,
    roughness: 0.92,
    aoMap: textures.bodyAO,
    aoMapIntensity: 0.72,
    metalness: 0.02,
    vertexColors: true,
  });

  // Raised osteoderm scutes: darker weathered olive-charcoal with slightly lower roughness on ridges
  const scuteMat = new THREE.MeshStandardMaterial({
    color: new THREE.Color('#6a6e5e'),
    map: textures.bodyAlbedo,
    normalMap: textures.bodyNormal,
    normalScale: new THREE.Vector2(1.3, 1.3),
    roughnessMap: textures.bodyRoughness,
    roughness: 0.64,
    metalness: 0.04,
  });

  const limbSkinMat = new THREE.MeshStandardMaterial({
    color: 0xffffff,
    map: textures.limbAlbedo,
    normalMap: textures.limbNormal,
    normalScale: new THREE.Vector2(1.35, 1.35),
    roughnessMap: textures.limbRoughness,
    roughness: 0.90,
    metalness: 0.02,
  });

  const webMat = new THREE.MeshStandardMaterial({
    color: new THREE.Color('#4a4e42'),
    roughness: 0.80,
    metalness: 0.02,
    side: THREE.DoubleSide,
  });

  const mouthFleshMat = new THREE.MeshStandardMaterial({
    color: new THREE.Color('#d89b91'),
    roughness: 0.42,
    metalness: 0.01,
  });

  const tongueMat = new THREE.MeshStandardMaterial({
    color: new THREE.Color('#cc847b'),
    roughness: 0.36,
    metalness: 0.01,
  });

  const toothMat = new THREE.MeshPhysicalMaterial({
    color: new THREE.Color('#e6dfd0'),
    roughness: 0.24,
    metalness: 0.02,
    clearcoat: 0.35,
    clearcoatRoughness: 0.25,
  });

  const clawMat = new THREE.MeshStandardMaterial({
    color: new THREE.Color('#23221f'),
    roughness: 0.34,
    metalness: 0.06,
  });

  const eyeIrisMat = new THREE.MeshStandardMaterial({
    color: 0xffffff,
    map: textures.eyeIris,
    roughness: 0.28,
    metalness: 0.02,
  });

  const corneaMat = new THREE.MeshPhysicalMaterial({
    color: 0xffffff,
    roughness: 0.04,
    metalness: 0.0,
    clearcoat: 1.0,
    clearcoatRoughness: 0.04,
    transparent: true,
    opacity: 0.28,
    depthWrite: false,
  });

  const nostrilInteriorMat = new THREE.MeshStandardMaterial({
    color: new THREE.Color('#181915'),
    roughness: 0.72,
    metalness: 0.0,
  });

  const contactShadowMat = new THREE.MeshBasicMaterial({
    map: textures.contactShadow,
    transparent: true,
    opacity: 0.56,
    depthWrite: false,
  });

  const skinMaterials = [bodySkinMat, scuteMat, limbSkinMat];
  const allCrocMaterials: THREE.Material[] = [
    bodySkinMat,
    scuteMat,
    limbSkinMat,
    webMat,
    mouthFleshMat,
    tongueMat,
    toothMat,
    clawMat,
    eyeIrisMat,
    corneaMat,
    nostrilInteriorMat,
  ];

  // =========================================================================
  // 1. ARTICULATED SPINE CHAIN (Pelvis -> MidTorso -> Chest -> Neck x3 -> Head)
  // =========================================================================

  // --- PELVIS ROOT ---
  const pelvisRoot = new THREE.Group();
  pelvisRoot.name = 'Joint_Pelvis';
  pelvisRoot.position.set(0, 0.51, 0);
  modelRoot.add(pelvisRoot);

  const pelvisMesh = createSpineSegmentMesh(
    [
      { z: -0.22, yOffset: -0.01, rx: 0.255, ry: 0.182, flatTop: 0.28, flatBottom: 0.42, lowerBulge: 0.12, uvV: 0.435 },
      { z: -0.06, yOffset: 0.0, rx: 0.295, ry: 0.205, flatTop: 0.30, flatBottom: 0.46, lowerBulge: 0.15, uvV: 0.465 },
      { z: 0.10, yOffset: 0.01, rx: 0.318, ry: 0.218, flatTop: 0.32, flatBottom: 0.48, lowerBulge: 0.16, uvV: 0.495 },
      { z: 0.27, yOffset: 0.015, rx: 0.312, ry: 0.214, flatTop: 0.32, flatBottom: 0.48, lowerBulge: 0.15, uvV: 0.525 },
    ],
    bodySkinMat
  );
  pelvisMesh.name = 'PelvisMesh';
  pelvisRoot.add(pelvisMesh);

  // Pelvis dorsal osteoderms (4 rows + lateral flank bosses)
  const pelvisScutes = [];
  for (const z of [-0.12, 0.02, 0.15]) {
    pelvisScutes.push(
      { x: -0.068, y: 0.152, z, length: 0.12, width: 0.085, height: 0.044, roll: 0.12 },
      { x: 0.068, y: 0.152, z, length: 0.12, width: 0.085, height: 0.044, roll: -0.12 },
      { x: -0.165, y: 0.130, z: z - 0.01, length: 0.105, width: 0.075, height: 0.036, roll: 0.36 },
      { x: 0.165, y: 0.130, z: z - 0.01, length: 0.105, width: 0.075, height: 0.036, roll: -0.36 }
    );
  }
  addDorsalOsteoderms(pelvisRoot, pelvisScutes, scuteMat);

  // --- MID-TORSO (Ribcage / Barrel Abdomen) ---
  const midTorsoPivot = new THREE.Group();
  midTorsoPivot.name = 'Joint_MidTorso';
  midTorsoPivot.position.set(0, 0.01, 0.23);
  pelvisRoot.add(midTorsoPivot);

  const ribcageMesh = createSpineSegmentMesh(
    [
      { z: -0.08, yOffset: 0.004, rx: 0.315, ry: 0.215, flatTop: 0.32, flatBottom: 0.48, lowerBulge: 0.16, uvV: 0.52 },
      { z: 0.12, yOffset: 0.016, rx: 0.358, ry: 0.242, flatTop: 0.36, flatBottom: 0.52, lowerBulge: 0.20, uvV: 0.56 },
      { z: 0.32, yOffset: 0.018, rx: 0.362, ry: 0.246, flatTop: 0.36, flatBottom: 0.52, lowerBulge: 0.20, uvV: 0.60 },
      { z: 0.48, yOffset: 0.010, rx: 0.342, ry: 0.234, flatTop: 0.34, flatBottom: 0.50, lowerBulge: 0.18, uvV: 0.635 },
    ],
    bodySkinMat
  );
  ribcageMesh.name = 'MidTorsoMesh';
  midTorsoPivot.add(ribcageMesh);

  // Mid-torso dorsal osteoderm shield (6 rows: 2 inner, 2 mid, 2 outer + lateral flank tubercles!)
  const midScutes = [];
  for (const z of [0.02, 0.16, 0.30, 0.43]) {
    midScutes.push(
      { x: -0.068, y: 0.175, z, length: 0.128, width: 0.092, height: 0.048, roll: 0.10 },
      { x: 0.068, y: 0.175, z, length: 0.128, width: 0.092, height: 0.048, roll: -0.10 },
      { x: -0.175, y: 0.155, z: z - 0.01, length: 0.118, width: 0.082, height: 0.040, roll: 0.30 },
      { x: 0.175, y: 0.155, z: z - 0.01, length: 0.118, width: 0.082, height: 0.040, roll: -0.30 },
      { x: -0.275, y: 0.112, z: z + 0.02, length: 0.095, width: 0.068, height: 0.032, roll: 0.64 },
      { x: 0.275, y: 0.112, z: z + 0.02, length: 0.095, width: 0.068, height: 0.032, roll: -0.64 },
      // Lateral flank osteoderm bosses visible on the mottled sides in the reference!
      { x: -0.345, y: 0.042, z: z - 0.03, length: 0.072, width: 0.056, height: 0.026, roll: 1.08 },
      { x: 0.345, y: 0.042, z: z - 0.03, length: 0.072, width: 0.056, height: 0.026, roll: -1.08 }
    );
  }
  addDorsalOsteoderms(midTorsoPivot, midScutes, scuteMat);

  // --- CHEST / MUSCULAR SHOULDERS ---
  const chestPivot = new THREE.Group();
  chestPivot.name = 'Joint_Chest';
  chestPivot.position.set(0, 0.008, 0.44);
  midTorsoPivot.add(chestPivot);

  const chestMesh = createSpineSegmentMesh(
    [
      { z: -0.08, yOffset: 0.005, rx: 0.340, ry: 0.232, flatTop: 0.34, flatBottom: 0.50, lowerBulge: 0.18, uvV: 0.63 },
      { z: 0.10, yOffset: 0.002, rx: 0.335, ry: 0.228, flatTop: 0.34, flatBottom: 0.50, lowerBulge: 0.18, uvV: 0.665 },
      { z: 0.28, yOffset: -0.005, rx: 0.305, ry: 0.205, flatTop: 0.32, flatBottom: 0.48, lowerBulge: 0.15, uvV: 0.70 },
      { z: 0.44, yOffset: -0.010, rx: 0.268, ry: 0.178, flatTop: 0.30, flatBottom: 0.46, lowerBulge: 0.12, uvV: 0.728 },
    ],
    bodySkinMat
  );
  chestMesh.name = 'ChestMesh';
  chestPivot.add(chestMesh);

  const chestScutes = [];
  for (const z of [0.06, 0.20, 0.33]) {
    const taper = 1 - (z / 0.44) * 0.18;
    chestScutes.push(
      { x: -0.062 * taper, y: 0.158 * taper + 0.005, z, length: 0.122, width: 0.086, height: 0.046, roll: 0.11 },
      { x: 0.062 * taper, y: 0.158 * taper + 0.005, z, length: 0.122, width: 0.086, height: 0.046, roll: -0.11 },
      { x: -0.162 * taper, y: 0.138 * taper, z: z - 0.01, length: 0.110, width: 0.076, height: 0.038, roll: 0.34 },
      { x: 0.162 * taper, y: 0.138 * taper, z: z - 0.01, length: 0.110, width: 0.076, height: 0.038, roll: -0.34 },
      { x: -0.245 * taper, y: 0.095 * taper, z: z + 0.01, length: 0.086, width: 0.062, height: 0.028, roll: 0.66 },
      { x: 0.245 * taper, y: 0.095 * taper, z: z + 0.01, length: 0.086, width: 0.062, height: 0.028, roll: -0.66 }
    );
  }
  addDorsalOsteoderms(chestPivot, chestScutes, scuteMat);

  // --- NECK SEGMENTS (x3 articulated) + NUCHAL SHIELD & POST-OCCIPITAL SCUTES ---
  const neck1Pivot = new THREE.Group();
  neck1Pivot.name = 'Joint_Neck_1';
  neck1Pivot.position.set(0, -0.008, 0.38);
  chestPivot.add(neck1Pivot);

  const neck1Mesh = createSpineSegmentMesh(
    [
      { z: -0.06, yOffset: 0.0, rx: 0.268, ry: 0.178, flatTop: 0.28, flatBottom: 0.45, lowerBulge: 0.12, uvV: 0.725 },
      { z: 0.08, yOffset: 0.006, rx: 0.252, ry: 0.166, flatTop: 0.28, flatBottom: 0.44, lowerBulge: 0.12, uvV: 0.752 },
      { z: 0.20, yOffset: 0.010, rx: 0.238, ry: 0.156, flatTop: 0.28, flatBottom: 0.44, lowerBulge: 0.12, uvV: 0.774 },
    ],
    bodySkinMat,
    22
  );
  neck1Pivot.add(neck1Mesh);

  // Prominent Nuchal Shield (large raised bony osteoderms on the neck — iconic Nile croc trait!)
  addDorsalOsteoderms(
    neck1Pivot,
    [
      { x: -0.058, y: 0.128, z: 0.05, length: 0.118, width: 0.088, height: 0.054, roll: 0.10 },
      { x: 0.058, y: 0.128, z: 0.05, length: 0.118, width: 0.088, height: 0.054, roll: -0.10 },
      { x: -0.145, y: 0.106, z: 0.08, length: 0.095, width: 0.068, height: 0.042, roll: 0.36 },
      { x: 0.145, y: 0.106, z: 0.08, length: 0.095, width: 0.068, height: 0.042, roll: -0.36 },
    ],
    scuteMat
  );

  const neck2Pivot = new THREE.Group();
  neck2Pivot.name = 'Joint_Neck_2';
  neck2Pivot.position.set(0, 0.010, 0.16);
  neck1Pivot.add(neck2Pivot);

  const neck2Mesh = createSpineSegmentMesh(
    [
      { z: -0.05, yOffset: 0.0, rx: 0.236, ry: 0.156, flatTop: 0.28, flatBottom: 0.44, lowerBulge: 0.14, uvV: 0.768 },
      { z: 0.08, yOffset: 0.008, rx: 0.228, ry: 0.152, flatTop: 0.28, flatBottom: 0.44, lowerBulge: 0.14, uvV: 0.790 },
      { z: 0.18, yOffset: 0.012, rx: 0.220, ry: 0.146, flatTop: 0.28, flatBottom: 0.44, lowerBulge: 0.14, uvV: 0.808 },
    ],
    bodySkinMat,
    22
  );
  neck2Pivot.add(neck2Mesh);
  // Reference to neck2Mesh for subtle gular throat pulsing during idle breathing
  const throatMesh = neck2Mesh;

  // Front row of Nuchal Shield
  addDorsalOsteoderms(
    neck2Pivot,
    [
      { x: -0.054, y: 0.122, z: 0.05, length: 0.112, width: 0.084, height: 0.052, roll: 0.10 },
      { x: 0.054, y: 0.122, z: 0.05, length: 0.112, width: 0.084, height: 0.052, roll: -0.10 },
    ],
    scuteMat
  );

  const neck3Pivot = new THREE.Group();
  neck3Pivot.name = 'Joint_Neck_3';
  neck3Pivot.position.set(0, 0.012, 0.15);
  neck2Pivot.add(neck3Pivot);

  const neck3Mesh = createSpineSegmentMesh(
    [
      { z: -0.05, yOffset: 0.0, rx: 0.222, ry: 0.144, flatTop: 0.28, flatBottom: 0.46, lowerBulge: 0.15, uvV: 0.802 },
      { z: 0.06, yOffset: 0.008, rx: 0.214, ry: 0.136, flatTop: 0.30, flatBottom: 0.48, lowerBulge: 0.15, uvV: 0.818 },
      { z: 0.15, yOffset: 0.012, rx: 0.202, ry: 0.124, flatTop: 0.30, flatBottom: 0.50, lowerBulge: 0.14, uvV: 0.830 },
    ],
    bodySkinMat,
    22
  );
  neck3Pivot.add(neck3Mesh);

  // Post-occipital scutes (row of 4 compact keeled plates right behind the skull)
  addDorsalOsteoderms(
    neck3Pivot,
    [
      { x: -0.044, y: 0.106, z: 0.05, length: 0.072, width: 0.056, height: 0.036, roll: 0.08 },
      { x: 0.044, y: 0.106, z: 0.05, length: 0.072, width: 0.056, height: 0.036, roll: -0.08 },
      { x: -0.112, y: 0.094, z: 0.04, length: 0.064, width: 0.050, height: 0.030, roll: 0.28 },
      { x: 0.112, y: 0.094, z: 0.04, length: 0.064, width: 0.050, height: 0.030, roll: -0.28 },
    ],
    scuteMat
  );

  // =========================================================================
  // 2. SKULL, FESTOONED SNOUT, EYES, PALATE, TONGUE, ARTICULATED JAW & 34 TEETH
  // =========================================================================
  const headPivot = new THREE.Group();
  headPivot.name = 'Joint_Head';
  headPivot.position.set(0, 0.022, 0.12);
  neck3Pivot.add(headPivot);

  const upperSkullMesh = new THREE.Mesh(buildUpperSkullGeometry(), bodySkinMat);
  upperSkullMesh.name = 'UpperSkullSnout';
  upperSkullMesh.castShadow = true;
  upperSkullMesh.receiveShadow = true;
  headPivot.add(upperSkullMesh);

  // Upper fleshy palate inside mouth roof
  const upperPalateMesh = new THREE.Mesh(buildUpperPalateGeometry(), mouthFleshMat);
  upperPalateMesh.name = 'UpperPalate';
  upperPalateMesh.receiveShadow = true;
  headPivot.add(upperPalateMesh);

  // Twin recessed crescent nostril apertures atop the raised nasal boss at snout tip
  for (const side of [-1, 1]) {
    const nostrilRim = new THREE.Mesh(
      buildOsteodermScuteGeometry(0.052, 0.040, 0.016, 0.0),
      bodySkinMat
    );
    nostrilRim.position.set(side * 0.034, 0.068, 0.658);
    nostrilRim.rotation.y = side * 0.22;
    headPivot.add(nostrilRim);

    const nostrilHole = new THREE.Mesh(
      new THREE.CylinderGeometry(0.013, 0.010, 0.016, 10),
      nostrilInteriorMat
    );
    nostrilHole.position.set(side * 0.035, 0.075, 0.662);
    nostrilHole.scale.set(1.25, 1.0, 0.65);
    nostrilHole.rotation.y = side * 0.28;
    headPivot.add(nostrilHole);
  }

  // Multi-layered Reptile Eyes recessed naturally under supraorbital brow ridges
  const eyelids: THREE.Mesh[] = [];
  for (const side of [-1, 1]) {
    const eye = createReptileEye(side, eyeIrisMat, corneaMat, bodySkinMat, eyelids);
    eye.position.set(side * 0.098, 0.072, 0.165);
    headPivot.add(eye);
  }

  // --- INDEPENDENT ARTICULATED LOWER JAW (hinged at rear skull joint) ---
  const jawPivot = new THREE.Group();
  jawPivot.name = 'Joint_LowerJaw';
  jawPivot.position.set(0, -0.022, 0.035);
  headPivot.add(jawPivot);

  const lowerJawMesh = new THREE.Mesh(buildLowerJawGeometry(), bodySkinMat);
  lowerJawMesh.name = 'LowerJawMandible';
  lowerJawMesh.castShadow = true;
  lowerJawMesh.receiveShadow = true;
  jawPivot.add(lowerJawMesh);

  // Fleshy pinkish-salmon tongue & palatal glottal fold
  const tongueMesh = new THREE.Mesh(buildTongueGeometry(), tongueMat);
  tongueMesh.name = 'TongueMesh';
  tongueMesh.receiveShadow = true;
  jawPivot.add(tongueMesh);

  // --- 34 INDIVIDUAL CONICAL INTERLOCKING TEETH (18 Upper + 16 Lower) ---
  // Carefully positioned along the festooned jawline so overhanging upper fangs & the iconic
  // 4th mandibular tooth are visibly exposed even with the jaw closed (matching the reference photo!).
  const upperTeethSpecs: Array<{ z: number; x: number; y: number; len: number; r: number; cantOut: number }> = [
    // Anterior premaxillary fangs (front snout overhang)
    { z: 0.722, x: 0.038, y: -0.008, len: 0.036, r: 0.009, cantOut: 0.14 },
    { z: 0.688, x: 0.074, y: -0.008, len: 0.044, r: 0.010, cantOut: 0.18 },
    { z: 0.642, x: 0.104, y: -0.010, len: 0.052, r: 0.012, cantOut: 0.22 },
    // Behind notch for 4th lower tooth -> prominent large 5th maxillary fang!
    { z: 0.530, x: 0.116, y: -0.014, len: 0.056, r: 0.013, cantOut: 0.20 },
    { z: 0.468, x: 0.128, y: -0.018, len: 0.046, r: 0.011, cantOut: 0.16 },
    { z: 0.408, x: 0.138, y: -0.020, len: 0.040, r: 0.010, cantOut: 0.14 },
    { z: 0.348, x: 0.146, y: -0.020, len: 0.038, r: 0.010, cantOut: 0.12 },
    { z: 0.286, x: 0.156, y: -0.018, len: 0.034, r: 0.0095, cantOut: 0.10 },
    { z: 0.224, x: 0.164, y: -0.016, len: 0.028, r: 0.009, cantOut: 0.08 },
  ];

  for (const side of [-1, 1]) {
    for (const ts of upperTeethSpecs) {
      const tGeo = buildConicalToothGeometry(ts.r, ts.len, 0.22);
      const tooth = new THREE.Mesh(tGeo, toothMat);
      tooth.position.set(side * ts.x, ts.y, ts.z);
      // Point downward (-Y) with slight outward flare so tips peek past lower lip
      tooth.rotation.x = Math.PI;
      tooth.rotation.z = side * ts.cantOut;
      tooth.castShadow = true;
      headPivot.add(tooth);
    }
  }

  // Lower jaw teeth (coordinates relative to jawPivot at (0, -0.022, 0.035))
  const lowerTeethSpecs: Array<{ z: number; x: number; y: number; len: number; r: number; cantOut: number }> = [
    { z: 0.648, x: 0.036, y: 0.004, len: 0.038, r: 0.009, cantOut: 0.12 },
    { z: 0.605, x: 0.072, y: 0.005, len: 0.034, r: 0.0085, cantOut: 0.12 },
    // ICONIC EXPOSED 4TH MANDIBULAR TOOTH! Tall, projecting visibly outside upper snout notch!
    { z: 0.542, x: 0.106, y: 0.006, len: 0.062, r: 0.0135, cantOut: 0.24 },
    { z: 0.455, x: 0.108, y: 0.004, len: 0.034, r: 0.009, cantOut: 0.10 },
    { z: 0.392, x: 0.118, y: 0.002, len: 0.036, r: 0.0095, cantOut: 0.10 },
    { z: 0.330, x: 0.126, y: 0.000, len: 0.038, r: 0.010, cantOut: 0.10 },
    { z: 0.268, x: 0.136, y: -0.002, len: 0.034, r: 0.0095, cantOut: 0.08 },
    { z: 0.205, x: 0.146, y: -0.004, len: 0.028, r: 0.009, cantOut: 0.06 },
  ];

  for (const side of [-1, 1]) {
    for (const ts of lowerTeethSpecs) {
      const tGeo = buildConicalToothGeometry(ts.r, ts.len, 0.18);
      const tooth = new THREE.Mesh(tGeo, toothMat);
      tooth.position.set(side * ts.x, ts.y, ts.z);
      // Point upward (+Y)
      tooth.rotation.z = -side * ts.cantOut;
      tooth.castShadow = true;
      jawPivot.add(tooth);
    }
  }

  // =========================================================================
  // 3. MULTI-SEGMENT ARTICULATED TAIL (11 Segments + Double -> Single Fin Crest)
  // =========================================================================
  const tailPivots: THREE.Group[] = [];
  let tailParent: THREE.Group = pelvisRoot;

  // Tail runs posteriorly along -Z from pelvis (-0.18) to tip (~ -2.18m)
  const tailSegLengths = [0.22, 0.21, 0.20, 0.19, 0.19, 0.18, 0.17, 0.16, 0.15, 0.14, 0.14];

  for (let i = 0; i < TAIL_SEGMENT_COUNT; i++) {
    const frac0 = i / TAIL_SEGMENT_COUNT;
    const frac1 = (i + 1) / TAIL_SEGMENT_COUNT;
    const segLen = tailSegLengths[i];

    const pivot = new THREE.Group();
    pivot.name = `Joint_Tail_${i}`;
    if (i === 0) {
      pivot.position.set(0, -0.015, -0.17);
    } else {
      pivot.position.set(0, -0.004, -tailSegLengths[i - 1] * 0.94);
    }
    tailParent.add(pivot);
    tailPivots.push(pivot);

    // Smooth taper & progressive lateral compression (swimming oar cross-section!)
    const rxAt = (f: number) => 0.245 * Math.pow(1 - f, 1.38) + 0.012;
    const ryAt = (f: number) => 0.186 * Math.pow(1 - f, 0.96) + 0.016;
    // UV v runs from 0.44 at tail base down to 0.0 at tail tip
    const uvAt = (f: number) => 0.44 * (1 - f);

    const overlapStart = 0.045 * (1 - frac0 * 0.65);
    const isLast = i === TAIL_SEGMENT_COUNT - 1;
    const rings: LoftRingSpec[] = [
      {
        z: overlapStart,
        yOffset: 0,
        rx: rxAt(Math.max(0, frac0 - 0.02)),
        ry: ryAt(Math.max(0, frac0 - 0.02)),
        flatTop: 0.22 * (1 - frac0),
        flatBottom: 0.36 * (1 - frac0 * 0.6),
        dorsalRidge: 0.012 * frac0,
        lowerBulge: 0.08 * (1 - frac0),
        uvV: uvAt(Math.max(0, frac0 - 0.015)),
      },
      {
        z: -segLen * 0.48,
        yOffset: -0.003,
        rx: rxAt((frac0 + frac1) * 0.5),
        ry: ryAt((frac0 + frac1) * 0.5),
        flatTop: 0.18 * (1 - (frac0 + frac1) * 0.5),
        flatBottom: 0.32 * (1 - (frac0 + frac1) * 0.3),
        dorsalRidge: 0.016 * ((frac0 + frac1) * 0.5),
        lowerBulge: 0.06 * (1 - (frac0 + frac1) * 0.5),
        uvV: uvAt((frac0 + frac1) * 0.5),
      },
      {
        z: -segLen * (isLast ? 1.05 : 1.02),
        yOffset: -0.006,
        rx: isLast ? 0.008 : rxAt(frac1),
        ry: isLast ? 0.014 : ryAt(frac1),
        flatTop: 0.14 * (1 - frac1),
        flatBottom: 0.25 * (1 - frac1 * 0.5),
        dorsalRidge: isLast ? 0 : 0.018 * frac1,
        lowerBulge: 0.04 * (1 - frac1),
        uvV: uvAt(frac1),
      },
    ];

    const segGeo = buildLoftedGeometry(rings, {
      radialSegments: Math.max(12, Math.round(22 - frac0 * 8)),
      capStart: true,
      capEnd: true,
      bumpAmp: 0.004 * (1 - frac0 * 0.6),
      bumpFreq: 18,
      bakeVertexAO: true,
    });
    const segMesh = new THREE.Mesh(segGeo, bodySkinMat);
    segMesh.name = `TailSegment_${i}`;
    segMesh.castShadow = true;
    segMesh.receiveShadow = true;
    pivot.add(segMesh);

    // DRAMATIC SERRATED CAUDAL FIN CRESTS!
    // Segments 0..5: DOUBLE-CRESTED RIDGE converging smoothly toward midline.
    // Segments 6..10: SINGLE SERRATED VERTICAL BLADE CREST tapering cleanly to tip!
    const scuteCount = i < 7 ? 2 : i < 10 ? 2 : 1;
    for (let k = 0; k < scuteCount; k++) {
      const subT = scuteCount === 1 ? 0.45 : (k + 0.45) / scuteCount;
      const fLocal = frac0 + (frac1 - frac0) * subT;
      const zPos = -segLen * subT;
      const topY = ryAt(fLocal) * (1 - 0.16 * (1 - fLocal)) - 0.008;

      if (i <= 5) {
        // Double-crested dorsal ridges converging smoothly from x=±0.088 down to x=±0.018
        const converge = i / 5.4;
        const xSpread = 0.090 * (1 - converge * 0.78);
        const finHeight = 0.056 + Math.sin(fLocal * Math.PI * 1.15) * 0.052;
        const finLen = segLen * 0.56;
        const finThick = 0.028 * (1 - fLocal * 0.35);

        for (const side of [-1, 1]) {
          const finGeo = buildTailFinScuteGeometry(finLen, finThick, finHeight, side * (0.28 - converge * 0.18));
          const finMesh = new THREE.Mesh(finGeo, scuteMat);
          finMesh.position.set(side * xSpread, topY, zPos);
          finMesh.castShadow = true;
          finMesh.receiveShadow = true;
          pivot.add(finMesh);
        }
      } else {
        // Single tall serrated vertical blade fin crest along posterior tail midline!
        const tailTipFade = 1 - Math.pow((fLocal - 0.52) / 0.48, 1.55);
        const finHeight = Math.max(0.024, 0.105 * Math.max(0.18, tailTipFade));
        const finLen = segLen * 0.58;
        const finThick = 0.024 * Math.max(0.35, tailTipFade);
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
  // 4. FOUR SPRAWLING MUSCULAR REPTILIAN LIMBS (3-Joint Hierarchy Each)
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
  fl.hipPivot.position.set(0.265, -0.055, 0.21);
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
  fr.hipPivot.position.set(-0.265, -0.055, 0.21);
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
  bl.hipPivot.position.set(0.245, -0.042, 0.01);
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
  br.hipPivot.position.set(-0.245, -0.042, 0.01);
  pelvisRoot.add(br.hipPivot);

  const limbs: Record<LimbId, LimbRig> = { FL: fl, FR: fr, BL: bl, BR: br };

  // Soft studio contact shadows anchored to root (world Y = 0.008)
  const torsoShadowGeo = new THREE.PlaneGeometry(1.55, 3.05);
  torsoShadowGeo.rotateX(-Math.PI / 2);
  const torsoShadow = new THREE.Mesh(torsoShadowGeo, contactShadowMat);
  torsoShadow.position.set(0, 0.006, 0.35);
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

  // Ordered parent->child bone connections for line rendering
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

  const jointGeo = new THREE.SphereGeometry(0.032, 10, 8);
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
  // 6. REAL-TIME POSE APPLICATION & EXACT GROUND-LOCK FOOT PLANTING
  // =========================================================================
  const applyPose = (pose: CrocPose, wetness: number, showShadows: boolean): void => {
    // World root position & heading
    root.position.set(pose.posX, 0, pose.posZ);
    root.rotation.y = pose.heading;

    // Pelvis transform
    pelvisRoot.position.y = pose.pelvisY;
    pelvisRoot.rotation.set(pose.pelvisPitch, 0, pose.pelvisRoll, 'YXZ');

    // Anterior spine chain: [midTorso, chest, neck1, neck2, neck3, head]
    midTorsoPivot.rotation.set(pose.anteriorPitch[0], pose.anteriorYaw[0], 0, 'YXZ');
    chestPivot.rotation.set(pose.anteriorPitch[1], pose.anteriorYaw[1], 0, 'YXZ');
    neck1Pivot.rotation.set(pose.anteriorPitch[2], pose.anteriorYaw[2], 0, 'YXZ');
    neck2Pivot.rotation.set(pose.anteriorPitch[3], pose.anteriorYaw[3], 0, 'YXZ');
    neck3Pivot.rotation.set(pose.anteriorPitch[4], pose.anteriorYaw[4], 0, 'YXZ');
    headPivot.rotation.set(pose.anteriorPitch[5], pose.anteriorYaw[5], 0, 'YXZ');

    // Breathing ribcage expansion & subtle throat gular pulse
    ribcageMesh.scale.set(pose.breathScale, 1 + (pose.breathScale - 1) * 0.75, 1);
    throatMesh.scale.set(1, 1 + pose.throatPulse, 1);

    // Articulated lower jaw opening (rotates downward +X cleanly up to 45 deg)
    jawPivot.rotation.x = pose.jawAngleRad;

    // Eyelid blink
    for (const lid of eyelids) {
      const s = lid.userData.sideSign as number;
      // Open: tucked at -0.78*s, Closed: rotates smoothly to +0.06*s covering cornea
      lid.rotation.z = s * (-0.78 + pose.blinkAmount * 0.84);
    }

    // 11-segment tail travelling wave
    for (let i = 0; i < tailPivots.length; i++) {
      tailPivots[i].rotation.set(pose.tailPitch[i], pose.tailYaw[i], 0, 'YXZ');
    }

    // Apply sprawling 3-joint limb kinematics
    const limbList = [fl, fr, bl, br];
    for (const limb of limbList) {
      const a = pose.limbs[limb.id];
      limb.hipPivot.rotation.set(a.hipPitch, a.hipYaw, a.hipRoll, 'YXZ');
      // a.kneeBend already encodes -sideSign * kneeReturnAngle so forearm drops cleanly downward!
      limb.kneePivot.rotation.set(
        limb.isFront ? -0.14 : 0.18,
        a.kneeYaw,
        a.kneeBend,
        'YXZ'
      );
      limb.anklePivot.rotation.set(a.anklePitch, 0, a.ankleRoll, 'YXZ');
    }

    // Update world matrices and run 2-pass exact Y=0 ground-lock on planted feet!
    root.updateMatrixWorld(true);

    for (const limb of limbList) {
      const a = pose.limbs[limb.id];
      if (pose.limbTuck < 0.85) {
        const targetY = a.footLiftY;
        // 2 quick refinement passes guarantee sole rests flush on Y=0 (zero floating, zero floor clipping!)
        for (let pass = 0; pass < 2; pass++) {
          limb.footSoleMarker.getWorldPosition(_worldPos);
          const yError = _worldPos.y - targetY;
          if (Math.abs(yError) <= 0.002) break;
          // Increasing sprawlOut (making hipRoll larger in magnitude for +sideSign / more negative for -sideSign) LIFTS foot!
          // When yError < 0 (foot below ground), -yError > 0 -> increases sprawlOut & lifts foot cleanly to Y=0!
          const corr = THREE.MathUtils.clamp(-yError * 2.1, -0.55, 0.55) * (1 - pose.limbTuck);
          limb.hipPivot.rotation.z += limb.sideSign * corr;
          limb.kneePivot.rotation.z -= limb.sideSign * corr;
          limb.hipPivot.updateMatrixWorld(true);
        }

        // Level the foot flat against the ground plane during stance (blended smoothly off in swim/swing)
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

          // Final micro-check after flattening foot pad so claws & sole sit right on top of Y=0
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

      // Position soft contact shadow directly under each foot on Y=0.008
      limb.footShadow.visible = showShadows;
      if (showShadows) {
        limb.footSoleMarker.getWorldPosition(_worldPos);
        root.worldToLocal(_localPos.copy(_worldPos));
        limb.footShadow.position.set(_localPos.x, 0.008, _localPos.z);
        const heightFade = THREE.MathUtils.clamp(1 - Math.max(0, _worldPos.y) * 3.8, 0.08, 1);
        (limb.footShadow.material as THREE.MeshBasicMaterial).opacity =
          0.52 * heightFade * (1 - pose.limbTuck * 0.85);
      }
    }

    // Update main torso studio contact shadow
    torsoShadow.visible = showShadows;
    if (showShadows) {
      const liftFade = THREE.MathUtils.clamp(1.12 - pose.pelvisY * 0.72, 0.22, 0.62);
      contactShadowMat.opacity = liftFade * (1 - pose.limbTuck * 0.65);
    }

    // Update roughness / wetness across reptile skin materials
    const dryRough = 0.94 - wetness * 0.56; // Dry matte (~0.94) -> wet glossy (~0.38)
    const scuteRough = 0.66 - wetness * 0.42;
    bodySkinMat.roughness = dryRough;
    limbSkinMat.roughness = dryRough;
    scuteMat.roughness = scuteRough;

    // Update Skeleton Debug geometry if visible
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
    eyelids,
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
