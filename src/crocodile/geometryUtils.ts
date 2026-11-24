import * as THREE from 'three';

export interface LoftRingSpec {
  /** Longitudinal position along segment axis (typically Z) */
  z: number;
  /** Vertical center offset (Y) */
  yOffset: number;
  /** Horizontal half-width (X) */
  rx: number;
  /** Vertical half-height (Y) */
  ry: number;
  /** Flattening on dorsal top (0 = round, 0.5 = broad flat back) */
  flatTop?: number;
  /** Flattening on ventral bottom (0 = round, 0.65 = flat belly) */
  flatBottom?: number;
  /** Dorsal ridge sharpness (0 = smooth, >0 = tall fin/keel ridge) */
  dorsalRidge?: number;
  /** Lateral lower cheek / flank bulge (pushes lower sides outward) */
  lowerBulge?: number;
  /** Festooning / scalloped lip wave multiplier for snout */
  lipWave?: number;
  /** Longitudinal UV V coordinate (0 = tail tip, 1 = snout tip) */
  uvV: number;
  /** Extra custom vertex modifier per ring */
  deform?: (angle: number, x: number, y: number, z: number) => [number, number, number];
}

export interface LoftOptions {
  radialSegments?: number;
  capStart?: boolean;
  capEnd?: boolean;
  /** Scale of organic scale-bump surface displacement */
  bumpAmp?: number;
  /** Spatial frequency of organic bumps */
  bumpFreq?: number;
  /** Bake subtle vertex-color ambient occlusion into belly/crevices */
  bakeVertexAO?: boolean;
}

/** Deterministic fast 3D hash noise for organic sculptural displacement */
function hash3(x: number, y: number, z: number): number {
  const n = Math.sin(x * 127.1 + y * 311.7 + z * 74.7) * 43758.5453123;
  return n - Math.floor(n);
}

function smoothNoise3(x: number, y: number, z: number): number {
  const ix = Math.floor(x);
  const iy = Math.floor(y);
  const iz = Math.floor(z);
  const fx = x - ix;
  const fy = y - iy;
  const fz = z - iz;
  const ux = fx * fx * (3 - 2 * fx);
  const uy = fy * fy * (3 - 2 * fy);
  const uz = fz * fz * (3 - 2 * fz);

  const n000 = hash3(ix, iy, iz);
  const n100 = hash3(ix + 1, iy, iz);
  const n010 = hash3(ix, iy + 1, iz);
  const n110 = hash3(ix + 1, iy + 1, iz);
  const n001 = hash3(ix, iy, iz + 1);
  const n101 = hash3(ix + 1, iy, iz + 1);
  const n011 = hash3(ix, iy + 1, iz + 1);
  const n111 = hash3(ix + 1, iy + 1, iz + 1);

  const nx00 = n000 + (n100 - n000) * ux;
  const nx10 = n010 + (n110 - n010) * ux;
  const nx01 = n001 + (n101 - n001) * ux;
  const nx11 = n011 + (n111 - n011) * ux;

  const nxy0 = nx00 + (nx10 - nx00) * uy;
  const nxy1 = nx01 + (nx11 - nx01) * uy;
  return nxy0 + (nxy1 - nxy0) * uz;
}

/**
 * Builds a watertight, anatomically sculpted lofted BufferGeometry from cross-section rings.
 * UV convention:
 *   u = 0.0 -> dorsal midline (+Y)
 *   u = 0.25 -> right flank (+X)
 *   u = 0.5 -> ventral belly midline (-Y)
 *   u = 0.75 -> left flank (-X)
 *   u = 1.0 -> dorsal midline (+Y)
 *   v = ring.uvV (longitudinal body coordinate)
 */
export function buildLoftedGeometry(
  rings: LoftRingSpec[],
  options: LoftOptions = {}
): THREE.BufferGeometry {
  const radialSegs = options.radialSegments ?? 24;
  const capStart = options.capStart ?? true;
  const capEnd = options.capEnd ?? true;
  const bumpAmp = options.bumpAmp ?? 0.004;
  const bumpFreq = options.bumpFreq ?? 14.0;
  const bakeVertexAO = options.bakeVertexAO ?? false;

  const positions: number[] = [];
  const uvs: number[] = [];
  const colors: number[] = [];
  const indices: number[] = [];

  const ringStride = radialSegs + 1;

  for (let r = 0; r < rings.length; r++) {
    const spec = rings[r];
    const flatTop = spec.flatTop ?? 0.18;
    const flatBottom = spec.flatBottom ?? 0.35;
    const dorsalRidge = spec.dorsalRidge ?? 0;
    const lowerBulge = spec.lowerBulge ?? 0.08;

    for (let s = 0; s <= radialSegs; s++) {
      const u = s / radialSegs;
      // theta = 0 at dorsal top (+Y), PI/2 at +X, PI at ventral bottom (-Y), 3PI/2 at -X
      const theta = u * Math.PI * 2;
      const sinT = Math.sin(theta);
      const cosT = Math.cos(theta);

      let x = sinT * spec.rx;
      let y = cosT * spec.ry;

      // Dorsal flattening (broad armored crocodile back)
      if (cosT > 0 && flatTop > 0) {
        const flatten = Math.pow(cosT, 2.2) * flatTop;
        y *= 1 - flatten;
        x *= 1 + flatten * 0.22;
      }

      // Ventral flattening (flat tiled crocodile belly)
      if (cosT < 0 && flatBottom > 0) {
        const absC = -cosT;
        const flatten = Math.pow(absC, 1.8) * flatBottom;
        y *= 1 - flatten;
        x *= 1 + flatten * 0.18;
      }

      // Lateral lower muscle/flank bulge (gives muscular pear/trapezoid cross-section)
      if (lowerBulge !== 0 && cosT < 0.25 && cosT > -0.85) {
        const flankBell = Math.sin(((cosT + 0.85) / 1.1) * Math.PI);
        x *= 1 + flankBell * lowerBulge;
      }

      // Dorsal ridge / crest elevation on top midline
      if (dorsalRidge > 0 && cosT > 0.55) {
        const ridgeProfile = Math.pow((cosT - 0.55) / 0.45, 2.4);
        y += ridgeProfile * dorsalRidge;
      }

      y += spec.yOffset;
      let z = spec.z;

      if (spec.deform) {
        const [dx, dy, dz] = spec.deform(theta, x, y, z);
        x = dx;
        y = dy;
        z = dz;
      }

      // Subtle organic micro-displacement (skip at seam endpoints so radial seam stays watertight)
      if (bumpAmp > 0 && spec.rx > 0.008 && spec.ry > 0.008) {
        // Use sinT/cosT instead of raw u so s=0 and s=radialSegs evaluate identically!
        const nVal =
          smoothNoise3(sinT * 2.5 + z * bumpFreq, cosT * 2.5 + y * bumpFreq, z * bumpFreq * 0.7) -
          0.5;
        const radialLen = Math.hypot(sinT, cosT) || 1;
        x += (sinT / radialLen) * nVal * bumpAmp;
        y += (cosT / radialLen) * nVal * bumpAmp;
      }

      positions.push(x, y, z);
      uvs.push(u, spec.uvV);

      if (bakeVertexAO) {
        // Subtle anatomical shading: slightly darker under belly crease and joint collars
        const endFade = r === 0 || r === rings.length - 1 ? 0.88 : 1.0;
        const ventralShade = cosT < -0.4 ? 0.94 : 1.0;
        const ao = endFade * ventralShade;
        colors.push(ao, ao, ao);
      }
    }
  }

  // Connect adjacent rings with outward-facing CCW triangles.
  // Because theta runs clockwise around +Z (x=sin(t), y=cos(t)), dP/dz x dP/dtheta points outward when dz >= 0.
  for (let r = 0; r < rings.length - 1; r++) {
    const rowA = r * ringStride;
    const rowB = (r + 1) * ringStride;
    const dz = rings[r + 1].z - rings[r].z;
    const forward = dz >= 0;

    for (let s = 0; s < radialSegs; s++) {
      const a0 = rowA + s;
      const a1 = rowA + s + 1;
      const b0 = rowB + s;
      const b1 = rowB + s + 1;

      if (forward) {
        indices.push(a0, b0, a1);
        indices.push(a1, b0, b1);
      } else {
        indices.push(a0, a1, b0);
        indices.push(a1, b1, b0);
      }
    }
  }

  // Optional start cap (outward normal -Z when forward, +Z when backward)
  if (capStart && rings.length > 0) {
    const first = rings[0];
    const centerIdx = positions.length / 3;
    positions.push(0, first.yOffset, first.z);
    uvs.push(0.5, first.uvV);
    if (bakeVertexAO) colors.push(0.85, 0.85, 0.85);

    const dz = rings.length > 1 ? rings[1].z - rings[0].z : 1;
    const forward = dz >= 0;
    for (let s = 0; s < radialSegs; s++) {
      if (forward) {
        indices.push(centerIdx, s, s + 1);
      } else {
        indices.push(centerIdx, s + 1, s);
      }
    }
  }

  // Optional end cap (outward normal +Z when forward, -Z when backward)
  if (capEnd && rings.length > 0) {
    const lastIdx = rings.length - 1;
    const last = rings[lastIdx];
    const rowStart = lastIdx * ringStride;
    const centerIdx = positions.length / 3;
    positions.push(0, last.yOffset, last.z);
    uvs.push(0.5, last.uvV);
    if (bakeVertexAO) colors.push(0.85, 0.85, 0.85);

    const dz = rings.length > 1 ? rings[lastIdx].z - rings[lastIdx - 1].z : 1;
    const forward = dz >= 0;
    for (let s = 0; s < radialSegs; s++) {
      if (forward) {
        indices.push(centerIdx, rowStart + s + 1, rowStart + s);
      } else {
        indices.push(centerIdx, rowStart + s, rowStart + s + 1);
      }
    }
  }

  const geo = new THREE.BufferGeometry();
  geo.setAttribute('position', new THREE.Float32BufferAttribute(positions, 3));
  geo.setAttribute('uv', new THREE.Float32BufferAttribute(uvs, 2));
  geo.setAttribute('uv2', new THREE.Float32BufferAttribute(uvs.slice(), 2));
  if (bakeVertexAO) {
    geo.setAttribute('color', new THREE.Float32BufferAttribute(colors, 3));
  }
  geo.setIndex(indices);
  geo.computeVertexNormals();
  geo.computeBoundingSphere();
  geo.computeBoundingBox();

  return geo;
}

/**
 * Builds a realistic keeled crocodile dorsal osteoderm scute (bony shield plate with raised central ridge).
 * Oriented along +Z (forward), base seated at Y = 0, keel rising along +Y.
 */
export function buildOsteodermScuteGeometry(
  length: number,
  width: number,
  height: number,
  sweepBack = 0.22
): THREE.BufferGeometry {
  const hw = width * 0.5;
  const hl = length * 0.5;
  const peakZ = -hl * sweepBack;

  // 3-tier sculpted bony shield: flared octagonal base -> beveled shoulder -> sharp central keel crest
  const pts: [number, number, number][] = [
    // Base ring (0..7)
    [0, -0.008, hl], // 0 front
    [hw * 0.72, -0.006, hl * 0.55], // 1 front-right
    [hw, -0.005, 0], // 2 mid-right
    [hw * 0.68, -0.006, -hl * 0.62], // 3 back-right
    [0, -0.008, -hl], // 4 back
    [-hw * 0.68, -0.006, -hl * 0.62], // 5 back-left
    [-hw, -0.005, 0], // 6 mid-left
    [-hw * 0.72, -0.006, hl * 0.55], // 7 front-left

    // Mid-slope beveled armor collar (8..15)
    [0, height * 0.34, hl * 0.76], // 8
    [hw * 0.44, height * 0.38, hl * 0.38], // 9
    [hw * 0.56, height * 0.42, -hl * 0.05], // 10
    [hw * 0.38, height * 0.36, -hl * 0.52], // 11
    [0, height * 0.30, -hl * 0.82], // 12
    [-hw * 0.38, height * 0.36, -hl * 0.52], // 13
    [-hw * 0.56, height * 0.42, -hl * 0.05], // 14
    [-hw * 0.44, height * 0.38, hl * 0.38], // 15

    // Sharp dorsal keel crest (16..18)
    [0, height * 0.68, hl * 0.48], // 16 front keel
    [0, height, peakZ], // 17 apex keel peak
    [0, height * 0.72, -hl * 0.62], // 18 rear keel
  ];

  const positions: number[] = [];
  const uvs: number[] = [];
  for (const [x, y, z] of pts) {
    positions.push(x, y, z);
    // Map onto dorsal dark-olive ridge band in texture space (u near 0.04..0.12)
    uvs.push(0.06 + (x / (width + 0.001)) * 0.08, 0.58 + (z / (length + 0.001)) * 0.06);
  }

  const indices: number[] = [];
  // Base to mid collar
  for (let i = 0; i < 8; i++) {
    const next = (i + 1) % 8;
    const b0 = i;
    const b1 = next;
    const m0 = 8 + i;
    const m1 = 8 + next;
    indices.push(b0, m0, b1);
    indices.push(b1, m0, m1);
  }

  // Mid collar to keel ridge (right side + left side)
  indices.push(
    8, 16, 9,
    9, 16, 17,
    9, 17, 10,
    10, 17, 11,
    11, 17, 18,
    11, 18, 12,
    12, 18, 13,
    13, 18, 17,
    13, 17, 14,
    14, 17, 15,
    15, 17, 16,
    15, 16, 8
  );

  const geo = new THREE.BufferGeometry();
  geo.setAttribute('position', new THREE.Float32BufferAttribute(positions, 3));
  geo.setAttribute('uv', new THREE.Float32BufferAttribute(uvs, 2));
  geo.setAttribute('uv2', new THREE.Float32BufferAttribute(uvs.slice(), 2));
  geo.setIndex(indices);
  geo.computeVertexNormals();
  geo.computeBoundingSphere();
  return geo;
}

/**
 * Builds a dramatic tall serrated caudal fin scute (swept-back triangular vertical blade)
 * matching the iconic double-crest & single-crest crocodile tail silhouette in the reference.
 */
export function buildTailFinScuteGeometry(
  length: number,
  thickness: number,
  height: number,
  cantOutward = 0
): THREE.BufferGeometry {
  const ht = thickness * 0.5;
  const hl = length * 0.5;
  const cant = cantOutward * height;

  const pts: [number, number, number][] = [
    // Flared triangular base (0..5)
    [0, -0.008, hl * 0.92], // 0 front base
    [ht, -0.006, hl * 0.25], // 1 front-right base
    [ht * 0.85, -0.006, -hl * 0.65], // 2 back-right base
    [0, -0.008, -hl * 0.95], // 3 back base
    [-ht * 0.85, -0.006, -hl * 0.65], // 4 back-left base
    [-ht, -0.006, hl * 0.25], // 5 front-left base

    // Mid-blade ridges (6..9)
    [cant * 0.5 + ht * 0.35, height * 0.48, hl * 0.12], // 6
    [cant * 0.5 + ht * 0.28, height * 0.54, -hl * 0.56], // 7
    [cant * 0.5 - ht * 0.28, height * 0.54, -hl * 0.56], // 8
    [cant * 0.5 - ht * 0.35, height * 0.48, hl * 0.12], // 9

    // Swept-back sharp serrated blade apex (10..11)
    [cant, height * 0.94, -hl * 0.42], // 10 main swept peak
    [cant * 0.85, height * 0.72, -hl * 0.82], // 11 trailing hook edge
  ];

  const positions: number[] = [];
  const uvs: number[] = [];
  for (const [x, y, z] of pts) {
    positions.push(x, y, z);
    uvs.push(0.05 + (y / (height + 0.01)) * 0.08, 0.25 + (z / (length + 0.01)) * 0.08);
  }

  const indices = [
    // Right face
    0, 6, 1,
    1, 6, 2,
    2, 6, 7,
    2, 7, 3,
    // Left face
    0, 5, 9,
    5, 4, 9,
    4, 8, 9,
    4, 3, 8,
    // Upper swept fin blade
    0, 9, 6,
    6, 9, 10,
    6, 10, 7,
    7, 10, 11,
    7, 11, 3,
    9, 8, 10,
    8, 11, 10,
    8, 3, 11,
  ];

  const geo = new THREE.BufferGeometry();
  geo.setAttribute('position', new THREE.Float32BufferAttribute(positions, 3));
  geo.setAttribute('uv', new THREE.Float32BufferAttribute(uvs, 2));
  geo.setAttribute('uv2', new THREE.Float32BufferAttribute(uvs.slice(), 2));
  geo.setIndex(indices);
  geo.computeVertexNormals();
  geo.computeBoundingSphere();
  return geo;
}

/**
 * Builds a slightly recurved conical ivory crocodile tooth (root at Y=0, tip at Y=height).
 */
export function buildConicalToothGeometry(
  baseRadius: number,
  height: number,
  inwardCurve = 0.18
): THREE.BufferGeometry {
  const rings: LoftRingSpec[] = [
    { z: 0, yOffset: 0, rx: baseRadius * 1.05, ry: baseRadius * 0.95, flatTop: 0, flatBottom: 0, lowerBulge: 0, uvV: 0.0 },
    { z: height * 0.28, yOffset: height * 0.015 * inwardCurve, rx: baseRadius * 0.82, ry: baseRadius * 0.78, flatTop: 0, flatBottom: 0, lowerBulge: 0, uvV: 0.3 },
    { z: height * 0.65, yOffset: height * 0.07 * inwardCurve, rx: baseRadius * 0.45, ry: baseRadius * 0.44, flatTop: 0, flatBottom: 0, lowerBulge: 0, uvV: 0.7 },
    { z: height * 0.94, yOffset: height * 0.14 * inwardCurve, rx: baseRadius * 0.14, ry: baseRadius * 0.14, flatTop: 0, flatBottom: 0, lowerBulge: 0, uvV: 0.95 },
    { z: height, yOffset: height * 0.16 * inwardCurve, rx: 0.0012, ry: 0.0012, flatTop: 0, flatBottom: 0, lowerBulge: 0, uvV: 1.0 },
  ];
  const geo = buildLoftedGeometry(rings, { radialSegments: 10, capStart: true, capEnd: true, bumpAmp: 0 });
  // Rotate so tooth grows along +Y instead of +Z
  geo.rotateX(-Math.PI / 2);
  geo.computeVertexNormals();
  return geo;
}

/**
 * Builds a sharp downturned keratin reptile claw (root at Z=0, curved tip forward & down along +Z, -Y).
 */
export function buildCurvedClawGeometry(baseRadius: number, length: number): THREE.BufferGeometry {
  const rings: LoftRingSpec[] = [
    { z: 0, yOffset: 0, rx: baseRadius * 0.85, ry: baseRadius * 1.05, flatTop: 0, flatBottom: 0.2, lowerBulge: 0, uvV: 0.0 },
    { z: length * 0.35, yOffset: length * 0.04, rx: baseRadius * 0.65, ry: baseRadius * 0.86, flatTop: 0, flatBottom: 0.25, lowerBulge: 0, uvV: 0.35 },
    { z: length * 0.72, yOffset: -length * 0.09, rx: baseRadius * 0.36, ry: baseRadius * 0.50, flatTop: 0, flatBottom: 0.3, lowerBulge: 0, uvV: 0.72 },
    { z: length, yOffset: -length * 0.28, rx: 0.0015, ry: 0.0018, flatTop: 0, flatBottom: 0, lowerBulge: 0, uvV: 1.0 },
  ];
  return buildLoftedGeometry(rings, { radialSegments: 10, capStart: true, capEnd: true, bumpAmp: 0 });
}

/**
 * Builds a scaly articulated reptilian toe with subtle phalangeal knuckle rings.
 */
export function buildReptileToeGeometry(
  length: number,
  baseRadius: number,
  tipRadius: number
): THREE.BufferGeometry {
  const rings: LoftRingSpec[] = [
    { z: -0.012, yOffset: 0, rx: baseRadius * 1.08, ry: baseRadius * 0.92, flatTop: 0.15, flatBottom: 0.45, lowerBulge: 0.05, uvV: 0.1 },
    { z: length * 0.32, yOffset: -0.003, rx: baseRadius * 0.92, ry: baseRadius * 0.80, flatTop: 0.18, flatBottom: 0.48, lowerBulge: 0.04, uvV: 0.35 },
    { z: length * 0.52, yOffset: 0.002, rx: baseRadius * 0.86, ry: baseRadius * 0.76, flatTop: 0.18, flatBottom: 0.48, lowerBulge: 0.04, uvV: 0.55 },
    { z: length * 0.78, yOffset: -0.004, rx: tipRadius * 1.12, ry: tipRadius * 0.98, flatTop: 0.15, flatBottom: 0.45, lowerBulge: 0.02, uvV: 0.8 },
    { z: length, yOffset: -0.005, rx: tipRadius * 0.85, ry: tipRadius * 0.78, flatTop: 0.12, flatBottom: 0.4, lowerBulge: 0, uvV: 1.0 },
  ];
  return buildLoftedGeometry(rings, { radialSegments: 12, capStart: true, capEnd: true, bumpAmp: 0.0015, bumpFreq: 42 });
}

/**
 * Verifies a BufferGeometry has zero NaN/Infinity values in positions, normals, and UVs.
 */
export function validateBufferGeometry(geo: THREE.BufferGeometry): {
  valid: boolean;
  vertexCount: number;
  triangleCount: number;
  error?: string;
} {
  const pos = geo.attributes.position;
  const norm = geo.attributes.normal;
  const uv = geo.attributes.uv;
  if (!pos || pos.count === 0) {
    return { valid: false, vertexCount: 0, triangleCount: 0, error: 'Missing position attribute' };
  }
  if (!norm || norm.count !== pos.count) {
    return { valid: false, vertexCount: pos.count, triangleCount: 0, error: 'Missing or mismatched normal attribute' };
  }
  for (let i = 0; i < pos.array.length; i++) {
    if (!Number.isFinite(pos.array[i])) {
      return { valid: false, vertexCount: pos.count, triangleCount: 0, error: `Non-finite position at index ${i}` };
    }
  }
  for (let i = 0; i < norm.array.length; i++) {
    if (!Number.isFinite(norm.array[i])) {
      return { valid: false, vertexCount: pos.count, triangleCount: 0, error: `Non-finite normal at index ${i}` };
    }
  }
  if (uv) {
    for (let i = 0; i < uv.array.length; i++) {
      if (!Number.isFinite(uv.array[i])) {
        return { valid: false, vertexCount: pos.count, triangleCount: 0, error: `Non-finite UV at index ${i}` };
      }
    }
  }
  const triCount = geo.index ? geo.index.count / 3 : pos.count / 3;
  return { valid: true, vertexCount: pos.count, triangleCount: Math.floor(triCount) };
}
