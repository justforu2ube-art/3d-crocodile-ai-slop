import * as THREE from 'three';

export interface CrocodileTextureSet {
  bodyAlbedo: THREE.CanvasTexture;
  bodyNormal: THREE.CanvasTexture;
  bodyRoughness: THREE.CanvasTexture;
  bodyAO: THREE.CanvasTexture;
  limbAlbedo: THREE.CanvasTexture;
  limbNormal: THREE.CanvasTexture;
  limbRoughness: THREE.CanvasTexture;
  eyeIris: THREE.CanvasTexture;
  contactShadow: THREE.CanvasTexture;
  waterNormal: THREE.CanvasTexture;
  dispose: () => void;
}

/** Ultra-fast deterministic integer 2D hash returning [0, 1) — ~15x faster than Math.sin */
function hash2i(ix: number, iy: number): number {
  let h = (Math.imul(ix | 0, 374761393) + Math.imul(iy | 0, 668265263)) | 0;
  h = Math.imul(h ^ (h >>> 13), 1274126177);
  return ((h ^ (h >>> 16)) >>> 0) * 2.3283064365386963e-10;
}

/** Deterministic float-seed 2D hash for eye/helper routines */
function hash2(ix: number, iy: number): number {
  return hash2i(Math.round(ix * 131), Math.round(iy * 257));
}

/** 2D value noise with smoothstep interpolation */
function valueNoise2(x: number, y: number): number {
  const ix = Math.floor(x);
  const iy = Math.floor(y);
  const fx = x - ix;
  const fy = y - iy;
  const ux = fx * fx * (3 - 2 * fx);
  const uy = fy * fy * (3 - 2 * fy);
  const a = hash2i(ix, iy);
  const b = hash2i(ix + 1, iy);
  const c = hash2i(ix, iy + 1);
  const d = hash2i(ix + 1, iy + 1);
  return a + (b - a) * ux + (c - a) * uy + (a - b - c + d) * ux * uy;
}

function fbm2(x: number, y: number, octaves = 3): number {
  let value = 0;
  let amp = 0.5;
  let freq = 1;
  for (let i = 0; i < octaves; i++) {
    value += amp * valueNoise2(x * freq, y * freq);
    freq *= 2.03;
    amp *= 0.5;
  }
  return value;
}

// Reusable scratch output for zero-GC Voronoi evaluation inside pixel loops
let vorF1 = 0;
let vorEdge = 0;
let vorH1 = 0;
let vorH2 = 0;

/** Zero-allocation tileable Voronoi cellular distance (uses squared distances inside 3x3 loop) */
function evalVoronoiTileable(
  u: number,
  v: number,
  cellsX: number,
  cellsY: number,
  jitter = 0.72,
  seedOffset = 0
): void {
  const gx = u * cellsX;
  const gy = v * cellsY;
  const ix = Math.floor(gx);
  const iy = Math.floor(gy);
  const fx = gx - ix;
  const fy = gy - iy;

  let d1Sq = 99;
  let d2Sq = 99;
  let bestHash = 0;
  let bestHash2 = 0;

  for (let dy = -1; dy <= 1; dy++) {
    let cy = (iy + dy) % cellsY;
    if (cy < 0) cy += cellsY;
    const cyS = cy + seedOffset;
    for (let dx = -1; dx <= 1; dx++) {
      let cx = (ix + dx) % cellsX;
      if (cx < 0) cx += cellsX;
      const h1 = hash2i(cx + seedOffset, cyS + 17);
      const h2 = hash2i(cx + 911 + seedOffset, cyS + 353);
      const px = dx + 0.5 + (h1 - 0.5) * jitter - fx;
      const py = dy + 0.5 + (h2 - 0.5) * jitter - fy;
      const distSq = px * px + py * py;
      if (distSq < d1Sq) {
        d2Sq = d1Sq;
        d1Sq = distSq;
        bestHash = h1;
        bestHash2 = h2;
      } else if (distSq < d2Sq) {
        d2Sq = distSq;
      }
    }
  }
  const d1 = Math.sqrt(d1Sq);
  vorF1 = d1;
  vorEdge = Math.sqrt(d2Sq) - d1;
  vorH1 = bestHash;
  vorH2 = bestHash2;
}

function clamp01(v: number): number {
  return v < 0 ? 0 : v > 1 ? 1 : v;
}

function smoothstep(e0: number, e1: number, x: number): number {
  const t = clamp01((x - e0) / (e1 - e0));
  return t * t * (3 - 2 * t);
}

function mix(a: number, b: number, t: number): number {
  return a + (b - a) * t;
}

/** Converts a non-square float height buffer [0..1] into a crisp tangent-space normal map */
function heightFieldToNormalCanvas(
  heights: Float32Array,
  width: number,
  height: number,
  strength: number
): HTMLCanvasElement {
  const canvas = document.createElement('canvas');
  canvas.width = width;
  canvas.height = height;
  const ctx = canvas.getContext('2d')!;
  const img = ctx.createImageData(width, height);
  const data = img.data;

  for (let y = 0; y < height; y++) {
    const ym = (y - 1 + height) % height;
    const yp = (y + 1) % height;
    for (let x = 0; x < width; x++) {
      const xm = (x - 1 + width) % width;
      const xp = (x + 1) % width;

      const tl = heights[ym * width + xm];
      const tc = heights[ym * width + x];
      const tr = heights[ym * width + xp];
      const ml = heights[y * width + xm];
      const mr = heights[y * width + xp];
      const bl = heights[yp * width + xm];
      const bc = heights[yp * width + x];
      const br = heights[yp * width + xp];

      const dX = (tr + 2 * mr + br - (tl + 2 * ml + bl)) * strength;
      const dY = (bl + 2 * bc + br - (tl + 2 * tc + tr)) * strength;
      const dZ = 1.0;

      const invLen = 1 / Math.sqrt(dX * dX + dY * dY + dZ * dZ);
      const idx = (y * width + x) * 4;
      data[idx] = Math.round(((-dX * invLen) * 0.5 + 0.5) * 255);
      data[idx + 1] = Math.round(((-dY * invLen) * 0.5 + 0.5) * 255);
      data[idx + 2] = Math.round(((dZ * invLen) * 0.5 + 0.5) * 255);
      data[idx + 3] = 255;
    }
  }
  ctx.putImageData(img, 0, 0);
  return canvas;
}

/**
 * Generates ultra-detailed 1024x2048 anatomically zoned Nile Crocodile PBR maps
 * closely matching the photographic skin density, weathering, and markings of Crocodylus niloticus:
 * - Dense weathered slate-olive dorsal scutes with central keels & dark mud-crevice patina
 * - Fine pebbled dusty khaki-ochre flanks heavily dappled with organic charcoal blotches & lateral bosses
 * - High-contrast alternating charcoal-slate & dusty tan-ochre tail cross-bands
 * - Warm muted sand-ivory rectangular ventral belly plates
 * - Heavily rugose pitted grey-olive snout & jaw plates with fine sensory pores
 */
function generateBodyTextures(width = 1024, height = 2048): {
  albedoCanvas: HTMLCanvasElement;
  normalCanvas: HTMLCanvasElement;
  roughnessCanvas: HTMLCanvasElement;
  aoCanvas: HTMLCanvasElement;
} {
  const albedoCanvas = document.createElement('canvas');
  albedoCanvas.width = width;
  albedoCanvas.height = height;
  const albedoCtx = albedoCanvas.getContext('2d')!;
  const albedoImg = albedoCtx.createImageData(width, height);

  const roughCanvas = document.createElement('canvas');
  roughCanvas.width = width;
  roughCanvas.height = height;
  const roughCtx = roughCanvas.getContext('2d')!;
  const roughImg = roughCtx.createImageData(width, height);

  const aoCanvas = document.createElement('canvas');
  aoCanvas.width = width;
  aoCanvas.height = height;
  const aoCtx = aoCanvas.getContext('2d')!;
  const aoImg = aoCtx.createImageData(width, height);

  const heights = new Float32Array(width * height);

  for (let py = 0; py < height; py++) {
    // y=0 -> v=1 (snout tip), y=height-1 -> v=0 (tail tip)
    const v = 1 - py / (height - 1);

    const isTail = 1 - smoothstep(0.37, 0.47, v);
    const isHead = smoothstep(0.76, 0.84, v);
    const isNeck = smoothstep(0.67, 0.72, v) * (1 - smoothstep(0.77, 0.83, v));
    const isTorso = clamp01(1 - isTail - isHead);

    for (let px = 0; px < width; px++) {
      const u = px / width;
      // 0 = dorsal midline, 0.25 = mid-flank, 0.5 = ventral belly midline
      const spineDist = Math.min(u, 1 - u);

      const dorsalMask = 1 - smoothstep(0.14, 0.23, spineDist);
      const ventralMask = smoothstep(0.35, 0.44, spineDist);
      const flankMask = clamp01(1 - dorsalMask - ventralMask);

      // Multi-scale organic warp so scales look living and anatomical rather than rigid grids
      const warpU = (fbm2(u * 16 + 3.1, v * 36 + 1.7, 2) - 0.5) * 0.012;
      const warpV = (fbm2(u * 18 + 7.4, v * 40 + 5.2, 2) - 0.5) * 0.008;

      // 1. MULTI-ZONE HIGH-DENSITY REPTILE SCALE HEIGHTFIELD (zone-gated for instant startup!)
      let bellyHeight = 0.5;
      if (ventralMask > 0.01) {
        const bellyCols = 18;
        const bellyRows = 104;
        const bx = ((u + warpU * 0.3) * bellyCols) % 1;
        const by = ((v + warpV * 0.3) * bellyRows) % 1;
        const bxP = bx < 0 ? bx + 1 : bx;
        const byP = by < 0 ? by + 1 : by;
        const bEdgeX = bxP < 0.5 ? bxP : 1 - bxP;
        const bEdgeY = byP < 0.5 ? byP : 1 - byP;
        const bellyBevel = smoothstep(0.0, 0.13, bEdgeX) * smoothstep(0.0, 0.15, bEdgeY);
        bellyHeight = 0.16 + 0.76 * Math.sqrt(bellyBevel);
      }

      // Dorsal armor: structured armored bony plates with subtle raised central keel
      let dorsalHeight = 0.5;
      let dorsalCellHash = 0.5;
      if (dorsalMask > 0.01 && isHead < 0.99 && isTail < 0.99) {
        evalVoronoiTileable(u + warpU * 0.6, v + warpV * 0.5, 28, 86, 0.36, 11);
        const dorsalDome = smoothstep(0.02, 0.22, vorEdge);
        const dK = (vorF1 - 0.04) * 3.8;
        const dorsalKeel = Math.exp(-dK * dK) * 0.38;
        dorsalHeight = 0.14 + 0.68 * Math.pow(dorsalDome, 0.62) + dorsalKeel;
        dorsalCellHash = vorH1;
      }

      // Flank & Neck: FINE DENSE PEBBLED GRANULAR SCALES + scattered lateral osteoderm bosses!
      let flankHeight = 0.5;
      let flankCellHash = 0.5;
      let flankCellHash2 = 0.5;
      let bossHeight = 0;
      if (flankMask > 0.01 && isHead < 0.99 && isTail < 0.99) {
        evalVoronoiTileable(u + warpU, v + warpV, 64, 128, 0.76, 37);
        const flankDome = smoothstep(0.02, 0.22, vorEdge);
        flankCellHash = vorH1;
        flankCellHash2 = vorH2;
        if (isTorso > 0.02 && spineDist > 0.17 && spineDist < 0.39) {
          evalVoronoiTileable(u, v, 24, 52, 0.52, 79);
          const bossMask =
            smoothstep(0.74, 0.86, vorH2) *
            smoothstep(0.18, 0.30, spineDist) *
            (1 - smoothstep(0.30, 0.38, spineDist)) *
            isTorso;
          if (bossMask > 0.01) {
            bossHeight = Math.exp(-vorF1 * vorF1 * 9.5) * 0.65 * bossMask;
          }
        }
        flankHeight = 0.14 + 0.66 * Math.pow(flankDome, 0.65) + bossHeight;
      }

      // Head & Snout: heavily rugose, pitted crocodilian dermal skull texture + fine sensory pits
      let headHeight = 0.5;
      let skullRugosity = 0.5;
      let pitNoise = 0;
      if (isHead > 0.01) {
        evalVoronoiTileable(u + warpU * 1.2, v + warpV * 1.1, 68, 136, 0.86, 131);
        const headDome = smoothstep(0.02, 0.19, vorEdge);
        skullRugosity = fbm2(u * 104, v * 208, 3);
        pitNoise = smoothstep(0.72, 0.86, valueNoise2(u * 180, v * 340));
        headHeight = clamp01(0.18 + 0.56 * headDome + (skullRugosity - 0.5) * 0.34 - pitNoise * 0.26);
      }

      // Tail: crisp banded rectangular-shield caudal scales wrapping circumference
      let tailHeight = 0.5;
      if (isTail > 0.01) {
        const tailCols = 32;
        const tailRows = 90;
        const tx = ((u + warpU * 0.25) * tailCols) % 1;
        const ty = ((v + warpV * 0.25) * tailRows) % 1;
        const txPos = tx < 0 ? tx + 1 : tx;
        const tyPos = ty < 0 ? ty + 1 : ty;
        const tEdge = Math.min(Math.min(txPos, 1 - txPos), Math.min(tyPos, 1 - tyPos));
        const tailDome = smoothstep(0.0, 0.16, tEdge);
        tailHeight = 0.14 + 0.78 * Math.sqrt(tailDome);
      }

      // Blend zones cleanly
      let scaleH =
        dorsalHeight * dorsalMask +
        flankHeight * flankMask +
        bellyHeight * ventralMask;
      scaleH = mix(scaleH, tailHeight, isTail * (1 - ventralMask * 0.45));
      scaleH = mix(scaleH, headHeight, isHead * (1 - ventralMask * 0.55));

      // Fine leather micro-wrinkles & organic pore grain
      const microGrain = fbm2(u * 170, v * 320, 2);
      const finalH = clamp01(scaleH * 0.87 + microGrain * 0.13);
      heights[py * width + px] = finalH;

      // Crisp crevice factor (deep recessed shadow between every scale)
      const crevice = smoothstep(0.18, 0.50, scaleH);

      // 2. PHOTOREALISTIC MUTED NILE CROCODILE ALBEDO (matching reference photo!)
      // Deep weathered olive-slate dorsum (#3d4137) with subtle mossy/dusty scale-to-scale patina
      const cellTint = (dorsalCellHash - 0.5) * 0.12 + (microGrain - 0.5) * 0.08;
      let r = 64 + cellTint * 74;
      let g = 68 + cellTint * 72;
      let b = 57 + cellTint * 62;

      // Subtle dusty-olive mineral weathering along top ridge crowns
      if (dorsalMask > 0.02) {
        const dustPatina = smoothstep(0.54, 0.78, fbm2(u * 20 + 2.5, v * 48 + 9.1, 2));
        r = mix(r, 88, dustPatina * dorsalMask * 0.35);
        g = mix(g, 90, dustPatina * dorsalMask * 0.35);
        b = mix(b, 77, dustPatina * dorsalMask * 0.35);
      }

      // Head & Snout: weathered grey-olive (#55594e) with darker rugose pits & subtle pale jawline
      if (isHead > 0.01) {
        const jawLipFade = smoothstep(0.24, 0.36, spineDist);
        let hr = 84 + (skullRugosity - 0.5) * 34 - pitNoise * 26;
        let hg = 87 + (skullRugosity - 0.5) * 32 - pitNoise * 25;
        let hb = 75 + (skullRugosity - 0.5) * 28 - pitNoise * 22;
        const snoutMottle = smoothstep(0.48, 0.70, fbm2(u * 14, v * 44, 2));
        hr = mix(hr, 52, snoutMottle * 0.48);
        hg = mix(hg, 55, snoutMottle * 0.48);
        hb = mix(hb, 47, snoutMottle * 0.48);
        hr = mix(hr, 126, jawLipFade * 0.48);
        hg = mix(hg, 124, jawLipFade * 0.48);
        hb = mix(hb, 104, jawLipFade * 0.48);

        r = mix(r, hr, isHead * (1 - ventralMask * 0.85));
        g = mix(g, hg, isHead * (1 - ventralMask * 0.85));
        b = mix(b, hb, isHead * (1 - ventralMask * 0.85));
      }

      // FLANKS: Muted warm dusty khaki-ochre (#a59a6e) richly dappled with dark charcoal blotches & vertical bars!
      if (flankMask > 0.01) {
        const scaleRand = (flankCellHash - 0.5) * 26;
        const ochreR = 166 + scaleRand;
        const ochreG = 156 + scaleRand * 0.94;
        const ochreB = 112 + scaleRand * 0.82;

        const mWarp = fbm2(u * 12 + 4.1, v * 28 + 1.3, 2) * 0.38;
        const vertBars = Math.sin((v * 34 + mWarp * 9.5) * Math.PI);
        const macroBlotch = fbm2(u * 18 + mWarp * 2.2, v * 44, 3);
        const microSpots = smoothstep(0.62, 0.78, flankCellHash2);
        const darkPattern = clamp01(
          smoothstep(0.42, 0.59, macroBlotch + vertBars * 0.19) * 0.92 +
            microSpots * 0.38 * (1 - smoothstep(0.28, 0.38, spineDist))
        );

        const bossDark = clamp01(bossHeight * 1.45);
        const flDark = clamp01(Math.max(darkPattern, bossDark));

        const flR = mix(ochreR, 37, flDark * 0.88);
        const flG = mix(ochreG, 40, flDark * 0.88);
        const flB = mix(ochreB, 34, flDark * 0.88);

        const neckTone = isNeck * 0.42;
        const flFinalR = mix(flR, 108 + scaleRand * 0.6, neckTone);
        const flFinalG = mix(flG, 106 + scaleRand * 0.6, neckTone);
        const flFinalB = mix(flB, 86 + scaleRand * 0.5, neckTone);

        const flankStrength = flankMask * (1 - isHead * 0.72);
        r = mix(r, flFinalR, flankStrength);
        g = mix(g, flFinalG, flankStrength);
        b = mix(b, flFinalB, flankStrength);
      }

      // TAIL BANDING: Bold alternating dusty tan-ochre & deep charcoal-black cross-bands!
      if (isTail > 0.01) {
        const bandWarp = (fbm2(u * 10, v * 24, 2) - 0.5) * 0.055;
        const bandWave = Math.sin(((v + bandWarp) / 0.43) * Math.PI * 14.0);
        const darkBand = smoothstep(-0.16, 0.24, bandWave);
        const bandSpeckle = smoothstep(0.58, 0.78, fbm2(u * 38, v * 72, 2)) * 0.42;

        const lightR = 172 + (microGrain - 0.5) * 20;
        const lightG = 160 + (microGrain - 0.5) * 18;
        const lightB = 114 + (microGrain - 0.5) * 16;

        const darkR = 34 + (microGrain - 0.5) * 12;
        const darkG = 37 + (microGrain - 0.5) * 12;
        const darkB = 33 + (microGrain - 0.5) * 10;

        const topDarken = dorsalMask * 0.52;
        const bandMix = clamp01(darkBand + topDarken + bandSpeckle * (1 - darkBand) * 0.5);
        const tbR = mix(lightR, darkR, bandMix);
        const tbG = mix(lightG, darkG, bandMix);
        const tbB = mix(lightB, darkB, bandMix);

        r = mix(r, tbR, isTail * (1 - ventralMask * 0.70));
        g = mix(g, tbG, isTail * (1 - ventralMask * 0.70));
        b = mix(b, tbB, isTail * (1 - ventralMask * 0.70));
      }

      // VENTRAL BELLY & THROAT: Muted warm cream-sand rectangular tiles (#c5bc97) with warm taupe grout
      if (ventralMask > 0.01) {
        const bellyTint = (fbm2(u * 18, v * 42, 3) - 0.5) * 18;
        const bellyR = 192 + bellyTint;
        const bellyG = 183 + bellyTint * 0.95;
        const bellyB = 144 + bellyTint * 0.85;
        r = mix(r, bellyR, ventralMask);
        g = mix(g, bellyG, ventralMask);
        b = mix(b, bellyB, ventralMask);
      }

      // Subtle dark dorsal saddles across torso back
      if (isTorso > 0.01 && dorsalMask > 0.1) {
        const saddle = smoothstep(0.10, 0.62, Math.sin(v * Math.PI * 18 + fbm2(u * 8, v * 16, 2) * 1.4));
        const darken = 1 - saddle * 0.28 * isTorso * dorsalMask;
        r *= darken;
        g *= darken;
        b *= darken;
      }

      // Deep realistic inter-scale crevices + subtle warm dry-dust edge catchlight on raised scale ridges
      const creviceFactor = mix(0.28, 1.0, crevice);
      r *= creviceFactor;
      g *= creviceFactor;
      b *= creviceFactor;

      const crownLift = smoothstep(0.74, 0.96, finalH) * 15;
      r = Math.min(255, Math.max(0, Math.round(r + crownLift)));
      g = Math.min(255, Math.max(0, Math.round(g + crownLift * 0.96)));
      b = Math.min(255, Math.max(0, Math.round(b + crownLift * 0.84)));

      const idx = (py * width + px) * 4;
      albedoImg.data[idx] = r;
      albedoImg.data[idx + 1] = g;
      albedoImg.data[idx + 2] = b;
      albedoImg.data[idx + 3] = 255;

      // 3. ROUGHNESS MAP: Dry dusty hide crevices (~0.92), smoother satin highlights on scale crowns & bosses (~0.42)
      const crownSmooth = smoothstep(0.56, 0.94, finalH);
      const baseRough = mix(0.92, 0.44, crownSmooth * (0.62 + dorsalMask * 0.38));
      const roughByte = Math.round(clamp01(baseRough + (microGrain - 0.5) * 0.08) * 255);
      roughImg.data[idx] = roughByte;
      roughImg.data[idx + 1] = roughByte;
      roughImg.data[idx + 2] = roughByte;
      roughImg.data[idx + 3] = 255;

      // 4. AMBIENT OCCLUSION: Deep darkened scale grout + limb socket & throat shadows
      const shDist = Math.hypot((v - 0.67) * 4.5, (spineDist - 0.28) * 3.5);
      const hipDist = Math.hypot((v - 0.48) * 4.5, (spineDist - 0.27) * 3.5);
      const socketAO =
        1 -
        Math.exp(-shDist * shDist * 2.2) * 0.34 -
        Math.exp(-hipDist * hipDist * 2.2) * 0.34;
      const aoVal = clamp01(mix(0.36, 1.0, crevice) * socketAO);
      const aoByte = Math.round(aoVal * 255);
      aoImg.data[idx] = aoByte;
      aoImg.data[idx + 1] = aoByte;
      aoImg.data[idx + 2] = aoByte;
      aoImg.data[idx + 3] = 255;
    }
  }

  albedoCtx.putImageData(albedoImg, 0, 0);
  roughCtx.putImageData(roughImg, 0, 0);
  aoCtx.putImageData(aoImg, 0, 0);

  const normalCanvas = heightFieldToNormalCanvas(heights, width, height, 4.8);
  return { albedoCanvas, normalCanvas, roughnessCanvas: roughCanvas, aoCanvas };
}

/**
 * Generates crisp 1024x1024 Limb PBR textures: dense dark charcoal-olive armored outer shield scales
 * transitioning cleanly into mottled warm dusty-ochre inner thighs & elbows matching the reference!
 */
function generateLimbTextures(size = 1024): {
  albedoCanvas: HTMLCanvasElement;
  normalCanvas: HTMLCanvasElement;
  roughnessCanvas: HTMLCanvasElement;
} {
  const albedoCanvas = document.createElement('canvas');
  albedoCanvas.width = size;
  albedoCanvas.height = size;
  const albedoCtx = albedoCanvas.getContext('2d')!;
  const albedoImg = albedoCtx.createImageData(size, size);

  const roughCanvas = document.createElement('canvas');
  roughCanvas.width = size;
  roughCanvas.height = size;
  const roughCtx = roughCanvas.getContext('2d')!;
  const roughImg = roughCtx.createImageData(size, size);

  const heights = new Float32Array(size * size);

  for (let py = 0; py < size; py++) {
    const v = 1 - py / (size - 1); // 0 = shoulder/hip, 1 = foot/toes
    for (let px = 0; px < size; px++) {
      const u = px / size;
      const innerSide = smoothstep(0.24, 0.44, Math.min(u, 1 - u));

      // Dense overlapping hexagonal/shield limb scales (finer near toes)
      const scaleX = Math.round(mix(28, 40, v));
      const scaleY = Math.round(mix(42, 56, v));
      evalVoronoiTileable(u, v, scaleX, scaleY, 0.56, 211);
      const dome = smoothstep(0.02, 0.22, vorEdge);
      const grain = fbm2(u * 110, v * 130, 2);
      const h = clamp01(0.14 + Math.pow(dome, 0.62) * 0.74 + grain * 0.12);
      heights[py * size + px] = h;

      const crevice = smoothstep(0.20, 0.52, h);

      // Outer limb: dark weathered charcoal-slate (#383c34) with subtle ochre flecks & knee wear
      const blotch = smoothstep(0.46, 0.66, fbm2(u * 14 + 2.3, v * 18 + 5.1, 2));
      const scaleVar = (vorH1 - 0.5) * 20;
      let r = mix(54 + scaleVar, 134 + scaleVar, blotch * (1 - v * 0.52) * 0.45);
      let g = mix(58 + scaleVar, 126 + scaleVar, blotch * (1 - v * 0.52) * 0.45);
      let b = mix(49 + scaleVar, 92 + scaleVar, blotch * (1 - v * 0.52) * 0.45);

      // Inner/ventral limb: warm dusty ochre (#b8ad7f) with dark charcoal bands & spots
      const innerBand = smoothstep(0.48, 0.66, fbm2(u * 18, v * 28, 3));
      const ir = mix(178, 44, innerBand * 0.76);
      const ig = mix(168, 47, innerBand * 0.76);
      const ib = mix(124, 39, innerBand * 0.76);

      r = mix(r, ir, innerSide * (1 - v * 0.35));
      g = mix(g, ig, innerSide * (1 - v * 0.35));
      b = mix(b, ib, innerSide * (1 - v * 0.35));

      const cf = mix(0.26, 1.0, crevice);
      const crownLift = smoothstep(0.76, 0.95, h) * 12;
      const idx = (py * size + px) * 4;
      albedoImg.data[idx] = Math.min(255, Math.max(0, Math.round(r * cf + crownLift)));
      albedoImg.data[idx + 1] = Math.min(255, Math.max(0, Math.round(g * cf + crownLift * 0.95)));
      albedoImg.data[idx + 2] = Math.min(255, Math.max(0, Math.round(b * cf + crownLift * 0.85)));
      albedoImg.data[idx + 3] = 255;

      const rough = Math.round(mix(0.90, 0.46, smoothstep(0.56, 0.92, h)) * 255);
      roughImg.data[idx] = rough;
      roughImg.data[idx + 1] = rough;
      roughImg.data[idx + 2] = rough;
      roughImg.data[idx + 3] = 255;
    }
  }

  albedoCtx.putImageData(albedoImg, 0, 0);
  roughCtx.putImageData(roughImg, 0, 0);
  const normalCanvas = heightFieldToNormalCanvas(heights, size, size, 4.4);
  return { albedoCanvas, normalCanvas, roughnessCanvas: roughCanvas };
}

/**
 * Generates a high-impact luminous golden-amber Nile Crocodile iris texture with detailed radial fibers,
 * dark limbal ring, and a razor-sharp vertical black slit pupil!
 */
function generateEyeIrisCanvas(size = 512): HTMLCanvasElement {
  const canvas = document.createElement('canvas');
  canvas.width = size;
  canvas.height = size;
  const ctx = canvas.getContext('2d')!;

  // Dark olive-black sclera surrounding the large golden reptile iris
  ctx.fillStyle = '#181a14';
  ctx.fillRect(0, 0, size, size);

  const cx = size * 0.5;
  const cy = size * 0.5;
  // Slightly larger iris radius in equirectangular UV space so the whole visible eye opening shows rich gold!
  const irisRx = size * 0.28;
  const irisRy = size * 0.40;

  ctx.save();
  ctx.translate(cx, cy);
  ctx.scale(irisRx / irisRy, 1);

  const grad = ctx.createRadialGradient(0, 0, irisRy * 0.08, 0, 0, irisRy);
  grad.addColorStop(0.0, '#f4e08a');
  grad.addColorStop(0.42, '#d8b654');
  grad.addColorStop(0.75, '#9d8236');
  grad.addColorStop(0.92, '#3c3419');
  grad.addColorStop(1.0, '#12140f');

  ctx.fillStyle = grad;
  ctx.beginPath();
  ctx.arc(0, 0, irisRy, 0, Math.PI * 2);
  ctx.fill();

  // Radial fibrous reptilian iris striations & delicate gold highlights
  ctx.save();
  ctx.beginPath();
  ctx.arc(0, 0, irisRy * 0.95, 0, Math.PI * 2);
  ctx.clip();

  for (let i = 0; i < 220; i++) {
    const angle = (i / 220) * Math.PI * 2;
    const h = hash2(i * 1.7, 3.9);
    const len = irisRy * (0.48 + h * 0.50);
    ctx.strokeStyle = h > 0.45 ? 'rgba(255, 240, 165, 0.34)' : 'rgba(38, 30, 14, 0.34)';
    ctx.lineWidth = 1.4 + h * 2.0;
    ctx.beginPath();
    ctx.moveTo(Math.cos(angle) * irisRy * 0.10, Math.sin(angle) * irisRy * 0.10);
    ctx.lineTo(Math.cos(angle) * len, Math.sin(angle) * len);
    ctx.stroke();
  }

  for (let i = 0; i < 110; i++) {
    const a = hash2(i * 3.1, 7.2) * Math.PI * 2;
    const r = irisRy * (0.26 + hash2(i * 5.7, 1.3) * 0.66);
    ctx.fillStyle = 'rgba(26, 22, 12, 0.36)';
    ctx.beginPath();
    ctx.arc(Math.cos(a) * r, Math.sin(a) * r, 2.5 + hash2(i, 9.1) * 4.5, 0, Math.PI * 2);
    ctx.fill();
  }
  ctx.restore();

  // Razor-sharp vertical cat/croc slit pupil with luminous gold inner rim
  ctx.beginPath();
  ctx.moveTo(0, -irisRy * 0.74);
  ctx.bezierCurveTo(irisRy * 0.21, -irisRy * 0.32, irisRy * 0.21, irisRy * 0.32, 0, irisRy * 0.74);
  ctx.bezierCurveTo(-irisRy * 0.21, irisRy * 0.32, -irisRy * 0.21, -irisRy * 0.32, 0, -irisRy * 0.74);
  ctx.closePath();
  ctx.lineWidth = 5;
  ctx.strokeStyle = 'rgba(255, 238, 152, 0.80)';
  ctx.stroke();
  ctx.fillStyle = '#040403';
  ctx.fill();

  ctx.restore();
  return canvas;
}

/** Soft oval studio contact shadow texture */
function generateContactShadowCanvas(size = 512): HTMLCanvasElement {
  const canvas = document.createElement('canvas');
  canvas.width = size;
  canvas.height = size;
  const ctx = canvas.getContext('2d')!;
  ctx.clearRect(0, 0, size, size);

  const grad = ctx.createRadialGradient(
    size * 0.5,
    size * 0.5,
    size * 0.05,
    size * 0.5,
    size * 0.5,
    size * 0.48
  );
  grad.addColorStop(0.0, 'rgba(16, 18, 20, 0.68)');
  grad.addColorStop(0.45, 'rgba(16, 18, 20, 0.38)');
  grad.addColorStop(0.78, 'rgba(16, 18, 20, 0.11)');
  grad.addColorStop(1.0, 'rgba(16, 18, 20, 0.0)');
  ctx.fillStyle = grad;
  ctx.fillRect(0, 0, size, size);
  return canvas;
}

/** Tileable procedural water ripple normal map */
function generateWaterNormalCanvas(size = 512): HTMLCanvasElement {
  const heights = new Float32Array(size * size);
  const twoPi = Math.PI * 2;
  for (let y = 0; y < size; y++) {
    const v = y / size;
    for (let x = 0; x < size; x++) {
      const u = x / size;
      let h = 0;
      h += Math.sin(twoPi * (u * 3 + v * 2)) * 0.28;
      h += Math.sin(twoPi * (u * 5 - v * 4 + 0.3)) * 0.22;
      h += Math.sin(twoPi * (u * 8 + v * 7 + 0.7)) * 0.16;
      h += Math.sin(twoPi * (u * 13 - v * 11 + 0.15)) * 0.11;
      h += Math.sin(twoPi * (u * 21 + v * 17 + 0.5)) * 0.07;
      heights[y * size + x] = h * 0.5 + 0.5;
    }
  }
  return heightFieldToNormalCanvas(heights, size, size, 2.2);
}

/**
 * Creates and configures the full high-resolution procedural PBR texture set.
 */
export function createCrocodileTextures(maxAnisotropy: number): CrocodileTextureSet {
  const body = generateBodyTextures(1024, 1024);
  const limb = generateLimbTextures(512);
  const eyeCanvas = generateEyeIrisCanvas(512);
  const shadowCanvas = generateContactShadowCanvas(512);
  const waterNormCanvas = generateWaterNormalCanvas(256);

  const setupTex = (canvas: HTMLCanvasElement, isColor: boolean, repeat = true): THREE.CanvasTexture => {
    const tex = new THREE.CanvasTexture(canvas);
    tex.colorSpace = isColor ? THREE.SRGBColorSpace : THREE.NoColorSpace;
    tex.wrapS = repeat ? THREE.RepeatWrapping : THREE.ClampToEdgeWrapping;
    tex.wrapT = repeat ? THREE.RepeatWrapping : THREE.ClampToEdgeWrapping;
    tex.anisotropy = Math.max(1, Math.min(16, maxAnisotropy));
    tex.needsUpdate = true;
    return tex;
  };

  const bodyAlbedo = setupTex(body.albedoCanvas, true, true);
  const bodyNormal = setupTex(body.normalCanvas, false, true);
  const bodyRoughness = setupTex(body.roughnessCanvas, false, true);
  const bodyAO = setupTex(body.aoCanvas, false, true);

  const limbAlbedo = setupTex(limb.albedoCanvas, true, true);
  const limbNormal = setupTex(limb.normalCanvas, false, true);
  const limbRoughness = setupTex(limb.roughnessCanvas, false, true);

  const eyeIris = setupTex(eyeCanvas, true, false);
  const contactShadow = setupTex(shadowCanvas, false, false);
  const waterNormal = setupTex(waterNormCanvas, false, true);
  waterNormal.repeat.set(10, 10);

  return {
    bodyAlbedo,
    bodyNormal,
    bodyRoughness,
    bodyAO,
    limbAlbedo,
    limbNormal,
    limbRoughness,
    eyeIris,
    contactShadow,
    waterNormal,
    dispose: () => {
      bodyAlbedo.dispose();
      bodyNormal.dispose();
      bodyRoughness.dispose();
      bodyAO.dispose();
      limbAlbedo.dispose();
      limbNormal.dispose();
      limbRoughness.dispose();
      eyeIris.dispose();
      contactShadow.dispose();
      waterNormal.dispose();
    },
  };
}
