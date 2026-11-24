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

/** Fast deterministic 2D hash */
function hash2(ix: number, iy: number): number {
  const n = Math.sin(ix * 127.1 + iy * 311.7) * 43758.5453123;
  return n - Math.floor(n);
}

/** 2D value noise with smoothstep interpolation */
function valueNoise2(x: number, y: number): number {
  const ix = Math.floor(x);
  const iy = Math.floor(y);
  const fx = x - ix;
  const fy = y - iy;
  const ux = fx * fx * (3 - 2 * fx);
  const uy = fy * fy * (3 - 2 * fy);
  const a = hash2(ix, iy);
  const b = hash2(ix + 1, iy);
  const c = hash2(ix, iy + 1);
  const d = hash2(ix + 1, iy + 1);
  return a + (b - a) * ux + (c - a) * uy + (a - b - c + d) * ux * uy;
}

function fbm2(x: number, y: number, octaves = 4): number {
  let value = 0;
  let amp = 0.5;
  let freq = 1;
  for (let i = 0; i < octaves; i++) {
    value += amp * valueNoise2(x * freq, y * freq);
    freq *= 2.04;
    amp *= 0.5;
  }
  return value;
}

/** Tileable Voronoi cellular distance for organic pebbled reptile scales */
function voronoiTileable(
  u: number,
  v: number,
  cellsX: number,
  cellsY: number,
  jitter = 0.72
): { f1: number; edgeDist: number; cellHash: number } {
  const gx = u * cellsX;
  const gy = v * cellsY;
  const ix = Math.floor(gx);
  const iy = Math.floor(gy);
  const fx = gx - ix;
  const fy = gy - iy;

  let d1 = 99;
  let d2 = 99;
  let bestHash = 0;

  for (let dy = -1; dy <= 1; dy++) {
    for (let dx = -1; dx <= 1; dx++) {
      const cx = ((ix + dx) % cellsX + cellsX) % cellsX;
      const cy = ((iy + dy) % cellsY + cellsY) % cellsY;
      const h1 = hash2(cx * 1.13 + 3.1, cy * 1.71 + 7.4);
      const h2 = hash2(cx * 2.37 + 19.3, cy * 0.91 + 11.7);
      const px = dx + 0.5 + (h1 - 0.5) * jitter - fx;
      const py = dy + 0.5 + (h2 - 0.5) * jitter - fy;
      const dist = Math.sqrt(px * px + py * py);
      if (dist < d1) {
        d2 = d1;
        d1 = dist;
        bestHash = h1;
      } else if (dist < d2) {
        d2 = dist;
      }
    }
  }
  return { f1: d1, edgeDist: d2 - d1, cellHash: bestHash };
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

/** Converts a float height buffer [0..1] into a tangent-space normal map CanvasTexture */
function heightFieldToNormalCanvas(
  heights: Float32Array,
  size: number,
  strength: number
): HTMLCanvasElement {
  const canvas = document.createElement('canvas');
  canvas.width = size;
  canvas.height = size;
  const ctx = canvas.getContext('2d')!;
  const img = ctx.createImageData(size, size);
  const data = img.data;

  for (let y = 0; y < size; y++) {
    const ym = (y - 1 + size) % size;
    const yp = (y + 1) % size;
    for (let x = 0; x < size; x++) {
      const xm = (x - 1 + size) % size;
      const xp = (x + 1) % size;

      // Sobel kernel for smooth, artifact-free normal derivatives
      const tl = heights[ym * size + xm];
      const tc = heights[ym * size + x];
      const tr = heights[ym * size + xp];
      const ml = heights[y * size + xm];
      const mr = heights[y * size + xp];
      const bl = heights[yp * size + xm];
      const bc = heights[yp * size + x];
      const br = heights[yp * size + xp];

      const dX = (tr + 2 * mr + br - (tl + 2 * ml + bl)) * strength;
      const dY = (bl + 2 * bc + br - (tl + 2 * tc + tr)) * strength;
      const dZ = 1.0;

      const invLen = 1 / Math.sqrt(dX * dX + dY * dY + dZ * dZ);
      const idx = (y * size + x) * 4;
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
 * Generates high-resolution 1024x1024 anatomically zoned Nile Crocodile PBR maps
 * matching the attached reference specimen:
 * - Olive-charcoal weathered dorsal armor
 * - Pale ochre/sand flanks with bold dark charcoal mottling & blotches
 * - Crisp alternating cream-ochre & charcoal tail cross-bands
 * - Warm ivory-cream rectangular tiled ventral belly scales
 * - Rugose pitted snout & dark inter-scale crevices
 */
function generateBodyTextures(size = 1024): {
  albedoCanvas: HTMLCanvasElement;
  normalCanvas: HTMLCanvasElement;
  roughnessCanvas: HTMLCanvasElement;
  aoCanvas: HTMLCanvasElement;
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

  const aoCanvas = document.createElement('canvas');
  aoCanvas.width = size;
  aoCanvas.height = size;
  const aoCtx = aoCanvas.getContext('2d')!;
  const aoImg = aoCtx.createImageData(size, size);

  const heights = new Float32Array(size * size);

  for (let py = 0; py < size; py++) {
    // Canvas y=0 corresponds to UV v=1 (snout tip), y=size-1 to UV v=0 (tail tip)
    const v = 1 - py / (size - 1);

    // Anatomical longitudinal zones:
    // Tail: v in [0, 0.44], Torso/Neck: [0.44, 0.80], Head/Snout: [0.80, 1.0]
    const isTail = 1 - smoothstep(0.38, 0.48, v);
    const isHead = smoothstep(0.77, 0.85, v);
    const isTorso = clamp01(1 - isTail - isHead);

    for (let px = 0; px < size; px++) {
      const u = px / size;
      // Circumferential distance from dorsal spine midline:
      // 0 = dorsal top, 0.25 = mid-flank, 0.5 = ventral belly midline
      const distFromSpine = Math.abs(((u + 0.5) % 1) - 0.5);
      // Convert: 0 at spine (u=0 or 1), 0.5 at belly (u=0.5)
      const dorsalToVentral = 0.5 - distFromSpine; // wait: at u=0 => |0.5 - 0.5| = 0 -> let's be exact!
      const spineDist = Math.min(u, 1 - u); // 0 at u=0/1 (dorsal), 0.25 at flank, 0.5 at u=0.5 (ventral)
      void dorsalToVentral;

      const dorsalMask = 1 - smoothstep(0.14, 0.24, spineDist);
      const ventralMask = smoothstep(0.34, 0.43, spineDist);
      const flankMask = clamp01(1 - dorsalMask - ventralMask);

      // 1. MULTI-ZONE SCALE HEIGHTFIELD
      // Rectangular tiled ventral belly plates (classic crocodile belly pattern!)
      const bellyCols = 14;
      const bellyRows = 68;
      const bx = (u * bellyCols) % 1;
      const by = (v * bellyRows) % 1;
      const bellyEdgeX = Math.min(bx, 1 - bx);
      const bellyEdgeY = Math.min(by, 1 - by);
      const bellyBevel = smoothstep(0.0, 0.14, bellyEdgeX) * smoothstep(0.0, 0.16, bellyEdgeY);
      const bellyHeight = 0.22 + 0.72 * Math.pow(bellyBevel, 0.55);

      // Dorsal armored rows (large rectangular-oval scute tiles along back)
      const dorsalCols = 22;
      const dorsalRows = 54;
      const dJitter = (fbm2(u * 14, v * 28, 2) - 0.5) * 0.12;
      const vorDorsal = voronoiTileable(u + dJitter * 0.015, v, dorsalCols, dorsalRows, 0.42);
      const dorsalScaleDome = smoothstep(0.03, 0.22, vorDorsal.edgeDist);
      const dorsalKeel = Math.exp(-Math.pow((vorDorsal.f1 - 0.05) * 3.5, 2)) * 0.35;
      const dorsalHeight = 0.18 + 0.72 * dorsalScaleDome + dorsalKeel;

      // Flank pebbled cellular scales (smaller rounded oval pebbles + occasional raised lateral tubercles)
      const vorFlank = voronoiTileable(u, v, 44, 72, 0.78);
      const flankScaleDome = smoothstep(0.02, 0.20, vorFlank.edgeDist);
      const flankHeight = 0.20 + 0.68 * Math.pow(flankScaleDome, 0.7);

      // Head & snout finer rugose pitted scales (matches crocodile cranial bone texture)
      const vorHead = voronoiTileable(u, v, 56, 84, 0.85);
      const headScale = smoothstep(0.02, 0.18, vorHead.edgeDist);
      const microPit = fbm2(u * 90, v * 120, 3);
      const headHeight = 0.25 + 0.55 * headScale + (microPit - 0.5) * 0.25;

      // Tail banded scales (structured rectangular bands around tail circumference)
      const tailCols = 26;
      const tailRows = 62;
      const tx = (u * tailCols) % 1;
      const ty = (v * tailRows) % 1;
      const tEdge = Math.min(Math.min(tx, 1 - tx), Math.min(ty, 1 - ty));
      const tailScaleDome = smoothstep(0.0, 0.16, tEdge);
      const tailHeight = 0.18 + 0.74 * Math.pow(tailScaleDome, 0.6);

      // Combine scale height smoothly across anatomical zones
      let scaleH =
        dorsalHeight * dorsalMask +
        flankHeight * flankMask +
        bellyHeight * ventralMask;
      scaleH = mix(scaleH, tailHeight, isTail * (1 - ventralMask * 0.5));
      scaleH = mix(scaleH, headHeight, isHead * (1 - ventralMask * 0.6));

      // Fine leather grain micro-relief
      const microGrain = fbm2(u * 130, v * 130, 2);
      const finalH = clamp01(scaleH * 0.88 + microGrain * 0.12);
      heights[py * size + px] = finalH;

      // Crevice mask (0 inside deep scale grout lines, 1 on scale crown)
      const crevice = smoothstep(0.22, 0.56, scaleH);

      // 2. REFERENCE-MATCHED NILE CROCODILE ALBEDO PALETTE
      // Base dorsal olive-charcoal (#494d42) with subtle scale-to-scale organic variation
      const cellTint = (vorDorsal.cellHash - 0.5) * 0.10 + (microGrain - 0.5) * 0.06;
      let r = 74 + cellTint * 85;
      let g = 77 + cellTint * 82;
      let b = 65 + cellTint * 70;

      // Dusty olive-grey head & snout (#62665b)
      if (isHead > 0.01) {
        const hr = 96 + (microPit - 0.5) * 32;
        const hg = 99 + (microPit - 0.5) * 30;
        const hb = 86 + (microPit - 0.5) * 26;
        r = mix(r, hr, isHead * (1 - ventralMask));
        g = mix(g, hg, isHead * (1 - ventralMask));
        b = mix(b, hb, isHead * (1 - ventralMask));
      }

      // FLANKS: Warm pale cream-ochre (#c8bc8a) broken by bold charcoal-black vertical blotches & mottling!
      if (flankMask > 0.01) {
        // Pale ochre/sand base
        const ochreR = 198 + (vorFlank.cellHash - 0.5) * 22;
        const ochreG = 186 + (vorFlank.cellHash - 0.5) * 20;
        const ochreB = 136 + (vorFlank.cellHash - 0.5) * 18;

        // Bold charcoal mottling blotches (using warped multi-octave fbm + vertical stripes)
        const warp = fbm2(u * 10 + 4.2, v * 16 + 1.3, 3) * 0.35;
        const stripeWave = Math.sin((v * 26 + warp * 8) * Math.PI);
        const blotchNoise = fbm2(u * 16 + warp * 2.5, v * 26, 4);
        const darkBlotch = smoothstep(0.46, 0.62, blotchNoise + stripeWave * 0.16);

        // Dark charcoal-olive blotch color (#2a2d27)
        const flR = mix(ochreR, 42, darkBlotch * 0.88);
        const flG = mix(ochreG, 45, darkBlotch * 0.88);
        const flB = mix(ochreB, 38, darkBlotch * 0.88);

        // Blend flanks smoothly with torso/neck
        const flankStrength = flankMask * (1 - isHead * 0.65);
        r = mix(r, flR, flankStrength);
        g = mix(g, flG, flankStrength);
        b = mix(b, flB, flankStrength);
      }

      // TAIL BANDING: Dramatic alternating pale cream-ochre & dark charcoal cross-bands!
      if (isTail > 0.01) {
        const bandWarp = (fbm2(u * 8, v * 18, 3) - 0.5) * 0.06;
        // ~7 bold dark charcoal bands along the tail matching reference
        const bandPhase = Math.sin(((v + bandWarp) / 0.44) * Math.PI * 13.5);
        const darkBand = smoothstep(-0.18, 0.26, bandPhase);

        // Cream-ochre band vs charcoal-slate band
        const bandLightR = 196 + (microGrain - 0.5) * 18;
        const bandLightG = 183 + (microGrain - 0.5) * 16;
        const bandLightB = 133 + (microGrain - 0.5) * 14;

        const bandDarkR = 39 + (microGrain - 0.5) * 12;
        const bandDarkG = 42 + (microGrain - 0.5) * 12;
        const bandDarkB = 37 + (microGrain - 0.5) * 10;

        // Dorsal top of tail stays slightly darker than flanks
        const topDarken = dorsalMask * 0.45;
        const tbR = mix(bandLightR, bandDarkR, clamp01(darkBand + topDarken));
        const tbG = mix(bandLightG, bandDarkG, clamp01(darkBand + topDarken));
        const tbB = mix(bandLightB, bandDarkB, clamp01(darkBand + topDarken));

        r = mix(r, tbR, isTail * (1 - ventralMask * 0.72));
        g = mix(g, tbG, isTail * (1 - ventralMask * 0.72));
        b = mix(b, tbB, isTail * (1 - ventralMask * 0.72));
      }

      // VENTRAL BELLY & THROAT: Pale warm cream / ivory rectangular plates (#d4cbab)
      if (ventralMask > 0.01) {
        const bellyR = 212 + (microGrain - 0.5) * 16;
        const bellyG = 202 + (microGrain - 0.5) * 15;
        const bellyB = 166 + (microGrain - 0.5) * 14;
        r = mix(r, bellyR, ventralMask);
        g = mix(g, bellyG, ventralMask);
        b = mix(b, bellyB, ventralMask);
      }

      // Subtle dark dorsal saddles across torso back
      if (isTorso > 0.01 && dorsalMask > 0.1) {
        const saddle = smoothstep(0.15, 0.65, Math.sin(v * Math.PI * 16 + fbm2(u * 6, v * 12, 2) * 1.4));
        const darken = 1 - saddle * 0.24 * isTorso * dorsalMask;
        r *= darken;
        g *= darken;
        b *= darken;
      }

      // Crisp dark crevices between scales (critical for tactile realism!)
      const creviceFactor = mix(0.36, 1.0, crevice);
      r *= creviceFactor;
      g *= creviceFactor;
      b *= creviceFactor;

      // Subtle warm highlight dust on scale crowns
      const crownLift = smoothstep(0.76, 0.96, finalH) * 14;
      r = Math.min(255, Math.max(0, Math.round(r + crownLift)));
      g = Math.min(255, Math.max(0, Math.round(g + crownLift * 0.96)));
      b = Math.min(255, Math.max(0, Math.round(b + crownLift * 0.85)));

      const idx = (py * size + px) * 4;
      albedoImg.data[idx] = r;
      albedoImg.data[idx + 1] = g;
      albedoImg.data[idx + 2] = b;
      albedoImg.data[idx + 3] = 255;

      // 3. ROUGHNESS MAP: Dry matte hide (0.76..0.89), slightly smoother worn scale crowns (0.45..0.58)
      const crownSmoothness = smoothstep(0.60, 0.95, finalH);
      const baseRough = mix(0.88, 0.52, crownSmoothness * (0.65 + dorsalMask * 0.35));
      const roughByte = Math.round(clamp01(baseRough + (microGrain - 0.5) * 0.08) * 255);
      roughImg.data[idx] = roughByte;
      roughImg.data[idx + 1] = roughByte;
      roughImg.data[idx + 2] = roughByte;
      roughImg.data[idx + 3] = 255;

      // 4. AMBIENT OCCLUSION MAP: Darkened scale grout + leg socket / jaw hinge zones
      // Leg sockets roughly around v=0.68 (shoulder) and v=0.47 (hip) at flank (spineDist ~0.28)
      const shDist = Math.hypot((v - 0.68) * 4.5, (spineDist - 0.28) * 3.5);
      const hipDist = Math.hypot((v - 0.48) * 4.5, (spineDist - 0.27) * 3.5);
      const socketAO =
        1 -
        Math.exp(-shDist * shDist * 2.2) * 0.28 -
        Math.exp(-hipDist * hipDist * 2.2) * 0.28;
      const aoVal = clamp01(mix(0.45, 1.0, crevice) * socketAO);
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

  const normalCanvas = heightFieldToNormalCanvas(heights, size, 3.6);
  return { albedoCanvas, normalCanvas, roughnessCanvas: roughCanvas, aoCanvas };
}

/**
 * Generates specialized 512x512 Limb PBR textures: dark charcoal armored outer leg scales
 * transitioning to pale mottled cream-olive on the inner thigh/forearm (just like reference!).
 */
function generateLimbTextures(size = 512): {
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
      const innerSide = smoothstep(0.26, 0.45, Math.min(u, 1 - u)); // ventral/inner surface

      // Larger shield scales on outer forearm/shin, finer scales near toes
      const scaleFreq = mix(18, 28, v);
      const vor = voronoiTileable(u, v, Math.round(scaleFreq), Math.round(scaleFreq * 1.4), 0.62);
      const dome = smoothstep(0.02, 0.22, vor.edgeDist);
      const grain = fbm2(u * 80, v * 80, 2);
      const h = clamp01(0.18 + dome * 0.72 + grain * 0.10);
      heights[py * size + px] = h;

      const crevice = smoothstep(0.24, 0.56, h);

      // Outer limb: dark weathered charcoal-olive (#3c4037) with subtle ochre flecks
      const blotch = smoothstep(0.44, 0.64, fbm2(u * 11 + 2.3, v * 14 + 5.1, 3));
      let r = mix(62, 165, blotch * (1 - v * 0.55) * 0.48);
      let g = mix(66, 155, blotch * (1 - v * 0.55) * 0.48);
      let b = mix(57, 114, blotch * (1 - v * 0.55) * 0.48);

      // Inner/ventral limb: warm cream-ochre (#c8bd90) with dark spots
      const innerSpot = smoothstep(0.55, 0.70, fbm2(u * 18, v * 22, 3));
      const ir = mix(198, 52, innerSpot * 0.7);
      const ig = mix(187, 54, innerSpot * 0.7);
      const ib = mix(142, 44, innerSpot * 0.7);

      r = mix(r, ir, innerSide * (1 - v * 0.35));
      g = mix(g, ig, innerSide * (1 - v * 0.35));
      b = mix(b, ib, innerSide * (1 - v * 0.35));

      const cf = mix(0.32, 1.0, crevice);
      const idx = (py * size + px) * 4;
      albedoImg.data[idx] = Math.round(clamp01((r * cf) / 255) * 255);
      albedoImg.data[idx + 1] = Math.round(clamp01((g * cf) / 255) * 255);
      albedoImg.data[idx + 2] = Math.round(clamp01((b * cf) / 255) * 255);
      albedoImg.data[idx + 3] = 255;

      const rough = Math.round(mix(0.88, 0.54, smoothstep(0.58, 0.92, h)) * 255);
      roughImg.data[idx] = rough;
      roughImg.data[idx + 1] = rough;
      roughImg.data[idx + 2] = rough;
      roughImg.data[idx + 3] = 255;
    }
  }

  albedoCtx.putImageData(albedoImg, 0, 0);
  roughCtx.putImageData(roughImg, 0, 0);
  const normalCanvas = heightFieldToNormalCanvas(heights, size, 3.2);
  return { albedoCanvas, normalCanvas, roughnessCanvas: roughCanvas };
}

/**
 * Generates a realistic 512x512 Nile Crocodile golden-amber iris texture with fibrous striations
 * and a razor-sharp vertical black slit pupil!
 */
function generateEyeIrisCanvas(size = 512): HTMLCanvasElement {
  const canvas = document.createElement('canvas');
  canvas.width = size;
  canvas.height = size;
  const ctx = canvas.getContext('2d')!;

  // Dark sclera outer base
  ctx.fillStyle = '#191a14';
  ctx.fillRect(0, 0, size, size);

  const cx = size * 0.5;
  const cy = size * 0.5;
  const irisR = size * 0.34;

  // Radial golden-amber / olive-gold gradient
  const grad = ctx.createRadialGradient(cx, cy, irisR * 0.08, cx, cy, irisR);
  grad.addColorStop(0.0, '#e3ca74');
  grad.addColorStop(0.45, '#c7a44a');
  grad.addColorStop(0.78, '#8a7335');
  grad.addColorStop(0.94, '#39321b');
  grad.addColorStop(1.0, '#141510');

  ctx.fillStyle = grad;
  ctx.beginPath();
  ctx.arc(cx, cy, irisR, 0, Math.PI * 2);
  ctx.fill();

  // Procedural radial iris fibers & reptilian speckles
  ctx.save();
  ctx.beginPath();
  ctx.arc(cx, cy, irisR * 0.96, 0, Math.PI * 2);
  ctx.clip();

  for (let i = 0; i < 180; i++) {
    const angle = (i / 180) * Math.PI * 2;
    const h = hash2(i * 1.7, 3.9);
    const len = irisR * (0.45 + h * 0.52);
    ctx.strokeStyle = h > 0.5 ? 'rgba(246, 226, 148, 0.28)' : 'rgba(42, 34, 18, 0.32)';
    ctx.lineWidth = 1.2 + h * 1.8;
    ctx.beginPath();
    ctx.moveTo(cx + Math.cos(angle) * irisR * 0.1, cy + Math.sin(angle) * irisR * 0.1);
    ctx.lineTo(cx + Math.cos(angle) * len, cy + Math.sin(angle) * len);
    ctx.stroke();
  }

  // Dark fine reptilian iris flecks
  for (let i = 0; i < 90; i++) {
    const a = hash2(i * 3.1, 7.2) * Math.PI * 2;
    const r = irisR * (0.28 + hash2(i * 5.7, 1.3) * 0.65);
    ctx.fillStyle = 'rgba(28, 22, 12, 0.35)';
    ctx.beginPath();
    ctx.arc(cx + Math.cos(a) * r, cy + Math.sin(a) * r, 2 + hash2(i, 9.1) * 4, 0, Math.PI * 2);
    ctx.fill();
  }
  ctx.restore();

  // Subtle golden inner rim around vertical slit pupil
  ctx.save();
  ctx.translate(cx, cy);
  ctx.beginPath();
  ctx.moveTo(0, -irisR * 0.72);
  ctx.bezierCurveTo(irisR * 0.15, -irisR * 0.32, irisR * 0.15, irisR * 0.32, 0, irisR * 0.72);
  ctx.bezierCurveTo(-irisR * 0.15, irisR * 0.32, -irisR * 0.15, -irisR * 0.32, 0, -irisR * 0.72);
  ctx.closePath();
  ctx.lineWidth = 4;
  ctx.strokeStyle = 'rgba(245, 224, 130, 0.65)';
  ctx.stroke();
  ctx.fillStyle = '#050504';
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
  grad.addColorStop(0.0, 'rgba(18, 20, 22, 0.62)');
  grad.addColorStop(0.45, 'rgba(18, 20, 22, 0.34)');
  grad.addColorStop(0.78, 'rgba(18, 20, 22, 0.10)');
  grad.addColorStop(1.0, 'rgba(18, 20, 22, 0.0)');
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
      // Superposition of periodic wave harmonics (guaranteed seamless tiling!)
      let h = 0;
      h += Math.sin(twoPi * (u * 3 + v * 2)) * 0.28;
      h += Math.sin(twoPi * (u * 5 - v * 4 + 0.3)) * 0.22;
      h += Math.sin(twoPi * (u * 8 + v * 7 + 0.7)) * 0.16;
      h += Math.sin(twoPi * (u * 13 - v * 11 + 0.15)) * 0.11;
      h += Math.sin(twoPi * (u * 21 + v * 17 + 0.5)) * 0.07;
      heights[y * size + x] = h * 0.5 + 0.5;
    }
  }
  return heightFieldToNormalCanvas(heights, size, 2.2);
}

/**
 * Creates and configures the full procedural PBR texture set with proper color spaces & anisotropy.
 */
export function createCrocodileTextures(maxAnisotropy: number): CrocodileTextureSet {
  const body = generateBodyTextures(1024);
  const limb = generateLimbTextures(512);
  const eyeCanvas = generateEyeIrisCanvas(512);
  const shadowCanvas = generateContactShadowCanvas(512);
  const waterNormCanvas = generateWaterNormalCanvas(512);

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
