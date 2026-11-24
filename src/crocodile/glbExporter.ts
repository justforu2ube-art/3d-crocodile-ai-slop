import * as THREE from 'three';
import { GLTFExporter } from 'three/addons/exporters/GLTFExporter.js';

export interface GlbExportResult {
  filename: string;
  byteLength: number;
  meshCount: number;
}

/**
 * Exports the procedurally generated Nile Crocodile Three.js hierarchy to a binary .GLB file
 * and triggers an immediate file download in the browser.
 */
export async function exportCrocodileToGLB(
  modelRoot: THREE.Object3D,
  filename = 'nile-crocodile-crocodylus-niloticus.glb'
): Promise<GlbExportResult> {
  const exporter = new GLTFExporter();
  let meshCount = 0;
  modelRoot.traverse((obj) => {
    if (obj instanceof THREE.Mesh) {
      meshCount++;
    }
  });

  // Update all matrices so exported pose matches live viewport
  modelRoot.updateMatrixWorld(true);

  const glbBuffer = await new Promise<ArrayBuffer>((resolve, reject) => {
    exporter.parse(
      modelRoot,
      (result) => {
        if (result instanceof ArrayBuffer) {
          resolve(result);
        } else {
          const json = JSON.stringify(result);
          resolve(new TextEncoder().encode(json).buffer);
        }
      },
      (error) => {
        reject(error instanceof Error ? error : new Error(String(error)));
      },
      {
        binary: true,
        onlyVisible: true,
        embedImages: true,
      }
    );
  });

  const blob = new Blob([glbBuffer], { type: 'model/gltf-binary' });
  const url = URL.createObjectURL(blob);
  const anchor = document.createElement('a');
  anchor.href = url;
  anchor.download = filename;
  anchor.style.display = 'none';
  document.body.appendChild(anchor);
  anchor.click();
  document.body.removeChild(anchor);
  // Delay revoke slightly so browser finishes initiating download
  window.setTimeout(() => URL.revokeObjectURL(url), 4000);

  return {
    filename,
    byteLength: glbBuffer.byteLength,
    meshCount,
  };
}
