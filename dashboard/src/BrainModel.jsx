import { Suspense, useRef, useMemo } from 'react';
import { Canvas, useFrame } from '@react-three/fiber';
import { OrbitControls, useGLTF, Center, Resize } from '@react-three/drei';
import * as THREE from 'three';

// ── Heatmap: 0-1 → yellow/orange/red ────────────────────────────────
function getHeatColor(v) {
  v = Math.max(0, Math.min(1, v));
  if (v < 0.4) {
    const t = v / 0.4;
    return [1.0, 0.92 - t * 0.47, 0.0];
  }
  const t = (v - 0.4) / 0.6;
  return [0.95, 0.45 - t * 0.4, 0.05 * t];
}

function lerp3(a, b, t) {
  return [a[0]*(1-t)+b[0]*t, a[1]*(1-t)+b[1]*t, a[2]*(1-t)+b[2]*t];
}

// ── Classify vertex to Yeo 7 region from MNI coordinates ────────────
// fsaverage vertices are in MNI space:
//   X: left(-) → right(+)
//   Y: posterior(-) → anterior(+)
//   Z: inferior(-) → superior(+)
function classifyVertex(x, y, z) {
  if (y < -65) return 0; // visual (occipital)
  if (z > 55 && Math.abs(y) < 25) return 1; // somatomotor (central strip)
  if (z > 35 && y < -20 && y > -65) return 2; // dorsal_attention (parietal)
  if (Math.abs(x) > 40 && z < 15 && z > -20) return 3; // ventral_attention (temporal)
  if (z < -5 && y > -20) return 4; // limbic (orbitofrontal/medial temporal)
  if (y > 20 && Math.abs(x) > 25 && z > 10) return 5; // frontoparietal (lateral prefrontal)
  if (Math.abs(x) < 20 && (y > 30 || (y < -40 && z > 20))) return 6; // default_mode (medial PFC/PCC)
  // Fallback
  if (y > 0) return Math.abs(x) < 25 ? 6 : 5;
  return z > 20 ? 2 : 0;
}

const REGION_NAMES = [
  "visual", "somatomotor", "dorsal_attention", "ventral_attention",
  "limbic", "frontoparietal", "default_mode"
];

// ── Brain Component ─────────────────────────────────────────────────
const GLBBrain = ({ currentData }) => {
  const { scene } = useGLTF('/brain.glb');
  const brainRef = useRef();
  const geoDataRef = useRef([]);

  const clonedScene = useMemo(() => {
    const clone = scene.clone(true);
    const geoList = [];

    clone.traverse((child) => {
      if (!child.isMesh) return;
      
      const geo = child.geometry;
      if (!geo.attributes.normal) geo.computeVertexNormals();
      
      const posAttr = geo.attributes.position;
      const count = posAttr.count;
      
      // Classify EVERY vertex by its MNI coordinate position
      const regionPerVertex = new Int8Array(count);
      for (let i = 0; i < count; i++) {
        const x = posAttr.getX(i);
        const y = posAttr.getY(i);
        const z = posAttr.getZ(i);
        regionPerVertex[i] = classifyVertex(x, y, z);
      }
      
      // Debug: count regions
      const rc = {};
      for (let i = 0; i < count; i++) {
        rc[regionPerVertex[i]] = (rc[regionPerVertex[i]] || 0) + 1;
      }
      console.log(`Brain mesh: ${count} vertices, regions:`, rc);
      
      // Create color buffer for dynamic heatmap
      const colors = new Float32Array(count * 3);
      for (let i = 0; i < count; i++) {
        colors[i*3] = 0.85;
        colors[i*3+1] = 0.85;
        colors[i*3+2] = 0.85;
      }
      geo.setAttribute('color', new THREE.BufferAttribute(colors, 3));
      
      child.material = new THREE.MeshStandardMaterial({
        vertexColors: true,
        roughness: 0.45,
        metalness: 0.05,
      });
      
      geoList.push({ geometry: geo, regionPerVertex });
    });

    geoDataRef.current = geoList;
    return clone;
  }, [scene]);

  // ── Paint heatmap every frame using REAL data ─────────────────────
  useFrame(() => {
    const nets = currentData?._networks || {};
    const grey = [0.85, 0.85, 0.85];
    
    const acts = REGION_NAMES.map(n => Math.abs(nets[n]?.activation || 0));

    geoDataRef.current.forEach(({ geometry, regionPerVertex }) => {
      const colors = geometry.attributes.color;
      if (!colors) return;

      for (let i = 0; i < colors.count; i++) {
        const region = regionPerVertex[i];
        const activation = acts[region] || 0;
        const heat = Math.max(0, Math.min(1, (activation - 0.05) / 0.55));
        
        if (heat > 0.02) {
          const hc = getHeatColor(heat);
          const fc = lerp3(grey, hc, heat * 0.92);
          colors.setXYZ(i, fc[0], fc[1], fc[2]);
        } else {
          colors.setXYZ(i, 0.85, 0.85, 0.85);
        }
      }
      colors.needsUpdate = true;
    });
  });

  return (
    <Center>
      <Resize scale={2.8}>
        <primitive ref={brainRef} object={clonedScene} />
      </Resize>
    </Center>
  );
};

const FallbackBrain = () => (
  <mesh>
    <sphereGeometry args={[2, 32, 32]} />
    <meshStandardMaterial color="#cccccc" wireframe />
  </mesh>
);

export default function BrainEngine({ currentData }) {
  return (
    <div style={{ width: '100%', height: '100%', background: '#0a0a0a' }}>
      <Canvas camera={{ position: [0, 0, 5], fov: 45 }}>
        <ambientLight intensity={0.8} />
        <directionalLight position={[5, 5, 5]} intensity={1.2} />
        <directionalLight position={[-3, 3, -3]} intensity={0.5} />
        <Suspense fallback={<FallbackBrain />}>
          <GLBBrain currentData={currentData} />
        </Suspense>
        <OrbitControls enableZoom={false} enablePan={false} autoRotate autoRotateSpeed={0.8} />
      </Canvas>
    </div>
  );
}
