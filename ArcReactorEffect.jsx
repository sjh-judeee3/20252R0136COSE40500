import React, { useRef } from "react";
import { Canvas, useFrame, useLoader } from "@react-three/fiber";
import * as THREE from "three";

function ArcReactor() {
  const texture = useLoader(THREE.TextureLoader, "/arcReactor.png");

  return (
    <mesh>
      <planeGeometry args={[2, 2]} />
      <meshBasicMaterial map={texture} transparent={true} />
    </mesh>
  );
}


//ring
export function SpinningArc({
  reverse = false,
  radius = 1,
  thickness = 0.4,
  color = 'cyan',
}) {
  const meshRef = useRef();
  const direction = reverse ? -1 : 1;

  useFrame(() => {
    if (meshRef.current) {
      meshRef.current.rotation.z += 0.01 * direction;
    }
  });

  const innerRadius = radius;
  const outerRadius = radius + thickness;
  const thetaStart = 0;
  const thetaLength = (300 / 360) * Math.PI * 2;

  return (
    <mesh ref={meshRef} rotation={[0, 0, 0]}>
      <ringGeometry
        args={[innerRadius, outerRadius, 128, 1, thetaStart, thetaLength]}
      />
      <meshBasicMaterial
        color={new THREE.Color(color)}
        transparent={true}
        opacity={0.6}
        blending={THREE.AdditiveBlending}
        side={THREE.DoubleSide}
        depthWrite={false}
        toneMapped={false} 
      />
    </mesh>
  );
}



export default function ArcReactorEffect() {
  return (
    <Canvas camera={{ position: [0, 0, 5] }} style={{ background: 'transparent', width: '100%', height: '100%' }}>
      <ArcReactor />
      <SpinningArc radius={1.1} color="cyan" />
      <SpinningArc radius={1.3} color="deepskyblue" reverse />
    </Canvas>
  );
}
