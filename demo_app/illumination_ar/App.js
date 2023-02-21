import { Camera } from 'react-native-pytorch-core';
import ModelingStage from './ModelingStage';
import { useRef, useState } from "react";
import { Button, StyleSheet, View } from "react-native";
import { Canvas, useFrame } from "@react-three/fiber";

function Box(props) {
  // This reference will give us direct access to the mesh
  const mesh = useRef();

  // Set up state for the hovered and active state
  const [hovered, setHover] = useState(false);
  const [active, setActive] = useState(false);

  // Rotate mesh every frame, this is outside of React without overhead
  useFrame(() => {
    if (mesh && mesh.current) {
      // mesh.current.rotation.x = mesh.current.rotation.y += 0.01;
      mesh.current.rotation.x = mesh.current.rotation.y = 0.5;
    }
  });

  return (
    <mesh
      {...props}
      ref={mesh}
      scale={active ? [1.5, 1.5, 1.5] : [1, 1, 1]}
      onClick={(e) => setActive(!active)}
      onPointerOver={(e) => setHover(true)}
      onPointerOut={(e) => setHover(false)}
    >
      <boxGeometry attach="geometry" args={[1, 1, 1]} />
      <meshStandardMaterial
        attach="material"
        color={hovered ? "hotpink" : "orange"}
      />
    </mesh>
  );
}

export default function App() {
  const [resetText, setResetText] = useState("reset");
  const [reset, setReset] = useState(false);
  const [lightDirection, setLightDirection] = useState([1., 0., 0.]);
  const [lightIntensity, setLightIntensity] = useState(1.);

  async function handleImage(image) {
    // Call the classify image function with the camera image
    const result = await ModelingStage(image, reset);
    // 2. Set result as top class label state
    const _lightDirection = result[0];
    const _lightIntensity = result[1];
    console.log(_lightDirection);
    setLightDirection([_lightDirection[0] * 2, _lightDirection[1] * 2, _lightDirection[2] * 2]);
    setLightIntensity(_lightIntensity[0]);
    // Release the image from memory
    image.release();
  }

  return (
    <View style={styles.container}>
      <Camera
        style={[styles.full, {zIndex: 0, position: "absolute"}]}
        onFrame={handleImage}
        hideCaptureButton={true}
        hideFlipButton={true}
      />
      <Canvas style={[styles.center, {zIndex: 2}]}>
        <directionalLight position={lightDirection} intensity={lightIntensity * 1}/>
        <Box position={[0, 0, 0]} />
      </Canvas>
      <Button title={resetText} onPress={() => {
        setReset(true);
        setResetText("restart");
        console.log("reset");
      }}/>
    </View>
  );
}

const styles = StyleSheet.create({
  container: {
    zIndex: 0,
    width: "100%",
    height: "100%",
  },
  center: {
    width: "100%",
    height: "100%",
    position: 'absolute'
  },
  full: {
    width: "100%",
    height: "100%",
  }
});
