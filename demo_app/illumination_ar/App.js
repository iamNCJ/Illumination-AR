import * as React from 'react';
import { Button } from 'react-native';
import { Camera } from 'react-native-pytorch-core';
import ModelingStage from './ModelingStage';
// import { GLView } from 'expo-gl';
// import { Renderer, THREE, TextureLoader, loadAsync } from 'expo-three';
// import { Asset } from 'expo-asset';
// import { GLTFLoader } from 'three/examples/jsm/loaders/GLTFLoader';
// import * as FileSystem from "expo-file-system";
// import { decode } from 'base64-arraybuffer';
// import {
//   AmbientLight,
//   BoxGeometry,
//   Fog,
//   GridHelper,
//   Mesh,
//   MeshStandardMaterial,
//   PerspectiveCamera,
//   PointLight,
//   Scene,
//   SpotLight,
// } from 'three';

// class IconMesh extends Mesh {
//   constructor() {
//     super(
//       new BoxGeometry(1.0, 1.0, 1.0),
//       new MeshStandardMaterial({
//         map: new TextureLoader().load(require('./assets/icon.png')),
//         // color: 0xff0000
//       })
//     );
//   }
// }

// export default function App() {
//   const [topClass, setTopClass] = React.useState(
//     "Move your lighting around to see the effect",
//   );

//   const [isRunning, setRunning] = React.useState(true);
//   const [lastImage, setLastImage] = React.useState(null);
//   const [lightDirection, setLightDirection] = React.useState(null);
//   const [lightIntensity, setLightIntensity] = React.useState(null);

//   async function handleImage(image, reset = false) {
//     // console.log("handleImage")
//     setLastImage(image)
//     if (isRunning) {
//       // Call the classify image function with the camera image
//       const result = await ModelingStage(image, reset);
//       // 2. Set result as top class label state
//       const _lightDirection = result[0];
//       const _lightIntensity = result[1];
//       // setTopClass(lightDirection.toString() + " " + lightIntensity.toString());
//       console.log(_lightDirection.toString() + " " + _lightIntensity.toString());
//       setLightDirection(_lightDirection);
//       setLightIntensity(_lightIntensity[0]);
//     }
//     // Release the image from memory
//     image.release();
//   }

//   return (
//     <View style={{flex: 1}}>
//       <Camera
//         style={[styles.cameraView]}
//         onFrame={handleImage}
//         // onCapture={(img) => handleImage(img, reset = true)}
//         hideCaptureButton={true}
//         hideFlipButton={true}
//       />
//       {/* <View style={styles.labelContainer}>
//         <Text>{topClass}</Text>
//       </View> */}
//       <GLView
//         style={{flex: 1}}
//         onContextCreate={async (gl) => {
//           // Create a WebGLRenderer without a DOM element
//           const renderer = new Renderer({ gl });
//           renderer.setSize(gl.drawingBufferWidth, gl.drawingBufferHeight);

//           const quad = new THREE.PlaneGeometry((2 * 9) / 16, 2);
//           const mesh = new THREE.Mesh(quad, null);
//           const material = new THREE.MeshBasicMaterial();
//           material.needsUpdate = true;
//           const aspect = window.innerWidth / window.innerHeight;
//           console.log(`window: ${window.innerWidth} ${window.innerHeight}`)
//           console.log(`aspect: ${aspect}`);
//           bgcamera = new THREE.OrthographicCamera(-aspect, aspect, 1, -1, 0, 1);
//           bgcamera.position.set(0, 0, 0);
//           bgcamera.lookAt(0, 0, -1);
//           bgscene = new THREE.Scene();
//           bgscene.add(mesh);

//           scene = new THREE.Scene();
//           // // Create an Asset from a resource
//           // const asset = Asset.fromModule(require("./assets/duduko/scene.gltf"));
//           // await asset.downloadAsync();

//           // // This is the local URI
//           // const uri = asset.localUri;
//           // console.log(uri);
//           // const base64 = await FileSystem.readAsStringAsync(uri, {
//           //   encoding: FileSystem.EncodingType.Base64,
//           // });

//           // const arrayBuffer = decode(base64);
//           // // const loader = new GLTFLoader();

//           // new GLTFLoader().parse(
//           //   arrayBuffer,
//           //   null,
//           //   gltf => {
//           //     arobj = gltf.scene;
//           //     arobjPos
//           //       .set(
//           //         -0.007205986622358538,
//           //         0.06654940586531129,
//           //         -0.9977571098898631
//           //       )
//           //       .multiplyScalar(raydist);
//           //     arobj.position.set(arobjPos.x, arobjPos.y, arobjPos.z);
//           //     // set rotation with guide of normal
//           //     var quat = new THREE.Quaternion();
//           //     quat.setFromUnitVectors(new THREE.Vector3(0, 1, 0), normal);
//           //     arobj.setRotationFromQuaternion(quat);
//           //     //   arobj.rotation.x = (60 / 180.0) * Math.PI;
//           //     scene.add(arobj);
//           //   },
//           //   err => {
//           //     console.log("error");
//           //   },
//           // );

//           // const model = {
//           //   '3d.obj': 'https://raw.githubusercontent.com/mrdoob/three.js/dev/examples/models/obj/walt/WaltHead.obj',
//           //   '3d.mtl': 'https://raw.githubusercontent.com/mrdoob/three.js/dev/examples/models/obj/walt/WaltHead.mtl'
//           // };

//           // const object = await loadAsync([model['3d.obj'], model['3d.mtl']], null, name => model[name]);

//           // object.position.y += 2;
//           // object.position.z -= 2;
//           // object.scale.set(.02, .02, .02);

//           // scene.add(object);

//           light = new THREE.DirectionalLight(0xffffff, 10);
//           light.position.set(0, 1, 0);
//           // light.add(new THREE.Mesh(lightMarker, new THREE.MeshBasicMaterial({ color: 0xffffff })))
//           scene.add(light);

//           // const pointLight = new PointLight(0xffffff, 2, 1000, 1);
//           // pointLight.position.set(0, 200, 200);
//           // scene.add(pointLight);

//           const cube = new IconMesh();
//           cube.position.set(0, 0, -3);
//           scene.add(cube);

//           // camera = new THREE.PerspectiveCamera(45, aspect, 0.001, 1000);
//           camera = new PerspectiveCamera(70, aspect, 0.01, 1000)
//           camera.position.set(0, 0, 0);
//           camera.lookAt(0, 0, -1);
//           // camera.position.set(2, 5, 5);
//           // camera.lookAt(cube.position);
//           // console.log(cube.position); // {x: 0, y: 0, z: 0}

//           const render = () => {
//             timeout = requestAnimationFrame(render);
//             console.log("render")
//             // if (lastImage !== null) {
//             //   const texture = new TextureLoader().load(lastImage);
//             //   texture.wrapS = THREE.RepeatWrapping;
//             //   texture.wrapT = THREE.RepeatWrapping;
//             //   // texture.repeat.y = - 1;
//             //   material.map = texture;
//             //   material.map.needsUpdate = true;
//             //   mesh.material = material;
//             //   mesh.material.depthTest = false;
//             //   mesh.material.depthWrite = false;
//             //   renderer.autoClear = false;
//             //   renderer.clear();
//             //   renderer.render(bgscene, bgcamera);
//             //   gl.endFrameEXP();
//             // }
//             // renderer.autoClear = false;
//             // renderer.clear();
//             // renderer.render(bgscene, bgcamera);
//             console.log(lightDirection);
//             if (lastImage !== null) {
//               light.position.set(lightDirection[0], lightDirection[1], lightDirection[2]);
//             }
//             renderer.render(scene, camera);
//             gl.endFrameEXP();
//           };
//           render();
//         }}
//       />
//       {/* <GLView
//         style={{ flex: 1 }}
//         onContextCreate={async (gl) => {
//           const { drawingBufferWidth: width, drawingBufferHeight: height } = gl;
//           const sceneColor = 0x6ad6f0;

//           // Create a WebGLRenderer without a DOM element
//           const renderer = new Renderer({ gl });
//           renderer.setSize(width, height);
//           renderer.setClearColor(sceneColor);

//           const camera = new PerspectiveCamera(70, width / height, 0.01, 1000);
//           camera.position.set(2, 5, 5);

//           const scene = new Scene();
//           scene.fog = new Fog(sceneColor, 1, 10000);
//           scene.add(new GridHelper(10, 10));

//           const ambientLight = new AmbientLight(0x101010);
//           scene.add(ambientLight);

//           const pointLight = new PointLight(0xffffff, 2, 1000, 1);
//           pointLight.position.set(0, 200, 200);
//           scene.add(pointLight);

//           const spotLight = new SpotLight(0xffffff, 0.5);
//           spotLight.position.set(0, 500, 100);
//           spotLight.lookAt(scene.position);
//           scene.add(spotLight);

//           // Load and add a texture
//           const cube = new IconMesh();
//           scene.add(cube);

//           // Load and add an obj model
//           const model = {
//             '3d.obj': 'https://raw.githubusercontent.com/mrdoob/three.js/dev/examples/models/obj/walt/WaltHead.obj',
//             '3d.mtl': 'https://raw.githubusercontent.com/mrdoob/three.js/dev/examples/models/obj/walt/WaltHead.mtl'
//           };

//           const object = await loadAsync([model['3d.obj'], model['3d.mtl']], null, name => model[name]);

//           object.position.y += 2;
//           object.position.z -= 2;
//           object.scale.set(.02, .02, .02);

//           scene.add(object);

//           camera.lookAt(cube.position);

//           function update() {
//             cube.rotation.y += 0.05;
//             cube.rotation.x += 0.025;
//           }

//           // Setup an animation loop
//           const render = () => {
//             timeout = requestAnimationFrame(render);
//             update();
//             renderer.render(scene, camera);
//             gl.endFrameEXP();
//           };
//           render();
//         }}
//       /> */}
//       <Button title='start / puase' onPress={() => {
//         if (isRunning) {
//           setRunning(false)
//           console.log("pause")
//         } else {
//           setRunning(true)
//           console.log("start")
//         }
//       }} />
//     </View>
//   );
// }

// const styles = StyleSheet.create({
//   container: {
//     flex: 1,
//     backgroundColor: '#fff',
//     alignItems: 'center',
//     justifyContent: 'center',
//   },
//   labelContainer: {
//     padding: 20,
//     margin: 20,
//     marginTop: 40,
//     borderRadius: 10,
//     backgroundColor: 'white',
//   },
//   cameraView: {
//     display: 'none',
//   }
// });


// import * as THREE from 'three'
import { useRef, useState, Suspense } from "react";
import { StyleSheet, View } from "react-native";
import { Canvas, useFrame, useThree, useLoader } from "@react-three/fiber";

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
  const [isRunning, setRunning] = React.useState(true);
  const [resetText, setResetText] = React.useState("reset");
  const [reset, setReset] = React.useState(false);
  const [lastImage, setLastImage] = React.useState(null);
  const [lightDirection, setLightDirection] = React.useState([1., 0., 0.]);
  const [lightIntensity, setLightIntensity] = React.useState(1.);

  async function handleImage(image) {
    // console.log("handleImage")
    setLastImage(image)
    if (isRunning) {
      // Call the classify image function with the camera image
      const result = await ModelingStage(image, reset);
      // 2. Set result as top class label state
      const _lightDirection = result[0];
      const _lightIntensity = result[1];
      // setTopClass(lightDirection.toString() + " " + lightIntensity.toString())
      // console.log(_lightDirection.toString() + " " + _lightIntensity.toString());
      console.log(_lightDirection);
      setLightDirection([_lightDirection[0] * 2, _lightDirection[1] * 2, _lightDirection[2] * 2]);
      setLightIntensity(_lightIntensity[0]);
    }
    // Release the image from memory
    image.release();
  }

  // const set = useThree((state) => state.set)
  // useEffect(() => {
  //   set({ camera: new THREE.PerspectiveCamera(70, aspect, 0.01, 1000) })
  // }, [])

  return (
    <View style={styles.container}>
      <Camera
        style={[styles.full, {zIndex: 0, position: "absolute"}]}
        onFrame={handleImage}
        // onCapture={(img) => handleImage(img, reset = true)}
        hideCaptureButton={true}
        hideFlipButton={true}
      />
      <Canvas style={[styles.center, {zIndex: 2}]}>
        {/* <ambientLight /> */}
        {/* <pointLight position={[10, 10, 10]} /> */}
        <directionalLight position={lightDirection} intensity={lightIntensity * 1}/>
        <Box position={[0, 0, 0]} />
        {/* <Suspense>
          <Environment />
        </Suspense> */}
        {/* <Box position={[1.2, 0, 0]} /> */}
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
    // flex: 1,
    // alignItems: 'center',
    // justifyContent: 'center',
    // backgroundColor: "black",
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
