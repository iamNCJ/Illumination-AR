import * as React from 'react';
import { StyleSheet, Text, View } from 'react-native';
import { Camera } from 'react-native-pytorch-core';
import ModelingStage from './ModelingStage';

export default function App() {
  const [topClass, setTopClass] = React.useState(
    "Move your lighting around to see the effect",
  );

  async function handleImage(image, reset=false) {
    // Call the classify image function with the camera image
    const result = await ModelingStage(image, reset);
    // 2. Set result as top class label state
    lightDirection = result[0];
    lightIntensity = result[1];
    setTopClass(lightDirection.toString() + " " + lightIntensity.toString());
    // Release the image from memory
    image.release();
  }

  return (
    <View style={styles.container}>
      <Camera
        style={[StyleSheet.absoluteFill]}
        onFrame={handleImage}
        onCapture={(img) => handleImage(img, reset=true)}
        // hideCaptureButton={true}
        hideFlipButton={true}
      />
      <View style={styles.labelContainer}>
        <Text>{topClass}</Text>
      </View>
    </View>
  );
}

const styles = StyleSheet.create({
  container: {
    flex: 1,
    backgroundColor: '#fff',
    alignItems: 'center',
    justifyContent: 'center',
  },
  labelContainer: {
    padding: 20,
    margin: 20,
    marginTop: 40,
    borderRadius: 10,
    backgroundColor: 'white',
  },
});