import { MobileModel, torch, torchvision, media } from 'react-native-pytorch-core';

const T = torchvision.transforms;

let model = null;
let fusedFeature = torch.ones([1, 256, 8, 8]).mul(-10000);


export default async function ModelingStage(image, reset = False) {
    if (reset) {
        fusedFeature = torch.ones([1, 256, 8, 8]).mul(-10000);
    }

    // Get image width and height
    const width = image.getWidth();
    const height = image.getHeight();

    // Convert image to blob, which is a byte representation of the image
    // in the format height (H), width (W), and channels (C), or HWC for short
    const blob = media.toBlob(image);

    // Get a tensor from image the blob and also define in what format the image blob is.
    let tensor = torch.fromBlob(blob, [height, width, 3]);

    // 3.iv. Rearrange the tensor shape to be [CHW]
    tensor = tensor.permute([2, 0, 1]);

    // Divide the tensor values by 255 to get values between [0, 1]
    tensor = tensor.div(255);

    // Crop the image in the center to be a squared image
    const centerCrop = T.centerCrop(Math.min(width, height));
    tensor = centerCrop(tensor);

    // Resize the image tensor to 3 x 128 x 128
    const resize = T.resize(128);
    tensor = resize(tensor);

    tensor = tensor.unsqueeze(0);

    if (model == null) {
        const filePath = await MobileModel.download(require('./assets/models/online_stage.ptl'));
        model = await torch.jit._loadForMobile(filePath);
    }

    const output = await model.forward(tensor, fusedFeature);
    const lightDirection = output[0];
    const lightIntensity = output[1];
    fusedFeature = output[2];
    console.log(fusedFeature.sum().div(256 * 8 * 8).item())

    return [lightDirection.data(), lightIntensity.data()];
}