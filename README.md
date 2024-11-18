# EC523
# High-Resolution Image Deblurring with Spatially Adaptive Techniques

## Problem Statement
This project addresses the challenge of reconstructing high-quality, high-resolution images from blurred, wide Field of View (FoV) inputs. Directly deblurring a full high-resolution image is computationally intensive, so we employ a patch-based training approach. The challenge arises because the Point Spread Function (PSF) varies across different spatial locations in the image, complicating traditional convolutional neural network (CNN) methods. CNNs use shared convolutions, approximating global non-uniform deconvolution with a uniform one, which compromises accuracy or efficiency. To overcome this, we propose a location-aware MLP for spatially adaptive deblurring, coupled with a context mechanism to retain broader scene information for optimized resource use.

## Objective
1. **Develop a deblurring model** that handles spatially varying PSFs across different patches of a high-resolution image.
2. **Improve the model's receptive field** without significantly increasing computational demand by integrating context information from the original image.
3. **Evaluate the model’s performance** by comparing it to baseline models using Mean Squared Error (MSE) and Structural Similarity Index (SSIM) metrics.
4. **Dataset Description**: The dataset includes measurement images obtained from a microlens array, processed through demixing to yield images without overlapping information.

## Methodology

### 1. Patch-Based Training with Spatial Adaptation
To address spatially varying PSFs, we propose a patch-based training approach. By incorporating the spatial location of each patch into a standard Multilayer Perceptron (MLP), we generate a spatially adaptive mask. This mask, when combined with CNN outputs, integrates location-specific spatial information, accurately simulating the unique PSF of each patch. The MLP adapts the CNN outputs based on patch location, preserving both local and global deblurring accuracy.

### 2. Contextual Information Integration
Patch-based training often results in a loss of surrounding contextual information, which can lead to less accurate reconstructions. To mitigate this, we introduce a hierarchical context mechanism. By using information from a low-resolution version of the original image at each network layer, we incrementally incorporate broader context, effectively expanding the receptive field without overloading early layers. This approach balances contextual awareness and computational efficiency.

### 3. Backbone Network: Nonlinear Activation-Free Net (NAFNet)
We utilize NAFNet as our backbone due to its state-of-the-art performance in image restoration with low computational cost. NAFNet replaces conventional nonlinear activation functions with gating functions, reducing complexity while maintaining performance. This provides a robust foundation for our patch-based deblurring model.

## Comparative Evaluation
We will evaluate the model's performance using three configurations:
1. **Baseline**: NAFNet alone.
2. **Experiment 1**: NAFNet + Coordinate MLP.
3. **Experiment 2**: NAFNet + Coordinate MLP + Context Mechanism.

We will compare these configurations using MSE and SSIM metrics to assess deblurring accuracy and computational efficiency.

## References
1. Patch-based image restoration methods.
2. Studies on CNN limitations with non-uniform deconvolution.
3. Context-aware techniques for image reconstruction.
4. NAFNet: A lightweight and efficient model for image restoration.

