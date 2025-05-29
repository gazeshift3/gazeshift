# GazeShift: Unsupervised Gaze Estimation and Dataset for VR

Official github repository for GazeShift: Unsupervised Gaze Estimation and Dataset for VR


> Abstract: Gaze estimation is instrumental in modern virtual reality (VR) systems. Despite significant progress in remote-camera gaze estimation, VR gaze research remains constrained by data scarcity -- especially the lack of large-scale, accurately labeled datasets captured with off-axis camera configurations common in modern VR systems. Gaze annotation is inherently difficult, as gaze cannot be inferred from raw eye images and user fixation on the intended on-screen target cannot be guaranteed. To address this, we first introduce \textit{VRGaze} -- the first large-scale off-axis gaze estimation dataset for VR -- comprising 2.1 million near-eye infrared images.
Second, we introduce \textit{GazeShift}, the first unsupervised training method for VR-based gaze estimation, eliminating the need for accurate gaze labels.
We rely on a key assumption applicable for near-eye cameras: for a given subject, temporal variations between images are primarily caused by changes in gaze direction. 
Leveraging this insight, we train a model to redirect a source eye image toward a target image, conditioned on a learned latent representation of the target. 
As most of the difference between image pairs is gaze-related, the learned latent representation is rich in gaze information. 
To further guide the model, we propose a gaze-aware loss that utilizes attention maps to emphasize gaze-relevant regions.
Our method demonstrates 1.84\degree\ mean angular error on VRGaze. 
Additionally, our method generalizes well to remote-camera gaze benchmarks, achieving a new state-of-the-art accuracy for unsupervised approaches, 7.15\degree on MPIIGaze, with a 10$\times$ reduction in parameters. Implementation on a custom headset's mobile GPU measures 5 ms, demonstrating that GazeShift is well-suited for efficient deployment on edge devices. 
VRGaze and GazeShift are released under \url{https://github.com/gazeshift3/gazeshift}.



#<img src="assets/arch.png"  />
# 1. VRGaze dataset 
We curated VRGaze, the first large-scale dataset captured using off-axis camera configurations typical of modern VR systems.
The dataset is here: https://huggingface.co/datasets/gazeshift/VRGaze


# Installation and Preparation

pip install -r requirements.txt

# Training from scratch
To reproduce the main experiment of the paper on the VRGaze dataset, run the Train.sh scripts (remember to update your VRGaze dataset and output locations). The 1.84 [deg] reported average error after per-person calibration is expected after 400K steps.




