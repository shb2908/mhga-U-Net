# Multi-Headed-Graph-based-Attention-aided-U-Net
Accurate segmentation of nuclei in histopathology images is
critical for understanding tissue morphology and aiding in disease diag-
nosis, particularly cancer. However, this task is challenging due to the
high variability in staining and diverse morphological features. In this
study, we propose a novel approach that integrates a graph-based at-
tention mechanism into the U-Net architecture. Our method utilizes a
state-of-the-art encoder backbone and introduces a Pairwise Node Sim-
ilarity Attention Module (PNSAM), which computes the similarity be-
tween feature channels using a kernel function that inherently applies a
dot product to capture spatial information. This module enhances the
relationships between local and non-local feature vectors within a feature
map obtained from multiple encoder layers, forming a graph attention
map. Additionally, we incorporate a channel pruning mechanism that
leverages predefined statistical knowledge to select important individual
channels for graph attention map creation. The resulting graph atten-
tion map enhances encoder features for skip connections. Furthermore,
we combine activated features from multiple trainable PNSAM heads
to generate a more diverse and robust feature map. We evaluated our
novel architecture on three widely recognized datasets: Monuseg, TNBC,
and CryoNuSeg. 
