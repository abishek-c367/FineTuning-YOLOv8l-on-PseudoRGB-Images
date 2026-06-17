GATE-IR: Gated Adaptive Thermal Enhancement for Infrared Detection
1. Problem Statement
Thermal (infrared) cameras are widely used for detecting people, vehicles, and animals in low-light or night-time 
conditions. Unlike regular cameras, they do not rely on visible light — instead they detect heat emitted by objects. This 
makes them valuable for security, search and rescue, autonomous driving, and military applications.
However, real-world deployment of thermal cameras faces a major challenge: adverse weather conditions severely degrade 
the quality of thermal images, making object detection unreliable. For instance, fog scatters the infrared radiation reducing 
image contrast and blurring object boundaries while rain reduces the temperature gradients of objects and background.
In addition, object detection systems based on heavy deep learning frameworks are often slow during inference and also 
cover large memory space which is infeasible for deployment in edge devices.
Existing detection systems have less consideration for such adaptive deployable systems.
2. Objective
The goal of GATE-IR is to build a complete, end-to-end pipeline that can reliably detect objects in thermal images under 
all weather conditions — clear, foggy, and rainy — with low computational cost and high accuracy.
Specifically, GATE-IR aims to:
• Automatically identify the current weather condition from a thermal image using a fast, lightweight classifier.
• Apply the most appropriate image enhancement method for that specific weather — or skip processing entirely 
if the image is already clear.
• Detect small and distant objects (such as people at range) more accurately than standard detectors by using a 
modified detection architecture tuned for thermal imagery.
• Transfer knowledge from powerful visible-light (RGB) detection models to the thermal domain to capture the 
RGB textural information that increases detection accuracy.
• Achieve all of this in real-time, making it practically deployable on real systems.
Overall Pipeline Flow
Thermal Image 
Input
→ Stage A Weather 
Gate (Classify)
→ Stage B 
Preprocessing 
(Enhance)
→ Stage C 
YOLOv8 
(Detect)
→ Final Detections 
(Output)
Figure 1: The three-stage GATE-IR pipeline
3. Methodology
GATE-IR is structured as a three-stage pipeline, where each stage has a specific, well-defined role. The stages are designed 
to be modular — each can be improved or replaced independently.
Stage Component Purpose
A – Gating WeatherGate (MLP) Classify: Clear / Fog / Rain
B – Preprocessing Fog Enhancer / Rain Remover Restore image quality
C – Detection YOLOv8-Thermal Find and locate objects
3.1 Stage A — Weather Classification (The Gating Mechanism)
The first stage acts as a "gatekeeper." Before any processing or detection occurs, the system quickly analyzes the incoming 
thermal image and decides what weather condition it belongs to: Clear, Fog, or Rain. It makes use of thermal variance, 
entropy and Laplacian variance of images to classify thermal images related to foggy, rain or normal condition. A small
MLP is used for the task.
3.2 Stage B — Weather-Specific Preprocessing (The Enhancement)
Based on Stage A's decision, the image is routed to the appropriate preprocessing module — or skipped entirely if the 
weather is clear.
• For Fog — Adaptive Gamma Correction: The image brightness curve is adjusted based on how dark and lowcontrast the image is. A correction factor (gamma) is calculated from the mean intensity, and applied to lift dark 
regions and restore visible temperature differences. This is computationally inexpensive and fully differentiable.
• For Rain — LSRB (Lightweight Spatial Residual Block) Network: A small convolutional neural network learns 
to predict and remove rain streaks from the image. It uses depth wise separable convolutions — a technique that 
achieves roughly 8-9x faster computation than standard convolutions. The network subtracts the predicted rain 
mask from the original image, leaving a cleaner thermal frame. This is followed by Local Contrast 
Normalization (LCN) to improve local sharpness in a fully GPU-accelerated way.
3.3 Stage C — Object Detection (YOLOv8-Thermal)
The enhanced image is passed to a YOLOv8 model for object detection. The standard YOLOv8s is slightly modified by 
adding a Vision Transformer neck in between the backbone and detection head.
3.4 Cross-Modal Knowledge Distillation (Improving Accuracy with RGB Models)
Thermal datasets are much smaller than RGB datasets, which limits training performance. To overcome this, GATE-IR 
uses a teacher-student knowledge distillation framework:
• A CycleGAN model is trained to translate thermal images into realistic pseudo-RGB images — without needing 
matched pairs of thermal and RGB images from the same scene.
• A large, powerful YOLOv8-Large model (the Teacher), pre-trained on RGB images, processes the pseudo-RGB 
versions and generates rich feature representations.
• The smaller YOLOv8-Small model (the Student) trains on the actual thermal images, but is guided by the 
teacher's features through a Feature Mimic Loss — encouraging the student to learn similarly rich 
representations even from single-channel thermal data.
This approach effectively bridges the gap between the RGB and thermal domains, improving detection accuracy without 
requiring large amounts of labeled thermal training data.
GATE-IR combines fast weather-aware routing, targeted image enhancement, and knowledge transfer to achieve robust thermal 
object detection across all weather conditions.
