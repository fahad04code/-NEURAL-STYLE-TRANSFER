# -NEURAL-STYLE-TRANSFER

*COMPANY*: CODTECH IT SOLUTIONS

*NAME*: HASHMI SYED FAHAD

*INTERN ID*: CT08DL815

*DOMAIN*: ARTIFICIAL INTELLIGENCE

*DURATION*: 8 WEEEKS

*MENTOR*: NEELA SANTOSH



## Description :

## Project Overview :-

I’m a student passionate about AI, and neural-style-transfer is my latest project—a Python script that blends the content of one image with the style of another using PyTorch and VGG19. After exploring NLP with projects like BART summarization and speech-to-text transcription, I wanted to dive into computer vision. My earlier LSTM attempts produced messy outputs like “musensiblet,” and memory constraints (~1025 MB) taught me to optimize. This script takes a content image (e.g., a landscape) and a style image (e.g., a painting), creating a stylized result, like a photo in Van Gogh’s style. It’s a portfolio piece for AI internships, showcasing my growth in coding, optimization, and creative AI applications.


## What It Does :-

This script performs neural style transfer using VGG19’s pre-trained features. It extracts content features from one image and style features (via Gram matrices) from another, then optimizes a target image to combine both using a loss function. The script includes memory monitoring, image preprocessing with normalization, and error handling for file paths. It resizes images to 64x64 to save memory, runs 300 optimization steps, and saves the stylized image as stylized_image.jpg. It’s a creative tool for generating art, experimenting with styles, or learning about deep learning in computer vision.


## Lesson Learned :-

This project represents my inaugural journey into computer vision following my work with BART and GPT-2  models for natural language processing. Working with neural style transfer provided me an opportunity to generate images when  my LSTM model faced difficulties with text production. I acquired knowledge about using PyTorch’s torchvision to  process VGG19 which enabled me to extract features from specific image layers (conv_1 through conv_5  for style and conv_4 for content). The quantification of artistic style through Gram matrix calculations was  a new learning experience which showed me how to measure artistic style through mathematical operations while creating connections between art  and technology.

VGG19 has a heavy memory requirements because it consumes more than 500 megabytes  of storage space which exceeded my laptop's 1025 MB memory capacity. By reducing images to 64x64  dimensions my memory usage decreased according to psutil while larger images would increase system memory to about  1.5 GB. The experience of managing quality in relation to limited resources provides valuable skills for developing artificial  intelligence applications. My development of the StyleContentModel class to analyze feature extraction methods through PyTorch’s  nn.Module functions helped me develop better knowledge about neural network structure beyond my previous works.

The script developed a robust structure because I implemented image path validation and error handling for optimization that was  different from the LSTM scripts which encountered file problems. My debugging process improved when I implemented memory usage logging after learning the technique from my BART project. The script triggered my creative process because I experienced the transformation of a landscape into starry night  inspired artwork. The experience enhanced my understanding of AI's creative abilities which led me to develop concepts for potential  internship projects involving real-time style transfer and video stylization.


## Requirements:-

- torch>=1.9.0 
- torchvision>=0.10.0 
- pillow>=8.0.0 
- matplotlib>=3.3.0
- psutil>=5.8.0


## Setup and Usage :-

Setup and Usage





1. Install Dependencies: Install required libraries using requirements.txt:

```
pip install -r requirements.txt

```

2. Prepare Images: Provide a content image (content.jpg, e.g., a photo) and a style image (style4.jpg, e.g., a painting).



Run:

```
cd neural-style-transfer
python style_transfer.py

```

Customize: Edit content_path and style_path in style_transfer.py with your image paths. Adjust size or steps for quality vs. speed.


## OUTPUT:-

![Image](https://github.com/user-attachments/assets/b5807e67-97c7-4830-be47-c5a446a4dd09)

![Image](https://github.com/user-attachments/assets/2f50df2e-9271-4620-90e3-3d7d8719538f)
