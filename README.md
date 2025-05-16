# Neural Style Transfer Tool

This is a **Neural Style Transfer Tool** that allows users to blend the *content of one image* with the *style of another*, generating a stunning stylized image using deep learning techniques. The tool uses a pre-trained VGG19 model to extract features and apply artistic styles from one image to another.


## Features

- **Style Transfer**: Apply artistic style from a reference image to a content image.
- **Image Upload**: Upload your own content and style images.
- **Custom Parameters**: Adjust content and style weights for different output styles.
- **Download Result**: Save the generated image after style transfer.
- **Web Interface**: Simple web UI using Flask and HTML/CSS.
  

## Technologies Used

- **PyTorch**: For model implementation and style transfer logic.
- **TorchVision**: To load and work with the VGG19 model.
- **PIL**: For image loading and saving.
- **Flask**: To build the backend server and web UI.
- **HTML/CSS**: For the front-end interface.


## How It Works

1. **Upload Images**: Users upload a content image and a style image through the browser.
2. **Feature Extraction**: A VGG19 model extracts features from both images.
3. **Style Transfer**: The content image is optimized to adopt the style image's patterns using Gram matrices.
4. **Output Generation**: The final image is rendered and displayed in the browser with an option to save it.

##  How to Use

### 1. **Clone the repository**

```bash
git clone [https://github.com/sairao4102/SPEECH-TRANSCRIPTION-TOOL.git]
cd GPT-TEXT-GENERATOR-TOOL
```

### 2.  Requirements

Install all dependencies:

```bash
pip install -r requirements.txt
```
### 3. **Run the application**

```bash
python app.py
```

Access the Application:
Web Interface: Open [http://127.0.0.1:5000] in your browser

-- >This project was developed as part of my internship at ** CODTECH IT Solutions** (Task 3)
