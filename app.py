"""This is a simple Flask web application for performing style transfer on images."""


# Import necessary libraries
from flask import Flask, render_template, request, redirect, url_for
import os
from style_transfer import perform_style_transfer  # Import the style transfer function

# Initialize Flask app
app = Flask(__name__)

# Define folder for image uploads
app.config['UPLOAD_FOLDER'] = 'static/uploads'

# Create required folders if they don't exist
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs('static/generated', exist_ok=True)

# Route for home page: handles both GET and POST requests
@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        # Get uploaded content and style images from form
        content_img = request.files['content']
        style_img = request.files['style']

        # Define file paths to save uploaded images
        content_path = os.path.join(app.config['UPLOAD_FOLDER'], 'content.jpg')
        style_path = os.path.join(app.config['UPLOAD_FOLDER'], 'style.jpg')

        # Save uploaded files to the server
        content_img.save(content_path)
        style_img.save(style_path)

        # Perform style transfer and get output image path
        output_path = perform_style_transfer(content_path, style_path)

        # Render result page with all images (content, style, and output)
        return render_template('index.html',
                               content_url=content_path,
                               style_url=style_path,
                               output_url=output_path)

    # If GET request, render upload form
    return render_template('index.html')

# Run the app in debug mode
if __name__ == '__main__':
    app.run(debug=True)
