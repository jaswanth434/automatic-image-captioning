from flask import Flask, request, jsonify, render_template
from werkzeug.utils import secure_filename
import os,time

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads/'

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return jsonify({"error": "No file part"})
    file = request.files['file']
    if file.filename == '':
        return jsonify({"error": "No selected file"})
    if file:
        filename = secure_filename(file.filename)
        file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
        # Process the image here and generate response
        time.sleep(3)
        response_text = "A flow chart showing the database schema on a black canvas"

        return jsonify({"message": response_text})

if __name__ == '__main__':
    app.run(debug=True)
