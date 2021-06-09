import os
import urllib.request
from flask import Flask, flash, request, redirect, url_for, render_template
from flask_wtf import FlaskForm
from flask_wtf.file import FileField, FileRequired, FileAllowed
from wtforms import SubmitField
from flask_bootstrap import Bootstrap
from werkzeug.utils import secure_filename

from predictor import run_predictor

app = Flask(__name__)

UPLOAD_FOLDER = 'static/uploads/'
ALLOWED_EXTENSIONS = ['png', 'jpg', 'jpeg', 'gif']

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['SECRET_KEY'] = 'C2HWGVoMGfNTBsrYQg8EcMrdTimkZfAb24'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024

# Flask-Bootstrap requires this line
Bootstrap(app)

class UploadForm(FlaskForm):
	validators = [
		FileRequired(message='There was no file!'),
		FileAllowed(ALLOWED_EXTENSIONS, message='Not a supported extension...')
		]
	file = FileField('', validators=validators)
	submit = SubmitField(label='Upload')
	
@app.route('/', methods=['GET', 'POST'])
def upload_form():
	form = UploadForm()
	if request.method == 'POST' and form.validate_on_submit():
		file = request.files['file']
		filename = secure_filename(file.filename)
		print(filename)
		file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
		flash('Image successfully uploaded!')

		results = run_predictor(os.path.join(app.config['UPLOAD_FOLDER'], filename))
		return render_template('upload.html', filename=filename, results=results, form=form)
	else:
		return render_template('upload.html', form=form)

@app.route('/display/<filename>')
def display_image(filename):
	return redirect(url_for('static', filename='uploads/' + filename), code=301)

if __name__ == "__main__":
    app.run()