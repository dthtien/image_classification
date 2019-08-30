from flask import Flask, render_template, request
from labeler import Labeler

app = Flask(__name__)

@app.route('/', methods = ['POST', 'GET'])
def index():
  detected_name = False
  probability = False

  if request.method == 'POST':
    file = request.files['file']
    labeler = Labeler(file, type = 'file')
    results = labeler.execute()
    detected_name = max(results, key=results.get)
    probability = results[detected_name]

  return render_template('index.html', detected_name = detected_name, probability = probability)

if __name__ == '__main__':
  app.run(debug = True)