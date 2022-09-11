from flask import Flask, render_template,  request, jsonify
from predict import *
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/api/respond', methods=['POST'])
def api_respond():
    query = request.get_json()
    res = respond(query['q'])
    return jsonify({'answer':res}) 

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5000, debug=True)