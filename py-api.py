from flask import Flask ,request
import json
reponse = {}

app =Flask(__name__)


@app.route('/camera', methods=['POST'])
def onmobileCamera():
    if(request.methods == 'POST'):
        request_data =request_data
        request_data = json.loads(request_data.decode('utf-8'))
        image = request_data[image]

if __name__ == "__main__":
    app.run()