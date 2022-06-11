import flask
import io
from PIL import Image

app = flask.Flask(__name__)

def transform_image(image_bytes):
    image = Image.open(io.BytesIO(image_bytes))
    image.save('hello.jpg')
    return 'hi'

@app.route('/')
def status():
    return 'OK'

@app.route('/estimation', methods=["POST"])
def estimation():
    if flask.request.method == "POST":
        if flask.request.files.get("img"):
            image = flask.request.files["img"].read()

            transform_image()
            print(image)

            # class_id, class_name = get_prediction(image_bytes=image)
            return flask.jsonify({
                'hi': 'hihi'
            })


if __name__ == "__main__":
    app.run(host='0.0.0.0', port=9090)