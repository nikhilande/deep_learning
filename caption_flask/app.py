from flask import Flask,render_template,request,jsonify
from caption import captionimage
app=Flask(__name__)
@app.route('/')
def hello():
    return render_template('index.html')
@app.route('/',methods=['POST'])
def result():
    if request.method == 'POST':
        f=request.files['fileuser']
        path='./static/{}'.format(f.filename)
        f.save(path)
        captions=captionimage(path)
        imagedict={
            'image':path,
            'cap':captions,
        }
    return render_template('index.html',your_result=imagedict)
# @app.route('/api')
# def api():
#     if request.method=='POST':
#         im=request.json['image']
#         out=captionimage(im)
#         result=request.json(out)
#     return jsonify(result)
if __name__ == '__main__':
    app.run(debug=True)