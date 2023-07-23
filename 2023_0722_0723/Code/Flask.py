from flask import Flask, render_template

app = Flask(__name__)

class Fan:
    def __init__(self, fanname, email):
        self.fanname = fanname
        self.email = email

@app.route('/')
def hello_world():
    return render_template("index.html")

@app.route('/fans/<fans_id>')
def fans_detail(fans_id):
    # 类形式
    fan = Fan(fanname="WongYi", email="12282822@qq.com")
    # 字典形式
    fen ={
        "fanname":"王艺",
        "email":"WongYi@qq.com"

    }
    return render_template("fans_detail.html", fans_id=fans_id, fan=fan, fen=fen)

@app.route('/control')
def control():
    age = 17
    books = [{
        'name':"西游记",
        'author':'吴承恩'
    },{
        'name':"三国演义",
        'author':'罗贯中'
    },]
    return render_template("control.html", age=age, books=books)

if __name__ == '__main__':
    app.run(debug=True)
