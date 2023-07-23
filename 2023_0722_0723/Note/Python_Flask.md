# Python后台

# Flask

```python
from flask import Flask

app = Flask(__name__)
@app.route('/')
def hello_world():
    return 'Hello Flask!'

if __name__ == '__main__':
    app.run(debug=True, port=80, host='0.0.0.0')
    # http://10.33.92.241:80/
```

官方文档

https://dormousehole.readthedocs.io/en/latest/

### 轻量级 后端框架

#### 1.flask路由

用于匹配url

```python
from flask import Flask

app = Flask(__name__)
@app.route('/')
def hello_world():
    return 'Hello Flask!'

if __name__ == '__main__':
    app.run()
# http://127.0.0.1:5000/

->@app.route('/hello')
# http://127.0.0.1:5000/hello

->@app.route('/', methods=['POST'])
# 提示：Method Not ALLowed
# 无法访问
# methods=['GET', 'POST']

# 出现重名优先匹配第一个
```

##### debug模式

```python
->app.run(debug=True)
```

##### 修改host

让其他电脑访问到我的电脑上的flask项目

```python
->app.run(host='0.0.0.0')
```

##### 修改port端口号

```python
->app.run(port=80)
```

##### URL与视图的映射

```python
http[80]
https[443]
->@app.route('/blog/list')
# 带参数的url
->@app.route('/blog/<blog_id>')
->@app.route('/blog/<int:blog_id>')

# 查询字符串的方式传参
from flask import request
@app.route('/book/list')
def book_list()
	# arguments:参数
    # request.args:类字典类型
    page = request.args.get("page", default=1, type=int)
    return f"您获取的是第{page}的图书列表!"
```

#### 2.request对象 abort函数

#### 3.模板

```python
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
    # 您的信息：{{ fan.fanname }} / {{ fan.email }}
    # 字典形式
    fen ={
        "fanname":"王艺",
        "email":"WongYi@qq.com"

    }
    # <div>您的信息：{{ fen['fanname'] }} / {{ fen.email }}</div>
    return render_template("fans_detail.html", fans_id=fans_id, fan=fan, fen=fen)

if __name__ == '__main__':
    app.run(debug=True)

```

```python
<p>表示段落或文本块</P>
<div>用于创建块级容器</div>
<h1>变粗变大变黑</h1>
```

#### 过滤器

管道符 | 

```python
# username = WongYi
{{user.username|length}}
# 输出：6

# 自定义过滤器
def 函数名(value)
	return 目标

app.add_template_filter(函数名, "过滤器名")

->html
->{{value|过滤器名}}
```

#### 控制语句

```python8
{% if age>18 %}
<div>您已满18岁，可以进入网吧！</div>
{% elif age==18 %}
<div>您刚满18岁，需要父母陪同才可以进入网吧！</div>
{% else %}
<div>您未满18岁，不可以进入网吧！</div>
{% endif %}
<p></p>
{% for book in books %}
<div>图书名称：{{book.name}}，图书作者：{{book.author}}</div>
{% endfor %}
```

#### 模板继承

#### 4.flask数据库 

#### 