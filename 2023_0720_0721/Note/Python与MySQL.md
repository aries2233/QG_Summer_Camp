# Python与MySQL

```python
# 常用模块
import pymysql

def get_conn():
    # 获取MySQL的链接
    returen pymysql.connect(
    	host='',
        user='root',
        password='123456',
        database='',
        charset='utf8'
    )
    
def query_data(sql):
    # 根据SQL查询数据并返回
    conn = get_conn()
    try:
        cursor = conn.cursor(pymysql.cursors.DictCursor)
        cursor.execute(sql)
        return cursor.fetchall()
    finally:
        conn.close()
        
def insert_data(sql):
def update_data(sql):
    # 新增或更新sql
    conn = get_conn()
    try:
        cursor = conn.cursor()
        cursor.execute(sql)
        conn.commit()# 重点
    finally:
        conn.close()
```

```python
try:
	db = pymysql.connect(DBHOST, DBUSER, DBPASS, DBNAME)
	print('数据库连接成功！')
except pymysql.Error as e:
	print('数据库连接失败'+str(e))
	
	
# 创建新表
# 1.声明一个游标
cur = db.cursor()
# 2.检查是否存在
cur.execute('DROP TABLE IF EXISTS Student')
# 3.编写SQL语句
sqlQuery = "CREATE TABLE Student(Name CHAR(20)NOT NULL, Email CHAR(20), Age int)"
cur.execute(sqlQuery)

# 插入数据
sqlQuery = "INSERT INTO Student(Name, Email, Age)VALUE(%s, %s, %s)"
value = ('Mike', '123456@163.com', 20)
try:
    cur.execute(sqlQuery, value)
    db.commit()
    print('数值插入成功！')
except pymysql.Error as e:
	print('数值插入失败'+str(e))
    
# 查询数据
sqlQuery = "SELECT * FROM Student"
try:
    cur.execute(sqlQuery)
    result = cur.fetchall()
    for row in results:
        name = row[0]
        email = row [1]
        age = row[2]
        print('Name:%s, Email:%s, Age:%s'%(name, email, age))
except pymysql.Error as e:
	print('数值查询失败'+str(e))
    
# 更新数据
sqlQuery = "updata Student set Name = %s where Name = %s"
value = ('John', 'Mike')
try:
    cur.execute(sqlQuery, value)
    db.commit()
    print('数值更新成功！')
except pymysql.Error as e:
	print('数值更新失败'+str(e))
    
# 删除数据
sqlQuery = "delete from Student where Name = %s"
value = ('John')
try:
    cur.execute(sqlQuery, value)
    db.commit()
    print('数值删除成功！')
except pymysql.Error as e:
	print('数值删除失败'+str(e))
    
# 删除表
sqlQuery = "DROP TABLE IF EXISTS Student"
cur.execute(sqlQuery)
print('表删除成功！'
```

