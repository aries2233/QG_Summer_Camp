# MySQL

## SQL语言

## 基础

DataBase

DAMS数据库管理系统

SQL编程语言

#### 数据库类型

##### 关系型数据库

表头、行、列、可以关联其他表格

![bcc1795b879704118220bae6524fb53](https://cdn.jsdelivr.net/gh/aries2233/image_0/bcc1795b879704118220bae6524fb53.png)

##### MySQL数据库

![7def733cdad657d9bdd1d41139941ce](https://cdn.jsdelivr.net/gh/aries2233/image_0/7def733cdad657d9bdd1d41139941ce.png)

### SQL

#### 通用语法

以分号结尾

不分大小写

注释--或#

多行注释/**/

#### 分类

#### DDL数据定义>数据库对象（数据库、表、字段）

```sql
SHOW DATABASES; # 查询所有数据库
SELECT DATABASE(); # 查询当前数据库
CREATE DATABASE 数据库名[字符集][排序规则]; # 创建
DROP DATABASE 数据库名 # 删除、
USE 数据库名;

```

```sql
# 表结构
SHOW TABLES; # 查询当前数据库所有表
DESC 表名;  # 查询表结构
SHOW CREATE TABLE 表名; # 查询指定表
# 创建表
CREATE TABLE 表名(
	字段1 字段1类型[COMMFNT 字段1注释],
	字段2 字段2类型[COMMFNT 字段2注释],
	字段3 字段3类型[COMMFNT 字段3注释]
)[COMMFNT 表注释];

# 表结构的修改
# 添加字段
ALTER TABLE 表名 ADD 字段名 类型(长度) [COMMENT 注释][约束];
# 修改字段
ALTER TABLE 表名 MODIFY 字段名 新数据类型(长度);
ALTER TABLE 表名 CHANGE 旧字段名 新字段名 类型(长度) [COMMENT 注释][约束];
# 删除字段
ALTER TABLE 表名 DROP 字段名;
# 修改表名
ALTER TABLE 表名 RENAME TO 新表名;
# 删除表
DROP TABLE [IF EXISTS] 表名；
TRUNCATE TABLE 表名; # 删除后重构

```

数据类型

![bb72e6720010ded9a4c75de54d5a7bb](https://cdn.jsdelivr.net/gh/aries2233/image_0/bb72e6720010ded9a4c75de54d5a7bb.png)

M精度：整个数的位数

D标度：小数部分的位数

无符号类型UNSIGNED

float(M,D)

![ed66071ea657c97b8518d81a7e89353](https://cdn.jsdelivr.net/gh/aries2233/image_0/ed66071ea657c97b8518d81a7e89353.png)

![f2e10c014362e270af2f5ee2d2fd0f1](https://cdn.jsdelivr.net/gh/aries2233/image_0/f2e10c014362e270af2f5ee2d2fd0f1.png)

```sql
create table emp(
	id int comment '编号',
	workno varchar(10) comment '工号',
	name varchar(10) comment '姓名',
	gender char(1) comment '性别',
	age tinyint unsigned comment '年龄',
	idcard char(18) comment '身份证号',
	entrydate date comment '入职时间'
) comment '员工表';
```



#### DML数据操作

增

```sql
# 给指定字段添加数据
INSERT INTO 表名(字段1, 字段2...) VALUES (值1, 值2...);
# 给全部字段添加数据
INSERT INTO 表名 VALUES (值1, 值2...);
# 批量添加数据
INSERT INTO 表名(字段1, 字段2...) VALUES (值1, 值2...), (值1, 值2...), (值1, 值2...);
INSERT INTO 表名 VALUES (值1, 值2...), (值1, 值2...), (值1, 值2...);
```

删

```sql
DELETE FROM 表名 [WHERE 条件];
```

改

```sql
UPDATA 表名 SET 字段名1 = 值1, 字段名2 = 值2, ...[WHERE 条件];
```



#### DQL数据查询

```sql
# 基本查询
SELECT 字段1， 字段2, 字段3... FROM 表名;
SELECT * FROM 表名;
SELECT 字段1[AS 别名1], 字段2[AS 别名2]... FROM 表名; # 设置别名
SELECT DISTINCT 字段列表 FROM 表名; # 去除重复记录
# 分组查询
# 排序查询
# 分页查询
```

```sql
# 条件查询
SELECT 字段列表 FROM 表名 WHERE 条件列表;

### 模糊匹配 (_匹配单个字符, %匹配任意个字符)
# 查询两个字符
SELECT * FROM 表名 WHERE 字段 LIKE '__';
# 查询最后一位为X
SELECT * FROM 表名 WHERE 字段 LIKE '%X';
```

![06e1093eaa0ac1f1b6b53be5837e036](https://cdn.jsdelivr.net/gh/aries2233/image_0/06e1093eaa0ac1f1b6b53be5837e036.png)



#### DCL数据控制>创建用户，用户权限

##### 管理用户

```sql
# 查询用户
USE mysql;
SELECT * FROM user;
# 创建用户
CREAT USER '用户名' @ '主机名' IDENTIFIED BY '密码';
# 在任意主机 主机名->%
# 修改用户密码
ALTER USER '用户名' @ '主机名' IDENTIFIED WITH mysql_native_password BY '新密码';
# 删除用户
DROP USER '用户名' @ '主机名';
```

权限控制

```sql
# 查询权限
SHOW GRANTS FOR '用户名' @ '主机名';
# 授予权限
GRANT 权限列表 ON 数据库名.表名 TO '用户名' @ '主机名';
# 撤销权限
REVOKE 权限列表 ON 数据库名.表名 FROM '用户名' @ '主机名';
```

![bb058744d4c9b575aa76daaad771a77](https://cdn.jsdelivr.net/gh/aries2233/image_0/bb058744d4c9b575aa76daaad771a77.png)

### 函数

字符串函数

![0fb1ffcd1765111a3ad665628af388c](https://cdn.jsdelivr.net/gh/aries2233/image_0/0fb1ffcd1765111a3ad665628af388c.png)

数值函数

![54fc5b1939b3899e013829ca0b60c4d](https://cdn.jsdelivr.net/gh/aries2233/image_0/54fc5b1939b3899e013829ca0b60c4d.png)

日期函数

![ce38947c1aa2231054c88eefa74030e](https://cdn.jsdelivr.net/gh/aries2233/image_0/ce38947c1aa2231054c88eefa74030e.png)

流程函数

![a2c7d2294d403fe4704e818cfa90933](https://cdn.jsdelivr.net/gh/aries2233/image_0/a2c7d2294d403fe4704e818cfa90933.png)

```sql
select
	name,
	(case workaddress when '北京' then '一线城市' when '上海' then '一线城市' else '二线城市' end) as '工作地址' from emp;
```



### 约束

限制存储在表中的数据

![d81d2a5bb748df598a2b59e4202594b](https://cdn.jsdelivr.net/gh/aries2233/image_0/d81d2a5bb748df598a2b59e4202594b.png)

```sql
create table emp(
	id int primary key auto_increment comment '编号',
	name varchar(10) not null unique comment '姓名',
    status char(1) default '1' comment '状态' # 默认值为1
)comment '用户表';
```

##### 外键约束

![0549607a7ebe75ae9e669a30edf2814](https://cdn.jsdelivr.net/gh/aries2233/image_0/0549607a7ebe75ae9e669a30edf2814.png)

```sql
# 添加外键
CREATE TABLE 表名(
	字段名 数据类型, 
    ...
    [CONSTRAINT][外键名称]FOREIGN KEY (外键字段名) REFERENCES 主表(主表列名)
);
ALTER TABLE 表名 ADD CONSTRAINT 外键名称 FOREIGN KEY (外键字段名) REFERENCES 主表(主表列名);

# 删除外键
ALTER TABLE 表名 DROP FOREIGN KEY 外键名称;
```



### 多表查询

### 事务

## 进阶

## 运维