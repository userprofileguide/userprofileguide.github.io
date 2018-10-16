---
id: 7
title: Python变量类型
date: 2018-10-14T23:17:07+00:00
author: Yan Xu
layout: single
permalink: /python-variable/
categories: Python
---
#### (二). Python变量类型

变量存储在内存中的值。这就意味着在创建变量时会在内存中开辟一个空间。
基于变量的数据类型，解释器会分配指定内存，并决定什么数据可以被存储在内存中。
因此，变量可以指定不同的数据类型，这些变量可以存储整数，小数或字符。

##### 1. 变量赋值
Python 中的变量赋值不需要类型声明。
每个变量在内存中创建，都包括变量的标识，名称和数据这些信息。
每个变量在使用前都必须赋值，变量赋值以后该变量才会被创建。
等号（=）用来给变量赋值。
等号（=）运算符左边是一个变量名,等号（=）运算符右边是存储在变量中的值。例如：

```
counter = 100 # 赋值整型变量
miles = 1000.0 # 浮点型
name = "John" # 字符串
print counter
print miles
print name
```

以上实例中，100，1000.0和"John"分别赋值给counter，miles，name变量。
执行以上程序会输出如下结果：


```
100
1000.0
John
```

##### 2.标准数据类型
在内存中存储的数据可以有多种类型。<br>
例如，一个人的年龄可以用数字来存储，他的名字可以用字符来存储。<br>
Python 定义了一些标准类型，用于存储各种类型的数据。
Python有五个标准的数据类型：

* Numbers（数字）
* String（字符串）
* List（列表）
* Tuple（元组）
* Dictionary（字典）

##### (1) Numbers
Python支持四种不同的数字类型：

* int（有符号整型）
* long（长整型[也可以代表八进制和十六进制]）
* float（浮点型）
* complex（复数）

|      | int       |long        |float      |complex  |
| :--- | :-------: |:-------- -:|:---------:|:-------: |
| 实例  |10        |51924361L   | 0.0       |    3.14j |
| 实例  |100       |-0x19323L   | -21.9      |    9.322e-36j |
| 实例  |10        |535633629843L  | 15.20      |    3e+26J|
| 实例  |10        |-052318172735L |70.2E-12     |   4.53e-7j |

长整型也可以使用小写 l，但是还是建议您使用大写 L，避免与数字 1 混淆。Python使用 L 来显示长整型。

Python 还支持复数，复数由实数部分和虚数部分构成，可以用 a + bj,或者 complex(a,b) 表示， 复数的实部 a 和虚部 b 都是浮点型<br>

**注意**: long 类型只存在于 Python2.X 版本中，在 2.2 以后的版本中，int 类型数据溢出后会自动转为long类型。在 Python3.X 版本中 long 类型被移除，使用 int 替代。


##### (2)字符串 String
##### 定义
字符串或串(String)是由数字、字母、下划线组成的一串字符。<br>
一般记为 :<br>
s = "a1a2···an"(n>=0)<br>
它是编程语言中表示文本的数据类型。

##### 字符串截取与拼接
python的字串列表有2种取值顺序:
* 从左到右索引默认0开始的，最大范围是字符串长度少1

* 从右到左索引默认-1开始的，最大范围是字符串开头

如果你要实现从字符串中获取一段子字符串的话，可以使用 [头下标:尾下标] 来截取相应的字符串，其中下标是从 0 开始算起，可以是正数或负数，下标可以为空表示取到头或尾。

[头下标:尾下标] 获取的子字符串包含头下标的字符，但不包含尾下标的字符。<br>
例如：s[1,5]<br>
当使用以冒号分隔的字符串，python 返回一个新的对象，结果包含了以这对偏移标识的连续的内容，左边的开始是包含了下边界。
上面的结果包含了 s[1] 的值，而取到的最大范围不包括尾下标，就是 s[5] 的值。

加号（+）是字符串连接运算符，星号（\*）是重复操作。如下实例：

```
str = 'Hello World!'
print str           
print str[0]        
print str[2:5]     
print str[2:]       
print str * 2       
print str + "TEST"
```

以上实例输出结果:
```
Hello World!
H
llo
llo World!
Hello World!Hello World!
Hello World!TEST
```

##### (3) 列表
List（列表） 是 Python 中使用最频繁的数据类型。
列表可以完成大多数集合类的数据结构实现。它支持字符，数字，字符串甚至可以包含列表（即嵌套）。<br>

列表用 [ ] 标识，是 python 最通用的复合数据类型。
列表中值的切割也可以用到变量 [头下标:尾下标] ，就可以截取相应的列表，从左到右索引默认 0 开始，从右到左索引默认 -1 开始，下标可以为空表示取到头或尾。

加号 + 是列表连接运算符，星号 * 是重复操作。如下实例：

```
list = [ 'runoob', 786 , 2.23, 'john', 70.2 ]
tinylist = [123, 'john']
print list               
print list[0]            
print list[1:3]          
print list[2:]           
print tinylist * 2       
print list + tinylist    
```
以上实例输出结果：

```
['runoob', 786, 2.23, 'john', 70.2]
runoob
[786, 2.23]
[2.23, 'john', 70.2]
[123, 'john', 123, 'john']
['runoob', 786, 2.23, 'john', 70.2, 123, 'john']
```
##### (4) 元组

元组是另一个数据类型，类似于List（列表）。
元组用"()"标识。内部元素用逗号隔开。但是元组不能二次赋值，相当于只读列表。

##### (5) 字典
字典(dictionary)是除列表以外python之中最灵活的内置数据结构类型。列表是有序的对象集合，字典是无序的对象集合。

两者之间的区别在于：字典当中的元素是通过键来存取的，而不是通过偏移存取。
字典用"{ }"标识。字典由索引(key)和它对应的值value组成。

```
dict = {}
dict['one'] = "This is one"
dict[2] = "This is two"
tinydict = {'name': 'john','code':6734, 'dept': 'sales'}
print dict['one']          
print dict[2]              
print tinydict             
print tinydict.keys()      
print tinydict.values()    
```

输出结果为：

```
This is one
This is two
{'dept': 'sales', 'code': 6734, 'name': 'john'}
['dept', 'code', 'name']
['sales', 6734, 'john']
```


##### (6) 数据类型转换


|函数             | 描述            |
|:-------------:|:-------------:|
| int(x [,base])  | 将x转换为一个整数 |
|  long(x[,base]) | 将x转换为一个长整数 |
| float(x)        | 将x转换到一个浮点数 |
| str(x)          | 将对象 x 转换为字符串 |
| tuple(s)        | 将序列 s 转换为一个元组 |
| list(s)         | 将序列 s 转换为一个列表 |