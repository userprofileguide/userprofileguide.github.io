---
id: 3
title: Python文件I/O
date: 2018-10-14T23:17:07+00:00
author: Yan Xu
layout: single
permalink: /python-fileio/
categories: Python
---
#### (六). Python文件I/O

<em>这里只讲述所有基本的的I/O函数，更多函数请参考Python标准文档</em>

##### 1. 打开和关闭文件
Python 提供了必要的函数和方法进行默认情况下的文件基本操作。你可以用 file 对象做大部分的文件操作

##### (1). open函数

你必须先用Python内置的open()函数打开一个文件，创建一个file对象，相关的方法才可以调用它进行读写<br>
语法：
```
file object = open(file_name , access_mode, buffering)
```

各个参数的细节如下：
* file_name：file_name变量是一个包含了你要访问的文件名称的字符串值

* access_mode：access_mode决定了打开文件的模式：只读，写入，追加等。所有可取值见如下的完全列表。这个参数是非强制的，默认文件访问模式为只读(r)

* buffering:如果buffering的值被设为0，就不会有寄存。如果buffering的值取1，访问文件时会寄存行。如果将buffering的值设为大于1的整数，表明了这就是的寄存区的缓冲大小。如果取负值，寄存区的缓冲大小则为系统默认

**不同模式打开文件的完全列表:**

| 模式 | 描述            |
| :----| :------------- |
| t    | 文本模式 (默认) |
| x    | 写模式，新建一个文件，如果该文件已存在则会报错  |
| b    | 二进制模式       |
| +    | 打开一个文件进行更新(可读可写)       |
| U    | 通用换行模式（不推荐）      |
| r    | 以只读方式打开文件。文件的指针将会放在文件的开头。这是默认模式   |
| rb   | 以二进制格式打开一个文件用于只读。文件指针将会放在文件的开头。这是默认模式。一般用于非文本文件如图片等  |
| r+   | 打开一个文件用于读写。文件指针将会放在文件的开头  |
| rb+  | 以二进制格式打开一个文件用于读写。文件指针将会放在文件的开头。一般用于非文本文件如图片等    |
| w    | 打开一个文件只用于写入。如果该文件已存在则打开文件，并从开头开始编辑，即原有内容会被删除。如果该文件不存在，创建新文件 |
| wb   | 以二进制格式打开一个文件只用于写入。如果该文件已存在则打开文件，并从开头开始编辑，即原有内容会被删除。如果该文件不存在，创建新文件。一般用于非文本文件如图片等 |
| w+   | 打开一个文件用于读写。如果该文件已存在则打开文件，并从开头开始编辑，即原有内容会被删除。如果该文件不存在，创建新文件  |
| wb+  | 以二进制格式打开一个文件用于读写。如果该文件已存在则打开文件，并从开头开始编辑，即原有内容会被删除。如果该文件不存在，创建新文件。一般用于非文本文件如图片等 |
| a    | 打开一个文件用于追加。如果该文件已存在，文件指针将会放在文件的结尾。也就是说，新的内容将会被写入到已有内容之后。如果该文件不存在，创建新文件进行写入 |
| ab   | 以二进制格式打开一个文件用于追加。如果该文件已存在，文件指针将会放在文件的结尾。也就是说，新的内容将会被写入到已有内容之后。如果该文件不存在，创建新文件进行写入 |
| a+   | 打开一个文件用于读写。如果该文件已存在，文件指针将会放在文件的结尾。文件打开时会是追加模式。如果该文件不存在，创建新文件用于读写  |
| ab+  | 以二进制格式打开一个文件用于追加。如果该文件已存在，文件指针将会放在文件的结尾。如果该文件不存在，创建新文件用于读写  |

##### (2). file对象的属性

一个文件被打开后，你有一个file对象，你可以得到有关该文件的各种信息
以下是和file对象相关的所有属性的列表：

| 属性           | 描述            |
| :------------- | :------------- |
| file.closed    | 返回true如果文件已被关闭，否则返回false  |
| file.mode      | 返回被打开文件的访问模式  |
| file.name      | 返回文件的名称   |
| file.softspace | 如果用print输出后，必须跟一个空格符，则返回false。否则返回true  |

如下实例：

```
#!/usr/bin/python
# -*- coding: UTF-8 -*-
# 打开一个文件
fo = open("foo.txt", "w")
print("文件名: ", fo.name)
print ("是否已关闭 : ", fo.closed)
print ("访问模式 : ", fo.mode)
print ("末尾是否强制加空格 : ", fo.softspace)
```

得到结果：
```
文件名:  foo.txt
是否已关闭 :  False
访问模式 :  w
末尾是否强制加空格 :  0
```

##### (3). close()方法

File 对象的 close（）方法刷新缓冲区里任何还没写入的信息，并关闭该文件，这之后便不能再进行写入。<br>
当一个文件对象的引用被重新指定给另一个文件时，Python 会关闭之前的文件。用 close（）方法关闭文件是一个很好的习惯。 <br>
语法：

``fileObject.close()``

例子:

```
#!/usr/bin/python
# -*- coding: UTF-8 -*-
# 打开一个文件
fo = open("foo.txt", "w")
print "文件名: ", fo.name
# 关闭打开的文件
fo.close()
```

结果如下：

```
文件名:  foo.txt
```

##### (4). write()方法

ite()方法可将任何字符串写入一个打开的文件。需要重点注意的是，Python字符串可以是二进制数据，而不是仅仅是文字。 <br>
write()方法不会在字符串的结尾添加换行符('\n')：

语法：

```
fileObject.write(string)
```

在这里，被传递的参数是要写入到已打开文件的内容

例子：
```
#!/usr/bin/python
# -*- coding: UTF-8 -*-
# 打开一个文件
fo = open("foo.txt", "w")
fo.write( "www.runoob.com!\nVery good site!\n")
# 关闭打开的文件
fo.close()
```

结果如下：

上述方法会创建foo.txt文件，并将收到的内容写入该文件，并最终关闭文件。如果你打开这个文件，将看到以下内容:
```
$ cat foo.txt
www.runoob.com!
Very good site!
```

##### (5). read()方法

read（）方法从一个打开的文件中读取一个字符串。需要重点注意的是，Python字符串可以是二进制数据，而不是仅仅是文字。

语法：
```
fileObject.read([count])
```

在这里，被传递的参数是要从已打开文件中读取的字节计数。该方法从文件的开头开始读入，如果没有传入count，它会尝试尽可能多地读取更多的内容，很可能是直到文件的末尾。

例子：
这里我们用到以上创建的 foo.txt 文件

```
#!/usr/bin/python
# -*- coding: UTF-8 -*-
# 打开一个文件
fo = open("foo.txt", "r+")
str = fo.read(10)
print "读取的字符串是 : ", str
# 关闭打开的文件
fo.close()
```

结果如下：

``读取的字符串是 :  www.runoob``

**注**：

| 序号 |      方法及描述        |
| :-- | :----------------------|
| 1| file.close() 关闭文件,关闭后文件不能再进行读写操作   |
| 2| file.flush() 刷新文件内部缓冲，直接把内部缓冲区的数据立刻写入文件, 而不是被动的等待输出缓冲区写入 |
| 3| file.fileno() 返回一个整型的文件描述符(file descriptor FD 整型), 可以用在如os模块的read方法等一些底层操作上 |
| 4| file.isatty() 如果文件连接到一个终端设备返回 True，否则返回 Fals |
| 5| file.next() 返回文件下一行   |
| 6| file.read([size]) 从文件读取指定的字节数，如果未给定或为负则读取所有 |
| 7| file.readline([size] 读取整行，包括 "\n" 字符  |
| 8| file.readlines([sizeint]) 读取所有行并返回列表，若给定sizeint>0，则是设置一次读多少字节，这是为了减轻读取压力    |
| 9| file.seek(offset[, whence])设置文件当前位置  |
| 10| file.tell() 返回文件当前位置  |
| 11| file.truncate([size]) 截取文件，截取的字节通过size指定，默认为当前文件位置   |
| 12| file.write(str) 将字符串写入文件，返回的是写入的字符长度   |
| 13| file.writelines(sequence) 向文件写入一个序列字符串列表，如果需要换行则要自己加入每行的换行符     |
