#本模块用于初始化各种全局数据

label_encode = { #每个标签对应一个编号
    "1 1" : 0,
    "1 0" : 1,
    "1 -1" : 2,
    "0 1" : 3,
    "0 0" : 4,
    "0 -1" : 5,
    "-1 1" : 6,
    "-1 0" : 7,
    "-1 -1" : 8
}

label_decode = { #每个编号对应一个标签
    0 : "1 1",
    1 : "1 0",
    2 : "1 -1",
    3 : "0 1",
    4 : "0 0",
    5 : "0 -1",
    6 : "-1 1",
    7 : "-1 0",
    8 : "-1 -1"
}
all_labels = [
    "1 1",
    "1 0",
    "1 -1",
    "0 1",
    "0 0",
    "0 -1",
    "-1 1",
    "-1 0",
    "-1 -1"
]

label_msg = {
    "1 1" : "A greater y is more preferred at any time",
    "1 0" : "A greater y is more preferred when y < x; Any y is equally preferred when y > x",
    "1 -1" : "A greater y is more preferred when y < x; A smaller y is more preferred when y > x",
    "0 1" : "Any y is equally preferred when y < x; A greater y is more preferred when y > x",
    "0 0" : "Any y is equally preferred at any time",
    "0 -1" : "Any y is equally preferred when y < x; A smaller y is more preferred when y > x",
    "-1 1" : "A smaller y is more preferred when y < x; A greater y is more preferred when y > x",
    "-1 0" : "A smaller y is more preferred when y < x; Any y is equally preferred when y > x",
    "-1 -1" : "A smaller y is more preferred at any time"   
}

label_inv = {
    "1 1" : "-1 -1",
    "1 0" : "0 -1",
    "1 -1" : "-1 1",
    "0 1" : "-1 0",
    "0 0" : "0 0",
    "0 -1" : "1 0",
    "-1 1" : "1 -1",
    "-1 0" : "0 1",
    "-1 -1" : "1 1"    
}


