# coding:utf-8
 
from flask import Flask, render_template, request, redirect, url_for, make_response,jsonify
from werkzeug.utils import secure_filename
import os
import cv2
import time 
import tensorflow as tf  
import numpy as np  
from sys import path  
from datetime import timedelta
import logging
from cassandra.cluster import Cluster
from cassandra.query import SimpleStatement


log = logging.getLogger()
log.setLevel('INFO')
handler = logging.StreamHandler()
handler.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(name)s: %(message)s"))
log.addHandler(handler)
#from cassandra.cluster import Cluster
#from cassandra import ConsistencyLevel

#设置允许的文件格式
ALLOWED_EXTENSIONS = set(['png', 'jpg', 'JPG', 'PNG', 'bmp'])
display_result= []

#cassandra数据库创建
KEYSPACE = "mydatabase"
def createKeySpace():
    cluster = Cluster(contact_points=['127.0.0.1'],port=9042)#contact_points根据需要进行更改
    session = cluster.connect()
    log.info("Creating keyspace...")
    try:
        session.execute("""
           CREATE KEYSPACE %s
           WITH replication = { 'class': 'SimpleStrategy', 'replication_factor': '2' }
           """ % KEYSPACE)
        log.info("setting keyspace...")
        
    except Exception as e:

        log.error("Unable to create keyspace")

        log.error(e)
    session.set_keyspace(KEYSPACE)
    log.info("creating table...")
    try:
        session.execute("""
           create table Pictures (
               name text,
               result text,
               time text,
               PRIMARY KEY (name)
           )
           """)
    except Exception as e:

        log.error("Unable to create tables")
        log.error(e)
    cluster.shutdown()# 关闭连接

def reversePic(src):
        # 图像反转  
    for i in range(src.shape[0]):
        for j in range(src.shape[1]):
            src[i,j] = 255 - src[i,j]
    return src 
 
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1] in ALLOWED_EXTENSIONS
 
app = Flask(__name__)
# 设置静态文件缓存过期时间
app.send_file_max_age_default = timedelta(seconds=600)
#createKeySpace();#创建databases
 
 
@app.route('/upload', methods=['POST', 'GET'])  # 添加路由
def upload():
    createKeySpace();#创建databases
    # 默认本机数据库集群(IP127.0.0.1).
    cluster = Cluster(contact_points=['127.0.0.1'],port=9042)#contact point根据需要进行IP更改
    # 连接并创建一个会话
    session = cluster.connect()
    session.set_keyspace(KEYSPACE)
    if request.method == 'POST':
        f = request.files['file']
        #设置时间戳
        file_time = time.strftime('%Y.%m.%d %H:%M:%S ',time.localtime(time.time()))
        if not (f and allowed_file(f.filename)):
            return jsonify({"error": 1001, "msg": "请检查上传的图片类型，仅限于png、PNG、jpg、JPG、bmp"})

        #读取mnist模型并进行识别工作
        sess = tf.InteractiveSession()  
        saver=tf.train.import_meta_graph('model_data/model.meta')
        saver.restore(sess, 'model_data/model')
        graph = tf.get_default_graph()
          # 获取输入tensor,,获取输出tensor
        input_x = sess.graph.get_tensor_by_name("Mul:0")
        y_conv2 = sess.graph.get_tensor_by_name("final_result:0")
        basepath = os.path.dirname(__file__)  # 当前文件所在路径
 
        upload_path = os.path.join(basepath, 'static/images', secure_filename(f.filename))  
        f.save(upload_path)
        filename = os.path.basename(upload_path)#读取文件名
 
        img = cv2.imread(upload_path)
        cv2.imwrite(os.path.join(basepath, 'static/images', 'test.jpg'), img)
        

        #运算出结果
        img =reversePic(img) 
        img = cv2.resize(img,(28,28),interpolation=cv2.INTER_CUBIC)  
        x_img = np.reshape(img , [-1 , 784])  
        output = sess.run(y_conv2 , feed_dict={input_x:x_img})  
        number_predict = np.argmax(output)
        a=str(number_predict) 
        
        
        session.execute('insert into mydatabase.Pictures (name,result,time) values (%s,%s,%s);', [filename, a, file_time])
        
        # table中查询数据
        #rows = session.execute('select * from DATA;')
        #final_result = rows
        
        cluster.shutdown()# 关闭连接
        final_result = {'filename':filename,
        'result': number_predict,
        'time': file_time
         }
        display_result.append(final_result)
        return render_template('upload_ok.html',userinput=final_result,val1=time.time())
 
    return render_template('upload.html')



if __name__ == '__main__':
    # app.debug = True
    app.run(host='0.0.0.0', port=5000, debug=True)
