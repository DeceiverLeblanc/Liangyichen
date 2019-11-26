FROM finalcassandra:new

WORKDIR ./myapp 
ADD . . 
RUN pip install -r requirements.txt -i  https://pypi.tuna.tsinghua.edu.cn/simple/ 
VOLUME ./myapp/model_data 
EXPOSE 9042 
EXPOSE 5000
CMD [ "python3", "./myapp/mnist_flask.py" ]

