FROM python:latest
RUN mkdir -p /usr/local/src/assg
WORKDIR /usr/local/src/assg

RUN mkdir -p static
RUN mkdir -p templates
RUN mkdir -p Output
RUN mkdir -p Output1
RUN mkdir -p Output2
RUN mkdir -p Models
RUN mkdir -p Models1
RUN mkdir -p Models2
ENV ak 1
ENV sak 1
ADD app.py .
ADD functions.py .
ADD Usernames.txt .
ADD static static/.
ADD Templates templates/.

RUN pip install flask
RUN pip install pandas
RUN pip install sklearn
RUN pip install scipy
RUN pip install boto
RUN pip install imblearn
EXPOSE 80
CMD ["sh","-c","python /usr/local/src/assg/app.py ${ak} ${sak}"]
