import os, zipfile
from flask import Flask, render_template, flash, request, url_for, redirect, Response, send_file
from functions import credentials, read_file,model_run,zip_file,aws_connect,unzip_file,form_output, clean_data,model_build,bucket_get,bucket_set,clean_run_data,clean_data_liver_run,clean_data_liver
from os import remove,path
from werkzeug.utils import secure_filename
import json
import pandas as pd
import sys

ak=sys.argv[1]
sak=sys.argv[2]

app = Flask(__name__)
app.config['UPLOAD_FOLDER']='Data'
# set as part of the config
SECRET_KEY = 'many random bytes'
app.config['S3']=''
# or set directly on the app
app.secret_key = 'many random bytes'

@app.route('/')
def homepage():
    print("main hiiiii")
    return render_template("main.html")

@app.route('/dashboard/', methods=["GET","POST"])
def dashboard():
    print("dashboard hiiiiiiiiiii")
    error = ''
    try:
        if request.method == "POST":
            attempted_username = request.form['username']
            attempted_password = request.form['password']
            credt=credentials()
            lst=credt[attempted_username]
            #bucket_get('modelsads','Models','Models.zip',aws_connect(ak,sak))
            #unzip_file('Models.zip','Models')
            #os.remove('Models.zip')
            if attempted_username in credt and lst[0]==attempted_password:
                print("sucess!!!!")
                if lst[1]=='admin':
                   return render_template("admindashboard.html")
                else:
                   return render_template("usrmenu.html")
            else:
                error = "Invalid credentials. Try Again."
                print(error)
                flash(error)
        return render_template("main.html", error = error)
    except Exception as e:
        flash(e)
        print(e)
        return render_template("main.html", error = error)

@app.route('/usr_menu/', methods=['GET','POST'])
def usr_menu():
    print("menu !!!!!!!!!!")
    error = ''
    try:
        if request.method=='POST':
           disease=request.form['disease']
           if disease=='Kidney':
              return render_template("dashboard.html")
           else:
              if disease=='Liver':
                 return render_template("dashboard1.html")
              else:
                 return render_template("dashboard2.html")
        error = "Error loading menu"
        print(error)
        return render_template("usrmenu.html", error = error)
    except Exception as e:
        flash(e)
        print(e)
        return render_template("usrmenu.html", error = error)

@app.route('/upload/', methods=['GET','POST'])
def upload_file(): 
    print("upload hiiiiiiiiiii")
    error = ''
    try:
        if request.method == 'POST':
           f=request.files['file']
           f.save(secure_filename(f.filename))
           df=read_file(f.filename)
           df=clean_run_data(df)
           df1,df_summ=model_run(df,'Kidney')
           os.remove(f.filename)
           flash("File uploaded")
           return render_template("view.html",tables=[df_summ.to_html(),df1.to_html()], titles = ['Error Metrics','Predicted O/P'])
        error = "File Not uploaded"
        return render_template("dashboard.html", error = error)
    except Exception as e:
        flash(e)
        print(e)
        return render_template("dashboard.html", error = error)

@app.route('/upload1/', methods=['GET','POST'])
def upload1_file(): 
    print("upload1 hiiiiiiiiiii")
    error = ''
    try:
        if request.method == 'POST':
           f=request.files['file']
           f.save(secure_filename(f.filename))
           df=read_file(f.filename)
           df=clean_data_liver_run(df)
           df1,df_summ=model_run(df,'Liver')
           os.remove(f.filename)
           flash("File uploaded")
           return render_template("view1.html",tables=[df_summ.to_html(),df1.to_html()], titles = ['Error Metrics','Predicted O/P'])
        error = "File Not uploaded"
        return render_template("dashboard1.html", error = error)
    except Exception as e:
        flash(e)
        print(e)
        return render_template("dashboard1.html", error = error)

@app.route('/upload2/', methods=['GET','POST'])
def upload2_file(): 
    print("upload2 hiiiiiiiiiii")
    error = ''
    try:
        if request.method == 'POST':
           f=request.files['file']
           f.save(secure_filename(f.filename))
           df=read_file(f.filename)
           #df=clean_data_diab_run(df)
           df1,df_summ=model_run(df,'Diab')
           os.remove(f.filename)
           flash("File uploaded")
           return render_template("view2.html",tables=[df_summ.to_html(),df1.to_html()], titles = ['Error Metrics','Predicted O/P'])
        error = "File Not uploaded"
        return render_template("dashboard2.html", error = error)
    except Exception as e:
        flash(e)
        print(e)
        return render_template("dashboard2.html", error = error)

@app.route('/adminupload/', methods=['GET','POST'])
def adminupload_file(): 
    print("upload admin hiiiiiiiiiii")
    error = ''
    try:
        if request.method == 'POST':
           f=request.files['file']
           disease=request.form['disease']
           if disease=='Kidney':
              f.save(secure_filename('kidney_disease.csv'))
              bucket_set('datasetads','kidney_disease.csv','kidney_disease.csv',aws_connect(ak,sak))
              df=read_file('kidney_disease.csv')
              df=clean_data(df)
              df_summ=model_build(df,disease)
              bucket_set('modelsads','Models','Models.zip',aws_connect(ak,sak))
              os.remove('kidney_disease.csv')
           else:
              if disease=='Liver':
                 f.save(secure_filename('liver_disease.csv'))
                 bucket_set('datasetads','liver_disease.csv','liver_disease.csv',aws_connect(ak,sak))
                 df=read_file('liver_disease.csv')
                 df=clean_data_liver(df)
                 df_summ=model_build(df,disease)
                 bucket_set('modelsads','Models1','Models1.zip',aws_connect(ak,sak))
                 os.remove('liver_disease.csv') 
              else:
                 f.save(secure_filename('diabetes_disease.csv'))
                 bucket_set('datasetads','diabetes_disease.csv','diabetes_disease.csv',aws_connect(ak,sak))
                 df=read_file('diabetes_disease.csv')
                 #df=clean_data_diab(df)
                 df_summ=model_build(df,disease)
                 bucket_set('modelsads','Models2','Models2.zip',aws_connect(ak,sak))
                 os.remove('diabetes_disease.csv') 
           flash("New Models Build")
           return render_template("admindashboard.html",table=df_summ.to_html(),title='Error Metrics',name=disease)
        error = "Error while Building Models"
        flash(error)
        return render_template("admindashboard.html",error = error)
    except Exception as e:
        flash(e)
        print(e)
        return render_template("admindashboard.html", error = error)

@app.route('/forminput/', methods=['GET','POST'])
def form_input():
    print("form input")
    error = ''
    try:
        if request.method == 'POST':
           ff=request.form['disease']
           pcv=request.form['pcv']
           hemo=request.form['hemo']
           sg=request.form['sg']
           sc=request.form['sc']
           rc=request.form['rc']
           al=request.form['al']
           dm=request.form['dm']
           bgr=request.form['bgr']
           if ff=='Kidney':
              sod=request.form['sod']
              htn=request.form['htn']
              bu=request.form['bu']
              x=[pcv,hemo,sg,sc,rc,al,dm,bgr,sod,htn,bu]
           else: 
              if ff=='Liver':
                 sod=request.form['sod']
                 htn=request.form['htn']
                 x=[pcv,hemo,sg,sc,rc,al,dm,bgr,sod,htn]
              else:
                 x=[pcv,hemo,sg,sc,rc,al,dm,bgr]
           df,df_out=form_output(x,ff)
           return render_template("formout.html",tables=[df.to_html(),df_out.to_html()], titles = ['Inputs Given','O/p Predicted by different Models'],name=ff)
        error= " Invalid Inputs"
        if ff=='Kidney':
           return render_template("dashboard.html", error = error)
        else:
           if ff=='Liver':
              return render_template("dashboard1.html", error = error)
    except Exception as e:
        flash(e)
        print(e)
        if ff=='Kidney':
           return render_template("dashboard.html", error = error)
        else:
           if ff=='Liver':
              return render_template("dashboard1.html", error = error)

@app.errorhandler(404)
def page_not_found(e):
    return render_template("404.html")

@app.route('/getCSV/',methods=['GET','POST'])
def getCSV():
    try:
        ff=request.form['disease'] 
        if ff=='Kidney':
           outfile='Output.zip'
        else:
           if ff=='Liver':
              outfile='Output1.zip'
           else:
              outfile='Output2.zip'
        file_path=os.path.join(app.root_path,outfile)
        return send_file(file_path, attachment_filename='predictedoutpt.zip',as_attachment=True)
    except Exception as e:
        return str(e)

@app.route('/predictoutput/', methods=['POST'])
def PredictOutput():
    data = request.get_json(force=True)
    df = pd.io.json.json_normalize(data)
    df=clean_run_data(df)
    df,df_summ=model_run(df,'Kidney')
    js=json.loads(df.to_json(orient='records'))
    resp = Response(json.dumps(js), status=200, mimetype='application/json')
    return resp

@app.route('/predictoutput1/', methods=['POST'])
def PredictOutput1():
    data = request.get_json(force=True)
    df = pd.io.json.json_normalize(data)
    df=clean_data_liver_run(df)
    df,df_summ=model_run(df,'Liver')
    js=json.loads(df.to_json(orient='records'))
    resp = Response(json.dumps(js), status=200, mimetype='application/json')
    return resp

@app.route('/predictoutput2/', methods=['POST'])
def PredictOutput2():
    data = request.get_json(force=True)
    df = pd.io.json.json_normalize(data)
    #df=clean_run_data(df)
    df,df_summ=model_run(df,'Diab')
    js=json.loads(df.to_json(orient='records'))
    resp = Response(json.dumps(js), status=200, mimetype='application/json')
    return resp

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=80)
