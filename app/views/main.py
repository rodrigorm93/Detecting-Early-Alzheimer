from flask import render_template, jsonify, Flask, redirect, url_for, request

from flask import render_template, jsonify
from app import app
import random
import pandas as pd
import numpy as np
import os
from werkzeug import secure_filename
from joblib import dump, load
import pickle
from sklearn.kernel_ridge import KernelRidge
import plotly.graph_objs as go
import plotly.tools as tls
import plotly.offline as py



def grafico (gb_feature,df):
    
    cols = df.columns.tolist()
    # Create a dataframe with features
    feature_dataframe = pd.DataFrame( {'features': cols,
        'Gradient Boost feature importances': gb_feature,
        })
    # Scatter plot 
    trace = go.Scatter(
        y = feature_dataframe['Gradient Boost feature importances'].values,
        x = feature_dataframe['features'].values,
        mode='markers',
        marker=dict(
            sizemode = 'diameter',
            sizeref = 1,
            size = 25,
    #       size= feature_dataframe['AdaBoost feature importances'].values,
            #color = np.random.randn(500), #set color equal to a variable
            color = feature_dataframe['Gradient Boost feature importances'].values,
            colorscale='Portland',
            showscale=True
        ),
        text = feature_dataframe['features'].values
    )
    data = [trace]

    layout= go.Layout(
        autosize= True,
        title= 'Gradient Boosting Feature Importance',
        hovermode= 'closest',
    #     xaxis= dict(
    #         title= 'Pop',
    #         ticklen= 5,
    #         zeroline= False,
    #         gridwidth= 2,
    #     ),
        yaxis=dict(
            title= 'Feature Importance',
            ticklen= 5,
            gridwidth= 2
        ),
        showlegend= False
    )
    fig = go.Figure(data=data, layout=layout)
    return fig
    
    



@app.route('/')


@app.route('/upload')
def upload_file2():
   return render_template('index.html')
@app.route('/uploaded_cdr', methods = ['GET', 'POST'])
def upload_file():
   if request.method == 'POST':
      f = request.files['file_cdr']
      filename = secure_filename(f.filename)
     
      path = os.path.join(app.config['UPLOAD_FOLDER'], f.filename)
      f.save(path)

      #clf = load('../Detecting Early Alzheimer/app/static/model/model_KRR2.joblib') 
      clf = load('../Detecting Early Alzheimer/app/static/model/model_gb.joblib') 

      UPLOAD_FOLDER = '../Detecting Early Alzheimer/app/static/data'

      
      df = pd.read_csv (path,delimiter =';')
	
      predict =clf.predict(df)


      print(predict)
      #predict = np.round(predict,2)


      #if(predict < 0):
      	#predict = 0

      lista_predic = []
   
      #for i in range(0,len(predict)):
      	#p = np.round(predict[i],2)
      	#if(p<0):
      		#p=0
      		
      	#lista_predic.append(p)

      for i in range(0,len(predict)):
      	if(predict[i] == 1):
      		predi = 0.5
      		lista_predic.append(predi)
      	elif(predict[i] == 2):
      		predi = 1
      		lista_predic.append(predi)
      	else:
      		lista_predic.append(predict[i])
	
   
      tam = len(lista_predic)


      return render_template('uploaded.html', title='Success',prediccion =lista_predic,tam=tam)


@app.route('/uploaded_dem', methods = ['GET', 'POST'])
def upload_file_dem():
   if request.method == 'POST':
      f = request.files['file_dem']
      filename = secure_filename(f.filename)
     
      path = os.path.join(app.config['UPLOAD_FOLDER'], f.filename)
      f.save(path)

      clf = load('../Detecting Early Alzheimer/app/static/model/model_svc.joblib') 
      #clf = load('../Detecting Early Alzheimer/app/static/model/model_gb.joblib') 

      UPLOAD_FOLDER = '../Detecting Early Alzheimer/app/static/data'

      
      df = pd.read_csv (path,delimiter =';')

	
      predict =clf.predict(df)


      print(predict)
      score =clf.predict_proba(df)

     

      lista_predic = []
      lista_prob = []
      for i in range(0,len(predict)):
     	 if(predict[i]==0):
     		 lista_predic.append('No')
     		 s = np.round(score[i,0],2)
     		 s= s*100
     		 lista_prob.append(s)
     	 else:
     		 lista_predic.append('Si')
     		 s = np.round(score[i,1],2)
     		 s= s*100
     		 lista_prob.append(s)
 
      tam = len(lista_predic)




      return render_template('uploaded_dem.html', title='Success',prediccion =lista_predic, score = lista_prob,tam=tam)

@app.route('/data_cdr', methods = ['GET', 'POST'])
def data_cdr():
   if request.method == 'POST':

      sexo = request.form['sexo']
      Edad = request.form['Edad']
      EDUC = request.form['EDUC']
      SES = request.form['SES']
      MMSE = request.form['MMSE']
      eTIV = request.form['eTIV']
      nWBV = request.form['nWBV']
      eTIV = request.form['eTIV']
      ASF = request.form['ASF']

     
      clf = load('../Detecting Early Alzheimer/app/static/model/KRR.joblib') 

      UPLOAD_FOLDER = '../Detecting Early Alzheimer/app/static/data'


      return render_template('pred_cdr.html')


@app.route('/index')
def index():
    return render_template('index.html', title='Home')


@app.route('/map')
def map():
    return render_template('map.html', title='Map')


@app.route('/map/refresh', methods=['POST'])
def map_refresh():
    points = [(random.uniform(48.8434100, 48.8634100),
               random.uniform(2.3388000, 2.3588000))
              for _ in range(random.randint(2, 9))]
    return jsonify({'points': points})


@app.route('/contact')
def contact():
    return render_template('contact.html', title='Contact')