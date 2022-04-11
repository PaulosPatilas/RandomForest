import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn import linear_model
from sklearn.preprocessing import LabelEncoder
from sklearn.neighbors import KNeighborsRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report

missing_values = ["n/a", "Unknown", "--"]
healthcare_df = pd.read_csv('healthcare-dataset-stroke-data.csv', na_values = missing_values)
sum_of_na=healthcare_df.isnull().sum()


def Multicollinearity():
	le = LabelEncoder()
	healthcare_df['gender'] = le.fit_transform(healthcare_df['gender'])
	healthcare_df['ever_married'] = le.fit_transform(healthcare_df['ever_married'])
	healthcare_df['work_type'] = le.fit_transform(healthcare_df['work_type'])
	healthcare_df['Residence_type'] = le.fit_transform(healthcare_df['Residence_type'])
	healthcare_df['smoking_status'] = le.fit_transform(healthcare_df['smoking_status'])
	corr = healthcare_df.corr().round(2)
	plt.figure(figsize=(10,7))
	sns.heatmap(corr, annot = True, cmap = 'RdYlGn');

def basic_analysis():

	#drop all rows with null values on a column then start analysis 
	
	
	print('Female med for bmi')
	bmi_Female_med = healthcare_df[healthcare_df['gender'] == 'Female']['bmi'].mean()
	
	print('Male med for bmi')
	bmi_Male_med = healthcare_df[healthcare_df['gender'] == 'Male']['bmi'].mean()
	
	print('bmi med over 50')
	healthcare_df[healthcare_df['age'] > 50]['bmi'].mean()
	
	print('bmi med under 50')
	healthcare_df[healthcare_df['age'] < 50]['bmi'].mean()
	
	print('Female med for avg_glucose_level')
	avg_glucose_level_Female_med = healthcare_df[healthcare_df['gender'] == 'Female']['avg_glucose_level'].mean()
	
	print('Male med for avg_glucose_level')
	avg_glucose_level_Female_med = healthcare_df[healthcare_df['gender'] == 'Male']['avg_glucose_level'].mean()
	
	print('avg_glucose_level med over 50')
	healthcare_df[healthcare_df['age'] > 50]['avg_glucose_level'].mean()
	
	print('avg_glucose_level med under 50')
	healthcare_df[healthcare_df['age'] < 50]['avg_glucose_level'].mean()
	
def nesteddonutchart_plot():	
	#gender frequency
	#sum = healthcare_df['gender'].notnull().sum()
	freq_df =healthcare_df[healthcare_df['Residence_type'] == 'Urban']['gender'].value_counts()
	freq1_df =healthcare_df[healthcare_df['Residence_type'] == 'Rural']['gender'].value_counts() 
	y = np.append(freq_df,freq1_df)
	
	#Residence_type frequency
	#sum = healthcare_df['Residence_type'].notnull().sum()
	freq2_df = healthcare_df['Residence_type'].value_counts()
	#freq1_df = freq1_df / sum 
	z = np.array(freq2_df)
	
	#donut nested charts
	residence_labels = ["Urban","Rural"]
	gender_labels = ["Female","Male","Female","Male"] 
	colors = ['#ff6666', '#ffcc99']
	colors_gender = ['#c2c2f0','#ffb3e6', '#c2c2f0','#ffb3e6']
	
	#myexplode = [0.2,0.2,0.2]
	plt.pie(z, labels=residence_labels, colors=colors, startangle=90,frame=True)
	plt.pie(y, labels=gender_labels,colors=colors_gender,radius=0.75,startangle=90)
	centre_circle = plt.Circle((0,0),0.5,color='black', fc='white',linewidth=0)
	fig = plt.gcf()
	fig.gca().add_artist(centre_circle)
 
	plt.axis('equal')
	plt.tight_layout()
	plt.show()	
	

def impute_model_basic(healthcare_df):
	cols_nan = healthcare_df.columns[healthcare_df.isna().any()].tolist() #Columns with missing values
	cols_no_nan = healthcare_df.columns.difference(cols_nan).values #Columns with no missing values
	train_data = healthcare_df.dropna() 
	for col in cols_nan:
		test_data = healthcare_df[healthcare_df[col].isna()] #row with nan value at column    
		knr = KNeighborsRegressor(n_neighbors=5).fit(train_data[cols_no_nan], train_data[col])
		healthcare_df.loc[healthcare_df[col].isna(), col] = knr.predict(test_data[cols_no_nan])
	return healthcare_df

def Knn():
	#replace categorical values with numeric ones
	healthcare_df['gender']= healthcare_df.gender.map({'Female' : 2 ,'Male' : 1 , 'Other' : 0 })
	healthcare_df['work_type']= healthcare_df.work_type.map({'Self-employed' : 2 , 'Private' : 1 , 'Govt_job' : 0 })
	healthcare_df['ever_married']= healthcare_df.ever_married.map({'Yes' : 1 , 'No' : 0 })
	healthcare_df['smoking_status'] = healthcare_df.smoking_status.map({'smokes': 2, 'formerly smoked': 1, 'never smoked':0})
	healthcare_df['Residence_type'] = healthcare_df.Residence_type.map({'Rural':0,'Urban':1})
	#temp = healthcare_df[['gender', 'age', 'hypertension', 'heart_disease', 'ever_married', 'work_type', 'Residence_type', 'avg_glucose_level', 'bmi', 'smoking_status', 'stroke']]
	healthcare_df = impute_model_basic(healthcare_df)
	return healthcare_df

def linearRegr():
	regr = linear_model.LinearRegression()
	le = LabelEncoder()
	
	
	#filling bmi values by regressing avg_glucose_level
	
	traintest = healthcare_df.dropna(subset=["bmi" ,"smoking_status"])
	
	bm = np.array(traintest.bmi).reshape(3426,1) #x einai posa menoun meta to dropna kai to y to idio  
	age = np.array(traintest.age).reshape(3426,1)

	mod = regr.fit(age,bm)
	#Rsquare=regr.score(agl,bm)

	#avg_glucose_level values on bmi missing values
	test = healthcare_df.avg_glucose_level[healthcare_df.bmi.isnull()]

	healthcare_df.loc[healthcare_df['bmi'].isna(), 'bmi'] = regr.predict(test.values.reshape(201,1))
	healthcare_df['bmi'] = healthcare_df['bmi'].round(1)
	
	#filling smoking_status values by regressing age
	healthcare_df['smoking_status'] = healthcare_df.smoking_status.map({'smokes': 2,'formerly smoked': 1,'never smoked':0})
	train = healthcare_df.dropna(subset=['smoking_status'])
	
	
	sm = np.array(train.smoking_status).reshape(3566,1)
	age = np.array(train.age).reshape(3566,1)
	
	mod2 = regr.fit(age,sm)
	test_ =  healthcare_df.age[healthcare_df.smoking_status.isnull()]
	healthcare_df.loc[healthcare_df['smoking_status'].isna(), 'smoking_status'] = regr.predict(test_.values.reshape(1544,1))
	healthcare_df['smoking_status'] = healthcare_df['smoking_status'].round()
	
	return healthcare_df

#delete column with missing values
def deleteColumn():
	print('Progress.....')
	healthcare_df.drop(healthcare_df.columns[healthcare_df.isnull().sum() > 0],axis=1) 
	return healthcare_df

def medforNaN():
	#fills with average numeric columns like bmi,avg_glucose_level etc. 
	healthcare_df['smoking_status'] = healthcare_df.smoking_status.map({'smokes': 2,'formerly smoked': 1,'never smoked':0})
	healthcare_df = healthcare_df.fillna(healthcare_df.mean()).round()
	return healthcare_df
	

def finalPreprocessing(healthcare_df):
	scaler = StandardScaler()
	le = LabelEncoder()
	
	healthcare_df['gender'] = le.fit_transform(healthcare_df['gender'])
	healthcare_df['ever_married'] = le.fit_transform(healthcare_df['ever_married'])
	healthcare_df['work_type'] = le.fit_transform(healthcare_df['work_type'])
	healthcare_df['Residence_type'] = le.fit_transform(healthcare_df['Residence_type'])
	healthcare_df['smoking_status'] = le.fit_transform(healthcare_df['smoking_status'])
	
	columns = ['avg_glucose_level','bmi','age']
	stand_scale = scaler.fit_transform(healthcare_df[['avg_glucose_level','bmi','age']])
	stand_scale = pd.DataFrame(stand_scale,columns=columns)
	stand_scale.head()
	
	df = pd.concat([healthcare_df, stand_scale], axis=1)
	RandomForest(df)
	
def RandomForest(df):
	model = RandomForestClassifier()
	x=df.drop(['stroke'], axis=1)
	y=df['stroke']
	x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.25, random_state= 124)
	model.fit(x_train, y_train)	
	y_pred = model.predict(x_test)
	arg_test = {'y_true':y_test, 'y_pred':y_pred}
	print(confusion_matrix(**arg_test))
	print(classification_report(**arg_test))		
		
def main():
		print('for 1 select delete ,for 2 median ,for 3 linear_regr ,for 4 knn ')
		x = input()
		if(x==1):
			healthcare_df = deleteColumn()
			finalPreprocessing(healthcare_df)
		elif (x == 2):
			healthcare_df = medforNaN()
			finalPreprocessing(healthcare_df)
		elif (x == 3):
			healthcare_df = linearRegr()
			finalPreprocessing(healthcare_df)
		elif (x == 4):
			healthcare_df = Knn()
			finalPreprocessing(healthcare_df)
if __name__ == "__main__":
    main()