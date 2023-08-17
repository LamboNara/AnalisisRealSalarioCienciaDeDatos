#Importación módulos
import pandas as pd
import seaborn as sns
import matplotlib as plt
import matplotlib.pyplot as plt

#Importación csv
archivo = pd.read_csv("https://drive.google.com/uc?id=1Bl6P4csc7Swwh8t8GNF2KZMVNxn72m3a")
archivo.head()

"""#EXPLORACIÓN Y TRANSFORMACIÓN DE DATOS"""

df= archivo.drop(columns= "Unnamed: 0")
df

df.info()

"""No hay missing values o datos nulos dentro de nuestro dataset. No es necesario utilizar técnicas de rellenamiento de datos."""

df.duplicated().sum()

#Cambiar nombres de categorías para mayor comprensión
# Remote ratio
df.remote_ratio.replace([100,50,0],['Fully Remote','Partially Remote','No remote work'], inplace=True)

# Experience Level
df.experience_level.replace(['EN','MI','SE','EX'],['Entry','Junior','Senior','Expert'],inplace=True)

# Employment Type
df.employment_type.replace(['FT','PT','CT','FL'],['Full Time','Part Time','Contract','Freelance'], inplace=True)

#Company size

df.company_size.replace(['S','M','L'],['Small','Medium','Large'], inplace=True)

"""#EXPLORATORY DATA ANALYSIS

##Posición de trabajo más frecuente
"""

sns.countplot(y="job_title", data=df, orient="h",)
sns.set(rc={'figure.figsize':(25,25)})
plt.ylabel('Job titles')
plt.xlabel('Count')
plt.title('Count of different job titles', y=1.01);

df['job_title'].value_counts()[:15]

"""El trabajo de Data Scientist es el más frecuente.

##Employment type más popular en el área de la ciencia de datos
"""

sns.countplot(x='employment_type', data=df)
sns.set(rc={'figure.figsize':(10,10)})
plt.ylabel('Count')
plt.xlabel('Employment type')
plt.title('Count of different employment types in the area of data science', y=1.01);

"""La mayoría de las personas dentro del área de la ciencia de datos trabajan Full time.

##Tipo de trabajo más común en la ciencia de datos
"""

sns.countplot(x="remote_ratio", data=df, orient="h",)
sns.set(rc={'figure.figsize':(25,25)})
plt.ylabel('Count')
plt.xlabel('Type of work')
plt.title('Conteo de personas en cada tipo de categoría de trabajo', y=1.01);

"""A partir de este gráfico vemos que en el área de la ciencia de datos los trabajos más comunes son los Totalmente Remotos

#Cantidad de trabajos en el área de la ciencia de datos a través de los años
"""

sns.countplot(x='work_year', data=df)
sns.set(rc={'figure.figsize':(10,10)})
plt.ylabel('Count')
plt.xlabel('Work Year')
plt.title('Count of different data science professionals in the different years', y=1.01)

"""Está en aumento los trabajos relacionados a la ciencia de datos.

##Salario promedio en base a distintas categorías
"""

#Cálculo de salarios promedio
avg_sal_per_job =df.groupby('job_title').mean()['salary_in_usd']
avg_sal_per_experiencie=df.groupby('experience_level').mean()['salary_in_usd']
avg_sal_per_company_size=df.groupby('company_size').mean()['salary_in_usd']
avg_sal_per_remote_work=df.groupby('remote_ratio').mean()['salary_in_usd']
avg_sal_per_employment_type=df.groupby('employment_type').mean()['salary_in_usd']

"""###Salario promedio por posición de trabajo"""

plt.figure(figsize=(20,20))
sns.barplot(x=avg_sal_per_job.values,y=avg_sal_per_job.index)
plt.xlabel('Average Salary(USD)')
plt.ylabel('Job title')
plt.title('Average Salary of Data Science jobs based on job title', y=1.01);

"""###Salario promedio por nivel de experiencia"""

plt.figure(figsize=(10,10))
sns.barplot(x=avg_sal_per_experiencie.index,y=avg_sal_per_experiencie.values)
plt.xlabel('Experience Levels')
plt.ylabel('Average Salary(USD)')
plt.title('Average Salary of Data Science jobs based on experience levels', y=1.01);

"""###Salario promedio por tamaño de la compañía"""

plt.figure(figsize=(10,10))
sns.barplot(x=avg_sal_per_company_size.index,y=avg_sal_per_company_size.values)
plt.xlabel('Company size')
plt.ylabel('Average Salary(USD)')
plt.title('Average Salary of Data Science jobs based on company size', y=1.01);

'''
fig, axes = plt.subplots(1,3, figsize = (15,4))

#fig.subtitle("El salario en dolares con respecto al titulo del trabajo, el nivel de experiencia y el trabajo remoto", fontsize = 19)

sns.barplot(ax = axes[0,0], data = df, x = 'job_title', y = 'salary_in_usd')
sns.barplot(ax = axes[0,1], data = df, x = 'experience_level', y = 'salary_in_usd')
sns.barplot(ax = axes[0,2], data = df, x = 'remote_radio', y = 'salary_in_usd')
'''

"""###Salario promedio por trabajo remoto

"""

avg_sal_per_remote_work=df.groupby('remote_ratio').mean()['salary_in_usd']
sns.barplot(x=avg_sal_per_remote_work.index,y=avg_sal_per_remote_work.values)
plt.xlabel('Percentage of remote work')
plt.ylabel('Average Salary(USD)')
plt.title('Average Salary of Data Science jobs based on remote work', y=1.01);

"""###Salario promedio por tipo de empleamiento

"""

plt.figure(figsize=(10,10))
sns.barplot(x=avg_sal_per_employment_type.index,y=avg_sal_per_employment_type.values)
plt.xlabel('Employment type')
plt.ylabel('Average Salary(USD)')
plt.title('Average Salary of Data Science jobs based on employment type', y=1.01);

sns.catplot(x='company_size', y='salary_in_usd', data=archivo)

"""#ARREGLO PARA ANÁLISIS PREDICTIVO"""

jobtitle = df.groupby('job_title').size().sort_values(ascending = False)
jobtitle.head(10)

dataset = df.loc[df.job_title.isin(['Data Scientist', 'Data Engineer',"Data Analyst","Machine Learning Engineer","Research Scientist","Data Science Manager","Data Architect","Big Data Engineer","Machine Learning Scientist","Director of Data Science"])]
dataset

predictive = dataset.drop(columns=["salary_currency","employee_residence","company_location","salary"],inplace =True)

predictive = pd.get_dummies(dataset)
predictive

predictive.columns.values

"""#ANÁLISIS PREDICTIVO"""

yvar = ['salary_in_usd']
Xvars = ['work_year', 'salary_in_usd', 'remote_ratio',
       'experience_level_EN', 'experience_level_EX',
       'experience_level_MI', 'experience_level_SE', 'employment_type_FL',
       'employment_type_FT', 'employment_type_PT',
       'job_title_Big Data Engineer', 'job_title_Data Analyst',
       'job_title_Data Architect', 'job_title_Data Engineer',
       'job_title_Data Science Manager', 'job_title_Data Scientist',
       'job_title_Director of Data Science',
       'job_title_Machine Learning Engineer',
       'job_title_Machine Learning Scientist',
       'job_title_Research Scientist', 'company_size_L', 'company_size_M',
       'company_size_S']

#Separando las variables
y = predictive[yvar]
X = predictive[Xvars]

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = .2, random_state = 0)

"""##Entrenamiento

"""

# Creando el modelo
# from sklearn.linear_model import LinearRegression
# modelo = LinearRegression()

from sklearn import tree
modelo = tree.DecisionTreeRegressor()

# Entrenando el modelo
modelo.fit(X_train, y_train)

"""##Evaluación"""

y_eval = modelo.predict(X_test)

from sklearn.metrics import r2_score

r2 = r2_score(y_test, y_eval)
r2

"""##Predicción"""

x_pred = pd.DataFrame({'work_year': [2020],
                       'salary_in_usd': [4000],
                       'remote_ratio' : [0],
                       'experience_level_EN': [0],
                       'experience_level_EX': [1],
                       'experience_level_MI': [0],
                       'experience_level_SE': [0],
                       'employment_type_FL': [0],
                       'employment_type_FT': [0],
                       'employment_type_PT': [1],
                       'job_title_Big Data Engineer': [0],
                       'job_title_Data Analyst': [0],
                       'job_title_Data Architect':[0],
                       'job_title_Data Engineer': [1],
                       'job_title_Data Science Manager': [0],
                       'job_title_Data Scientist': [0],
                       'job_title_Director of Data Science': [0],
                       'job_title_Machine Learning Engineer': [0],
                       'job_title_Machine Learning Scientist': [0],
                       'job_title_Research Scientist': [0],
                       'company_size_L': [0],
                       'company_size_M': [1],
                       'company_size_S': [0],})

y_pred = modelo.predict(x_pred)
y_pred
