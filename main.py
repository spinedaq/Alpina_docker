from explainerdashboard import ClassifierExplainer, ExplainerDashboard, ExplainerHub
import pandas as pd
import numpy as np
import pickle
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler,MinMaxScaler
import joblib
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer
from sklearn.base import BaseEstimator, TransformerMixin


# define a function that takes the outputs of the two models and combines them
def combine_probs(probas):
    proba1 = probas[0]
    proba2 = probas[1]

    # print("Proba_1:", proba1)
    # print("Proba_2:", proba2)
    # joint_proba = np.multiply(proba1, proba2)
    joint_proba = np.array([proba1[:,0]*proba2[:,0], proba1[:,0]*proba2[:,1], proba1[:,1]*proba2[:,0], proba1[:,1]*proba1[:,1]]).T
    # joint_proba = joint_proba.reshape(-1,4)
    # print("Joint_proba:", joint_proba)
    final_proba = joint_proba / np.sum(joint_proba, axis=1, keepdims=True)
    # print("Final_proba:",final_proba)
    return final_proba

class JointProbTransformer(BaseEstimator, TransformerMixin):
    
    def __init__(self):
        pass
    
    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        proba_joint = X
        return proba_joint
    
    def predict_proba(self, X):
        proba_joint = self.transform(X)
        return proba_joint

# Load the data

X_data_adn_enfermedad = pd.read_csv("Data/data_ADN_predict_enfermedad.csv", index_col = 0)
X_data_adn_metabolismo = pd.read_csv("Data/data_ADN_predict_metabolismo.csv", index_col = 0)
X_data_clinicos_metabolismo = pd.read_csv("Data/data_clinicos_predict_metabolismo.csv", index_col = 0)
X_data_microbiota_enfermedad = pd.read_csv("Data/data_microbiota_predict_enfermedad.csv", index_col = 0)
X_data_microbiota_metabolismo = pd.read_csv("Data/data_microbiota_predict_metabolismo.csv", index_col = 0)
X_data_clinicos_enfermedad = pd.read_csv("Data/data_clinicos_predict_enfermedad.csv", index_col = 0)

X_data_adn_enfermedad.columns = [X_data_adn_enfermedad.columns[i].replace(".","_").replace(",","_") for i in range(X_data_adn_enfermedad.shape[1])]
X_data_adn_metabolismo.columns = [X_data_adn_metabolismo.columns[i].replace(".","_").replace(",","_") for i in range(X_data_adn_metabolismo.shape[1])]
X_data_clinicos_enfermedad.columns = [X_data_clinicos_enfermedad.columns[i].replace(".","_").replace(",","_") for i in range(X_data_clinicos_enfermedad.shape[1])]
X_data_clinicos_metabolismo.columns = [X_data_clinicos_metabolismo.columns[i].replace(".","_").replace(",","_") for i in range(X_data_clinicos_metabolismo.shape[1])]
X_data_microbiota_enfermedad.columns = [X_data_microbiota_enfermedad.columns[i].replace(".","_").replace(",","_") for i in range(X_data_microbiota_enfermedad.shape[1])]
X_data_microbiota_metabolismo.columns = [X_data_microbiota_metabolismo.columns[i].replace(".","_").replace(",","_") for i in range(X_data_microbiota_metabolismo.shape[1])]

# Load the pipelines

model_adn_enfermedad = joblib.load("Models/model_ADN_predict_enfermedad.joblib")
model_adn_metabolismo = joblib.load("Models/model_ADN_predict_metabolismo.joblib")
model_clinicos_enfermedad = joblib.load("Models/model_clinicos_predict_enfermedad.joblib")
model_clinicos_metabolismo = joblib.load("Models/model_clinicos_predict_metabolismo.joblib")
model_microbiota_enfermedad = joblib.load("Models/model_microbiota_predict_enfermedad.joblib")
model_microbiota_metabolismo = joblib.load("Models/model_microbiota_predict_metabolismo.joblib")


# Define the model that predict the total risk:

# define a transformer that applies the predict_proba method of each model
# and returns a tuple of marginal probabilities
transformer = FunctionTransformer(lambda X: tuple((model_adn_metabolismo.predict_proba(X.iloc[:,:7]), model_clinicos_enfermedad.predict_proba(X.iloc[:,7:]))))

# define the pipeline
risk_model = Pipeline([
    ('transformer', transformer),
    ('combiner', FunctionTransformer(combine_probs, validate=False)),
    ('joint_proba', JointProbTransformer())
])

X_risk = X_data_adn_metabolismo.merge(X_data_clinicos_enfermedad, left_index = True, right_index = True)
# joints = risk_model.named_steps["transformer"].transform(X_risk.iloc[:2])

# print(joints)

# Define the explainers

explainer_adn_enfermedad = ClassifierExplainer(model_adn_enfermedad, X_data_adn_enfermedad, labels = ["Prediabetes", "Diabetes"])
explainer_adn_metabolismo = ClassifierExplainer(model_adn_metabolismo, X_data_adn_metabolismo, labels = ["Sin Dislipidemia", "Con Dislipidemia"])
explainer_clinicos_enfermedad = ClassifierExplainer(model_clinicos_enfermedad, X_data_clinicos_enfermedad, labels = ["Prediabetes", "Diabetes"])
explainer_clinicos_metabolismo = ClassifierExplainer(model_clinicos_metabolismo, X_data_clinicos_metabolismo, labels = ["Sin Dislipidemia", "Con Dislipidemia"])
explainer_microbiota_enfermedad = ClassifierExplainer(model_microbiota_enfermedad, X_data_microbiota_enfermedad, labels = ["Prediabetes", "Diabetes"])
explainer_microbiota_metabolismo = ClassifierExplainer(model_microbiota_metabolismo, X_data_microbiota_metabolismo, labels = ["Sin Dislipidemia", "Con Dislipidemia"])
explainer_risk = ClassifierExplainer(risk_model, X_risk, labels=["Prediabetes, sin dislipdemia", "Prediabetes, con dislipidemia", "Diabetes, sin dislipidemia", "Diabetes, con dislipidemia"])

# Define the dashboards

db_adn_enfermedad = ExplainerDashboard(explainer_adn_enfermedad, title="Modelo de diabetes según ADN", description="Modelo con exactitud estimada de 67%", shap_dependence= False)
db_adn_metabolismo = ExplainerDashboard(explainer_adn_metabolismo, title="Modelo de dislipidemia según ADN", description="Modelo con exactitud estimada de 89%", shap_dependence= False)
db_clinicos_enfermedad = ExplainerDashboard(explainer_clinicos_enfermedad, title="Modelo de diabetes según datos clínicos", description="Modelo con exactitud estimada de 76%", shap_dependence= False)
db_clinicos_metabolismo = ExplainerDashboard(explainer_clinicos_metabolismo, title="Modelo de dislipidemia según datos clínicos", description="Modelo con exactitud estimada de 85%", shap_dependence= False)
db_microbiota_enfermedad = ExplainerDashboard(explainer_microbiota_enfermedad, title="Modelo de diabetes según microbiota", description="Modelo con exactitud estimada de 67%", shap_dependence= False)
db_microbiota_metabolismo = ExplainerDashboard(explainer_microbiota_metabolismo, title="Modelo de dislipidemia según microbiota", description="Modelo con exactitud estimada de 78%", shap_dependence= False)
db_risk = ExplainerDashboard(explainer_risk, title="Modelo de Riesgo", description="Modelo que calcula la probabilidad conjunta del riesgo de dislipidemia y riesgo de diabetes de un paciente, utilizando datos de ADN y clínicos, respectivamente.", shap_dependence= False)

# Define the hub:

hub = ExplainerHub([db_risk,db_adn_enfermedad, db_adn_metabolismo, db_clinicos_enfermedad, db_clinicos_metabolismo, db_microbiota_enfermedad, db_microbiota_metabolismo], title = "Analítica predictiva", description= "Tableros de predicción de riesgo en pacientes según datos clínicos, microbióticos y genéticos.", logins=[['Alpina', 'alpina-bios2023']],
        db_users=dict(db_risk=['Alpina'], db_adn_enfermedad=['Alpina'], db_adn_metabolismo=['Alpina'], db_clinicos_enfermedad=['Alpina'], db_clinicos_metabolismo=['Alpina'], db_microbiota_enfermedad=['Alpina'], db_microbiota_metabolismo=['Alpina']), img = "https://upload.wikimedia.org/wikipedia/commons/thumb/d/d6/Alpina_S.A._logo.svg/2560px-Alpina_S.A._logo.svg.png")
hub.add_user("Alpina", "alpina-bios2023")
hub.run(port=3000)
