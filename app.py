import os
import joblib
import streamlit as st
from pandas import DataFrame, concat


class Data:
    def __init__(
        self,
        age,
        usage,
        plan,
        time_contract,
        customer_satisfaction,
        monthly_value,
        model,
        ohe,
        scaler,
    ):
        self.age = age
        self.usage = usage
        self.plan = plan
        self.time_contract = time_contract
        self.customer_satisfaction = customer_satisfaction
        self.monthly_value = monthly_value
        self.model = model
        self.ohe = ohe
        self.scaler = scaler
        self.df_processed = None

    def preprocess(self):
        df = DataFrame(
            {
                "age": self.age,
                "usage": self.usage,
                "plan": self.plan,
                "time_contract": self.time_contract,
                "customer_satisfaction": self.customer_satisfaction,
                "monthly_value": self.monthly_value,
            },
            index=[0],
        )

        categorical = ["plan", "time_contract"]
        df_ohe = DataFrame(
            self.ohe.transform(df[categorical].values),
            columns=self.ohe.get_feature_names_out(categorical),
        )

        df_processed = concat([df.drop(categorical, axis=1), df_ohe], axis=1)
        df_processed["age"] = self.scaler.transform(df_processed[["age"]].values)

        self.df_processed = df_processed

    def predict(self):
        self.preprocess()

        return model.predict(self.df_processed)[0]


@st.cache_resource
def load_artifacts():
    try:
        BASEDIR = os.path.abspath(os.path.dirname(__file__))
    except NameError:
        BASEDIR = os.getcwd()

    model = joblib.load(os.path.join(BASEDIR, "artifacts", "model.pkl"))
    ohe = joblib.load(os.path.join(BASEDIR, "artifacts", "ohe.pkl"))
    scaler = joblib.load(os.path.join(BASEDIR, "artifacts", "scaler.pkl"))

    return model, ohe, scaler


model, ohe, scaler = load_artifacts()

st.title("Prever :blue[Churn] do    Cliente")

container = st.container(border=True)
age = container.number_input("Idade do cliente", min_value=18, max_value=105, value=25)
usage = container.number_input("Consumo no mês", min_value=0, max_value=672, value=60)

plan = container.selectbox(
    "Selecione o plano atual do cliente",
    options=[
        "Basico",
        "Standard",
        "Premium",
    ],
)

time_contract = container.selectbox(
    "Tempo de contrato",
    options=[
        "Curto",
        "Medio",
        "Longo",
    ],
)

customer_satisfaction = container.number_input(
    "Satisfação do cliente",
    min_value=0,
    max_value=5,
    value=3,
)

monthly_value = container.number_input("Gasto mensal", value=120)

click = container.button(
    "Prever churn",
)

if click:
    data = Data(
        age,
        usage,
        plan,
        time_contract,
        customer_satisfaction,
        monthly_value,
        model,
        ohe,
        scaler,
    )

    print(
        data.age,
        data.usage,
        data.plan,
        data.time_contract,
        data.customer_satisfaction,
        data.monthly_value,
    )
    predict = data.predict()

    if predict == 0:
        st.success("O Cliente não irá cancelar", icon="✅")
    else:
        st.warning("Necessária ação, previsão de cancelamento do Cliente", icon="⚠️")
