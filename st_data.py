import streamlit as st
import pandas as pd
import pickle
def prep_st(data):
  y = data[["review_score"]]
  X = data.drop(columns=["review_score", "customer_id"])
  

  colunas = ['product_name_lenght', 'product_description_lenght',"order_products_value", 'order_freight_value', 'order_items_qty', 'order_sellers_qty', 
                'product_photos_qty', 'real_delivered_time', 
                'estimated_delivered_time', 'dif_estimated_real', 'is_late']

  X[colunas] = scaler.transform(X[colunas])

  return X, y

def predict_nota():
    data_sample = data.sample(5).reset_index(drop=True)
    X, y = prep_st(data.sample(5))
    st.write("Informacoes de compra dos novos consumidores")
    st.write(data_sample.drop(columns=["review_score"]))
    loaded_model = pickle.load(open(filename, 'rb'))
    st.write(X.shape)
    result = pd.DataFrame(loaded_model.predict(X))
    info = pd.concat([data_sample["customer_id"],result], axis=1)
    return info

@st.cache
def load_data():
    data = pd.read_csv("st_data.csv")
    data = data.drop(columns=["Unnamed: 0"])
    return data

scalerfile = 'scaler.sav'
scaler = pickle.load(open(scalerfile, 'rb'))

data = load_data()
filename = "finalized_model.sav"



#Seleção para quais análises deseja visualizar
select_anal = st.sidebar.radio('Menu:',
                                        ('Home','Dashboard', 'Equipe'))

# Todas análises para vereador em 2016
if select_anal == 'Home':

    st.image("Captura de tela 2022-03-13 224850.png")
    st.title("N3M Dashboard")

if select_anal == 'Dashboard':
    st.title("N3M Dashboard")
    st.header("Importar clientes e sua possivel nota para o processo de compra")

    if st.button('Buscar novos consumidores'):
        result = predict_nota()
        st.write("Nota predita para os consumidores (0 : negativo, 1 : positivo) \n")
        l1, l2 = st.columns(2)
        if (len(result[0].unique()) == 1):
            st.write("Foi encontrado apenas clientes preditos com nota positiva \n")
            with l1:
                st.dataframe(result)

        else:
            st.write("Cliente com nota negativa encontrado \n")
            with l1:
                st.dataframe(result)
            with l2:
                st.write("Informacao de clientes com predicao de nota negativa")
                st.dataframe(result[result[0]==0])

        
        

if select_anal == 'Equipe':
    st.title("Equipe")
    c1,c2,c3,c4 = st.columns(4)
    with c1:
        st.text("Nicolas Antero \n Co-founder")
    with c2:
        st.text("Maurici Meneghetti \n Co-founder")
    with c3:
        st.text("Maurício Salvador \n Co-founder")
    with c4:
        st.text("Márcio Fazolin \n Co-founder")









    
