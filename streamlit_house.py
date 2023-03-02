import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import folium
import geopandas


from streamlit_folium import folium_static
from folium.plugins import MarkerCluster
from datetime import datetime


def get_data(path):
    data = pd.read_csv(path)

    return data

def get_geofile(url):
    geofile = geopandas.read_file((url))

    return geofile

def set_attributes( df ):
    df['price_m2'] = df['price'] / df['sqft_lot']

# get geofile
url = 'https://opendata.arcgis.com/datasets/83fc2e72903343aabff6de8cb445b81c_2.geojson'
geofile = get_geofile(url)

# Definindo o tamanho da página
st.set_page_config(layout='wide')

# Lendo o arquivo
df = pd.read_csv('kc_house_data.csv')

# Adicionando uma nova coluna para converter ft³ para m³
df['price_m2'] = df['price'] / df['sqft_lot'] * 0.092903

# ==================
# Data Overview
# ==================
st.title('Data Overview')
st.dataframe(df)

def data_overview(df):
    # Selecionando as colunas para o filtro sidebar
    f_attributes = st.sidebar.multiselect('Escolha qual coluna deseja filtrar', df.columns)

    # Selecionando o zipcode para o filtro sidebar
    f_zipcode = st.sidebar.multiselect('Escolha o código postal', df['zipcode'].unique())

    # Esse print é para  mostrar o JSON
    # st.write(f_attributes)
    # st.write(f_zipcode)

    # zipcode + attributes = Selecionar linhas e colunas
    # attributes = Selecionar colunas
    # zipcode = Selecionar linhas
    # nenhum dos dois = exibe o dataframe original

    # Se os dois forem diferentes de vazio
    # Ele seleciona todas as linhas do zipcode e todas as colunas
    if (f_zipcode != []) & (f_attributes != []):
        df = df.loc[df['zipcode'].isin(f_zipcode), f_attributes]

    # Seleciona todos os atributos
    elif (f_zipcode == []) & (f_attributes != []):
        df = df.loc[:, f_attributes]

    # Seleciona todas as linhas do zipcode
    elif (f_zipcode != []) & (f_attributes == []):
        df = df.loc[df['zipcode'].isin(f_zipcode), :]

    else:
        df = df.copy()

    # Average metrics
    # Número total de imóveis
    df1 = df[['id', 'zipcode']].groupby('zipcode').count().reset_index()
    # Média de preço
    df2 = df[['price', 'zipcode']].groupby('zipcode').mean().reset_index()
    # Média do número das salas de estar
    df3 = df[['sqft_living', 'zipcode']].groupby('zipcode').mean().reset_index()
    # Preço por m²
    df4 = df[['price_m2', 'zipcode']].groupby('zipcode').mean().reset_index()

    # Merge
    m1 = pd.merge(df1, df2, on='zipcode', how='inner')
    df4 = pd.merge(m1, df3, on='zipcode', how='inner')
    # df = pd.merge(m2, df4,  on='zipcode', how='inner')

    # Renomeando as colunas
    df4.columns = ['zipcode', 'Total de Casas', 'Preço Total', 'Sqft Living']
    # Imprimindo

    # Estatística descritiva

    num_attributes = df.select_dtypes(['int64', 'float64'])

    media = pd.DataFrame(num_attributes.apply(np.mean))
    mediana = pd.DataFrame(num_attributes.apply(np.median))
    std = pd.DataFrame(num_attributes.apply(np.std))

    max_ = pd.DataFrame(num_attributes.apply(np.max))
    min_ = pd.DataFrame(num_attributes.apply(np.min))

    df5 = pd.concat([max_, min_, media, mediana, std], axis=1).reset_index()
    df5.columns = ['Attributes', 'Max', 'Min', 'Mean', 'Median', 'std']

    # Definição do número de colunas que as tabelas ocuparão
    c1, c2 = st.columns((1, 1))

    # Imprimindo
    c1.header('Analise Descritiva')
    c1.dataframe(df4, width=600)

    c2.header('Estatística Descritiva')
    c2.dataframe(df5, height=600)

    return df

# ==================
# Densidade  de Portifolio
# ==================
def region_overview( df, geofile):
    st.title('Regiões')
    c1, c2 = st.columns((1, 1))
    c1.header('Densidade de Portifólio')

    df = df.sample(1000)

    density_map = folium.Map(location=[df['lat'].mean(), df['long'].mean()], zoom_start=15)

    marker_cluster = MarkerCluster().add_to(density_map)

    # for name, row in df.iterrows():
    #     folium.Marker([row['lat'], row['long']], popup='Valor de venda R${0}, data da venda: {1}, Tamanho da sala de '
    #                                                    'estar: {2}, Número de quartos: {3}, Número de banheiros {4}, '
    #                                                    'Ano de construção {5}'.format((row['price'], row['date'],
    #                                                                                    row['sqft_living'],
    #                                                                                    row['bedrooms'], row['bathrooms'],
    #                                                                                    row['yr_built'])).add_to(
    #         marker_cluster) )

    for name, row in df.iterrows():
        folium.Marker([row['lat'], row['long']],
                      popup='Valor de venda R${0}, data da venda: {1}, Tamanho da sala de estar: {2}, Número de quartos: '
                            '{3}, Número de banheiros: {4}, Ano de construção: {5}'.format(
                          row['price'], row['date'], row['sqft_living'], row['bedrooms'], row['bathrooms'], row['yr_built'])
                      ).add_to(marker_cluster)

    with c1:
        folium_static(density_map)

    # Preços por região

    c2.header('Densidade de Preços')

    data = df[['price', 'zipcode']].groupby('zipcode').mean().reset_index()
    data.columns = ['ZIP', 'PRICE']

    df = df.sample(1000)

    geofile = geofile[geofile['ZIP'].isin(data['ZIP'].tolist())]

    region_price_map = folium.Map(location=[df['lat'].mean(),
                                            df['long'].mean()],
                                  zoom_start=15)

    region_price_map.choropleth(data=data,
                                geo_data=geofile,
                                columns=['ZIP', 'PRICE'],
                                key_on='feature.properties.ZIP',
                                fill_color='YlOrRd',
                                fill_opacity=0.7,
                                line_opacity=0.2,
                                legend_name='AVG PRICE')

    with c2:
        folium_static(region_price_map)

    return None

# ===========================================================
# Distribuição dos imóveis por categorias comerciais
# ===========================================================
def set_commercial(df):
    st.sidebar.title('Opções comerciais')
    st.title('Atributos Comerciais')

    # ============== Média de preço por ano =====================

    # Obtendo somente o dia da data
    df['date'] = pd.to_datetime(df['date']).dt.strftime('%Y-%m-%d')

    # Filtros
    min_year_built = int(df['yr_built'].min())
    max_year_built = int(df['yr_built'].max())

    st.sidebar.subheader('Média de preço por ano de construção')

    #(valor minímo, maxímo e default)
    f_year_built = st.sidebar.slider('Ano de construção', min_year_built, max_year_built, min_year_built)


    # Calculando a média de preço por ano

    df_yr_price = df.loc[df['yr_built'] < f_year_built]
    df_yr_price = df_yr_price[['price', 'yr_built']].groupby('yr_built').mean().reset_index()

    # Plotando o gráfico de média de preço por ano
    fig = px.line(df_yr_price, x="yr_built", y="price" )
    st.header('Variação média de preço por ano de construção')
    st.plotly_chart(fig, use_container_width=True)


    # ============ Média de preço por dia =======================

    st.sidebar.subheader('Selecione a data')

    # Filtros
    min_date = datetime.strptime(df['date'].min(), '%Y-%m-%d')
    max_date = datetime.strptime(df['date'].max(), '%Y-%m-%d')

    f_date = st.sidebar.slider('Data', min_date, max_date, min_date)

    df['date'] = pd.to_datetime((df['date'])) # Convertendo string para datetime
    df['date'] = pd.to_datetime(df['date'].dt.strftime('%Y-%m-%d'))

    df_day_price = df.loc[df['date'] < f_date]
    df_day_price = df[['price', 'date']].groupby('date').mean().reset_index()

    # Plotando o gráfico de média de preço por dia
    fig = px.line(df_day_price, x="date", y="price" )
    st.header('Variação média de preço por dia ')
    st.plotly_chart(fig, use_container_width=True)



    # ======================Histograma============================
    st.header('Distribuição de Preços')
    st.sidebar.subheader('Selecione o preço')

    # Filtro
    min_price = int(df['price'].min())
    max_price = int(df['price'].max())
    avg_price = int(df['price'].mean())

    # Filtragem dos dados
    f_price = st.sidebar.slider('Preço', min_price, max_price, min_price)
    df_price = df.loc[df['price'] < f_price]

    # Plotagem de dados
    fig = px.histogram(df_price, x='price', nbins=20)
    st.plotly_chart(fig, use_container_width=True)

    return None


# ===========================================================
# Distribuição dos imóveis por categorias físicas
# ===========================================================

def set_physical(df):
    st.sidebar.title('Opções de atributos')
    st.title("Atributos das casas")

    # Filtros de número de quartos e banheiros
    f_bedrooms = st.sidebar.selectbox("Número máximo de quartos", sorted(set(df['bedrooms'].unique())))
    f_bathrooms = st.sidebar.selectbox("Número máximo de banheiros", sorted(set(df['bathrooms'].unique())))

    # Gráfico de número de quartos
    with st.expander("Casas por número de quartos"):
        filtered_bedrooms = df[df['bedrooms'] < f_bedrooms]
        if len(filtered_bedrooms) > 0:
            fig_bedrooms = px.histogram(filtered_bedrooms, x='bedrooms', nbins=20)
            st.plotly_chart(fig_bedrooms, use_container_width=True)
        else:
            st.warning("Não há casas que atendam aos critérios de filtro selecionados.")

    # Gráfico de número de banheiros
    with st.expander("Casas por número de banheiros"):
        filtered_bathrooms = df[df['bathrooms'] < f_bathrooms]
        if len(filtered_bathrooms) > 0:
            fig_bathrooms = px.histogram(filtered_bathrooms, x='bathrooms', nbins=20)
            st.plotly_chart(fig_bathrooms, use_container_width=True)
        else:
            st.warning("Não há casas que atendam aos critérios de filtro selecionados.")

    # Filtros de número de andares e vista para água
    f_floors = st.sidebar.selectbox("Número de andares", sorted(set(df['floors'].unique())))
    f_waterfront = st.sidebar.checkbox('Somente com vista para água')

    # Gráfico de número de andares
    with st.expander("Casas por número de andares"):
        filtered_floors = df[df['floors'] < f_floors]
        if len(filtered_floors) > 0:
            fig_floors = px.histogram(filtered_floors, x='floors', nbins=20)
            st.plotly_chart(fig_floors, use_container_width=True)
        else:
            st.warning("Não há casas que atendam aos critérios de filtro selecionados.")

    # Gráfico de vista para água
    with st.expander("Casas com vista para água"):
        if f_waterfront:
            filtered_waterfront = df[df['waterfront'] == 1]
        else:
            filtered_waterfront = df.copy()
        if len(filtered_waterfront) > 0:
            fig_waterfront = px.histogram(filtered_waterfront, x='waterfront', nbins=20)
            st.plotly_chart(fig_waterfront, use_container_width=True)
        else:
            st.warning("Não há casas que atendam aos critérios de filtro selecionados.")

    return None

if __name__ == "__main__":
    # ETL
    path = 'kc_house_data.csv'
    url = 'https://opendata.arcgis.com/datasets/83fc2e72903343aabff6de8cb445b81c_2.geojson'

    # load data
    df = get_data(path)
    geofile = get_geofile(url)

    # transform data
    data = set_attributes(df)

    data_overview(df)

    region_overview(df, geofile)

    set_commercial(df)

    set_physical(df)

# # Filtros
# f_bedrooms = st.sidebar.selectbox("Número máximo de quartos", sorted(set(df['bedrooms'].unique())))
# f_bathrooms = st.sidebar.selectbox("Número máximo de banheiros", sorted(set(df['bathrooms'].unique())))
#
# c1, c2 = st.columns(2)
#
# # casas por número de quartos
# c1.header('Número de quartos')
# df_bedrooms = df[df['bedrooms'] < f_bedrooms]
# # ax = sns.barplot(data = df_floors, x='bedrooms', y='id')
# fig = px.histogram(df_bedrooms, x='bedrooms', nbins=20)
# c1.plotly_chart(fig, use_container_width=True)
#
# # casas por número de banheiros
# c2.header('Número de banheiros')
# df_bathrooms = df[df['bathrooms'] < f_bathrooms]
# fig = px.histogram(df_bathrooms, x='bathrooms', nbins=20)
# c2.plotly_chart(fig, use_container_width=True)
#
# # Filtros
# f_floors = st.sidebar.selectbox("Número de andares", sorted(set(df['floors'].unique())))
# f_waterfront = st.sidebar.checkbox('Somente com vista para água')
#
# c3, c4 = st.columns(2)
#
# # casas por número de amdares
# c3.header('Número de andares')
# df = df[df['floors'] < f_floors]
# fig = px.histogram(df, x='floors', nbins=20)
# c3.plotly_chart(fig, use_container_width=True)
#
# # casas por número de vista para água
# if f_waterfront:
#     df = df[df['waterfront'] == 1]
# else:
#     df = df.copy()
#
# fig = px.histogram(df, x='waterfront', nbins=20)
# c4.plotly_chart(fig, use_container_width=True)
