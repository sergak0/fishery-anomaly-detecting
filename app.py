import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import catboost
import pickle
import json
import plotly.express as px
# from transformers import AutoTokenizer, AutoModelForSequenceClassification
import plotly.graph_objects as go
import torch

# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# tokenizer = AutoTokenizer.from_pretrained("best_model")
# model = AutoModelForSequenceClassification.from_pretrained("best_model")
# model = model.to(device)


# @st.cache
# def load_data(df):
#
#     df_X = df[['fish', 'volume']]
#     df_y = df['unit']
#
#     return df_X, df_y


@st.cache
def convert_df(df):
    # IMPORTANT: Cache the conversion to prevent computation on every rerun
    return df.to_csv().encode('utf-8')


st.title('ЧП: Аномальные школьники')

st.markdown("""
После загрузки данных не закрывайте страницу. Обработка данных займет какое-то время.
""")

uploaded_file = st.file_uploader("Choose a file")
if uploaded_file is not None:
    top_n = st.slider('Top N?', 1, 10, 5)

    rating = pd.read_csv('rating.csv')
    rating_cnt = rating.groupby(['id_Plat', 'anomaly_type'])['date'].count().reset_index()
    rating_cnt = rating_cnt.sort_values('date', ascending=False)

    rating_cnt_bad_unit = rating_cnt[rating_cnt['anomaly_type'] == 'bad_unit']
    rating_cnt_bad_unit = rating_cnt_bad_unit[:top_n]

    rating_cnt_dupl = rating_cnt[(rating_cnt['id_Plat'].isin(rating_cnt_bad_unit['id_Plat'])) & (rating_cnt['anomaly_type'] == 'duplicate')]
    rating_cnt_oversales = rating_cnt[(rating_cnt['id_Plat'].isin(rating_cnt_bad_unit['id_Plat'])) & (rating_cnt['anomaly_type'] == 'over_sells')]

    rating_cnt_dupl['id_Plat'] = rating_cnt_dupl['id_Plat'].astype(str)
    rating_cnt_bad_unit['id_Plat'] = rating_cnt_bad_unit['id_Plat'].astype(str)
    rating_cnt_oversales['id_Plat'] = rating_cnt_oversales['id_Plat'].astype(str)


    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=rating_cnt_bad_unit['id_Plat'],
        y=rating_cnt_bad_unit['date'],
        name='Bad Unit',
        marker_color='indianred'
    ))
    fig.add_trace(go.Bar(
        x=rating_cnt_dupl['id_Plat'],
        y=rating_cnt_dupl['date'],
        name='Duplicate',
        marker_color='lightsalmon'
    ))
    fig.add_trace(go.Bar(
        x=rating_cnt_oversales['id_Plat'],
        y=rating_cnt_oversales['date'],
        name='Over sells',
        marker_color='coral'
    ))


    # Here we modify the tickangle of the xaxis, resulting in rotated labels.
    fig.update_layout(barmode='group', xaxis_tickangle=-45, title='Топ юр. лиц по количсетву аномалий (по убыванию слева направо)')

    st.plotly_chart(fig, use_container_width=True)

    left_shop = pd.read_csv('left_in_shop.csv')

    with open('selling_parts.pickle', 'rb') as f:
        selling_parts = pickle.load(f)

    option = st.text_input('Номер юр.лица')
    if option:
        if int(option) not in rating['id_Plat'].fillna(0).astype(int).values:
            st.error('Please write correct id plat (rounded value)')
        else:
            sel_fish = st.selectbox('Название рыбы для визуализации', left_shop.fish.unique())

            subset = left_shop[left_shop['fish'] == sel_fish]

            fig = px.line(subset, x='date', y=['volume_x', 'volume_y'])
            st.plotly_chart(fig)


    # Map Heatmap

    st.subheader('Тепловая карта происхождения аномалий')

    lat_lon = pd.read_csv('final_address_lat_lon.csv')
    fig = go.Figure(go.Densitymapbox(lat=lat_lon.latitude, lon=lat_lon.longitude, radius=8))
    fig.update_layout(mapbox_style="open-street-map", mapbox_center_lon=100)
    fig.update_layout(margin={"r": 0, "t": 0, "l": 0, "b": 0})
    st.plotly_chart(fig, use_container_width=True)






















# uploaded_file = st.file_uploader("Choose a file")
# if uploaded_file is not None:
#     df = pd.read_csv(uploaded_file)
#     ext1_df = pd.read_csv('data/db2/Ext.csv')
#
#     X_val, y_val = load_data(df)
#
#
#     # CATBOOST
#
#     with open('fish_cb.pkl', 'rb') as f:
#         model = pickle.load(f)
#
#     preds_cb = pd.read_csv('cb_preds.csv')
#     # preds_cb = pd.DataFrame()
#     # y_pred = model.predict(X_val)
#     # y_pred_proba = model.predict_proba(X_val)
#     # for i, (y_p, y_v, y_p_p) in enumerate(zip(y_pred.reshape(-1), y_val, y_pred_proba)):
#     #     if y_p != y_v and X_val.volume.iloc[i] != 0 and max(y_p_p) > 0.9999:
#     #         subset = pd.DataFrame(X_val.loc[i][['fish', 'volume']]).T
#     #         subset['id_vsd'] = df.id_vsd.loc[i]
#     #         preds_cb = pd.concat([preds_cb, subset])
#     # preds_cb = preds_cb.reset_index(drop=True)
#     st.subheader(f'CatBoost anomalies prediction: {len(preds_cb)}')
#
#     preds_cb_cnt = preds_cb.groupby(['fish'])['volume'].count()
#     preds_cb_cnt = preds_cb_cnt.reset_index().sort_values(by='volume', ascending=False)
#     preds_cb_cnt = preds_cb_cnt[preds_cb_cnt['volume'] > 1][:5]
#
#     fig = px.bar(preds_cb_cnt, x='fish', y='volume', title="Топ 5 видов рыбы по количеству найденных аномалий (CatBoost)",
#                  labels={'volume':'count'}, height=600)
#     fig.update_layout(xaxis_tickangle=-45, showlegend=False)
#     st.plotly_chart(fig, use_container_width=True)
#
#     # preds_cb = preds_cb.merge(ext1_df[['id_vsd', 'id_own', 'Region_Plat']], on='id_vsd', how='left').drop_duplicates('id_vsd')
#     csv = convert_df(preds_cb)
#
#     st.download_button('Скачать весь датасет', csv, file_name='cb_preds.csv')
#
#
#     # MIN MAX
#
#     with open('fish_min_max.json', 'rb') as f:
#         min_max_th = json.load(f)
#
#     preds_js = pd.read_csv('min_max_preds.csv')
#     # preds_js = pd.DataFrame()
#     # df_min_max = df[['fish', 'volume', 'id_vsd']].copy()
#     # for elem in df_min_max.fish.unique():
#     #     try:
#     #         subset = df_min_max[(df_min_max.fish == elem) & ((df_min_max.volume > min_max_th[elem]['max']) | (df_min_max.volume < min_max_th[elem]['min'])) & (df_min_max.volume != 0)]
#     #         preds_js = pd.concat([preds_js, subset])
#     #     except:
#     #         continue
#     # preds_js = preds_js.merge(ext1_df[['id_vsd', 'id_own', 'Region_Plat']], on='id_vsd', how='left').drop_duplicates(
#     #     'id_vsd')
#     #
#     # preds_js = preds_js.reset_index(drop=True)
#     preds_js_cnt = preds_js.groupby(['fish'])['volume'].count()
#     preds_js_cnt = preds_js_cnt.reset_index().sort_values(by='volume', ascending=False)
#     preds_js_cnt = preds_js_cnt[preds_js_cnt['volume'] > 1][:10]
#
#     st.subheader(f'Min Max anomalies prediction: {len(preds_js)}')
#     #, color='fish', color_discrete_sequence=px.colors.sequential.RdBu
#     fig = px.bar(preds_js_cnt, x='fish', y='volume', title="Топ 10 видов рыбы по количеству найденных аномалий (Min Max)",
#                  labels={'volume': 'count'}, height=600)
#     fig.update_layout(xaxis_tickangle=-45, showlegend=False)
#     st.plotly_chart(fig, use_container_width=True)
#     csv = convert_df(preds_js)
#
#     # st.dataframe(preds_js)
#     st.download_button('Скачать весь датасет', csv, file_name='min_max_preds.csv')
#
#
#     # RUBERT
#
#
#
#     st.subheader('Топ 10 недобросовестных собственников')
#
#     all_preds = pd.concat([preds_cb, preds_js])
#     all_preds_cnt = all_preds.groupby('id_own')['volume'].count()
#     all_preds_cnt = all_preds_cnt.reset_index().sort_values(by='volume', ascending=False)
#     all_preds_cnt = all_preds_cnt[all_preds_cnt['id_own'] != -1][:10]
#     all_preds_cnt['id_own'] = all_preds_cnt['id_own'].astype(str)
#
#     fig = px.bar(all_preds_cnt, x='id_own', y='volume',
#                  labels={'volume':'count', 'id_own': 'Собственник'}, height=600)
#     fig.update_layout(xaxis_tickangle=-45, showlegend=False)
#     st.plotly_chart(fig, use_container_width=True)
#
#
#     # Map Heatmap
#
#     st.subheader('Тепловая карта происхождения аномалий')
#
#     lat_lon = pd.read_csv('final_address_lat_lon.csv')
#     fig = go.Figure(go.Densitymapbox(lat=lat_lon.latitude, lon=lat_lon.longitude, radius=8))
#     fig.update_layout(mapbox_style="open-street-map", mapbox_center_lon=100)
#     fig.update_layout(margin={"r": 0, "t": 0, "l": 0, "b": 0})
#     st.plotly_chart(fig, use_container_width=True)



