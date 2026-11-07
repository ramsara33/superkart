"""Streamlit client: the model remains exclusively in the backend."""
import os
import pandas as pd
import requests
import streamlit as st

BACKEND_URL = os.environ.get('BACKEND_URL', 'http://backend:7860').rstrip('/')
st.set_page_config(page_title='SuperKart sales planning', layout='wide')
st.title('SuperKart sales planning')
st.caption('Estimate product–store revenue from product and store characteristics.')
st.info('These are estimates for records similar to the training data, not a validated next-quarter time-series forecast. Store age uses the training reference year 2025.')

def call_api(endpoint, **kwargs):
    try:
        response = requests.post(BACKEND_URL + endpoint, timeout=30, **kwargs)
        if not response.ok:
            try:
                detail = response.json().get('error', response.text)
            except ValueError:
                detail = response.text[:300]
            st.error(f'Request rejected ({response.status_code}): {detail}')
            return None
        return response
    except requests.RequestException:
        st.error('The prediction service is unavailable. Check that the backend is running and try again.')
        return None

st.subheader('Single prediction')
with st.form('single'):
    left, right = st.columns(2)
    with left:
        weight = st.number_input('Product weight', min_value=0.01, value=12.66)
        sugar = st.selectbox('Sugar content', ['Low Sugar', 'Regular', 'No Sugar'])
        area = st.number_input('Allocated display area ratio', min_value=0.0, max_value=1.0, value=0.027, format='%.3f')
        mrp = st.number_input('Maximum retail price', min_value=0.01, value=117.08)
        prefix = st.selectbox('Product ID prefix', ['FD', 'DR', 'NC'])
    with right:
        size = st.selectbox('Store size', ['Medium', 'Small', 'High'])
        city = st.selectbox('City tier', ['Tier 2', 'Tier 1', 'Tier 3'])
        store = st.selectbox('Store type', ['Supermarket Type2', 'Supermarket Type1', 'Departmental Store', 'Food Mart', 'Supermarket Type3'])
        age = st.number_input('Store age at reference year 2025', min_value=0, value=16, step=1)
        category = st.selectbox('Product category', ['Non Perishables', 'Perishables'])
    submitted = st.form_submit_button('Predict sales', type='primary')
if submitted:
    payload = dict(Product_Weight=weight, Product_Sugar_Content=sugar,
        Product_Allocated_Area=area, Product_MRP=mrp, Store_Size=size,
        Store_Location_City_Type=city, Store_Type=store, Product_Id_char=prefix,
        Store_Age_Years=age, Product_Type_Category=category)
    response = call_api('/v1/predict', json=payload)
    if response is not None:
        result = response.json()
        st.metric('Estimated revenue (dataset currency units)', f"{result['Sales']:,.2f}")
        for notice in result.get('warnings', []):
            st.warning('; '.join(notice['warnings']) + '. Treat this prediction cautiously.')
        st.caption('Request reference: ' + response.headers.get('X-Request-ID', 'unavailable'))

st.subheader('Batch predictions')
st.write('Upload the provided batch CSV or a file with the same ten feature columns.')
uploaded = st.file_uploader('Batch CSV', type=['csv'])
if uploaded is not None and st.button('Predict batch'):
    raw = uploaded.getvalue()
    response = call_api('/v1/predictbatch', files={'file': ('batch.csv', raw, 'text/csv')})
    if response is not None:
        predictions = response.json()
        from io import BytesIO
        result = pd.read_csv(BytesIO(raw))
        result.insert(0, 'input_row', range(len(result)))
        result['Predicted_Sales'] = [predictions[str(i)] for i in range(len(result))]
        warning_rows = response.headers.get('X-Input-Warning-Rows', '')
        flagged = set(int(i) for i in warning_rows.split(',') if i)
        result['Review_Required'] = [i in flagged for i in range(len(result))]
        if flagged:
            st.warning('Some rows contain categories or numerical values outside the training experience. Review flagged rows before business use.')
        st.success(f'{len(result)} predictions returned, in input order.')
        display_order = ['input_row', 'Predicted_Sales', 'Review_Required'] + [c for c in result if c not in ['input_row', 'Predicted_Sales', 'Review_Required']]
        st.dataframe(result[display_order], hide_index=True)
        st.download_button('Download predictions', result.to_csv(index=False), 'superkart_predictions.csv', 'text/csv')
