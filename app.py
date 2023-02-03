import streamlit as st  # pip install streamlit
import pandas as pd  # pip install pandas
import plotly.express as px  # pip install plotly-express
import base64  # Standard Python Module
from io import StringIO, BytesIO  # Standard Python Module
import plotly.graph_objs as go
import numpy as np

def generate_excel_download_link(df):
    # Credit Excel: https://discuss.streamlit.io/t/how-to-add-a-download-excel-csv-function-to-a-button/4474/5
    towrite = BytesIO()
    df.to_excel(towrite, encoding="utf-8", index=False, header=True)  # write to BytesIO buffer
    towrite.seek(0)  # reset pointer
    b64 = base64.b64encode(towrite.read()).decode()
    href = f'<a href="data:application/vnd.openxmlformats-officedocument.spreadsheetml.sheet;base64,{b64}" download="data_download.xlsx">Download Excel File</a>'
    return st.markdown(href, unsafe_allow_html=True)

def generate_html_download_link(fig):
    # Credit Plotly: https://discuss.streamlit.io/t/download-plotly-plot-as-html/4426/2
    towrite = StringIO()
    fig.write_html(towrite, include_plotlyjs="cdn")
    towrite = BytesIO(towrite.getvalue().encode())
    b64 = base64.b64encode(towrite.read()).decode()
    href = f'<a href="data:text/html;charset=utf-8;base64, {b64}" download="plot.html">Download Plot</a>'
    return st.markdown(href, unsafe_allow_html=True)

st.set_page_config(page_title='Visualisasi Data PT. SEITA SUKSES MAKMUR')
st.title('Visualisasi Data 📈 \nPT. :red[SEITA] SUKSES MAKMUR')


uploaded_file = st.file_uploader('Input data excel (XLSX)', type='xlsx')



if uploaded_file:
    st.markdown('---')

    # Pilih sheet dari file Excel yang diupload
    sheet_names = pd.read_excel(uploaded_file, sheet_name=None, engine='openpyxl').keys()
    selected_sheet = st.selectbox(
        'Pilih sheet yang akan dianalisis:',
        sheet_names,
    )
    df = pd.read_excel(uploaded_file, sheet_name=selected_sheet, engine='openpyxl')
    

    
    st.dataframe(df)
    if selected_sheet == "Harvest Result":
        groupby_column1 = st.selectbox(
            'Apa yang anda ingin analisis',
            ('Blok', 'Pond', 'Asal Benur', 'Merk Pakan','Teknisi'),
        )
        
        groupby_column2 = st.selectbox(
            'Apa yang anda ingin analisis lainnya',
            ('Blok', 'Pond', 'Asal Benur', 'Merk Pakan','Teknisi'),
        )
        
        output_columns = st.selectbox(
            'Output yang di inginkan',
            ('Total Biomass PP1', 'Total Biomass PP2', 'Biomass Panen Akhir', 'Total Nilai Jual'),
        )
        
        if groupby_column2 == groupby_column1:
            df_grouped = df.groupby(by=groupby_column1, as_index=False)[output_columns].sum()  
        else:
            df['new_groupby_column'] = df[groupby_column1].astype(str) + ' - ' + df[groupby_column2].astype(str)
            df_grouped = df.groupby(by='new_groupby_column', as_index=False)[output_columns].sum()
            df_grouped = df_grouped.rename(columns={'new_groupby_column': ''}) 
        
        total_output = df_grouped[output_columns].sum()
        if output_columns == "Total Nilai Jual":
            import locale
            locale.setlocale(locale.LC_ALL, 'id_ID.UTF-8')
            st.subheader(f'{output_columns} : {locale.currency(total_output, grouping=True)}')
        else:
            st.subheader(f'{output_columns} : {total_output:.2f}')

        if groupby_column2 == groupby_column1:
            x=groupby_column1
        else:
            x=''
        
        # -- PLOT DATAFRAME
        fig = px.bar(
            df_grouped,
            x=x,
            y=output_columns,
            template='plotly_white',
            color=output_columns,
            color_continuous_scale=['red', 'yellow', 'green'],
            facet_col=None,
            title=f'<b>Grafik {output_columns} berdasarkan {groupby_column1} dan {groupby_column2}'
            )
        st.plotly_chart(fig)
        
        # -- DOWNLOAD 
        st.subheader('Downloads:')
        generate_excel_download_link(df_grouped)
        generate_html_download_link(fig)
        
        
    elif selected_sheet == "DKA SSLA ":
        data_ke = df['Data ke-'].dropna().unique().tolist()
        data_ke = [int(x) for x in data_ke]
        data_ke_selection = st.slider('Data ke-:',
                            min_value= min(data_ke),
                            max_value= max(data_ke),
                            value=(min(data_ke),max(data_ke)))
               
        blok_1 = df['BLOK'].dropna().unique().tolist()
        pond_1 = df['Pond'].dropna().unique().tolist()
        all_groupby_columns = blok_1 + pond_1
        groupby_column1 = st.multiselect('Group by:',
                                            all_groupby_columns,
                                            default=blok_1
                                        )
        
        output_colums = st.multiselect(
                    'Apa yang anda ingin analisis',
                    ('PO4','NO2','  NH4','DO AM','Temp AM','Temp PM','DO PM','pH AM','pH PM'),
                )
        mask = (df['Data ke-'].between(*data_ke_selection)) & (df['BLOK'].isin(groupby_column1) | df['Pond'].isin(groupby_column1))
        number_of_result = df[mask].shape[0]
        df_grouped = df[mask].groupby(by=['Data ke-'])[output_colums].mean()
        df_grouped = df_grouped.reset_index()
        
        traces = []
        for column in output_colums:
            trace = go.Scatter(
                x=df_grouped['Data ke-'],
                y=df_grouped[column],
                name=column,
                mode='lines+markers'
            )
            traces.append(trace)
        layout = go.Layout(title='Line chart of '+ ','.join(output_colums), xaxis_title='Data ke-', yaxis_title='Value', hovermode='closest')
        fig = go.Figure(data=traces, layout=layout)
        st.plotly_chart(fig)
        
    
         
    elif selected_sheet == "Pakan Harian":
        doc = df['DOC'].dropna().unique().tolist()
        doc = [int(x) for x in doc]
        doc_selection = st.slider('DOC',
                                min_value= min(doc),
                                max_value= max(doc),
                                value=(min(doc),max(doc)))
               
        blok_1 = df['Blok'].dropna().unique().tolist()
        groupby_column1 = st.multiselect('Blok:',
                                                blok_1,
                                                default = blok_1)
        
        output_colums = st.multiselect(
                    'Apa yang anda ingin analisis',
                    ('ABW', 'ADG Aktual'),
                )
        mask = (df['DOC'].between(*doc_selection)) & (df['Blok'].isin(groupby_column1))
        number_of_result = df[mask].shape[0]
        df_grouped = df[mask].groupby(by=['DOC'])[output_colums].mean()
        df_grouped = df_grouped.reset_index()
        
        traces = []
        for column in output_colums:
            trace = go.Scatter(
                x=df_grouped['DOC'],
                y=df_grouped[column],
                name=column,
                mode='lines+markers'
            )
            traces.append(trace)
        layout = go.Layout(title='Line chart of '+ ','.join(output_colums), xaxis_title='DOC', yaxis_title='Value', hovermode='closest')
        fig = go.Figure(data=traces, layout=layout)
        st.plotly_chart(fig)
        
        
        # -- DOWNLOAD 
        st.subheader('Downloads:')
        generate_excel_download_link(df_grouped)
        generate_html_download_link(fig)
