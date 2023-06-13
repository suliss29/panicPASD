import pandas as pd
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
import streamlit as st
from streamlit_option_menu import option_menu

@st.cache_data()
def read_data():
    df = pd.read_csv('PASD.csv')
    return df
df = read_data()

def kategori_usia(usia):
    if 17 <= usia <= 25:
        return 'Remaja akhir'
    elif 26 <= usia <= 35:
        return 'Dewasa awal'
    elif 36 <= usia <= 45:
        return 'Dewasa akhir'
    elif 46 <= usia <= 55:
        return 'Lansia awal'
    elif 56 <= usia <= 65:
        return 'Lansia akhir'
    
# Menerapkan fungsi kategori_usia pada DataFrame
df['Kategori Usia'] = df['Age'].apply(kategori_usia)
Umur = df['Age']

def create_age_distribution_plot():
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))
    ax1.bar('Kategori Usia', age_counts)
    ax1.set_xlabel('Age Categories')
    ax1.set_ylabel('Count')
    ax1.set_title('Age Distribution')

    # Pie chart
    ax2.pie(age_counts, labels=age_categories, autopct='%1.1f%%', startangle=90)
    ax2.axis('equal')
    ax2.set_title('Age Distribution')

    # Menampilkan plot
    plt.tight_layout()
    st.pyplot(fig)

# Menggunakan Streamlit
st.title('Data Dummy - Age Distribution')

# Memanggil fungsi untuk membuat plot
create_age_distribution_plot()