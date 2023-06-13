import pandas as pd
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
import streamlit as st
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
from sklearn.ensemble import RandomForestClassifier

@st.cache_data()
def read_data():
    df = pd.read_csv('PASD.csv')
    return df
df = read_data()

def read_data1():
    df1 = pd.read_csv('panic2.csv')
    return df1
df1 = read_data1()


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

# Menghitung jumlah individu dalam setiap kategori usia
kategori_usia_count = df['Kategori Usia'].value_counts()

# Mengatur ukuran dan warna plot
plt.figure(figsize=(12, 6))
colors = ['#7FB3D5', '#1F77B4', '#5194E9', '#0080FF', '#005CBF']

# Membuat subplot 1 (bar plot)
plt.subplot(1, 2, 1)
kategori_usia_count.plot(kind='bar', color=colors)
plt.xlabel('Kategori Usia')
plt.ylabel('Jumlah Individu')
plt.title('Distribusi Kategori Usia')

# Membuat subplot 2 (pie chart)
plt.subplot(1, 2, 2)
kategori_usia_count.plot(kind='pie', colors=colors, autopct='%1.1f%%')
plt.ylabel('')
plt.title('Persentase Kategori Usia')

# Menampilkan plot
plt.tight_layout()
plt.show()

combined_df = df.groupby(['Kategori Usia', 'Panic Disorder Diagnosis']).size().unstack()

# Mengatur ukuran dan warna plot
plt.figure(figsize=(8, 6))
colors = ['#1F77B4', '#FF7F0E', '#2CA02C', '#D62728', '#9467BD']

# Membuat pie chart untuk setiap kategori usia
for age_group in combined_df.index:
    plt.pie(combined_df.loc[age_group], labels=combined_df.columns, colors=colors, autopct='%1.1f%%')
    plt.title(f'Perbandingan Panic Disorder Diagnosis untuk Usia {age_group}')
    plt.axis('equal')
    plt.show()
    
gender_count = df['Gender'].value_counts()

# Mengatur ukuran dan warna plot
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 6))
colors = ['#7FB3D5', '#FF7F50']

# Membuat pie chart
ax1.pie(gender_count, labels=gender_count.index, colors=colors, autopct='%1.1f%%')
ax1.set_title('Gender Comparison')

# Membuat bar plot
ax2.bar(gender_count.index, gender_count.values, color=colors)
ax2.set_xlabel('Gender')
ax2.set_ylabel('Count')
ax2.set_title('Gender Comparison')


# Menampilkan plot
plt.tight_layout()
plt.show()

history_count = df['Family History'].value_counts()

# Mengatur ukuran dan warna plot
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 6))
colors = ['#7FB3D5', '#FF7F50']

# Membuat pie chart
ax1.pie(history_count, labels=history_count.index, colors=colors, autopct='%1.1f%%')
ax1.set_title('Family History Comparison')

# Membuat bar plot
ax2.bar(history_count.index, history_count.values, color=colors)
ax2.set_xlabel('Family History')
ax2.set_ylabel('Count')
ax2.set_title('Family History Comparison')

# Menampilkan plot
plt.tight_layout()
plt.show()

personal_count = df['Personal History'].value_counts()

# Mengatur ukuran dan warna plot
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 6))
colors = ['#7FB3D5', '#FF7F50', '#90EE90']

# Membuat pie chart
ax1.pie(personal_count, labels=personal_count.index, colors=colors, autopct='%1.1f%%')
ax1.set_title('Personal History Comparison')

# Membuat bar plot
ax2.bar(personal_count.index, personal_count.values, color=colors)
ax2.set_xlabel('Personal History')
ax2.set_ylabel('Count')
ax2.set_title('Personal History Comparison')

# Menampilkan plot
plt.tight_layout()
plt.show()

stressors_count = df['Current Stressors'].value_counts()

# Mengatur ukuran dan warna plot
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 6))
colors = ['#7FB3D5', '#1F77B4', '#5194E9']

# Membuat pie chart
ax1.pie(stressors_count, labels=stressors_count.index, colors=colors, autopct='%1.1f%%')
ax1.set_title('Current Stressors Comparison')

# Membuat bar plot
ax2.bar(stressors_count.index, stressors_count.values, color=colors)
ax2.set_xlabel('Current Stressors')
ax2.set_ylabel('Count')
ax2.set_title('Current Stressors Comparison')

# Menampilkan plot
plt.tight_layout()
plt.show()

symptoms_count = df['Symptoms'].value_counts()

# Mengatur ukuran dan warna plot
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))
colors = ['#7FB3D5', '#1F77B4', '#5194E9', '#0080FF', '#005CBF']

# Membuat pie chart
ax1.pie(symptoms_count, labels=symptoms_count.index, colors=colors, autopct='%1.1f%%')
ax1.set_title('Symptoms Comparison')

# Membuat bar plot
ax2.bar(symptoms_count.index, symptoms_count.values, color=colors)
ax2.set_xlabel('Symptoms')
ax2.set_ylabel('Count')
ax2.set_title('Symptoms Comparison')

# Menampilkan plot
plt.tight_layout()
plt.show()

severity_count = df['Severity'].value_counts()

# Mengatur ukuran dan warna plot
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 6))
colors = ['#7FB3D5', '#1F77B4', '#5194E9']

# Membuat pie chart
ax1.pie(severity_count, labels=severity_count.index, colors=colors, autopct='%1.1f%%')
ax1.set_title('Severity Comparison')

# Membuat bar plot
ax2.bar(severity_count.index, severity_count.values, color=colors)
ax2.set_xlabel('Severity')
ax2.set_ylabel('Count')
ax2.set_title('Severity Comparison')

# Menampilkan plot
plt.tight_layout()
plt.show()

IOL_count = df['Impact on Life'].value_counts()

# Mengatur ukuran dan warna plot
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 6))
colors = ['#7FB3D5', '#1F77B4', '#5194E9']

# Membuat pie chart
ax1.pie(IOL_count, labels=IOL_count.index, colors=colors, autopct='%1.1f%%')
ax1.set_title('Impact on Life Comparison')

# Membuat bar plot
ax2.bar(IOL_count.index, IOL_count.values, color=colors)
ax2.set_xlabel('Impact on Life')
ax2.set_ylabel('Count')
ax2.set_title('Impact on Life Comparison')

# Menampilkan plot
plt.tight_layout()
plt.show()

Demographics_count = df['Demographics'].value_counts()

# Mengatur ukuran dan warna plot
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 6))
colors = ['#7FB3D5', '#FF7F50']

# Membuat pie chart
ax1.pie(Demographics_count, labels=Demographics_count.index, colors=colors, autopct='%1.1f%%')
ax1.set_title('Demographics Comparison')

# Membuat bar plot
ax2.bar(Demographics_count.index, Demographics_count.values, color=colors)
ax2.set_xlabel('Demographics')
ax2.set_ylabel('Count')
ax2.set_title('Demographics Comparison')

# Menampilkan plot
plt.tight_layout()
plt.show()

medical_count = df['Medical History'].value_counts()

# Mengatur ukuran dan warna plot
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 6))
colors = ['#7FB3D5', '#1F77B4', '#5194E9', '#0080FF']

# Membuat pie chart
ax1.pie(medical_count, labels=medical_count.index, colors=colors, autopct='%1.1f%%')
ax1.set_title('Medical History Comparison')

# Membuat bar plot
ax2.bar(medical_count.index, medical_count.values, color=colors)
ax2.set_xlabel('Medical History')
ax2.set_ylabel('Count')
ax2.set_title('Medical History Comparison')

# Menampilkan plot
plt.tight_layout()
plt.show()

Psychiatric_count = df['Psychiatric History'].value_counts()

# Mengatur ukuran dan warna plot
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 6))
colors = ['#7FB3D5', '#1F77B4', '#5194E9', '#0080FF']

# Membuat pie chart
ax1.pie(Psychiatric_count, labels=Psychiatric_count.index, colors=colors, autopct='%1.1f%%')
ax1.set_title('Psychiatric History Comparison')

# Membuat bar plot
ax2.bar(Psychiatric_count.index, Psychiatric_count.values, color=colors)
ax2.set_xlabel('Psychiatric History')
ax2.set_ylabel('Count')
ax2.set_title('Psychiatric History Comparison')

# Menampilkan plot
plt.tight_layout()
plt.show()

SU_count = df['Substance Use'].value_counts()

# Mengatur ukuran dan warna plot
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 6))
colors = ['#7FB3D5', '#1F77B4', '#5194E9', '#0080FF']

# Membuat pie chart
ax1.pie(SU_count, labels=SU_count.index, colors=colors, autopct='%1.1f%%')
ax1.set_title('Substance Use Comparison')

# Membuat bar plot
ax2.bar(SU_count.index, SU_count.values, color=colors)
ax2.set_xlabel('Substance Use')
ax2.set_ylabel('Count')
ax2.set_title('Substance Use Comparison')

# Menampilkan plot
plt.tight_layout()
plt.show()

cp_count = df['Coping Mechanisms'].value_counts()

# Mengatur ukuran dan warna plot
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 6))
colors = ['#7FB3D5', '#1F77B4', '#5194E9', '#0080FF']

# Membuat pie chart
ax1.pie(cp_count, labels=cp_count.index, colors=colors, autopct='%1.1f%%')
ax1.set_title('Coping Mechanisms Comparison')

# Membuat bar plot
ax2.bar(cp_count.index, cp_count.values, color=colors)
ax2.set_xlabel('Coping Mechanisms')
ax2.set_ylabel('Count')
ax2.set_title('Coping Mechanisms Comparison')

# Menampilkan plot
plt.tight_layout()
plt.show()

sl_count = df['Social Support'].value_counts()

# Mengatur ukuran dan warna plot
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 6))
colors = ['#7FB3D5', '#1F77B4', '#5194E9']

# Membuat pie chart
ax1.pie(sl_count, labels=sl_count.index, colors=colors, autopct='%1.1f%%')
ax1.set_title('Social Support Comparison')

# Membuat bar plot
ax2.bar(sl_count.index, sl_count.values, color=colors)
ax2.set_xlabel('Social Support')
ax2.set_ylabel('Count')
ax2.set_title('Social Support Comparison')

# Menampilkan plot
plt.tight_layout()
plt.show()

lf_count = df['Lifestyle Factors'].value_counts()

# Mengatur ukuran dan warna plot
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 6))
colors = ['#7FB3D5', '#1F77B4', '#5194E9']

# Membuat pie chart
ax1.pie(lf_count, labels=lf_count.index, colors=colors, autopct='%1.1f%%')
ax1.set_title('Lifestyle Factors Comparison')

# Membuat bar plot
ax2.bar(lf_count.index, lf_count.values, color=colors)
ax2.set_xlabel('Lifestyle Factors')
ax2.set_ylabel('Count')
ax2.set_title('Lifestyle Factors Comparison')

# Menampilkan plot
plt.tight_layout()
plt.show()

pdd_count = df['Panic Disorder Diagnosis'].value_counts()

# Mengatur ukuran dan warna plot
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 6))
colors = ['#7FB3D5', '#FF7F50']

# Membuat pie chart
ax1.pie(pdd_count, labels=pdd_count.index, colors=colors, autopct='%1.1f%%')
ax1.set_title('Panic Disorder Diagnosis Comparison')

# Membuat bar plot
ax2.bar(pdd_count.index, pdd_count.values, color=colors)
ax2.set_xlabel('Panic Disorder Diagnosis')
ax2.set_ylabel('Count')
ax2.set_title('Panic Disorder Diagnosis Comparison')

# Menampilkan plot
plt.tight_layout()
plt.show()

headerSection = st.container()

selected = st.sidebar.radio("Menu", ["Visualization", "Machine Learning"])

if selected == "Visualization":
    st.title("Visualization Menu")

    option = st.selectbox(
    'Pilih Variable',
    ('Age','Gender','Family History','Personal History','Coping Mechanisms','Current Stressors','Symptoms','Severity','Impact on Life','Demographics','Medical History','Psychiatric History','Substance Use','Social Support','Lifestyle Factors','Panic Disorder Diagnosis'))
    if (option == "Age") :
        
        st.title('Perbandingan Usia')
        gender_count = df['Kategori Usia'].value_counts()

        # Mengatur ukuran dan warna plot
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 6))
        colors = ['#7FB3D5', '#1F77B4', '#5194E9', '#0080FF', '#005CBF']

        # Membuat pie chart
        ax1.pie(kategori_usia_count, labels=kategori_usia_count.index, colors=colors, autopct='%1.1f%%')
        ax1.set_title('Age Comparison')

        # Membuat bar plot
        ax2.bar(kategori_usia_count.index, kategori_usia_count.values, color=colors)
        ax2.set_xlabel('Age')
        ax2.set_ylabel('Count')
        ax2.set_title('Age Comparison')

        # Menampilkan plot
        plt.tight_layout()
        st.pyplot(fig)

        # Daftar panic disorder diagnosis unik
        panic_diagnoses = df['Panic Disorder Diagnosis'].unique()

        # Mengganti nilai panic disorder diagnosis menjadi string
        panic_diagnoses_text = ["Tidak Terkena Penyakit" if d == 0 else "Terkena Penyakit" for d in panic_diagnoses]

        # Pilihan panic disorder diagnosis
        selected_diagnosis = st.selectbox("Pilih Panic Disorder Diagnosis", panic_diagnoses_text)

        # Menggabungkan data berdasarkan panic disorder diagnosis yang dipilih
        selected_data = df[df['Panic Disorder Diagnosis'] == (0 if selected_diagnosis == "Tidak Terkena Penyakit" else 1)]

        # Menghitung jumlah setiap kategori usia berdasarkan panic disorder diagnosis yang dipilih
        age_counts = selected_data['Age'].value_counts().reset_index()
        age_counts.columns = ['Kategori Usia', 'Jumlah']
        age_counts['Kategori Usia'] = age_counts['Kategori Usia'].astype(str)  # Mengonversi ke tipe string

        # Menampilkan hasil di Streamlit
        st.subheader("Jumlah Setiap Kategori Usia berdasarkan Panic Disorder Diagnosis: " + selected_diagnosis)
        st.dataframe(age_counts)
            # Visualisasi dalam bentuk barchart
        fig, ax = plt.subplots()
        ax.bar(age_counts['Kategori Usia'], age_counts['Jumlah'])
        ax.set_xlabel('Usia')
        ax.set_ylabel('Jumlah')
        ax.set_title('Jumlah Setiap Kategori Usia')
        st.pyplot(fig)
        
    elif (option == "Gender") :
        
        st.title('Perbandingan Gender')
        gender_count = df['Gender'].value_counts()

        # Mengatur ukuran dan warna plot
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 6))
        colors = ['#7FB3D5', '#FF7F50']

        # Membuat pie chart
        ax1.pie(gender_count, labels=gender_count.index, colors=colors, autopct='%1.1f%%')
        ax1.set_title('Gender Comparison')

        # Membuat bar plot
        ax2.bar(gender_count.index, gender_count.values, color=colors)
        ax2.set_xlabel('Gender')
        ax2.set_ylabel('Count')
        ax2.set_title('Gender Comparison')
        
        # Menampilkan plot
        plt.tight_layout()
        st.pyplot(fig)
        
        # Daftar panic disorder diagnosis unik
        panic_diagnoses = df['Panic Disorder Diagnosis'].unique()

        # Mengganti nilai panic disorder diagnosis menjadi string
        panic_diagnoses_text = ["Tidak Terkena Penyakit" if d == 0 else "Terkena Penyakit" for d in panic_diagnoses]

        # Pilihan panic disorder diagnosis
        selected_diagnosis = st.selectbox("Pilih Panic Disorder Diagnosis", panic_diagnoses_text)

        # Menggabungkan data berdasarkan panic disorder diagnosis yang dipilih
        selected_data = df[df['Panic Disorder Diagnosis'] == (0 if selected_diagnosis == "Tidak Terkena Penyakit" else 1)]

        # Menghitung jumlah setiap kategori gender
        gender_counts = selected_data['Gender'].value_counts().reset_index()
        gender_counts.columns = ['Gender', 'Jumlah']
        gender_counts['Gender'] = ["Laki-laki" if g == "Male" else "Perempuan" for g in gender_counts['Gender']]  # Mengganti nilai gender menjadi string

        # Menampilkan hasil di Streamlit
        if not gender_counts.empty:
            st.subheader("Jumlah Setiap Kategori Gender berdasarkan Panic Disorder Diagnosis: " + selected_diagnosis)
            st.table(gender_counts)
            # Visualisasi dalam bentuk barchart
            fig, ax = plt.subplots()
            ax.bar(gender_counts['Gender'], gender_counts['Jumlah'])
            ax.set_xlabel('Gender')
            ax.set_ylabel('Jumlah')
            ax.set_title('Jumlah Setiap Kategori Gender')
            st.pyplot(fig)
        else:
            st.subheader("Tidak ada data yang sesuai dengan pilihan yang dipilih.")
                
    elif (option == "Family History") :
        st.title ('Family History Comparison')
        history_count = df['Family History'].value_counts()

        # Mengatur ukuran dan warna plot
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 6))
        colors = ['#7FB3D5', '#FF7F50']

        # Membuat pie chart
        ax1.pie(history_count, labels=history_count.index, colors=colors, autopct='%1.1f%%')
        ax1.set_title('Family History Comparison')

        # Membuat bar plot
        ax2.bar(history_count.index, history_count.values, color=colors)
        ax2.set_xlabel('Family History')
        ax2.set_ylabel('Count')
        ax2.set_title('Family History Comparison')

        # Menampilkan plot
        plt.tight_layout()
        st.pyplot(fig)
        
        
        # Daftar panic disorder diagnosis unik
        panic_diagnoses = df['Panic Disorder Diagnosis'].unique()

        # Mengganti nilai panic disorder diagnosis menjadi string
        panic_diagnoses_text = ["Tidak Terkena Penyakit" if d == 0 else "Terkena Penyakit" for d in panic_diagnoses]

        # Pilihan panic disorder diagnosis
        selected_diagnosis = st.selectbox("Pilih Panic Disorder Diagnosis", panic_diagnoses_text)

        # Menggabungkan data berdasarkan panic disorder diagnosis yang dipilih
        selected_data = df[df['Panic Disorder Diagnosis'] == (0 if selected_diagnosis == "Tidak Terkena Penyakit" else 1)]

        # Menghitung jumlah setiap kategori family history
        family_history_counts = selected_data['Family History'].value_counts().reset_index()
        family_history_counts.columns = ['Family History', 'Jumlah']
        family_history_counts['Family History'] = ["Tidak" if f == "No" else "Ya" for f in family_history_counts['Family History']]  # Mengganti nilai family history menjadi string

        # Menampilkan hasil di Streamlit
        if not family_history_counts.empty:
            st.subheader("Jumlah Setiap Kategori Family History berdasarkan Panic Disorder Diagnosis: " + selected_diagnosis)
            st.table(family_history_counts)

            # Visualisasi dalam bentuk barchart
            fig, ax = plt.subplots()
            ax.bar(family_history_counts['Family History'], family_history_counts['Jumlah'])
            ax.set_xlabel('Family History')
            ax.set_ylabel('Jumlah')
            ax.set_title('Jumlah Setiap Kategori Family History')
            st.pyplot(fig)
        else:
            st.subheader("Tidak ada data yang sesuai dengan pilihan yang dipilih.")
    
    elif (option == "Personal History") :
        st.title ('Personal History Comparison')
        personal_count = df['Personal History'].value_counts()

        # Mengatur ukuran dan warna plot
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 6))
        colors = ['#7FB3D5', '#FF7F50', '#90EE90']

        # Membuat pie chart
        ax1.pie(personal_count, labels=personal_count.index, colors=colors, autopct='%1.1f%%')
        ax1.set_title('Personal History Comparison')

        # Membuat bar plot
        ax2.bar(personal_count.index, personal_count.values, color=colors)
        ax2.set_xlabel('Personal History')
        ax2.set_ylabel('Count')
        ax2.set_title('Personal History Comparison')

        # Menampilkan plot
        plt.tight_layout()
        st.pyplot(fig)
            # Daftar panic disorder diagnosis unik
        panic_diagnoses = df['Panic Disorder Diagnosis'].unique()

        # Mengganti nilai panic disorder diagnosis menjadi string
        panic_diagnoses_text = ["Tidak Terkena Penyakit" if d == 0 else "Terkena Penyakit" for d in panic_diagnoses]

        # Pilihan panic disorder diagnosis
        selected_diagnosis = st.selectbox("Pilih Panic Disorder Diagnosis", panic_diagnoses_text)

        # Menggabungkan data berdasarkan panic disorder diagnosis yang dipilih
        selected_data = df[df['Panic Disorder Diagnosis'] == (0 if selected_diagnosis == "Tidak Terkena Penyakit" else 1)]

        # Menghitung jumlah setiap kategori family history
        personal_history_counts = selected_data['Personal History'].value_counts().reset_index()
        personal_history_counts.columns = ['Personal History', 'Jumlah']
        personal_history_counts['Personal History'] = ["Tidak" if f == "No" else "Ya" for f in personal_history_counts['Personal History']]  # Mengganti nilai family history menjadi string

        # Menampilkan hasil di Streamlit
        if not personal_history_counts.empty:
            st.subheader("Jumlah Setiap Kategori Family History berdasarkan Panic Disorder Diagnosis: " + selected_diagnosis)
            st.table(personal_history_counts)

            # Visualisasi dalam bentuk barchart
            fig, ax = plt.subplots()
            ax.bar(personal_history_counts['Personal History'], personal_history_counts['Jumlah'])
            ax.set_xlabel('Personal History')
            ax.set_ylabel('Jumlah')
            ax.set_title('Jumlah Setiap Kategori Personal History')
            st.pyplot(fig)
        else:
            st.subheader("Tidak ada data yang sesuai dengan pilihan yang dipilih.")
    elif (option == "Current Stressors") :
        st.title ('Current Stressors Comparison')
        stressors_count = df['Current Stressors'].value_counts()

        # Mengatur ukuran dan warna plot
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 6))
        colors = ['#7FB3D5', '#1F77B4', '#5194E9']

        # Membuat pie chart
        ax1.pie(stressors_count, labels=stressors_count.index, colors=colors, autopct='%1.1f%%')
        ax1.set_title('Current Stressors Comparison')

        # Membuat bar plot
        ax2.bar(stressors_count.index, stressors_count.values, color=colors)
        ax2.set_xlabel('Current Stressors')
        ax2.set_ylabel('Count')
        ax2.set_title('Current Stressors Comparison')

        # Menampilkan plot
        plt.tight_layout()

        st.pyplot(fig)
    
    elif (option == "Symptoms") :
        st.title ('Symptoms Comparison')
        symptoms_count = df['Symptoms'].value_counts()

        # Mengatur ukuran dan warna plot
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))
        colors = ['#7FB3D5', '#1F77B4', '#5194E9', '#0080FF', '#005CBF']

        # Membuat pie chart
        ax1.pie(symptoms_count, labels=symptoms_count.index, colors=colors, autopct='%1.1f%%')
        ax1.set_title('Symptoms Comparison')

        # Membuat bar plot
        ax2.bar(symptoms_count.index, symptoms_count.values, color=colors)
        ax2.set_xlabel('Symptoms')
        ax2.set_ylabel('Count')
        ax2.set_title('Symptoms Comparison')

        # Menampilkan plot
        plt.tight_layout()
  
        st.pyplot(fig)
        
    elif (option == "Severity") :
        st.title ('Severity Comparison')
        severity_count = df['Severity'].value_counts()

        # Mengatur ukuran dan warna plot
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 6))
        colors = ['#7FB3D5', '#1F77B4', '#5194E9']

        # Membuat pie chart
        ax1.pie(severity_count, labels=severity_count.index, colors=colors, autopct='%1.1f%%')
        ax1.set_title('Severity Comparison')

        # Membuat bar plot
        ax2.bar(severity_count.index, severity_count.values, color=colors)
        ax2.set_xlabel('Severity')
        ax2.set_ylabel('Count')
        ax2.set_title('Severity Comparison')

        # Menampilkan plot
        plt.tight_layout()

        st.pyplot(fig)
    
    elif (option == "Impact on Life") :
        st.title ('Impact on Life Comparison')
        IOL_count = df['Impact on Life'].value_counts()

        # Mengatur ukuran dan warna plot
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 6))
        colors = ['#7FB3D5', '#1F77B4', '#5194E9']

        # Membuat pie chart
        ax1.pie(IOL_count, labels=IOL_count.index, colors=colors, autopct='%1.1f%%')
        ax1.set_title('Impact on Life Comparison')

        # Membuat bar plot
        ax2.bar(IOL_count.index, IOL_count.values, color=colors)
        ax2.set_xlabel('Impact on Life')
        ax2.set_ylabel('Count')
        ax2.set_title('Impact on Life Comparison')

        # Menampilkan plot
        plt.tight_layout()

        st.pyplot(fig)
        
    elif (option == "Demographics") :
        st.title ('Demographics Comparison')
        Demographics_count = df['Demographics'].value_counts()

        # Mengatur ukuran dan warna plot
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 6))
        colors = ['#7FB3D5', '#FF7F50']

        # Membuat pie chart
        ax1.pie(Demographics_count, labels=Demographics_count.index, colors=colors, autopct='%1.1f%%')
        ax1.set_title('Demographics Comparison')

        # Membuat bar plot
        ax2.bar(Demographics_count.index, Demographics_count.values, color=colors)
        ax2.set_xlabel('Demographics')
        ax2.set_ylabel('Count')
        ax2.set_title('Demographics Comparison')

        # Menampilkan plot
        plt.tight_layout()
      
        st.pyplot(fig)
    
    elif (option == "Medical History") :
        st.title ('Medical History Comparison')
        medical_count = df['Medical History'].value_counts()

        # Mengatur ukuran dan warna plot
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 6))
        colors = ['#7FB3D5', '#1F77B4', '#5194E9', '#0080FF']

        # Membuat pie chart
        ax1.pie(medical_count, labels=medical_count.index, colors=colors, autopct='%1.1f%%')
        ax1.set_title('Medical History Comparison')

        # Membuat bar plot
        ax2.bar(medical_count.index, medical_count.values, color=colors)
        ax2.set_xlabel('Medical History')
        ax2.set_ylabel('Count')
        ax2.set_title('Medical History Comparison')

        # Menampilkan plot
        plt.tight_layout()

        st.pyplot(fig)
        
    elif (option == "Psychiatric History") :
        st.title ('Psychiatric History Comparison')
        Psychiatric_count = df['Psychiatric History'].value_counts()

        # Mengatur ukuran dan warna plot
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 6))
        colors = ['#7FB3D5', '#1F77B4', '#5194E9', '#0080FF']

        # Membuat pie chart
        ax1.pie(Psychiatric_count, labels=Psychiatric_count.index, colors=colors, autopct='%1.1f%%')
        ax1.set_title('Psychiatric History Comparison')

        # Membuat bar plot
        ax2.bar(Psychiatric_count.index, Psychiatric_count.values, color=colors)
        ax2.set_xlabel('Psychiatric History')
        ax2.set_ylabel('Count')
        ax2.set_title('Psychiatric History Comparison')

        # Menampilkan plot
        plt.tight_layout()
       
        st.pyplot(fig)
    elif (option == "Substance Use") :
        st.title ('Substance Use Comparison')
        SU_count = df['Substance Use'].value_counts()

        # Mengatur ukuran dan warna plot
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 6))
        colors = ['#7FB3D5', '#1F77B4', '#5194E9', '#0080FF']

        # Membuat pie chart
        ax1.pie(SU_count, labels=SU_count.index, colors=colors, autopct='%1.1f%%')
        ax1.set_title('Substance Use Comparison')

        # Membuat bar plot
        ax2.bar(SU_count.index, SU_count.values, color=colors)
        ax2.set_xlabel('Substance Use')
        ax2.set_ylabel('Count')
        ax2.set_title('Substance Use Comparison')

        # Menampilkan plot
        plt.tight_layout()

        st.pyplot(fig)
        
    elif (option == "Coping Mechanisms") :
        st.title ('Coping Mechanisms Comparison')
        cp_count = df['Coping Mechanisms'].value_counts()

        # Mengatur ukuran dan warna plot
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 6))
        colors = ['#7FB3D5', '#1F77B4', '#5194E9', '#0080FF']

        # Membuat pie chart
        ax1.pie(cp_count, labels=cp_count.index, colors=colors, autopct='%1.1f%%')
        ax1.set_title('Coping Mechanisms Comparison')

        # Membuat bar plot
        ax2.bar(cp_count.index, cp_count.values, color=colors)
        ax2.set_xlabel('Coping Mechanisms')
        ax2.set_ylabel('Count')
        ax2.set_title('Coping Mechanisms Comparison')

        # Menampilkan plot
        plt.tight_layout()

        st.pyplot(fig)
        # Daftar panic disorder diagnosis unik
        panic_diagnoses = df['Panic Disorder Diagnosis'].unique()

        # Mengganti nilai panic disorder diagnosis menjadi string
        panic_diagnoses_text = ["Tidak Terkena Penyakit" if d == 0 else "Terkena Penyakit" for d in panic_diagnoses]

        # Pilihan panic disorder diagnosis
        selected_diagnosis = st.selectbox("Pilih Panic Disorder Diagnosis", panic_diagnoses_text)

        # Menggabungkan data berdasarkan panic disorder diagnosis yang dipilih
        selected_data = df[df['Panic Disorder Diagnosis'] == (0 if selected_diagnosis == "Tidak Terkena Penyakit" else 1)]

        # Menghitung jumlah setiap kategori coping mechanism
        coping_counts = selected_data['Coping Mechanisms'].value_counts().reset_index()
        coping_counts.columns = ['Coping Mechanisms', 'Jumlah']

        # Menampilkan hasil di Streamlit
        if not coping_counts.empty:
            st.subheader("Jumlah Setiap Kategori Coping Mechanism berdasarkan Panic Disorder Diagnosis: " + selected_diagnosis)
            st.table(coping_counts)

            # Visualisasi dalam bentuk barchart
            fig, ax = plt.subplots()
            ax.bar(coping_counts['Coping Mechanisms'], coping_counts['Jumlah'])
            ax.set_xlabel('Coping Mechanisms')
            ax.set_ylabel('Jumlah')
            ax.set_title('Jumlah Setiap Kategori Coping Mechanism')
            st.pyplot(fig)
        else:
            st.subheader("Tidak ada data yang sesuai dengan pilihan yang dipilih.")    
    elif (option == "Social Support") :
        st.title ('Social Support Comparison')
        sl_count = df['Social Support'].value_counts()

        # Mengatur ukuran dan warna plot
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 6))
        colors = ['#7FB3D5', '#1F77B4', '#5194E9']

        # Membuat pie chart
        ax1.pie(sl_count, labels=sl_count.index, colors=colors, autopct='%1.1f%%')
        ax1.set_title('Social Support Comparison')

        # Membuat bar plot
        ax2.bar(sl_count.index, sl_count.values, color=colors)
        ax2.set_xlabel('Social Support')
        ax2.set_ylabel('Count')
        ax2.set_title('Social Support Comparison')

        # Menampilkan plot
        plt.tight_layout()

        st.pyplot(fig)
        
    elif (option == "Lifestyle Factors") :
        st.title ('Lifestyle Factors Comparison')
        lf_count = df['Lifestyle Factors'].value_counts()

        # Mengatur ukuran dan warna plot
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 6))
        colors = ['#7FB3D5', '#1F77B4', '#5194E9']

        # Membuat pie chart
        ax1.pie(lf_count, labels=lf_count.index, colors=colors, autopct='%1.1f%%')
        ax1.set_title('Lifestyle Factors Comparison')

        # Membuat bar plot
        ax2.bar(lf_count.index, lf_count.values, color=colors)
        ax2.set_xlabel('Lifestyle Factors')
        ax2.set_ylabel('Count')
        ax2.set_title('Lifestyle Factors Comparison')

        # Menampilkan plot
        plt.tight_layout()

        st.pyplot(fig)
        
    elif (option == "Panic Disorder Diagnosis") :
        st.title ('Panic Disorder Diagnosis Comparison')
        pdd_count = df['Panic Disorder Diagnosis'].value_counts()

        # Mengatur ukuran dan warna plot
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 6))
        colors = ['#7FB3D5', '#FF7F50']

        # Membuat pie chart
        ax1.pie(pdd_count, labels=pdd_count.index, colors=colors, autopct='%1.1f%%')
        ax1.set_title('Panic Disorder Diagnosis Comparison')

        # Membuat bar plot
        ax2.bar(pdd_count.index, pdd_count.values, color=colors)
        ax2.set_xlabel('Panic Disorder Diagnosis')
        ax2.set_ylabel('Count')
        ax2.set_title('Panic Disorder Diagnosis Comparison')

        # Menampilkan plot
        plt.tight_layout()

        st.pyplot(fig)
    
elif selected == "Machine Learning":
    st.title("Machine Learning Menu")
# Pilihan metode Machine Learning
    variables = ['Age', 'Gender', 'Family History', 'Personal History', 'Current Stressors', 'Symptoms',
                'Severity', 'Impact on Life', 'Demographics', 'Medical History', 'Psychiatric History',
                'Substance Use', 'Coping Mechanisms', 'Social Support', 'Lifestyle Factors',
                'Panic Disorder Diagnosis']

    # Pilihan variabel
    selected_variables = st.multiselect("Pilih Variabel", variables[:-1], default=variables[:-1])

    # Pilihan metode Machine Learning
    selected_method = st.selectbox("Pilih Metode", ("Decision Tree", "Random Forest"))

    # Menggunakan subset data berdasarkan variabel yang dipilih
    df_selected = df1[selected_variables + ['Panic Disorder Diagnosis']]

    # Split data menjadi fitur dan label
    x = df_selected[selected_variables]
    y = df_selected["Panic Disorder Diagnosis"]

    # Train-test split
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2)

    # Model Machine Learning
    if selected_method == "Decision Tree":
        model = DecisionTreeClassifier()
    elif selected_method == "Random Forest":
        model = RandomForestClassifier()

    # Train model
    model.fit(x_train, y_train)

    # Prediksi pada data uji
    y_pred = model.predict(x_test)

    # Evaluasi model
    score = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred)

    # Menampilkan hasil di Streamlit
    st.subheader(selected_method)
    st.text("Akurasi: " + str(score))
    st.text("Classification Report:")
    st.text(report)
