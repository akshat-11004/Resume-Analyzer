#SET UP:

# 1. INSTALL BELOW LIBRARIES

        #pip install -r requirements.txt

        # pip install nltk

        # pip install spacy==2.3.5

        # pip install https://github.com/explosion/spacy-models/releases/download/en_core_web_sm-2.3.1/en_core_web_sm-2.3.1.tar.gz
    
        # pip install pyresparser

# 2. CREAT A FOLDER AND NAME IT (e.g. resume)
        #2.1 create two more folders inside this folder (Logo and Uploaded_Resumes)
        #2.2 create two python files (App.py and Courses.py)

# 3. START YOUR SQL DATABASE


# 4. CONTINUE WITH THE FOLLOWING CODE...

import streamlit as st
st.set_page_config(
    # page_title="CareerCompass Resume Analyzer",
    page_icon='./Logo/logo2.png',
    layout="wide",
    initial_sidebar_state="expanded"
)

import pandas as pd
import base64,random
import time,datetime
#libraries to parse the resume pdf files
from pyresparser import ResumeParser
from pdfminer3.layout import LAParams, LTTextBox
from pdfminer3.pdfpage import PDFPage
from pdfminer3.pdfinterp import PDFResourceManager
from pdfminer3.pdfinterp import PDFPageInterpreter
from pdfminer3.converter import TextConverter
import io,random
from streamlit_tags import st_tags
from PIL import Image
import pymysql
from Courses import ds_course,web_course,android_course,ios_course,uiux_course
# import pafy #for uploading youtube videos
import plotly.express as px #to create visualisations at the admin session
import nltk
nltk.download('stopwords')

from streamlit_option_menu import option_menu
selected = option_menu("Menu", ["Home", "Admin", "User"],
    icons=['house', 'gear', 'person'], menu_icon="cast", default_index=0)

def get_table_download_link(df,filename,text):
    """Generates a link allowing the data in a given panda dataframe to be downloaded
    in:  dataframe
    out: href string
    """
    csv = df.to_csv(index=False)
    b64 = base64.b64encode(csv.encode()).decode()  # some strings <-> bytes conversions necessary here
    # href = f'<a href="data:file/csv;base64,{b64}">Download Report</a>'
    href = f'<a href="data:file/csv;base64,{b64}" download="{filename}">{text}</a>'
    return href

def pdf_reader(file):
    resource_manager = PDFResourceManager()
    fake_file_handle = io.StringIO()
    converter = TextConverter(resource_manager, fake_file_handle, laparams=LAParams())
    page_interpreter = PDFPageInterpreter(resource_manager, converter)
    with open(file, 'rb') as fh:
        for page in PDFPage.get_pages(fh,
                                      caching=True,
                                      check_extractable=True):
            page_interpreter.process_page(page)
            print(page)
        text = fake_file_handle.getvalue()

    # close open handles
    converter.close()
    fake_file_handle.close()
    return text

def show_pdf(file_path):
    with open(file_path, "rb") as f:
        base64_pdf = base64.b64encode(f.read()).decode('utf-8')
    # pdf_display = f'<embed src="data:application/pdf;base64,{base64_pdf}" width="700" height="1000" type="application/pdf">'
    pdf_display = F'<iframe src="data:application/pdf;base64,{base64_pdf}" width="700" height="1000" type="application/pdf"></iframe>'
    st.markdown(pdf_display, unsafe_allow_html=True)

def course_recommender(course_list):
    st.subheader("**Courses & Certificates Recommendations 🎓**")
    c = 0
    rec_course = []
    no_of_reco = st.slider('Choose Number of Course Recommendations:', 1, 10, 5)
    random.shuffle(course_list)
    for c_name, c_link in course_list:
        c += 1
        st.markdown(f"({c}) [{c_name}]({c_link})")
        rec_course.append(c_name)
        if c == no_of_reco:
            break
    return rec_course




#CONNECT TO DATABASE

connection = pymysql.connect(host='localhost',user='root',password='akshat@2004',db='cv')
cursor = connection.cursor()

def insert_data(name,email,res_score,timestamp,no_of_pages,reco_field,cand_level,skills,recommended_skills,courses):
    DB_table_name = 'user_data'
    insert_sql = "insert into " + DB_table_name + """
    values (0,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)"""
    rec_values = (name, email, str(res_score), timestamp,str(no_of_pages), reco_field, cand_level, skills,recommended_skills,courses)
    cursor.execute(insert_sql, rec_values)
    connection.commit()

st.set_page_config(
   page_title="Resume Analyzer",
   page_icon='./Logo/logo2.png',  # this is for icon logo in tabs   
)
def run():
    img = Image.open('./Logo/logo3.png')
    # img = img.resize((250,250))
    st.image(img)
    st.title("AI Powered Resume Analyser")
    st.sidebar.markdown("# Choose User")
    activities = ["User", "Admin"]
    choice = st.sidebar.selectbox("Choose your login type:", activities)
    link = '[©Developed by akshat kadia](https://www.linkedin.com/in/akshat-kadia-588a85266/)'
    st.sidebar.markdown(link, unsafe_allow_html=True)


    # Create the DB
    db_sql = """CREATE DATABASE IF NOT EXISTS CV;"""
    cursor.execute(db_sql)

    # Create table
    DB_table_name = 'user_data'
    table_sql = "CREATE TABLE IF NOT EXISTS " + DB_table_name + """
                    (ID INT NOT NULL AUTO_INCREMENT,
                     Name varchar(500) NOT NULL,
                     Email_ID VARCHAR(500),
                     resume_score VARCHAR(8) NOT NULL,
                     Timestamp VARCHAR(50) NOT NULL,
                     Page_no VARCHAR(5) NOT NULL,
                     Predicted_Field BLOB NOT NULL,
                     User_level BLOB NOT NULL,
                     Actual_skills BLOB NOT NULL,
                     Recommended_skills BLOB NOT NULL,
                     Recommended_courses BLOB NOT NULL,
                     PRIMARY KEY (ID));
                    """
    cursor.execute(table_sql)
    if choice == 'User':
        st.markdown('''<h5 style='text-align: left; color: #021659;'> Upload your resume, and get smart recommendations</h5>''',
                    unsafe_allow_html=True)
        pdf_file = st.file_uploader("Choose your Resume", type=["pdf"])
        if pdf_file is not None:
            with st.spinner('Uploading your Resume...'):
                time.sleep(4)
            save_image_path = './Uploaded_Resumes/'+pdf_file.name
            with open(save_image_path, "wb") as f:
                f.write(pdf_file.getbuffer())
            show_pdf(save_image_path)
            resume_data = ResumeParser(save_image_path).get_extracted_data()
            if resume_data:
                ## Get the whole resume data
                resume_text = pdf_reader(save_image_path)

                st.header("**Resume Analysis**")
                st.success("Hello "+ resume_data['name'])
                st.subheader("**Your Basic info**")
                try:
                    st.text('Name: '+resume_data['name'])
                    st.text('Email: ' + resume_data['email'])
                    st.text('Contact: ' + resume_data['mobile_number'])
                    st.text('Resume pages: '+str(resume_data['no_of_pages']))
                except:
                    pass
                cand_level = ''
                if resume_data['no_of_pages'] == 1:
                    cand_level = "Fresher"
                    st.markdown( '''<h4 style='text-align: left; color: #d73b5c;'>You are at Fresher level!</h4>''',unsafe_allow_html=True)
                elif resume_data['no_of_pages'] == 2:
                    cand_level = "Intermediate"
                    st.markdown('''<h4 style='text-align: left; color: #1ed760;'>You are at intermediate level!</h4>''',unsafe_allow_html=True)
                elif resume_data['no_of_pages'] >=3:
                    cand_level = "Experienced"
                    st.markdown('''<h4 style='text-align: left; color: #fba171;'>You are at experience level!''',unsafe_allow_html=True)

                # st.subheader("**Skills Recommendation💡**")
                ## Skill shows
                keywords = st_tags(label='### Your Current Skills',
                text='See our skills recommendation below',
                    value=resume_data['skills'],key = '1  ')

                ##  keywords
                ds_keyword = ['tensorflow','keras','pytorch','machine learning','deep Learning','flask','streamlit']
                web_keyword = ['react', 'django', 'node jS', 'react js', 'php', 'laravel', 'magento', 'wordpress',
                               'javascript', 'angular js', 'c#', 'flask']
                android_keyword = ['android','android development','flutter','kotlin','xml','kivy']
                ios_keyword = ['ios','ios development','swift','cocoa','cocoa touch','xcode']
                uiux_keyword = ['ux','adobe xd','figma','zeplin','balsamiq','ui','prototyping','wireframes','storyframes','adobe photoshop','photoshop','editing','adobe illustrator','illustrator','adobe after effects','after effects','adobe premier pro','premier pro','adobe indesign','indesign','wireframe','solid','grasp','user research','user experience']

                recommended_skills = []
                reco_field = ''
                rec_course = ''
                ## Courses recommendation
                for i in resume_data['skills']:
                    ## Data science recommendation
                    if i.lower() in ds_keyword:
                        print(i.lower())
                        reco_field = 'Data Science'
                        st.success("** Our analysis says you are looking for Data Science Jobs.**")
                        recommended_skills = ['Data Visualization','Predictive Analysis','Statistical Modeling','Data Mining','Clustering & Classification','Data Analytics','Quantitative Analysis','Web Scraping','ML Algorithms','Keras','Pytorch','Probability','Scikit-learn','Tensorflow',"Flask",'Streamlit']
                        recommended_keywords = st_tags(label='### Recommended skills for you.',
                        text='Recommended skills generated from System',value=recommended_skills,key = '2')
                        st.markdown('''<h4 style='text-align: left; color: #1ed760;'>Adding this skills to resume will boost🚀 the chances of getting a Job</h4>''',unsafe_allow_html=True)
                        rec_course = course_recommender(ds_course)
                        break

                    ## Web development recommendation
                    elif i.lower() in web_keyword:
                        print(i.lower())
                        reco_field = 'Web Development'
                        st.success("** Our analysis says you are looking for Web Development Jobs **")
                        recommended_skills = ['React','Django','Node JS','React JS','php','laravel','Magento','wordpress','Javascript','Angular JS','c#','Flask','SDK']
                        recommended_keywords = st_tags(label='### Recommended skills for you.',
                        text='Recommended skills generated from System',value=recommended_skills,key = '3')
                        st.markdown('''<h4 style='text-align: left; color: #1ed760;'>Adding this skills to resume will boost🚀 the chances of getting a Job💼</h4>''',unsafe_allow_html=True)
                        rec_course = course_recommender(web_course)
                        break

                    ## Android App Development
                    elif i.lower() in android_keyword:
                        print(i.lower())
                        reco_field = 'Android Development'
                        st.success("** Our analysis says you are looking for Android App Development Jobs **")
                        recommended_skills = ['Android','Android development','Flutter','Kotlin','XML','Java','Kivy','GIT','SDK','SQLite']
                        recommended_keywords = st_tags(label='### Recommended skills for you.',
                        text='Recommended skills generated from System',value=recommended_skills,key = '4')
                        st.markdown('''<h4 style='text-align: left; color: #1ed760;'>Adding this skills to resume will boost🚀 the chances of getting a Job💼</h4>''',unsafe_allow_html=True)
                        rec_course = course_recommender(android_course)
                        break

                    ## IOS App Development
                    elif i.lower() in ios_keyword:
                        print(i.lower())
                        reco_field = 'IOS Development'
                        st.success("** Our analysis says you are looking for IOS App Development Jobs **")
                        recommended_skills = ['IOS','IOS Development','Swift','Cocoa','Cocoa Touch','Xcode','Objective-C','SQLite','Plist','StoreKit',"UI-Kit",'AV Foundation','Auto-Layout']
                        recommended_keywords = st_tags(label='### Recommended skills for you.',
                        text='Recommended skills generated from System',value=recommended_skills,key = '5')
                        st.markdown('''<h4 style='text-align: left; color: #1ed760;'>Adding this skills to resume will boost🚀 the chances of getting a Job💼</h4>''',unsafe_allow_html=True)
                        rec_course = course_recommender(ios_course)
                        break

                    ## Ui-UX Recommendation
                    elif i.lower() in uiux_keyword:
                        print(i.lower())
                        reco_field = 'UI-UX Development'
                        st.success("** Our analysis says you are looking for UI-UX Development Jobs **")
                        recommended_skills = ['UI','User Experience','Adobe XD','Figma','Zeplin','Balsamiq','Prototyping','Wireframes','Storyframes','Adobe Photoshop','Editing','Illustrator','After Effects','Premier Pro','Indesign','Wireframe','Solid','Grasp','User Research']
                        recommended_keywords = st_tags(label='### Recommended skills for you.',
                        text='Recommended skills generated from System',value=recommended_skills,key = '6')
                        st.markdown('''<h4 style='text-align: left; color: #1ed760;'>Adding this skills to resume will boost🚀 the chances of getting a Job💼</h4>''',unsafe_allow_html=True)
                        rec_course = course_recommender(uiux_course)
                        break

                
                ## Insert into table
                ts = time.time()
                cur_date = datetime.datetime.fromtimestamp(ts).strftime('%Y-%m-%d')
                cur_time = datetime.datetime.fromtimestamp(ts).strftime('%H:%M:%S')
                timestamp = str(cur_date+'_'+cur_time)

                ### Resume writing recommendation
                st.subheader("**Resume Tips & Ideas💡**")
                resume_score = 0
                if 'Objective' in resume_text:
                    resume_score = resume_score+20
                    st.markdown('''<h5 style='text-align: left; color: #1ed760;'>[+] Awesome! You have added Objective</h4>''',unsafe_allow_html=True)
                else:
                    st.markdown('''<h5 style='text-align: left; color: #000000;'>[-] Please add your career objective, it will give your career intension to the Recruiters.</h4>''',unsafe_allow_html=True)

                if 'Declaration'  in resume_text:
                    resume_score = resume_score + 20
                    st.markdown('''<h5 style='text-align: left; color: #1ed760;'>[+] Awesome! You have added Delcaration/h4>''',unsafe_allow_html=True)
                else:
                    st.markdown('''<h5 style='text-align: left; color: #000000;'>[-] Please add Declaration. It will give the assurance that everything written on your resume is true and fully acknowledged by you</h4>''',unsafe_allow_html=True)

                if 'Hobbies' or 'Interests'in resume_text:
                    resume_score = resume_score + 20
                    st.markdown('''<h5 style='text-align: left; color: #1ed760;'>[+] Awesome! You have added your Hobbies</h4>''',unsafe_allow_html=True)
                else:
                    st.markdown('''<h5 style='text-align: left; color: #000000;'>[-] Please add Hobbies. It will show your persnality to the Recruiters and give the assurance that you are fit for this role or not.</h4>''',unsafe_allow_html=True)

                if 'Achievements' in resume_text:
                    resume_score += 20
                    st.markdown('''<h5 style='text-align: left; color: #1ed760;'>[+] Awesome! You have added Achievements</h5>''', unsafe_allow_html=True)
                else:
                    st.markdown('''<h5 style='text-align: left; color: #000000;'>[-] Please add Achievements. It shows your accomplishments and makes your resume stand out.</h5>''', unsafe_allow_html=True)

                if 'Projects' in resume_text:
                    resume_score += 20
                    st.markdown(
                        "<h5 style='text-align: left; color: #1ed760;'>[+] Awesome! You have added your Projects </h5>",
                        unsafe_allow_html=True)
                else:
                    st.markdown(
                        "<h5 style='text-align: left; color: #000000;'>[-] Please add Projects. It reflects your hands-on experience and problem-solving ability.</h5>",
                        unsafe_allow_html=True)

                # Display Resume Score
                st.subheader("**Resume Score 📝**")
                st.markdown(
                    f"<h2 style='text-align: center; color: #ff4b4b;'>{resume_score}</h2>",
                    unsafe_allow_html=True)
                st.markdown("### ✔️ *You can download your resume report from the Admin Panel*")

                # Progress bar for visual effect
                my_bar = st.progress(0)
                for percent_complete in range(resume_score + 1):
                    time.sleep(0.05)  # Reduced delay for a faster effect
                    my_bar.progress(percent_complete)

                # Final messages
                st.subheader("**Your Resume Score📝**")
                st.markdown(f"<h3 style='text-align: center; color: #d73b5c;'>{resume_score} / 100</h3>", unsafe_allow_html=True)

                st.balloons()

                # Insert data into database or storage (assumes insert_data function is defined)
                insert_data(
                    resume_data['name'],
                    resume_data['email'],
                    str(resume_score),
                    timestamp,
                    str(resume_data['no_of_pages']),
                    reco_field,
                    cand_level,
                    str(resume_data['skills']),
                    str(recommended_skills),
                    str(rec_course)
                )

                connection.commit()
            else:
                st.error('Something went wrong..')
    else:
        ## Admin Side
        st.success('Welcome to Admin Side')
        # st.sidebar.subheader('**ID / Password Required!**')

        ad_user = st.text_input("Username")
        ad_password = st.text_input("Password", type='password')
        if st.button('Login'):
            if ad_user == 'akshat' and ad_password == 'ak@04':
                st.success("Welcome Akshat !")
                # Display Data
                cursor.execute('''SELECT*FROM user_data''')
                data = cursor.fetchall()
                st.header("**User's Data**")
                df = pd.DataFrame(data, columns=['ID', 'Name', 'Email', 'Resume Score', 'Timestamp', 'Total Page',
                                                 'Predicted Field', 'User Level', 'Actual Skills', 'Recommended Skills',
                                                 'Recommended Course'])
                st.dataframe(df)
                st.markdown(get_table_download_link(df,'User_Data.csv','Download Report'), unsafe_allow_html=True)
                ## Admin Side Data
                query = 'select * from user_data;'
                plot_data = pd.read_sql(query, connection)
                
                plot_data['Predicted_Field'] = plot_data['Predicted_Field'].apply(lambda x: x.decode('utf-8') if isinstance(x, bytes) else x)
                plot_data['User_level'] = plot_data['User_level'].apply(lambda x: x.decode('utf-8') if isinstance(x, bytes) else x)

                field_counts = plot_data['Predicted_Field'].value_counts().reset_index()
                field_counts.columns = ['Field', 'Count']

                # Pie chart for Predicted Field Recommendations
                st.subheader("**Pie-Chart for Predicted Field Recommendation**")
                fig = px.pie(field_counts, values='Count', names='Field',
                            title='Predicted Field according to the Skills')
                st.plotly_chart(fig)

                # Pie chart for User Experience Level
                level_counts = plot_data['User_level'].value_counts().reset_index()
                level_counts.columns = ['Level', 'Count']

                st.subheader("**Pie-Chart for User's Experienced Level**")
                fig = px.pie(level_counts, values='Count', names='Level',
                            title="Pie-Chart for User's Experienced Level")
                st.plotly_chart(fig)

            else:
                st.error("Wrong ID & Password Provided")
run()
