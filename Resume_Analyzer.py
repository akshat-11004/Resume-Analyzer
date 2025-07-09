import streamlit as st
st.set_page_config(
    # page_title="CareerCompass Resume Analyzer",
    page_icon='./Logo/logo2.png',
    layout="wide",
    initial_sidebar_state="expanded"
)
from functools import partial
if 'admin_logged_in' not in st.session_state:
    st.session_state.admin_logged_in = False

import pandas as pd
import base64, random
import re
import time, datetime
import nltk
nltk.download('stopwords')  
nltk.download('punkt')      # Download tokenizer data (often needed)
nltk.download('averaged_perceptron_tagger')  # For POS tagging
nltk.download('maxent_ne_chunker')  # For named entity recognition
nltk.download('words') 

from pyresparser import ResumeParser
from pdfminer.layout import LAParams, LTTextBox
from pdfminer.pdfpage import PDFPage
from pdfminer.pdfinterp import PDFResourceManager
from pdfminer.pdfinterp import PDFPageInterpreter
from pdfminer.converter import TextConverter
import io, random
from streamlit_tags import st_tags
from PIL import Image
import pymysql
from Courses import ds_course, web_course, android_course, ios_course, uiux_course
import plotly.express as px



# Custom CSS for styling
def local_css(file_name):
    with open(file_name) as f:
        st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)

local_css("styles.css")

def get_table_download_link(df, filename, text):
    csv = df.to_csv(index=False)
    b64 = base64.b64encode(csv.encode()).decode()
    href = f'<a href="data:file/csv;base64,{b64}" download="{filename}" class="download-link">{text}</a>'
    return href

def pdf_reader(file):
    resource_manager = PDFResourceManager()
    fake_file_handle = io.StringIO()
    converter = TextConverter(resource_manager, fake_file_handle, laparams=LAParams())
    page_interpreter = PDFPageInterpreter(resource_manager, converter)
    with open(file, 'rb') as fh:
        for page in PDFPage.get_pages(fh, caching=True, check_extractable=True):
            page_interpreter.process_page(page)
        text = fake_file_handle.getvalue()
    converter.close()
    fake_file_handle.close()
    return text

def show_pdf(file_path):
    with open(file_path, "rb") as f:
        base64_pdf = base64.b64encode(f.read()).decode('utf-8')
    pdf_display = f'<iframe src="data:application/pdf;base64,{base64_pdf}" width="700" height="1000" type="application/pdf" class="pdf-viewer"></iframe>'
    st.markdown(pdf_display, unsafe_allow_html=True)

def course_recommender(course_list):
    st.subheader("📚 Recommended Courses & Certifications")
    c = 0
    rec_course = []
    no_of_reco = st.slider('Select number of recommendations:', 1, 10, 4)
    random.shuffle(course_list)
    for c_name, c_link in course_list:
        c += 1
        st.markdown(f"({c}) [{c_name}]({c_link})")
        rec_course.append(c_name)
        if c == no_of_reco:
            break
    return rec_course

# Database Connection
connection = pymysql.connect(host='localhost', user='root', password='akshat@2004', db='CV')
cursor = connection.cursor()



def insert_data(name, email, res_score, timestamp, no_of_pages, reco_field, cand_level, skills, recommended_skills, courses):
    DB_table_name = 'user_data'
    email = email if email else 'not_provided@example.com'
    insert_sql = "INSERT INTO " + DB_table_name + """
    VALUES (0,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)"""
    rec_values = (
        name, 
        email, 
        str(res_score), 
        timestamp, 
        str(no_of_pages), 
        str(reco_field),  # Ensure string
        str(cand_level),   # Ensure string
        str(skills),       # Ensure string
        str(recommended_skills), 
        str(courses)
    )
    cursor.execute(insert_sql, rec_values)
    connection.commit()
        
def detect_responsibility_section(text):
    """Advanced detection of responsibility/leadership content"""
    # Check for section headers
    section_pattern = re.compile(
        r'(?:^|\n)\s*(.*?(responsibilit|leadership|role|activity|volunteer).*?)\s*(?:\n|:)',
        re.IGNORECASE
    )
    
    # Check for bullet points indicating responsibilities
    bullet_pattern = re.compile(
        r'(•|\d+\.|[-*])\s*(.*?(led|organized|managed|headed|founded|coordinated).*?)\n',
        re.IGNORECASE
    )
    
    return bool(section_pattern.search(text) or bullet_pattern.search(text))

# Add to your resume processing code
def extract_phone_number(text):
    """
    Comprehensive phone number extractor that handles:
    - International numbers with country codes (+91, +1, etc.)
    - Indian mobile numbers (10 digits with/without country code)
    - Various formatting styles (spaces, hyphens, parentheses)
    - Numbers with or without country codes
    """
    # Comprehensive phone number regex pattern
    phone_regex = r'''
        (?:\+?(\d{1,3}))?                # Optional country code (1-3 digits)
        [\s.-]?                           # Optional separator
        (?:\(?\d{1,4}\)?[\s.-]?)?        # Optional area code
        (\d{5,})                          # Main number (5+ digits)
        (?:[\s.-]?\d{1,5})*               # Optional extensions
        (?![\d])                          # Negative lookahead for more digits
    '''
    
    matches = re.finditer(phone_regex, text, re.VERBOSE)
    
    for match in matches:
        country_code = match.group(1) or ""
        main_number = match.group(2)
        
        # Remove all non-digit characters
        clean_number = re.sub(r'[^\d]', '', main_number)
        
        # Skip if the number is too short
        if len(clean_number) < 7:
            continue
            
        # Handle country code
        if country_code:
            full_number = f"+{country_code}{clean_number}"
            
            # Special formatting for Indian numbers
            if country_code == '91' and len(clean_number) == 10:
                return f"+{country_code} {clean_number[:5]} {clean_number[5:]}"
            return full_number
        else:
            # Handle local numbers (10 digits assumed to be Indian)
            if len(clean_number) == 10:
                return f"+91 {clean_number[:5]} {clean_number[5:]}"
            return clean_number
            
    return None


def run():
    col1 = st.container()
    with col1:
        img = Image.open('./Logo/logo4(1).png')
        img = img.resize((500,150))  # Adjust size as needed
        st.image(img)  # Center the logo    

    # Add some vertical space
    st.write("")  # Empty line for spacing
    st.write("")  # Add more empty lines if needed

    # Second row - Title and subtitle
    col2 = st.container()
    with col2:
        st.title("Your Resume Analyzer")
        st.markdown("""
        <div class="subheader">
            AI-powered resume analysis for your career growth
        </div>
        """, unsafe_allow_html=True)

    choice = st.sidebar.selectbox("Choose your login type:", ["User", "Admin"], index=0)
    st.sidebar.markdown("""
    <div class="sidebar-footer">
        <hr>
        <p>Developed with by Akshat Kadia</p>
        <a href="https://www.linkedin.com/in/akshat-kadia-588a85266/" target="_blank">Connect on LinkedIn</a>
    </div>
    """, unsafe_allow_html=True)

    # Create Database
    db_sql = """CREATE DATABASE IF NOT EXISTS CV;"""
    cursor.execute(db_sql)

    # Create table
    DB_table_name = 'user_data'
    table_sql = f"""CREATE TABLE IF NOT EXISTS {DB_table_name} (
            ID INT NOT NULL AUTO_INCREMENT,
            Name varchar(500) NOT NULL,  # Extra ) after 500
            Email_ID VARCHAR(500) NOT NULL,  # Extra ) after 500
            resume_score VARCHAR(8) NOT NULL,
            Timestamp VARCHAR(50) NOT NULL,
            Page_no VARCHAR(5) NOT NULL,
            Predicted_Field VARCHAR(500) NOT NULL,
            User_level VARCHAR(500) NOT NULL,
            Actual_skills VARCHAR(500) NOT NULL,
            Recommended_skills VARCHAR(500) NOT NULL,
            Recommended_courses VARCHAR(500) NOT NULL,
            PRIMARY KEY (ID)
        );"""
    cursor.execute(table_sql)

    if choice == 'User':
        st.markdown("""
        <div class="upload-header">
            <h3 style = 'color:#8d0606;'>Upload Your Resume for Analysis</h3>
            <p>Get personalized career recommendations based on your skills and experience</p>
        </div>
        """, unsafe_allow_html=True)
        
        pdf_file = st.file_uploader("Choose your resume (PDF only)", type=["pdf"], help="Upload a PDF version of your resume for analysis")

        if pdf_file is not None:
            with st.spinner('Analyzing your resume...'):
                time.sleep(2)
                
            save_image_path = './Uploaded_Resumes/' + pdf_file.name
            with open(save_image_path, "wb") as f:
                f.write(pdf_file.getbuffer())
                
            # Display PDF
            st.markdown("### Resume Preview")
            show_pdf(save_image_path)
            
            # Parse Resume
            resume_data = ResumeParser(save_image_path).get_extracted_data()
            if resume_data:
                resume_text = pdf_reader(save_image_path)
                # Personal Information
                st.markdown("""
                <div class="section-header">
                    <h3>Personal Information</h3>
                </div>
                """, unsafe_allow_html=True)
                if 'name' in resume_data:
                    # Check if extracted name looks like a job title
                    job_title_terms = ['developer', 'engineer', 'designer', 'manager', 'analyst','chef']
                    if any(term in resume_data['name'].lower() for term in job_title_terms):
                        # Look for proper name (all caps, between other fields)
                        name_match = re.search(r'(?:^|\n)([A-Z][A-Z\s]+[A-Z])(?:\n|$)', resume_text)
                        if name_match:
                            resume_data['name'] = name_match.group(1).strip()

                # 2. Fix phone number extraction
                if 'mobile_number' not in resume_data or not resume_data['mobile_number']:
                    resume_data['mobile_number'] = extract_phone_number(resume_text)
                    
                info_col1, info_col2 = st.columns(2)
                with info_col1:
                    st.info(f"👤 **Name:** {resume_data.get('name', 'Not specified')}")
                    st.info(f"📧 **Email:** {resume_data.get('email', 'Not specified')}")
                with info_col2:
                    st.info(f"📱 **Contact:** {resume_data.get('mobile_number', 'Not specified')}")
                    st.info(f"📄 **Pages:** {resume_data.get('no_of_pages', 'Not specified')}")

                # Experience Level
                cand_level = ''
                if resume_data['no_of_pages'] == 1:
                    cand_level = "Fresher"
                    st.markdown("""
                    <div class="level-fresher">
                        <h4>Career Level: Fresher</h4>
                        <p>You're just starting your professional journey. Focus on building strong fundamentals.</p>
                    </div>
                    """, unsafe_allow_html=True)
                elif resume_data['no_of_pages'] == 2:
                    cand_level = "Intermediate"
                    st.markdown("""
                    <div class="level-intermediate">
                        <h4>Career Level: Intermediate</h4>
                        <p>You have some professional experience. Time to specialize and deepen your skills.</p>
                    </div>
                    """, unsafe_allow_html=True)
                elif resume_data['no_of_pages'] >= 3:
                    cand_level = "Experienced"
                    st.markdown("""
                    <div class="level-experienced">
                        <h4>Career Level: Experienced</h4>
                        <p>You have substantial professional experience. Consider leadership opportunities.</p>
                    </div>
                    """, unsafe_allow_html=True)

                # Skills Analysis
                st.markdown("""
                <div class="section-header">
                    <h3>Skills Analysis</h3>
                </div>
                """, unsafe_allow_html=True)
                
                st.markdown("### Your Current Skills")
                keywords = st_tags(
                    label='',
                    text='These skills were identified in your resume:',
                    value=resume_data['skills'],
                    key='1'
                )

                # 1. First, create a domain scoring system
                domain_scores = {
                    'Data Science': 0,
                    'Web Development': 0,
                    'Android Development': 0,
                    'iOS Development': 0,
                    'UI/UX Design': 0
                }

                # 2. Define skill weights (core skills get higher weights)
                skill_weights = {
                    'Data Science': {
                        'tensorflow': 2, 'pytorch': 2, 'machine learning': 2,
                        'deep learning': 2, 'flask': 1, 'streamlit': 1
                    },
                    'Web Development': {
                        'react': 2, 'django': 2, 'node js': 2,
                        'javascript': 2, 'flask': 1, 'angular': 1
                    },
                    'Android Development': {
                        'android': 2, 'flutter': 2, 'kotlin': 2,
                        'android development': 2, 'xml': 1
                    },
                    'iOS Development': {
                        'ios': 2, 'swift': 2, 'xcode': 2,
                        'ios development': 2, 'cocoa': 1
                    },
                    'UI/UX Design': {
                        'figma': 2, 'adobe xd': 2, 'ui': 2,
                        'ux': 2, 'prototyping': 1, 'wireframes': 1
                    }
                }

                recommended_skills = []  # Empty list as default
                reco_field = 'General'   # Default field
                rec_course = []          # Empty course list
                
                # 3. Score each domain based on resume skills
                for skill in resume_data['skills']:
                    skill_lower = skill.lower()
                    for domain, weights in skill_weights.items():
                        if skill_lower in weights:
                            domain_scores[domain] += weights[skill_lower]

                # 4. Get the top scoring domain
                if domain_scores:
                    reco_field = max(domain_scores, key=domain_scores.get)
                    top_score = domain_scores[reco_field]
                    
                    # Only recommend if significant match found
                    if top_score > 0:
                        st.success(f"🎯 Our analysis suggests you're pursuing {reco_field} roles (confidence: {top_score}/10 points)")
                        
                        # Get recommendations based on domain
                        recommended_skills = {
                            'Data Science': ['Data Visualization', 'Predictive Analysis', 'Statistical Modeling', 
                                            'ML Algorithms', 'Scikit-learn', 'TensorFlow', 'PyTorch'],
                            'Web Development': ['React', 'Django', 'Node.js', 'JavaScript', 'REST APIs', 'GraphQL'],
                            'Android Development': ['Kotlin', 'Jetpack Compose', 'Android SDK', 'Firebase', 'Material Design'],
                            'iOS Development': ['SwiftUI', 'Combine', 'Core Data', 'ARKit', 'App Store Guidelines'],
                            'UI/UX Design': ['User Research', 'Prototyping', 'Accessibility', 'Design Systems', 'Figma Plugins']
                        }.get(reco_field, [])
                        
                        # Show recommendations
                        recommended_keywords = st_tags(
                            label='### Recommended Skills to Add',
                            text=f'These {reco_field} skills will make you more competitive',
                            value=recommended_skills,
                            key='skills_rec'
                        )
                        
                        # Show course recommendations
                        course_map = {
                            'Data Science': ds_course,
                            'Web Development': web_course,
                            'Android Development': android_course,
                            'iOS Development': ios_course,
                            'UI/UX Design': uiux_course
                        }
                        rec_course = course_recommender(course_map[reco_field])
                    else:
                        st.warning("🔍 Unable to determine a clear career field from skills")
                else:
                    recommended_skills = ['Add technical skills to your resume']
                    st.warning("⚠️ No skills found in resume")
                    
                    
                # Resume Quality Assessment
                st.markdown("""
                <div class="section-header">
                    <h3>Resume Quality Assessment</h3>
                </div>
                """, unsafe_allow_html=True)
                
                ts = time.time()
                cur_date = datetime.datetime.fromtimestamp(ts).strftime('%Y-%m-%d')
                cur_time = datetime.datetime.fromtimestamp(ts).strftime('%H:%M:%S')
                timestamp = str(cur_date+'_'+cur_time)

                resume_score = 0
                
                # Resume Section Checks
                check_col1, check_col2 = st.columns(2)
                
                def preprocess_text(text):
                    """Normalize text for reliable matching"""
                    # Convert to lowercase and handle special cases
                    text = text.lower()
                    # Replace common abbreviations with full forms
                    text = re.sub(r'\bproj\b', 'project', text)
                    text = re.sub(r'\bexp\b', 'experience', text)
                    text = re.sub(r'\bresp\b', 'responsibility', text)
                    text = re.sub(r'\bobj\b', 'objective', text)
                    text = re.sub(r'\bach\b', 'achievement', text)
                    # Handle hyphenated and spaced words
                    text = re.sub(r'(\w+)-(\w+)', r'\1 \2', text)  # "work-history" → "work history"
                    text = re.sub(r'\s+', ' ', text)  # Collapse multiple spaces
                    return text

                def detect_section(text, section_data):
                    """
                    Advanced section detection with:
                    - Abbreviation expansion
                    - Hyphen/slash handling
                    - Line break tolerance
                    - Spaceless matching
                    """
                    processed_text = preprocess_text(text)
                    spaceless_text = ''.join(processed_text.split())
                    
                    # Check standard keywords
                    for keyword in section_data['keywords']:
                        # Original keyword
                        if keyword in processed_text:
                            return True
                        # Hyphenated version
                        if keyword.replace(' ', '-') in text.lower():
                            return True
                        # Spaceless version
                        if ''.join(keyword.split()) in spaceless_text:
                            return True
                    
                    # Check regex patterns
                    for pattern in section_data.get('regex_patterns', []):
                        if re.search(pattern, text, re.IGNORECASE):
                            return True
                    
                    # Check abbreviations
                    for abbr_pattern in section_data.get('abbr_patterns', []):
                        if re.search(abbr_pattern, text, re.IGNORECASE):
                            return True
                    
                    return False

                SECTION_CONFIGS = [
                    {
                        'keywords': ['objective', 'career objective', 'professional summary', 
                                'summary', 'areas of interest', 'career interests','area(s) of interest',
                                'professional profile', 'career goals'],
                        'regex_patterns': [
                            r'career\s*(goal|obj)',
                            r'prof(essional)?\s*(sum|obj)'
                        ],
                        'abbr_patterns': [
                            r'obj\b',
                            r'prof\s*(sum|obj)'
                        ],
                        'section_name': 'Career Objective/Summary',
                        'score': 10,
                        'importance_msg': 'A strong career objective helps recruiters quickly understand your professional goals and alignment with the position.'
                    },
                    {
                        'keywords': ['experience', 'work history', 'professional experience',
                                'employment history', 'work experience', 'career history',
                                'professional background', 'relevant experience','internship'],
                        'regex_patterns': [
                            r'work[\s-]*(hist|exp)',
                            r'prof(essional)?[\s-]*exp'
                        ],
                        'abbr_patterns': [
                            r'exp\b',
                            r'work\s*hist'
                        ],
                        'section_name': 'Professional Experience',
                        'score': 25,
                        'importance_msg': 'Work experience demonstrates your practical skills and shows employers you can apply knowledge in real-world situations.'
                    },
                    {
                        'keywords': ['hobbies', 'interests', 'personal interests',
                                'extracurricular', 'activities', 'personal activities',
                                'leisure activities', 'other interests'],
                        'regex_patterns': [
                            r'(personal|other)[\s-]*interests?',
                            r'extra[\s-]*curricular'
                        ],
                        'section_name': 'Personal Interests',
                        'score': 10,
                        'importance_msg': 'Personal interests help showcase your personality and can demonstrate valuable soft skills beyond your technical qualifications.'
                    },
                    {
                        'keywords': ['achievements', 'accomplishments', 'awards',
                                'honors', 'recognitions', 'certifications',
                                'successes', 'key achievements'],
                        'regex_patterns': [
                            r'ach(ievements?)?\b',
                            r'awards?|honou?rs?'
                        ],
                        'abbr_patterns': [
                            r'ach\b',
                            r'awards?'
                        ],
                        'section_name': 'Achievements',
                        'score': 10,
                        'importance_msg': 'Achievements provide concrete evidence of your capabilities and help you stand out from other candidates.'
                    },
                    {
                        'keywords': ['projects', 'academic projects', 'professional projects',
                                'research projects', 'personal projects', 'selected projects',
                                'project experience', 'project work'],
                        'regex_patterns': [
                            r'proj(ects?)?[\s-]*(exp|work)?',
                            r'research[\s-]*proj'
                        ],
                        'abbr_patterns': [
                            r'proj\b',
                            r'acad\s*proj'
                        ],
                        'section_name': 'Projects',
                        'score': 25,
                        'importance_msg': 'Projects demonstrate your problem-solving abilities and show how you apply your skills to create tangible results.'
                    },
                    {
                        'keywords': ['positions of responsibility', 'responsibilities', 'roles',
                                'leadership', 'volunteer work', 'extracurricular activities',
                                'committee membership', 'organizational roles', 'team leadership',
                                'student organizations', 'additional activities', 'college societies',
                                'club participation', 'leadership experience'],
                        'regex_patterns': [
                            r'pos(itions?)?[\s-]*of[\s-]*resp',
                            r'leadership[\s-]*(roles?|exp)',
                            r'volunteer|extra[\s-]*curricular'
                        ],
                        'abbr_patterns': [
                            r'pos\s*resp',
                            r'lead\s*exp'
                        ],
                        'section_name': 'Positions of Responsibility',
                        'score': 20,
                        'importance_msg': 'Leadership roles show your ability to take initiative, manage responsibilities, and work effectively with others.'
                    }
                ]

                def check_resume_section(text, section_data, current_score):
                    """Enhanced section checking with all new features"""
                    if detect_section(text, section_data):
                        updated_score = current_score + section_data['score']
                        st.success(f"✅ {section_data['section_name']}: Found")
                    else:
                        updated_score = current_score
                        st.warning(f"⚠️ {section_data['section_name']}: Not found")
                        # with st.expander("Why is this important?"):
                        st.info(section_data['importance_msg'])
                    return updated_score

                # Usage example:
                resume_score = 0
                with check_col1:
                    resume_score = check_resume_section(resume_text, SECTION_CONFIGS[0], resume_score)  # Objective
                    resume_score = check_resume_section(resume_text, SECTION_CONFIGS[1], resume_score)  # Experience

                with check_col2:
                    resume_score = check_resume_section(resume_text, SECTION_CONFIGS[2], resume_score)  # Interests
                    resume_score = check_resume_section(resume_text, SECTION_CONFIGS[3], resume_score)  # Achievements

                resume_score = check_resume_section(resume_text, SECTION_CONFIGS[4], resume_score)  # Projects
                resume_score = check_resume_section(resume_text, SECTION_CONFIGS[5], resume_score)  # Positions
                               
                                
                st.markdown("""
                <div class="section-header">
                    <h3>Resume Score</h3>
                </div>
                """, unsafe_allow_html=True)
                
                score_col1, score_col2, score_col3 = st.columns([1,2,1])
                with score_col2:
                    st.markdown(f"""
                    <div class="score-display">
                        <h1>{resume_score}/100</h1>
                        <p>Overall Resume Quality</p>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    # Score Interpretation
                    if resume_score >= 80:
                        st.balloons()
                        st.success("Excellent resume! You're well-prepared for job applications.")
                    elif resume_score >= 60:
                        st.success("Good resume! Some improvements could make it even stronger.")
                    elif resume_score >= 40:
                        st.warning("Fair resume. Consider implementing the recommendations.")
                    elif resume_score >= 20:
                        st.warning("Basic resume. Significant improvements needed for competitive applications.")
                    else:
                        st.error("Lots of improvements needed")
                    
                    # Progress bar
                    st.progress(resume_score)

                # Data Storage
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
                
                # Additional Recommendations
                st.markdown("""
                <div class="section-header">
                    <h3>Additional Career Resources</h3>
                </div>
                """, unsafe_allow_html=True)
                
                tab1, tab2 = st.tabs(["📝 Resume Writing Tips", "🎥 Interview Preparation"])
                
                with tab1:
                    st.markdown("""
                    - Use action verbs to describe your experience (e.g., 'developed', 'managed', 'implemented')
                    - Quantify achievements where possible (e.g., 'Increased sales by 20%')
                    - Keep your resume to 1-2 pages maximum
                    - Use a clean, professional layout
                    - Tailor your resume for each job application
                    """)
                
                with tab2:
                    st.markdown("""
                    - Research the company thoroughly before your interview
                    - Prepare examples using the STAR method (Situation, Task, Action, Result)
                    - Practice common interview questions
                    - Prepare thoughtful questions to ask the interviewer
                    - Dress professionally and arrive early
                    """)
                
            else:
                st.error("Unable to process your resume. Please ensure it's a valid PDF file with readable text.")

    else:  # Admin Section
    # Initialize session state if not exists
        if 'admin_logged_in' not in st.session_state:
            st.session_state.admin_logged_in = False

        # Display header always
        st.markdown("""
        <div class="admin-header">
            <h2>Admin Dashboard</h2>
            <p>Access candidate analytics and system data</p>
        </div>
        """, unsafe_allow_html=True)

        # Login form if not logged in
        if not st.session_state.admin_logged_in:
            login_col1, login_col2 = st.columns(2)
            with login_col1:
                ad_user = st.text_input("Admin Username")
            with login_col2:
                ad_password = st.text_input("Admin Password", type='password')
            
            if st.button('Login', key='admin_login'):
                if ad_user == 'akshat' and ad_password == 'ak@04':
                    st.session_state.admin_logged_in = True
                    st.rerun()
                else:
                    st.error("Invalid credentials. Please try again.")
        
        # Show admin content if logged in
        if st.session_state.admin_logged_in:
            # Add logout button at top
            if st.button("🔒 Logout"):
                st.session_state.admin_logged_in = False
                st.rerun()
                return
            
            st.success("Welcome back, Admin!")

            # Candidate Data Overview
            st.markdown("### Candidate Database")
            cursor.execute('''SELECT * FROM user_data''')
            data = cursor.fetchall()

            # Create DataFrame with proper decoding
            decoded_data = []
            for row in data:
                decoded_row = list(row)
                # Decode fields that might have been stored as bytes
                for i in range(len(decoded_row)):
                    if isinstance(decoded_row[i], bytes):
                        try:
                            decoded_row[i] = decoded_row[i].decode('utf-8')
                        except:
                            decoded_row[i] = str(decoded_row[i])
                decoded_data.append(decoded_row)

            df = pd.DataFrame(decoded_data, columns=['ID', 'Name', 'Email', 'Resume Score', 'Timestamp', 'Total Page',
                                                'Predicted Field', 'User Level', 'Actual Skills', 'Recommended Skills',
                                                'Recommended Course'])
            st.dataframe(df)
            
            # Download option
            st.markdown(get_table_download_link(df, 'Candidate_Data.csv', '📥 Download Full Report'), unsafe_allow_html=True)
            
            # Database Management Section
            # with st.expander("⚠️ Database Management", expanded=False):
            #     if st.button("🚨 Clean Entire Database"):
            #             # Double confirmation
            #             confirm = st.checkbox("I understand this will delete ALL data permanently")
            #             if confirm:
            #                 try:
            #                     cursor.execute("TRUNCATE TABLE user_data")
            #                     connection.commit()
            #                     st.session_state.data_cleaned = True
            #                     st.success("Database successfully cleaned!")
            #                     st.rerun()
            #                 except Exception as e:
            #                     st.error(f"Error: {str(e)}")
                    
            
            # Analytics Section
            st.markdown("## Analytics Dashboard")
            
            query = 'SELECT * FROM user_data;'
            plot_data = pd.read_sql(query, connection)
            
            # Clean data
            plot_data['Predicted_Field'] = plot_data['Predicted_Field'].apply(lambda x: x.decode('utf-8') if isinstance(x, bytes) else x)
            plot_data['User_level'] = plot_data['User_level'].apply(lambda x: x.decode('utf-8') if isinstance(x, bytes) else x)
            
            # Field Distribut   ion
            # st.markdown("### Career Field Distribution")
            field_counts = plot_data['Predicted_Field'].value_counts().reset_index()
            field_counts.columns = ['Field', 'Count']
            
            fig1 = px.pie(field_counts, values='Count', names='Field',
                        title='Career Field Recommendations',
                        color_discrete_sequence=px.colors.sequential.Agsunset)
            st.plotly_chart(fig1, use_container_width=True)
            
            # Experience Level Distribution
            # st.markdown("### Candidate Experience Levels")
            level_counts = plot_data['User_level'].value_counts().reset_index()
            level_counts.columns = ['Level', 'Count']
            
            fig2 = px.bar(level_counts, x='Level', y='Count',
                        title='Distribution of Candidate Experience Levels',
                        color='Level',
                        color_discrete_sequence=px.colors.sequential.Plasma)
            st.plotly_chart(fig2, use_container_width=True)
            
            # Score Distribution
            # st.markdown("### Resume Score Distribution")
            fig3 = px.histogram(plot_data, x='resume_score',
                            title='Distribution of Resume Quality Scores',
                            nbins=10,
                            color_discrete_sequence=['#2a9df4'])
            fig3.update_layout(xaxis_title="Resume Score", yaxis_title="Number of Candidates")
            st.plotly_chart(fig3, use_container_width=True)

run()