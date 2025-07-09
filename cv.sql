CREATE DATABASE CV;
USE CV;

CREATE TABLE IF NOT EXISTS user_data (
    ID INT NOT NULL AUTO_INCREMENT,
    Name varchar(500) NOT NULL,
    Email_ID VARCHAR(500) NOT NULL,
    resume_score VARCHAR(8) NOT NULL,
    Timestamp VARCHAR(50) NOT NULL,
    Page_no VARCHAR(5) NOT NULL,
    Predicted_Field VARCHAR(500) NOT NULL,
    User_level VARCHAR(500) NOT NULL,
    Actual_skills VARCHAR(500) NOT NULL,
    Recommended_skills VARCHAR(500) NOT NULL,
    Recommended_courses VARCHAR(500) NOT NULL,
    PRIMARY KEY (ID)
);