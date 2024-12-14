job_title = [
    ("Core staff", "Core staff"),
    ("Drivers", "Drivers"),
    ("Cleaning staff", "Cleaning staff"),
    ("High skill tech staff", "High skill tech staff"),
    ("Sales staff", "Sales staff"),
    ("Laborers", "Laborers"),
    ("Accountants ", "Accountants"),
    ("Managers", "Managers"),
    ("Security staff", "Security staff"),
    ("Cooking staff", "Cooking staff"),
    ("Medicine staff", "Medicine staff"),
    ("IT staff", "IT staff"),
    ("Realty agents", "Realty agents"),
    ("Secretaries", "Secretaries"),
    ("Waiters/barmen staff", "Waiters/barmen staff"),
    ("Private service staff", "Private service staff"),
    ("HR staff            ", "HR staff"),
    ("Low-skill Laborers   ", "Low-skill Laborers"),
]
housing_type = [
    ("House / apartment", "House / Apartment"),
    ("With parents", "With Parents"),
    ("Municipal apartment", "Municipal Apartment"),
    ("Co-op apartment", "Co-op Apartment"),
    ("Rented apartment", "Rented Apartment"),
    ("Office apartment", "Office Apartment"),
]
family_status = [
    ("Married", "Married"),
    ("Separated", "Separated"),
    ("Civil marriage", "Civil Marriage"),
    ("Single / not married", "Single / Not Married"),
    ("Widow", "Widow"),
]
education_type = [
    ("Secondary / secondary special", "Secondary / Secondary Special"),
    ("Higher education", "Higher Education"),
    ("Incomplete higher", "Incomplete Higher"),
    ("Lower secondary", "Lower Secondary"),
    ("Academic degree", "Academic Degree"),
]
income_type = [
    ("Commercial associate", "Commercial Associate"),
    ("Working", "Working"),
    ("State servant", "State several"),
    ("Student", "Student"),
    ("Pensioner", "Pensioner"),
]

columns = [
    "Applicant_Gender",
    "Owned_Car",
    "Owned_Realty",
    "Total_Children",
    "Total_Income",
    "Income_Type",
    "Education_Type",
    "Family_Status",
    "Housing_Type",
    "Job_Title",
    "Total_Family_Members",
    "Applicant_Age",
    "Years_of_Working",
]

bad_user_data = """
**User's Shared Features**:
- Gender: Male
- Owned Car: No
- Owned Realty: No
- Total Children: 0
- Total Income (Annually): ฿200,000
- Income Type: Working
- Education Type: Secondary/Secondary Special
- Family Status: Single
- Housing Type: With Parents
- Job Title: Security Staff
- Total Family Members: 3
- Applicant Age: 23
- Years of Working: 2

**Feature Importance**:
- Years of Working: 637
- Total Income: 600
- Applicant Age: 581
- Owned Realty: 204
- Total Children: 176
- Total Family Members: 174
- Owned Car: 169

**Response**:
YOUR TURN (DO NOT SUGGEST ABOUT CREDIT HISTORY AND SCORE)
"""

good_user_data = """
**User's Shared Features**:
- Gender: Male
- Owned Car: No
- Owned Realty: No
- Total Children: 0
- Total Income (Annually): ฿200,000
- Income Type: Working
- Education Type: Secondary/Secondary Special
- Family Status: Single
- Housing Type: With Parents
- Job Title: Security Staff
- Total Family Members: 3
- Applicant Age: 23
- Years of Working: 2

**Feature Importance**:
- Years of Working: 637
- Total Income: 600
- Applicant Age: 581
- Owned Realty: 204
- Total Children: 176
- Total Family Members: 174
- Owned Car: 169

**Response**:
YOUR TURN (DO NOT SUGGEST ABOUT CREDIT HISTORY AND SCORE)
"""

few_shot_prompt = """Generate a reasoned response for a user's unsuccessful credit card application with actionable recommendations for improvement.

EXAMPLE 1: (Index 275)
**User's Shared Features**:
- Gender: Female
- Owned Car: Yes
- Owned Realty: Yes
- Total Children: 0
- Total Income (Annually): ฿637,500
- Income Type: Working
- Education Type: Lower Secondary
- Family Status: Married
- Housing Type: House/apartment
- Job Title: High Skill Tech Staff
- Total Family Members: 2
- Applicant Age: 45
- Years of Working: 2

**Feature Importance**:
- Years of Working: 637
- Total Income: 600
- Applicant Age: 581
- Owned Realty: 204
- Total Children: 176
- Total Family Members: 174
- Owned Car: 169

**Response**:
Your credit card application was unsuccessful because your limited work experience (2 years) significantly impacted the evaluation. Additionally, while your annual income is above average, increasing your years of work experience to at least 3-5 years would improve your chances. Strengthening your financial stability by maintaining consistent employment or pursuing opportunities for additional income can also enhance your eligibility.

---

EXAMPLE 2: (Index 356)
**User's Shared Features**:
- Gender: Male
- Owned Car: Yes
- Owned Realty: No
- Total Children: 1
- Total Income (Annually): ฿2,125,000
- Income Type: Working
- Education Type: Higher Education
- Family Status: Married
- Housing Type: House/apartment
- Job Title: Core Staff
- Total Family Members: 3
- Applicant Age: 27
- Years of Working: 2

**Feature Importance**:
- Years of Working: 637
- Total Income: 600
- Applicant Age: 581
- Owned Realty: 204
- Total Children: 176
- Total Family Members: 174
- Owned Car: 169


**Response**:
Your application was unsuccessful because of limited work experience (2 years). Although your income is well above the threshold, the lack of property ownership (`Owned Realty: No`) also lowered your creditworthiness. Consider increasing your work experience and exploring the option of acquiring real estate assets, as this will demonstrate greater financial stability.

---

EXAMPLE 3: (Index 22040)
**User's Shared Features**:
- Gender: Male
- Owned Car: Yes
- Owned Realty: No
- Total Children: 1
- Total Income (Annually): ฿531,250
- Income Type: Working
- Education Type: Secondary/Secondary Special
- Family Status: Separated
- Housing Type: Co-op apartment
- Job Title: Laborers
- Total Family Members: 2
- Applicant Age: 33
- Years of Working: 4

**Feature Importance**:
- Years of Working: 637
- Total Income: 600
- Applicant Age: 581
- Owned Realty: 204
- Total Children: 176
- Total Family Members: 174
- Owned Car: 169

**Response**:
Your credit card application was declined due to insufficient income. Increasing your annual income to at least $150,000, combined with steady employment, would significantly improve your eligibility. Additionally, acquiring real estate or demonstrating stronger financial assets could positively impact future applications.
"""

bad_user_data = """
**User's Shared Features**:
- Gender: Male
- Owned Car: No
- Owned Realty: No
- Total Children: 0
- Total Income (Annually): ฿200,000
- Income Type: Working
- Education Type: Secondary/Secondary Special
- Family Status: Single
- Housing Type: With Parents
- Job Title: Security Staff
- Total Family Members: 3
- Applicant Age: 23
- Years of Working: 2

**Feature Importance**:
- Years of Working: 637
- Total Income: 600
- Applicant Age: 581
- Owned Realty: 204
- Total Children: 176
- Total Family Members: 174
- Owned Car: 169

**Response**:
YOUR TURN (DO NOT SUGGEST ABOUT CREDIT HISTORY AND SCORE)
"""

good_user_data = """
**User's Shared Features**:
- Gender: Male
- Owned Car: No
- Owned Realty: No
- Total Children: 0
- Total Income (Annually): ฿200,000
- Income Type: Working
- Education Type: Secondary/Secondary Special
- Family Status: Single
- Housing Type: With Parents
- Job Title: Security Staff
- Total Family Members: 3
- Applicant Age: 23
- Years of Working: 2

**Feature Importance**:
- Years of Working: 637
- Total Income: 600
- Applicant Age: 581
- Owned Realty: 204
- Total Children: 176
- Total Family Members: 174
- Owned Car: 169

**Response**:
YOUR TURN (DO NOT SUGGEST ABOUT CREDIT HISTORY AND SCORE)
"""
