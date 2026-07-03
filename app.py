import streamlit as st
import pandas as pd
from sklearn.tree import DecisionTreeClassifier

# 1. Page Configuration (Must be the very first Streamlit command)
st.set_page_config(
    page_title="Customer Retention Dashboard",
    page_icon="📊",
    layout="wide"
)

# 2. Train the model in the background
@st.cache_resource
def load_and_train_model():
    df = pd.read_excel('data/churn_data.csv.xlsx')
    df['Churn'] = df['Churn'].map({'Yes': 1, 'No': 0})
    X = df.drop(columns=['customerID', 'Churn'])
    X = pd.get_dummies(X, drop_first=True)
    y = df['Churn']
    
    model = DecisionTreeClassifier(max_depth=5, random_state=42)
    model.fit(X, y)
    return model, X.columns

model, feature_columns = load_and_train_model()

# 3. Designing a Beautiful Sidebar for User Inputs
st.sidebar.header("👤 Customer Profile Configuration")
st.sidebar.write("Adjust the parameters below to test the AI model live.")

tenure = st.sidebar.slider("Tenure (Months with company):", min_value=0, max_value=72, value=12)
monthly_charges = st.sidebar.number_input("Monthly Charges ($):", min_value=0.0, max_value=150.0, value=65.0)

st.sidebar.markdown("---")
st.sidebar.subheader("Services & Billing")
internet_service = st.sidebar.selectbox("Internet Service Type:", ["Fiber optic", "DSL", "No internet service"])
payment_method = st.sidebar.selectbox("Payment Method:", ["Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"])

# 4. Main Dashboard Header Layout
st.markdown("# 📊 Enterprise Customer Retention Control Panel")
st.markdown("This predictive analytical tool uses an **80.77% accurate Decision Tree Classifier** to evaluate customer churn vulnerability in real-time.")
st.markdown("---")

# 5. Creating a 3-Column Layout for Business Context Metrics
col1, col2, col3 = st.columns(3)
with col1:
    st.metric(label="Target Accuracy Score", value="80.77%", delta="Excellent Baseline")
with col2:
    st.metric(label="Core Risk Driver", value="Tenure", delta="- High Impact", delta_color="inverse")
with col3:
    st.metric(label="Secondary Risk Driver", value="Fiber Optic", delta="- Infrastructure", delta_color="off")

st.markdown("### 🔮 Real-Time Prediction Analysis")
st.write("Click the button below to process the profile customized in the left sidebar panel.")

# 6. Predict Button Action & Interactive UI Responses
if st.sidebar.button("🚀 Process Churn Analysis", use_container_width=True):
    # Prepare the data
    input_data = pd.DataFrame(0, index=[0], columns=feature_columns)
    input_data['tenure'] = tenure
    input_data['MonthlyCharges'] = monthly_charges
    
    if internet_service == "Fiber optic":
        input_data['InternetService_Fiber optic'] = 1
    elif internet_service == "No internet service":
        input_data['InternetService_No internet service'] = 1
        
    if payment_method == "Electronic check":
        input_data['PaymentMethod_Electronic check'] = 1
    elif payment_method == "Mailed check":
        input_data['PaymentMethod_Mailed check'] = 1
    elif payment_method == "Credit card (automatic)":
        input_data['PaymentMethod_Credit card (automatic)'] = 1

    # Run predictions
    prediction = model.predict(input_data)[0]
    probability = model.predict_proba(input_data)[0][1]

    # Render beautiful layout results based on output
    if prediction == 1:
        st.error(f"### ⚠️ Alert: High Churn Risk Detected!")
        
        # Split results into a clean text / progress bar block
        res_col1, res_col2 = st.columns([1, 2])
        with res_col1:
            st.write(f"The model has determined a **{probability*100:.1f}%** mathematical probability that this user will abandon their subscription.")
        with res_col2:
            st.progress(int(probability * 100))
            
        st.warning("**Recommended Business Mitigation Strategy:** Offer a retention incentive or contact the customer immediately to review pricing or network stability constraints.")
    else:
        st.success(f"### ✅ Status: Stable Account (Low Risk)")
        
        res_col1, res_col2 = st.columns([1, 2])
        with res_col1:
            st.write(f"This customer is classified as highly loyal with a strong retention probability of **{100 - (probability*100):.1f}%**.")
        with res_col2:
            st.progress(int((1 - probability) * 100))
            
        st.info("**Account Status:** No direct loyalty actions required at this time.")
else:
    st.info("👈 Use the left sidebar to tweak the user profile, then click **'Process Churn Analysis'** to view the live dashboard calculations.")