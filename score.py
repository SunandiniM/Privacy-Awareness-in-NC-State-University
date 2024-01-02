import pandas as pd
from datetime import datetime
from scipy import stats
import matplotlib.pyplot as plt


# Define the CSV file path
csv_file_path = 'pri.csv'

# Read the CSV into a pandas DataFrame
df = pd.read_csv(csv_file_path)

# Display the original DataFrame
print("Original DataFrame:")
print(df)
df.columns = df.columns.str.strip()

# Define the weights dictionary
weights = {
    'Age': 2,
    'Gender': 0,
    'Educational Status': 2.5,
    'Data Breach Experience': 5,
    'Scam Recognition Confidence': 1,
    'Use of Strong Passwords': 2,
    'Use of 2FA': 2,
    'Reviewing Social Media Privacy': 2,
    'Reviewing Smartphone Privacy': 1,
    'Checking App Privacy Policy': 1,
    'Familiarity with Data Protection Laws': 2,
    'Action if Data Shared Without Consent': 3,
    'Advocacy for Privacy Practices': 3,
    'Preferred Learning Methods': 0,
    'Awareness of University\'s Privacy Statement': 2,
    'Understanding University\'s Data Use': 2
}

# Define mappings for scaling responses
response_mappings = {
    'Age': {
        'Under 18': 0,
        '18-24': 1,
        '25-34': 2,
        '35-44': 3,
        '45-54': 4,
        '55-64': 5
    },
    'Gender': {
        'Male': 0,
        'Female': 1,
        'Prefer not to say': 2
    },
    'Educational Status': {
        'Alumni': 0,
        'Undergraduate': 1,
        'Graduate': 2,
        'Staff': 3,
        'Faculty': 4
    },
    'Data Breach Experience': {
        'Yes': 1,
        'No': 0,
        'Maybe': 0.5
    },
    'Scam Recognition Confidence': {
        'Not confident at all': 0,
        'Not very confident': 1,
        'Somewhat confident': 2,
        'Very confident': 3
    },
    'Use of Strong Passwords': {
        'Rarely': 1,
        'Often': 2,
        'Always': 3
    },
    'Use of 2FA': {
        'Yes': 1,
        'No': 0
    },
    'Reviewing Social Media Privacy': {
        'Never': 0,
        'Rarely': 1,
        'Ocassionally': 2,
        'Monthly': 3
    },
    'Reviewing Smartphone Privacy': {
        'Never': 0,
        'Rarely': 1,
        'Ocassionally': 2,
        'Monthly': 3
    },
    'Checking App Privacy Policy': {
        'Yes': 1,
        'No': 0
    },
    'Familiarity with Data Protection Laws': {
        'Not familiar': 0,
        'Somewhat familiar': 1,
        'Very familiar': 2
    },
    'Action if Data Shared Without Consent': {
        'Nothing': 0,
        'Do nothing': 0,
        'Ignore it': 1,
        'Change passwords': 2,
        'Block it': 3,
        'Report to Police': 4
    },
    'Advocacy for Privacy Practices': {
        'Yes': 1,
        'No': 0
    },
    'Preferred Learning Methods': {
        # Add your options here
    },
    'Awareness of University\'s Privacy Statement': {
        'Yes': 1,
        'No': 0
    },
    'Understanding University\'s Data Use': {
        'Not confident': 0,
        'Somewhat confident': 1,
        'Very confident': 2
    }
}

# Define a function to calculate the privacy awareness score for each row
def calculate_privacy_score(row):
    score = 0
    for col, weight in weights.items():
        response = row[col].strip()
        if col in response_mappings.keys() and response in [x for x in response_mappings[col].keys()]:
            score += response_mappings[col][response] * weight
    return score

# Apply the function to each row to calculate the privacy awareness score
df['Privacy Awareness Score'] = df.apply(calculate_privacy_score, axis=1)

# Add a timestamp column with the current date and time
df['Timestamp'] = datetime.now()

# Display the updated DataFrame with privacy awareness scores and timestamps
print("\nUpdated DataFrame:")
print(df)

# Save the updated DataFrame to a new CSV file
df.to_csv('result.csv', index=False)

# Load your data into a DataFrame
df_final = pd.read_csv('result.csv')

average_pps = df_final['Privacy Awareness Score'].mean()
print(average_pps, max(df_final['Privacy Awareness Score']), min(df_final['Privacy Awareness Score']))

grouped_avg_gender = df_final.groupby('Gender')['Privacy Awareness Score'].mean()

t_statistic, p_value = stats.ttest_ind(
    df_final[df_final['Gender'] == ' Male']['Privacy Awareness Score'],
    df_final[df_final['Gender'] == ' Female']['Privacy Awareness Score']
)

f_statistic2, p_value_anova = stats.f_oneway(
    df_final[df_final['Educational Status'] == ' Undergraduate']['Privacy Awareness Score'],
    df_final[df_final['Educational Status'] == ' Graduate']['Privacy Awareness Score'],
    df_final[df_final['Educational Status'] == ' Alumni']['Privacy Awareness Score']
)

# Print the results
print(f"Average PPS by Gender:\n{grouped_avg_gender}")
print(f"\nT-test between Male and Female: T-statistic = {t_statistic}, P-value = {p_value}")
print(f"\nANOVA across Educational Status: F-statistic = {f_statistic2}, P-value = {p_value_anova}")

def age_range_midpoint(age_range):
    low, high = [int(x) for x in age_range.split('-')]
    return (low + high) / 2

# Apply this function to the Age column
df_final['Age Midpoint'] = df_final['Age'].apply(age_range_midpoint)

# Calculate correlation
correlation_matrix = df_final[['Age Midpoint', 'Privacy Awareness Score']].corr()

# Output the correlation matrix
print(correlation_matrix)

# One-hot encode the 'Educational Status' column
one_hot = pd.get_dummies(df_final['Educational Status'])
df_test = df_final['Privacy Awareness Score'].to_frame()
df_test = df_test.join(one_hot)

# Calculate correlations between each educational status and the Privacy Awareness Score
correlations = df_test.corr()['Privacy Awareness Score'].drop('Privacy Awareness Score')

print("\nCorrelation values of Privacy Awareness Score with educational status")

# Output the correlation values
print(correlations)

# Plot a histogram for PPS
plt.figure(figsize=(8, 6))
plt.hist(df['Privacy Awareness Score'], bins=20, color='skyblue', edgecolor='black')
plt.title('Distribution of Privacy Awareness Score')
plt.xlabel('PPS')
plt.ylabel('Frequency')
plt.show()

# Plot a histogram for PPS
plt.figure(figsize=(8, 6))
plt.hist(df['Privacy Awareness Score'], bins=20, color='skyblue', edgecolor='black')
plt.title('Distribution of Privacy Awareness Score')
plt.xlabel('PPS')
plt.ylabel('Frequency')
plt.show()
