"""
Student Grade Analyzer - Data Analysis with Python
Author: Tinka Max
Course: CSE 310 - Applied Programming
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime

print("=" * 50)
print("STUDENT GRADE ANALYZER")
print("=" * 50)
print()

# ============================================
# STEP 1: LOAD AND EXPLORE DATA
# ============================================
print("STEP 1: Loading data...")
print("-" * 30)

# Load the CSV file
df = pd.read_csv('student_grades.csv')

# Display first 5 rows
print("\nFirst 5 Students:")
print(df.head())

print("\nData Information:")
print(df.info())

print("\nBasic Statistics:")
print(df.describe())

# ============================================
# STEP 2: CALCULATE ADDITIONAL COLUMNS
# ============================================
print("\n" + "=" * 50)
print("STEP 2: Calculating Additional Columns")
print("=" * 50)

# Calculate average grade for each student
subject_cols = ['Math', 'Science', 'English', 'History']
df['Average'] = df[subject_cols].mean(axis=1).round(2)

# Assign letter grade
def get_letter_grade(avg):
    if avg >= 90:
        return 'A'
    elif avg >= 80:
        return 'B'
    elif avg >= 70:
        return 'C'
    elif avg >= 60:
        return 'D'
    else:
        return 'F'

df['Letter_Grade'] = df['Average'].apply(get_letter_grade)
df['Total_Points'] = df[subject_cols].sum(axis=1)

print("\nStudents with Calculated Averages:")
print(df[['Name', 'Average', 'Letter_Grade', 'Total_Points']].to_string(index=False))

# ============================================
# STEP 3: SUBJECT ANALYSIS
# ============================================
print("\n" + "=" * 50)
print("STEP 3: Subject Analysis")
print("=" * 50)

subject_averages = df[subject_cols].mean().round(2)
subject_medians = df[subject_cols].median().round(2)
subject_std = df[subject_cols].std().round(2)

subject_summary = pd.DataFrame({
    'Average': subject_averages,
    'Median': subject_medians,
    'Std Dev': subject_std,
    'Min': df[subject_cols].min(),
    'Max': df[subject_cols].max()
})

print("\nSubject Statistics:")
print(subject_summary)

best_subject = subject_averages.idxmax()
best_score = subject_averages.max()
print(f"\nBest Subject: {best_subject} (Avg: {best_score})")

worst_subject = subject_averages.idxmin()
worst_score = subject_averages.min()
print(f"Lowest Subject: {worst_subject} (Avg: {worst_score})")

# ============================================
# STEP 4: GRADE DISTRIBUTION
# ============================================
print("\n" + "=" * 50)
print("STEP 4: Grade Distribution")
print("=" * 50)

grade_counts = df['Letter_Grade'].value_counts().sort_index()

print("\nGrade Distribution:")
for grade, count in grade_counts.items():
    percentage = (count / len(df)) * 100
    print(f"{grade}: {count} students ({percentage:.1f}%)")

top_student = df.loc[df['Average'].idxmax()]
print(f"\nTop Student: {top_student['Name']} (Avg: {top_student['Average']})")

bottom_student = df.loc[df['Average'].idxmin()]
print(f"Student Needing Support: {bottom_student['Name']} (Avg: {bottom_student['Average']})")

# ============================================
# STEP 5: ATTENDANCE ANALYSIS
# ============================================
print("\n" + "=" * 50)
print("STEP 5: Attendance Analysis")
print("=" * 50)

print(f"\nAverage Attendance: {df['Attendance'].mean():.1f}%")
print(f"Highest Attendance: {df['Attendance'].max()}%")
print(f"Lowest Attendance: {df['Attendance'].min()}%")

correlation = df['Attendance'].corr(df['Average'])
print(f"\nCorrelation between Attendance and Grades: {correlation:.3f}")

if correlation > 0.7:
    print("Strong positive correlation: Higher attendance = Higher grades")
elif correlation > 0.3:
    print("Moderate positive correlation")
else:
    print("Weak correlation")

# ============================================
# STEP 6: CREATE VISUALIZATIONS
# ============================================
print("\n" + "=" * 50)
print("STEP 6: Creating Visualizations")
print("=" * 50)

# Set style
plt.style.use('seaborn-v0_8-darkgrid')

# 1. Bar Chart - Student Averages
plt.figure(figsize=(12, 6))
bars = plt.bar(df['Name'], df['Average'], color='skyblue', edgecolor='navy')
plt.title('Student Grade Averages', fontsize=16, fontweight='bold')
plt.xlabel('Student Name', fontsize=12)
plt.ylabel('Average Grade', fontsize=12)
plt.xticks(rotation=45, ha='right')
plt.ylim(60, 100)
plt.axhline(y=df['Average'].mean(), color='red', linestyle='--', 
            label=f'Class Average: {df["Average"].mean():.1f}')
plt.legend()
plt.tight_layout()
plt.savefig('student_averages.png', dpi=150, bbox_inches='tight')
plt.close()
print("✓ Created: student_averages.png")

# 2. Line Chart - Subject Comparison
plt.figure(figsize=(10, 6))
plt.plot(subject_averages.index, subject_averages.values, marker='o', 
         linewidth=2, markersize=10, color='green')
plt.title('Average Grades by Subject', fontsize=16, fontweight='bold')
plt.xlabel('Subject', fontsize=12)
plt.ylabel('Average Grade', fontsize=12)
plt.ylim(70, 95)
for i, (subject, avg) in enumerate(subject_averages.items()):
    plt.text(i, avg + 1, f'{avg:.1f}', ha='center', fontsize=10)
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('subject_averages.png', dpi=150, bbox_inches='tight')
plt.close()
print("✓ Created: subject_averages.png")

# 3. Pie Chart - Grade Distribution
colors = ['#2ecc71', '#3498db', '#f39c12', '#e74c3c']
plt.figure(figsize=(8, 8))
plt.pie(grade_counts.values, labels=grade_counts.index, autopct='%1.1f%%',
        colors=colors[:len(grade_counts)], startangle=90, explode=[0.05] * len(grade_counts))
plt.title('Grade Distribution', fontsize=16, fontweight='bold')
plt.savefig('grade_distribution.png', dpi=150, bbox_inches='tight')
plt.close()
print("✓ Created: grade_distribution.png")

# 4. Scatter Plot - Attendance vs Grades
plt.figure(figsize=(10, 6))
plt.scatter(df['Attendance'], df['Average'], s=100, alpha=0.6, color='purple')

# Add trend line
z = np.polyfit(df['Attendance'], df['Average'], 1)
p = np.poly1d(z)
plt.plot(df['Attendance'], p(df['Attendance']), "r--", alpha=0.8, label='Trend Line')

plt.title('Attendance vs Average Grade', fontsize=16, fontweight='bold')
plt.xlabel('Attendance (%)', fontsize=12)
plt.ylabel('Average Grade', fontsize=12)
plt.xlim(75, 100)
plt.ylim(70, 100)
plt.legend()
plt.grid(alpha=0.3)

# Add student labels
for i, row in df.iterrows():
    plt.annotate(row['Name'], (row['Attendance'], row['Average']),
                 fontsize=8, xytext=(5, 5), textcoords='offset points')

plt.tight_layout()
plt.savefig('attendance_vs_grades.png', dpi=150, bbox_inches='tight')
plt.close()
print("✓ Created: attendance_vs_grades.png")

# 5. Box Plot - Subject Distribution
plt.figure(figsize=(10, 6))
box_data = [df['Math'], df['Science'], df['English'], df['History']]
bp = plt.boxplot(box_data, labels=['Math', 'Science', 'English', 'History'],
                  patch_artist=True, showmeans=True)

colors_box = ['#FF9999', '#66B2FF', '#99FF99', '#FFCC99']
for patch, color in zip(bp['boxes'], colors_box):
    patch.set_facecolor(color)

plt.title('Grade Distribution by Subject', fontsize=16, fontweight='bold')
plt.ylabel('Grade', fontsize=12)
plt.grid(axis='y', alpha=0.3)
plt.tight_layout()
plt.savefig('subject_boxplot.png', dpi=150, bbox_inches='tight')
plt.close()
print("✓ Created: subject_boxplot.png")

# ============================================
# STEP 7: EXPORT RESULTS
# ============================================
print("\n" + "=" * 50)
print("STEP 7: Exporting Results")
print("=" * 50)

# Save processed data to CSV
df.to_csv('processed_student_grades.csv', index=False)
print("✓ Saved: processed_student_grades.csv")

# Save statistics to text file
with open('analysis_report.txt', 'w') as f:
    f.write("STUDENT GRADE ANALYSIS REPORT\n")
    f.write("=" * 50 + "\n\n")
    f.write(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
    f.write(f"Total Students: {len(df)}\n\n")
    f.write("SUBJECT AVERAGES:\n")
    for subject, avg in subject_averages.items():
        f.write(f"  {subject}: {avg:.2f}\n")
    f.write(f"\nOverall Class Average: {df['Average'].mean():.2f}\n")
    f.write(f"Highest Grade: {df['Average'].max():.2f}\n")
    f.write(f"Lowest Grade: {df['Average'].min():.2f}\n\n")
    f.write("GRADE DISTRIBUTION:\n")
    for grade, count in grade_counts.items():
        percentage = (count / len(df)) * 100
        f.write(f"  {grade}: {count} students ({percentage:.1f}%)\n")
    f.write(f"\nAttendance vs Grades Correlation: {correlation:.3f}\n")

print("✓ Saved: analysis_report.txt")

# ============================================
# STEP 8: SUMMARY
# ============================================
print("\n" + "=" * 50)
print("SUMMARY OF FINDINGS")
print("=" * 50)

print(f"\nTotal Students Analyzed: {len(df)}")
print(f"Class Average: {df['Average'].mean():.2f}")
print(f"Top Performer: {top_student['Name']} ({top_student['Average']})")
print(f"Lowest Performer: {bottom_student['Name']} ({bottom_student['Average']})")
print(f"Best Subject: {best_subject} ({best_score})")
print(f"Subject Needing Improvement: {worst_subject} ({worst_score})")
print(f"Attendance Correlation: {correlation:.3f}")

print("\n" + "=" * 50)
print("ANALYSIS COMPLETE!")
print("=" * 50)
print("\nOutput files created:")
print("  - processed_student_grades.csv")
print("  - analysis_report.txt")
print("  - student_averages.png")
print("  - subject_averages.png")
print("  - grade_distribution.png")
print("  - attendance_vs_grades.png")
print("  - subject_boxplot.png")
