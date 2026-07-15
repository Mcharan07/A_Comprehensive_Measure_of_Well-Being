import numpy as np
import pandas as pd

np.random.seed(42)

# Real-world-style country list with approximate realistic ranges per tier
countries_data = [
    # Very High HDI (life_exp 75-85, mean_school 11-14, exp_school 15-18, gni 25000-90000)
    ("Norway", "Very High"), ("Switzerland", "Very High"), ("Ireland", "Very High"),
    ("Germany", "Very High"), ("Iceland", "Very High"), ("Hong Kong", "Very High"),
    ("Australia", "Very High"), ("Sweden", "Very High"), ("Singapore", "Very High"),
    ("Netherlands", "Very High"), ("Denmark", "Very High"), ("Finland", "Very High"),
    ("United Kingdom", "Very High"), ("Belgium", "Very High"), ("New Zealand", "Very High"),
    ("Canada", "Very High"), ("United States", "Very High"), ("Austria", "Very High"),
    ("Japan", "Very High"), ("Israel", "Very High"), ("South Korea", "Very High"),
    ("Luxembourg", "Very High"), ("France", "Very High"), ("Slovenia", "Very High"),
    ("Spain", "Very High"), ("Italy", "Very High"), ("Czechia", "Very High"),
    ("Greece", "Very High"), ("Estonia", "Very High"), ("Cyprus", "Very High"),
    ("United Arab Emirates", "Very High"), ("Malta", "Very High"), ("Poland", "Very High"),
    ("Lithuania", "Very High"), ("Saudi Arabia", "Very High"), ("Portugal", "Very High"),
    ("Bahrain", "Very High"), ("Latvia", "Very High"), ("Croatia", "Very High"),
    ("Qatar", "Very High"), ("Slovakia", "Very High"), ("Hungary", "Very High"),
    ("Chile", "Very High"), ("Argentina", "High"), ("Montenegro", "High"),

    # High HDI (life_exp 70-77, mean_school 8-11, exp_school 13-15, gni 8000-25000)
    ("Russia", "High"), ("Romania", "High"), ("Kuwait", "High"), ("Bulgaria", "High"),
    ("Bahamas", "High"), ("Belarus", "High"), ("Uruguay", "High"), ("Panama", "High"),
    ("Turkey", "High"), ("Costa Rica", "High"), ("Serbia", "High"), ("Kazakhstan", "High"),
    ("Mexico", "High"), ("Mauritius", "High"), ("North Macedonia", "High"),
    ("Georgia", "High"), ("Albania", "High"), ("Brazil", "High"), ("Bosnia and Herzegovina", "High"),
    ("Iran", "High"), ("Armenia", "High"), ("Azerbaijan", "High"), ("China", "High"),
    ("Peru", "High"), ("Thailand", "High"), ("Ecuador", "High"), ("Colombia", "High"),
    ("Sri Lanka", "High"), ("Algeria", "High"), ("Mongolia", "High"), ("Dominican Republic", "High"),
    ("Tunisia", "High"), ("Jamaica", "High"), ("Jordan", "High"), ("Cuba", "High"),
    ("Paraguay", "High"), ("Maldives", "High"), ("Egypt", "High"), ("Moldova", "High"),
    ("Vietnam", "High"), ("Ukraine", "High"), ("Indonesia", "High"), ("Palestine", "High"),
    ("Philippines", "High"), ("Uzbekistan", "High"), ("Bolivia", "High"), ("Botswana", "High"),
    ("Kyrgyzstan", "High"), ("South Africa", "High"), ("Iraq", "High"), ("El Salvador", "High"),
    ("Tonga", "High"), ("Libya", "High"), ("Marshall Islands", "High"), ("Gabon", "High"),
    ("Morocco", "High"),

    # Medium HDI (life_exp 62-70, mean_school 5-8, exp_school 10-13, gni 2500-8000)
    ("India", "Medium"), ("Bangladesh", "Medium"), ("Bhutan", "Medium"), ("Nicaragua", "Medium"),
    ("Vanuatu", "Medium"), ("Eswatini", "Medium"), ("Ghana", "Medium"), ("Guatemala", "Medium"),
    ("Namibia", "Medium"), ("Kenya", "Medium"), ("Cabo Verde", "Medium"), ("Honduras", "Medium"),
    ("Myanmar", "Medium"), ("Nepal", "Medium"), ("Cambodia", "Medium"), ("Iraq2", "Medium"),
    ("Laos", "Medium"), ("Equatorial Guinea", "Medium"), ("Micronesia", "Medium"),
    ("Zambia", "Medium"), ("Congo", "Medium"), ("Angola", "Medium"), ("Solomon Islands", "Medium"),
    ("Tajikistan", "Medium"), ("Pakistan", "Medium"), ("Papua New Guinea", "Medium"),
    ("Syria", "Medium"), ("Comoros", "Medium"), ("Zimbabwe", "Medium"), ("Rwanda", "Medium"),
    ("Cameroon", "Medium"), ("Nigeria", "Medium"), ("Tanzania", "Medium"), ("Benin", "Medium"),
    ("Uganda", "Medium"), ("Senegal", "Medium"), ("Sudan", "Medium"), ("Togo", "Medium"),
    ("Haiti", "Medium"), ("Madagascar", "Medium"), ("Djibouti", "Medium"), ("Gambia", "Medium"),
    ("Lesotho", "Medium"), ("Mauritania", "Medium"), ("Timor-Leste", "Medium"),

    # Low HDI (life_exp 50-62, mean_school 2-5, exp_school 7-10, gni 600-2500)
    ("Malawi", "Low"), ("Ethiopia", "Low"), ("Sierra Leone", "Low"), ("Guinea-Bissau", "Low"),
    ("Liberia", "Low"), ("Yemen", "Low"), ("Afghanistan", "Low"), ("Guinea", "Low"),
    ("Eritrea", "Low"), ("Mozambique", "Low"), ("Burkina Faso", "Low"), ("Sierra2", "Low"),
    ("Mali", "Low"), ("Burundi", "Low"), ("South Sudan", "Low"), ("Chad", "Low"),
    ("Central African Republic", "Low"), ("Niger", "Low"), ("DR Congo", "Low"),
    ("Somalia", "Low"), ("Ivory Coast", "Low"), ("Gambia2", "Low"), ("Congo Republic", "Low"),
]

tier_ranges = {
    "Very High": dict(life=(75, 85), mean_school=(11, 14.5), exp_school=(15, 18.5), gni=(25000, 95000), hdi=(0.800, 0.965)),
    "High":      dict(life=(70, 77), mean_school=(8, 11.2),  exp_school=(13, 15.5), gni=(8000, 25500),  hdi=(0.700, 0.799)),
    "Medium":    dict(life=(62, 70), mean_school=(5, 8.2),   exp_school=(10, 13.2), gni=(2500, 8200),   hdi=(0.550, 0.699)),
    "Low":       dict(life=(50, 62), mean_school=(2, 5.2),   exp_school=(7, 10.2),  gni=(600, 2600),    hdi=(0.350, 0.549)),
}

rows = []
seen_names = set()
for name, tier in countries_data:
    clean_name = name.rstrip("0123456789")
    if clean_name in seen_names:
        continue
    seen_names.add(clean_name)
    r = tier_ranges[tier]
    life_exp = np.round(np.random.uniform(*r["life"]), 1)
    mean_school = np.round(np.random.uniform(*r["mean_school"]), 1)
    exp_school = np.round(np.random.uniform(*r["exp_school"]), 1)
    gni = int(np.random.uniform(*r["gni"]))

    # Compute HDI-like score using simplified geometric-mean style formula, then clip into tier range
    life_idx = (life_exp - 20) / (85 - 20)
    edu_idx = ((mean_school / 15) + (exp_school / 18)) / 2
    gni_idx = (np.log(gni) - np.log(100)) / (np.log(75000) - np.log(100))
    gni_idx = np.clip(gni_idx, 0, 1)
    hdi_raw = (life_idx * edu_idx * gni_idx) ** (1/3)
    hdi_score = np.clip(hdi_raw, r["hdi"][0], r["hdi"][1])
    hdi_score = np.round(hdi_score, 3)

    rows.append([clean_name, life_exp, mean_school, exp_school, gni, hdi_score, tier])

df = pd.DataFrame(rows, columns=[
    "Country", "Life_Expectancy", "Mean_Years_Schooling",
    "Expected_Years_Schooling", "GNI_Per_Capita", "HDI_Score", "HDI_Category"
])

df = df.drop_duplicates(subset="Country").reset_index(drop=True)
df.to_csv("HDI.csv", index=False)
print(df.shape)
print(df["HDI_Category"].value_counts())
print(df.head())
