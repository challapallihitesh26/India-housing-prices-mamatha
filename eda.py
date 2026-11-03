"""
eda.py - Exploratory Data Analysis (answers questions 1-20) 
Run:  python eda.py      (after clean_data.py)
Out:  eda_charts/*.png  and  Insights.txt
"""
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

df = pd.read_csv("cleaned_housing_data.csv")
os.makedirs("eda_charts", exist_ok=True)
plt.style.use("dark_background")
lines = []

def note(q, text, fig=None, name=None):
    lines.append(f"Q{q}. {text}\n")
    print(f"Q{q}. {text}")
    if fig is not None:
        fig.tight_layout(); fig.savefig(f"eda_charts/q{q:02d}_{name}.png", dpi=110); plt.close(fig)

def corr(a, b): return df[a].corr(df[b])

# ---- 1-5 Price & size ---------------------------------------------------------
fig, ax = plt.subplots(figsize=(7, 4)); df["Price_in_Lakhs"].hist(bins=50, ax=ax, color="#a855f7"); ax.set_title("Price distribution (Lakhs)")
note(1, f"Prices range {df.Price_in_Lakhs.min():.0f}-{df.Price_in_Lakhs.max():.0f} Lakhs; mean {df.Price_in_Lakhs.mean():.1f}, median {df.Price_in_Lakhs.median():.1f}. Distribution is almost flat (uniform) - no single dominant price band.", fig, "price_dist")
fig, ax = plt.subplots(figsize=(7, 4)); df["Size_in_SqFt"].hist(bins=50, ax=ax, color="#c084fc"); ax.set_title("Size distribution (sq ft)")
note(2, f"Sizes range {df.Size_in_SqFt.min()}-{df.Size_in_SqFt.max()} sq ft; median {df.Size_in_SqFt.median():.0f}. Also evenly spread across the range.", fig, "size_dist")
t = df.groupby("Property_Type")["Price_per_SqFt_INR"].mean().sort_values()
fig, ax = plt.subplots(figsize=(7, 4)); t.plot.bar(ax=ax, color="#8b5cf6"); ax.set_title("Avg price/sq ft (INR) by property type")
note(3, "Avg price per sq ft by type: " + "; ".join(f"{k} Rs {v:,.0f}" for k, v in t.items()) + f". Spread between types is only {(t.max()/t.min()-1):.1%}.", fig, "pps_type")
s = df.sample(5000, random_state=1)
fig, ax = plt.subplots(figsize=(7, 4)); ax.scatter(s.Size_in_SqFt, s.Price_in_Lakhs, s=4, alpha=.4, color="#d8b4fe"); ax.set_title("Size vs Price")
note(4, f"Correlation between size and total price = {corr('Size_in_SqFt','Price_in_Lakhs'):.3f} (essentially none). Bigger homes are NOT priced higher in this dataset, so price per sq ft falls as size rises (corr {corr('Size_in_SqFt','Price_per_SqFt_INR'):.2f}).", fig, "size_price")
fig, ax = plt.subplots(1, 2, figsize=(9, 4)); df.boxplot("Price_per_SqFt_INR", ax=ax[0]); df.boxplot("Size_in_SqFt", ax=ax[1])
note(5, f"IQR outliers: price/sq ft {df.Outlier_Price_per_SqFt_INR.sum():,} rows ({df.Outlier_Price_per_SqFt_INR.mean():.1%}), size {df.Outlier_Size_in_SqFt.sum()}, price {df.Outlier_Price_in_Lakhs.sum()}. Outliers sit at the top of price/sq ft - small flats at high prices; they were flagged, not deleted.", fig, "outliers")

# ---- 6-10 Location --------------------------------------------------------------
t = df.groupby("State")["Price_per_SqFt_INR"].mean().sort_values()
fig, ax = plt.subplots(figsize=(7, 6)); t.plot.barh(ax=ax, color="#a855f7"); ax.set_title("Avg price/sq ft by state")
note(6, f"Highest avg price/sq ft: {t.index[-1]} (Rs {t.iloc[-1]:,.0f}); lowest: {t.index[0]} (Rs {t.iloc[0]:,.0f}). Top-to-bottom gap is only {(t.iloc[-1]/t.iloc[0]-1):.1%}.", fig, "state_pps")
t = df.groupby("City")["Price_in_Lakhs"].mean().sort_values()
fig, ax = plt.subplots(figsize=(7, 7)); t.plot.barh(ax=ax, color="#c084fc"); ax.set_title("Avg price by city (Lakhs)")
note(7, f"Costliest city on average: {t.index[-1]} ({t.iloc[-1]:.1f} L); cheapest: {t.index[0]} ({t.iloc[0]:.1f} L). City averages sit in a narrow band around {df.Price_in_Lakhs.mean():.0f} L.", fig, "city_price")
t = df.groupby("Locality")["Age_of_Property"].median()
fig, ax = plt.subplots(figsize=(7, 4)); t.hist(bins=20, ax=ax, color="#8b5cf6"); ax.set_title("Median age by locality")
note(8, f"Median property age across {len(t)} localities ranges {t.min():.0f}-{t.max():.0f} years; typical locality median = {t.median():.0f} years.", fig, "locality_age")
t = pd.crosstab(df.City, df.BHK, normalize="index") * 100
fig, ax = plt.subplots(figsize=(8, 6)); t.plot.barh(stacked=True, ax=ax, colormap="Purples"); ax.set_title("BHK mix by city (%)")
note(9, f"BHK is evenly spread: each of 1-5 BHK is ~{100/5:.0f}% in every city (largest city-level deviation {abs(t-20).max().max():.1f} pts).", fig, "bhk_city")
top = df.groupby("Locality")["Price_per_SqFt_INR"].mean().nlargest(5)
fig, ax = plt.subplots(figsize=(8, 4))
for l in top.index:
    d = df[df.Locality == l].groupby(pd.cut(df[df.Locality == l].Year_Built, [1989, 1995, 2000, 2005, 2010, 2015, 2023]), observed=True)["Price_per_SqFt_INR"].mean()
    ax.plot(range(len(d)), d.values, marker="o", label=l)
ax.set_xticks(range(6)); ax.set_xticklabels(["1990-95", "96-00", "01-05", "06-10", "11-15", "16-23"]); ax.legend(fontsize=7); ax.set_title("Top-5 costliest localities: price/sq ft by build period")
note(10, "Top 5 costliest localities (avg Rs/sq ft): " + "; ".join(f"{l} Rs {v:,.0f}" for l, v in top.items()) + ". Build-period trend lines are noisy with no steady upward pattern.", fig, "top_localities")

# ---- 11-15 Correlation ------------------------------------------------------------
num = ["BHK", "Size_in_SqFt", "Price_in_Lakhs", "Price_per_SqFt_INR", "Age_of_Property", "Nearby_Schools", "Nearby_Hospitals", "Amenity_Count", "Infrastructure_Score", "Floor_No"]
c = df[num].corr()
fig, ax = plt.subplots(figsize=(9, 7)); im = ax.imshow(c, cmap="Purples"); ax.set_xticks(range(len(num))); ax.set_xticklabels(num, rotation=90); ax.set_yticks(range(len(num))); ax.set_yticklabels(num); fig.colorbar(im)
pairs = c.where(~(c.abs() >= .999)).stack().abs().sort_values(ascending=False)
note(11, f"Strongest relationship: Size vs Price/sq ft ({corr('Size_in_SqFt','Price_per_SqFt_INR'):.2f}); Price vs Price/sq ft ({corr('Price_in_Lakhs','Price_per_SqFt_INR'):.2f}). Apart from these derived pairs, raw features are almost uncorrelated (|r| < 0.02).", fig, "corr")
t = df.groupby("Nearby_Schools")["Price_per_SqFt_INR"].mean()
fig, ax = plt.subplots(figsize=(7, 4)); t.plot(marker="o", ax=ax, color="#d8b4fe"); ax.set_title("Price/sq ft vs nearby schools")
note(12, f"Correlation schools vs price/sq ft = {corr('Nearby_Schools','Price_per_SqFt_INR'):.3f}. Price per sq ft varies by just {(t.max()/t.min()-1):.1%} across 1-10 schools - schools do not drive price here.", fig, "schools")
t = df.groupby("Nearby_Hospitals")["Price_per_SqFt_INR"].mean()
fig, ax = plt.subplots(figsize=(7, 4)); t.plot(marker="o", ax=ax, color="#d8b4fe"); ax.set_title("Price/sq ft vs nearby hospitals")
note(13, f"Correlation hospitals vs price/sq ft = {corr('Nearby_Hospitals','Price_per_SqFt_INR'):.3f}; variation across hospital counts only {(t.max()/t.min()-1):.1%}.", fig, "hospitals")
t = df.groupby("Furnished_Status")["Price_in_Lakhs"].mean().sort_values()
fig, ax = plt.subplots(figsize=(7, 4)); t.plot.bar(ax=ax, color="#8b5cf6"); ax.set_title("Avg price by furnishing")
note(14, "Avg price by furnishing: " + "; ".join(f"{k} {v:.1f} L" for k, v in t.items()) + ". Furnishing adds no meaningful premium.", fig, "furnished")
t = df.groupby("Facing")["Price_per_SqFt_INR"].mean().sort_values()
fig, ax = plt.subplots(figsize=(7, 4)); t.plot.bar(ax=ax, color="#a855f7"); ax.set_title("Price/sq ft by facing")
note(15, "Price/sq ft by facing: " + "; ".join(f"{k} Rs {v:,.0f}" for k, v in t.items()) + f". Spread {(t.max()/t.min()-1):.1%} - direction is not a price driver.", fig, "facing")

# ---- 16-20 Investment / amenities ----------------------------------------------
t = df.Owner_Type.value_counts()
fig, ax = plt.subplots(figsize=(5, 5)); t.plot.pie(autopct="%1.1f%%", ax=ax, colors=["#7c3aed", "#a855f7", "#d8b4fe"]); ax.set_ylabel("")
note(16, "Owner types: " + "; ".join(f"{k} {v:,} ({v/len(df):.1%})" for k, v in t.items()) + ". Market is split almost equally.", fig, "owner")
t = df.Availability_Status.value_counts()
fig, ax = plt.subplots(figsize=(5, 5)); t.plot.pie(autopct="%1.1f%%", ax=ax, colors=["#7c3aed", "#d8b4fe"]); ax.set_ylabel("")
note(17, "Availability: " + "; ".join(f"{k} {v:,} ({v/len(df):.1%})" for k, v in t.items()) + ".", fig, "availability")
t = df.groupby("Parking_Space")[["Price_in_Lakhs", "Price_per_SqFt_INR"]].mean()
fig, ax = plt.subplots(figsize=(6, 4)); t.Price_in_Lakhs.plot.bar(ax=ax, color="#8b5cf6"); ax.set_title("Avg price by parking")
note(18, f"Parking: avg price {t.Price_in_Lakhs.get('Yes',0):.1f} L (with) vs {t.Price_in_Lakhs.get('No',0):.1f} L (without) - difference {(t.Price_in_Lakhs.get('Yes',0)/t.Price_in_Lakhs.get('No',1)-1):+.1%}.", fig, "parking")
t = df.groupby("Amenity_Count")["Price_per_SqFt_INR"].mean()
fig, ax = plt.subplots(figsize=(7, 4)); t.plot.bar(ax=ax, color="#c084fc"); ax.set_title("Price/sq ft by number of amenities")
note(19, f"Amenity count vs price/sq ft correlation = {corr('Amenity_Count','Price_per_SqFt_INR'):.3f}. Variation across 1-5 amenities is only {(t.max()/t.min()-1):.1%}.", fig, "amenities")
t = df.groupby("Public_Transport_Accessibility").agg(pps=("Price_per_SqFt_INR", "mean"), good=("Good_Investment", "mean"))
fig, ax = plt.subplots(figsize=(6, 4)); (t.good * 100).plot.bar(ax=ax, color="#a855f7"); ax.set_title("% good investments by transport access")
note(20, "Transport access vs avg price/sq ft: " + "; ".join(f"{k} Rs {v:,.0f}" for k, v in t.pps.items()) + ". Good-investment share: " + "; ".join(f"{k} {v:.1%}" for k, v in t.good.items()) + " - better transport clearly raises investment potential (it is a scoring rule).", fig, "transport")

# ---- Extra: investment & growth -------------------------------------------------
g = df.groupby("City").agg(good=("Good_Investment", "mean"), growth=("Annual_Growth_Rate_%", "mean"))
lines.append(f"\nEXTRA. {df.Good_Investment.mean():.1%} of properties are flagged Good Investment. Best city by good-investment share: {g.good.idxmax()} ({g.good.max():.1%}). "
             f"Avg modelled growth {df['Annual_Growth_Rate_%'].mean():.2f}%/yr -> avg 5-year price {df.Future_Price_5Y_Lakhs.mean():.1f} L vs today {df.Price_in_Lakhs.mean():.1f} L.\n")
lines.append("NOTE. Crime-rate data is not in the dataset, so Infrastructure_Score (schools, hospitals, transport, amenities) is used instead.\n")

with open("Insights.txt", "w", encoding="utf-8") as f:
    f.write("REAL ESTATE INVESTMENT ADVISOR - KEY INSIGHTS (auto-generated by eda.py)\n" + "=" * 70 + "\n\n" + "\n".join(lines))
print("\nSaved Insights.txt and eda_charts/")
