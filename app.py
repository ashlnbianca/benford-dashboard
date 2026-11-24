import io
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt

st.title("My Benford Dashboard")

# ---------- UPLOAD ----------
uploaded = st.file_uploader("Upload a CSV or Excel file", type=["csv", "xlsx"])
if uploaded is None:
    st.info("Please upload a file to begin.")
    st.stop()

df = pd.read_csv(uploaded) if uploaded.name.endswith(".csv") else pd.read_excel(uploaded)
st.write(f"Your data has {len(df):,} rows. Here are the first 100:")
st.dataframe(df.head(100))

# ---------- FILTERS ----------
st.sidebar.header("Filters")
filtered = df.copy()
text_cols = df.select_dtypes(include=["object", "string"]).columns.tolist()
num_cols = df.select_dtypes(include="number").columns.tolist()

# Filter 1: pick categories
col1 = st.sidebar.selectbox("Filter 1: choose a text column", text_cols)
options = df[col1].dropna().unique().tolist()
chosen = st.sidebar.multiselect("Keep these values", options, default=options)
filtered = filtered[filtered[col1].isin(chosen)]

# Filter 2: number range
col2 = st.sidebar.selectbox("Filter 2: choose a number column", num_cols)
low, high = float(df[col2].min()), float(df[col2].max())
picked = st.sidebar.slider("Keep values between", low, high, (low, high))
filtered = filtered[filtered[col2].between(picked[0], picked[1])]

# Filter 3: search for a word
col3 = st.sidebar.selectbox("Filter 3: search in this column", text_cols)
word = st.sidebar.text_input("Text contains")
if word:
    filtered = filtered[filtered[col3].str.contains(word, case=False, na=False)]

st.write(f"Rows left after filtering: {len(filtered):,}")
st.dataframe(filtered.head(100))

# ---------- BENFORD SETUP ----------
st.header("Benford Analysis")
default_col = next((i for i, c in enumerate(num_cols) if "amount" in c.lower()), 0)
target = st.selectbox("Pick the column to analyze", num_cols, index=default_col)

values = filtered[target].dropna()
values = values[values != 0].abs()
if values.empty:
    st.warning("No usable numbers to analyze. Try changing the filters.")
    st.stop()

# Each number as text with leading zeros/dots/decimal point removed: 0.045 -> "45"
text = values.astype(str).str.lstrip("0.").str.replace(".", "", regex=False).str.ljust(2, "0")


def benford(n_digits):
    """Compare the first 1 (or 2) digits to Benford and run the KS test."""
    digits = np.arange(10 ** (n_digits - 1), 10 ** n_digits)
    found = pd.to_numeric(text.str[:n_digits], errors="coerce").dropna().astype(int)
    counts = found[found.isin(digits)].value_counts().reindex(digits, fill_value=0)

    n = counts.sum()
    actual = counts.values / n * 100
    expected = np.log10(1 + 1 / digits) * 100

    # KS: the biggest gap between the two running totals
    D = np.max(np.abs(np.cumsum(actual) - np.cumsum(expected))) / 100
    critical = 1.36 / np.sqrt(n)
    k = np.arange(1, 101)
    p = float(np.clip(2 * np.sum((-1) ** (k - 1) * np.exp(-2 * k**2 * n * D**2)), 0, 1))

    table = pd.DataFrame({"digits": digits, "count": counts.values,
                          "actual_%": actual.round(2), "benford_%": expected.round(2)})
    table["difference"] = (table["actual_%"] - table["benford_%"]).round(2)

    fig, ax = plt.subplots(figsize=(8 if n_digits == 1 else 10, 4))
    ax.bar(digits, actual, color="steelblue", label="Actual")
    ax.plot(digits, expected, color="red", linewidth=2, label="Benford",
            marker="o" if n_digits == 1 else None)
    if n_digits == 1:
        ax.set_xticks(digits)
    ax.set_xlabel("First digit" if n_digits == 1 else "First two digits")
    ax.set_ylabel("Percent of numbers")
    ax.set_title(f"{'First digit' if n_digits == 1 else 'First two digits'} of {target}: actual vs. Benford")
    ax.legend()

    return dict(table=table, fig=fig, n=int(n), D=D, critical=critical, p=p,
                different=D > critical)


# ---------- RUN BOTH TESTS AND SHOW RESULTS ----------
names = {1: "First Digit", 2: "First Two Digits"}
results = {nd: benford(nd) for nd in (1, 2)}

for nd, r in results.items():
    st.header(f"{names[nd]} Analysis")
    c1, c2, c3 = st.columns(3)
    c1.metric("KS statistic (D)", f"{r['D']:.4f}")
    c2.metric("Critical value (5%)", f"{r['critical']:.4f}")
    c3.metric("p-value", f"{r['p']:.4f}")
    if r["different"]:
        st.error("D is bigger than the critical value, so this data is statistically DIFFERENT from Benford's distribution.")
    else:
        st.success("D is smaller than the critical value, so this data is NOT statistically different from Benford's distribution.")
    st.pyplot(r["fig"])
    st.dataframe(r["table"], hide_index=True)

# ---------- SUMMARY STATISTICS ----------
st.header("Summary Statistics")
summary = filtered[target].describe().to_frame()
st.dataframe(summary)

# ---------- DOWNLOADS ----------
st.header("Download Your Results")

verdict = lambda r: "Different from Benford" if r["different"] else "Not different from Benford"
ks_results = pd.DataFrame([{
    "test": names[nd], "numbers_analyzed": r["n"], "KS_statistic_D": r["D"],
    "critical_value_5pct": r["critical"], "p_value": r["p"], "conclusion": verdict(r),
} for nd, r in results.items()])

report = (f"BENFORD ANALYSIS REPORT\nFile: {uploaded.name}\nColumn analyzed: {target}\n"
          f"Rows after filtering: {len(filtered):,}\n\n"
          + "\n".join(f"{names[nd]}: D = {r['D']:.4f}, critical value = {r['critical']:.4f}, "
                      f"p-value = {r['p']:.4f} -> {verdict(r)}" for nd, r in results.items()))


def fig_to_png(fig):
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=150, bbox_inches="tight")
    return buf.getvalue()


downloads = [
    ("Summary stats (CSV)", summary.to_csv(), "summary_statistics.csv", "text/csv"),
    ("KS test results (CSV)", ks_results.to_csv(index=False), "ks_results.csv", "text/csv"),
    ("Written report (TXT)", report, "benford_report.txt", "text/plain"),
    ("First digit table (CSV)", results[1]["table"].to_csv(index=False), "first_digit_table.csv", "text/csv"),
    ("First two digits table (CSV)", results[2]["table"].to_csv(index=False), "first_two_digits_table.csv", "text/csv"),
    ("Filtered data (CSV)", filtered.to_csv(index=False), "filtered_data.csv", "text/csv"),
    ("First digit chart (PNG)", fig_to_png(results[1]["fig"]), "first_digit_chart.png", "image/png"),
    ("First two digits chart (PNG)", fig_to_png(results[2]["fig"]), "first_two_digits_chart.png", "image/png"),
]
cols = st.columns(3)
for i, (label, data, file_name, mime) in enumerate(downloads):
    cols[i % 3].download_button(label, data, file_name, mime)