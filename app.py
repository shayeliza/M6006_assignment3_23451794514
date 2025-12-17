from shiny import App, ui, render, reactive
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.miscmodels.ordinal_model import OrderedModel
import warnings
warnings.filterwarnings('ignore')

# Load data
hypothesis3_df = pd.read_csv("old_bailey_actual_final.csv")
hypothesis3_df_culled = hypothesis3_df[hypothesis3_df['punishment'].notna() & hypothesis3_df['defendant_occupation'].notna()]
hypothesis3_df_culled = hypothesis3_df_culled[hypothesis3_df_culled['defendant_gender'].isin(['male', 'female'])]
hypothesis3_df_culled['defendant_occupation'] = hypothesis3_df_culled['defendant_occupation'].str.lower().str.strip()

american_psycho_ahhh_comparison = hypothesis3_df_culled['defendant_occupation'].value_counts().nlargest(10).index
hypothesis3_df_culled['occupation_grouped'] = hypothesis3_df_culled['defendant_occupation'].apply(lambda x: x if x in american_psycho_ahhh_comparison else 'other')

# Imprisonment severity
def i_hate_historical_record_keeping(x):
    if pd.isna(x):
        return np.nan
    x = str(x).lower().strip()
    try:
        parts = x.split()
        number = int(parts[0])
        unit = parts[1]
        if 'day' in unit:
            return number / 30
        elif 'week' in unit:
            return number * 7 / 30
        elif 'month' in unit:
            return number
        elif 'year' in unit:
            return number * 12
        else:
            return np.nan
    except:
        return np.nan

hypothesis3_df_culled['imprison_months'] = hypothesis3_df_culled['imprisonment_length'].apply(i_hate_historical_record_keeping)

def imprisonment_severity(months):
    if pd.isna(months):
        return np.nan
    elif months <= 6:
        return 0
    elif months <= 24:
        return 1
    elif months <= 60:
        return 2
    elif months <= 120:
        return 3
    elif months < 240:
        return 4
    else:
        return 5

hypothesis3_df_culled['imprison_severity'] = hypothesis3_df_culled['imprison_months'].apply(imprisonment_severity)

# Fine severity
def how_many_pennies_innit(row):
    pounds = row.get('fine_value_pounds', 0) or 0
    shillings = row.get('fine_value_shillings', 0) or 0
    pence = row.get('fine_value_pence', 0) or 0
    guineas = row.get('fine_value_guineas', 0) or 0
    marks = row.get('fine_value_marks', 0) or 0
    return pounds*240 + shillings*12 + pence + guineas*252 + marks*160

hypothesis3_df_culled['fine_total_pence'] = hypothesis3_df_culled.apply(how_many_pennies_innit, axis=1)

def send_500_pennies_to_israel(pence):
    if pd.isna(pence) or pence == 0:
        return 0
    elif pence <= 50*12:
        return 1
    elif pence <= 200*12:
        return 2
    elif pence <= 1000*12:
        return 3
    else:
        return 4

hypothesis3_df_culled['fine_severity'] = hypothesis3_df_culled['fine_total_pence'].apply(send_500_pennies_to_israel)

hypothesis3_df_culled['punishment_sub_rank'] = 0
hypothesis3_df_culled.loc[hypothesis3_df_culled['punishment'] == 'imprison', 'punishment_sub_rank'] = hypothesis3_df_culled['imprison_severity']
hypothesis3_df_culled.loc[hypothesis3_df_culled['punishment'] == 'miscPunish', 'punishment_sub_rank'] = hypothesis3_df_culled['fine_severity']

# Build model
outcome_candidates = ["punishment_severity", "punishment_score", "punishment_sub_rank", "punishment_rank"]
outcome_col = next((c for c in outcome_candidates if c in hypothesis3_df_culled.columns), None)

the_code_bouncer = ["defendant_gender", "occupation_grouped", "session_year", outcome_col]
john_model = hypothesis3_df_culled.dropna(subset=the_code_bouncer).copy()

john_model["session_year"] = pd.to_numeric(john_model["session_year"], errors="coerce")
john_model[outcome_col] = pd.to_numeric(john_model[outcome_col], errors="coerce")
john_model = john_model.dropna(subset=["session_year", outcome_col])

john_model["y_cat"] = john_model[outcome_col].round().astype(int)

categories = sorted(john_model["y_cat"].unique())
df_model = john_model[john_model["y_cat"].isin(categories)].copy()

X = pd.get_dummies(john_model[["defendant_gender", "occupation_grouped"]], drop_first=True)
john_model["session_year_centred"] = john_model["session_year"] - john_model["session_year"].mean()
X["session_year_centred"] = john_model["session_year_centred"]
X = X.astype(float)

y = pd.Categorical(john_model["y_cat"], ordered=True, categories=categories)

model = OrderedModel(y, X, distr="logit")
res = model.fit(method="bfgs", disp=0)

# NOW THE SHINY APP
X_columns = list(X.columns)
year_mean = john_model["session_year"].mean()

punishment_labels = [f'Level {i}' for i in categories]
gender_options = ['female','male']
occupation_options = sorted(john_model['occupation_grouped'].unique())

def make_profile(gender="female", occ="other", year=1780):
    year_c = year - year_mean
    row = {col: 0.0 for col in X_columns}
    
    gcol = f"defendant_gender_{gender}"
    if gcol in row:
        row[gcol] = 1.0

    ocol = f"occupation_grouped_{occ}"
    if ocol in row:
        row[ocol] = 1.0

    if "session_year_centred" in row:
        row["session_year_centred"] = float(year_c)
    else:
        row["session_year_centered"] = float(year_c)

    return pd.DataFrame([row], columns=X_columns).astype(float)

def get_prediction(gender, occupation, year):
    profile = make_profile(gender=gender, occ=occupation, year=year)
    probs = res.model.predict(res.params, exog=profile)[0]
    return probs

app_ui = ui.page_fluid(
    ui.h2("Old Bailey Punishment Predictor"),
    ui.p("Select defendant characteristics to see predicted punishment probabilities"),
    ui.hr(),

    ui.layout_sidebar(
        ui.sidebar(
            ui.input_select("gender", "Defendant Gender:", choices=gender_options, selected="male"),
            ui.input_select("occupation", "Defendant Occupation:", choices=occupation_options, selected=occupation_options[0]),
            ui.input_slider("year", "Year of Trial:", min=1674, max=1913, value=1780, step=1, sep=""),
            width=300
        ),

        ui.card(
            ui.card_header("Predicted Punishment Probabilities"),
            ui.output_plot("prob_plot", height="400px"),
        ),
        ui.card(
            ui.card_header("Summary"),
            ui.output_text("summary_text")
        )
    )
)

def server(input, output, session):

    @reactive.calc
    def probabilities():
        return get_prediction(
            gender=input.gender(),
            occupation=input.occupation(),
            year=input.year()
        )

    @render.plot
    def prob_plot():
        probs = probabilities()

        fig, ax = plt.subplots(figsize=(10, 5))
        x_pos = range(len(punishment_labels))
        bars = ax.bar(x_pos, probs * 100, edgecolor="black", linewidth=0.8)

        ax.set_xticks(x_pos)
        ax.set_xticklabels(punishment_labels, rotation=45, ha="right")
        ax.set_ylabel("Probability (%)")
        ax.set_ylim(0, 100)
        ax.set_title(f"Predicted punishment for {input.gender().title()} {input.occupation().title()} in {input.year()}")

        for bar, prob in zip(bars, probs):
            if prob > 0.01:
                ax.annotate(f"{prob*100:.1f}%",
                            (bar.get_x() + bar.get_width()/2, bar.get_height()),
                            ha="center", va="bottom", fontsize=10)

        plt.tight_layout()
        return fig

    @render.text
    def summary_text():
        probs = probabilities()
        max_idx = int(np.argmax(probs))
        max_prob = probs[max_idx] * 100
        max_punishment = punishment_labels[max_idx]

        return (
            f"Most likely punishment: {max_punishment} ({max_prob:.1f}%)."
        )

app = App(app_ui, server)