import pandas as pd
import tkinter as tk
from tkinter import ttk, messagebox
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

# Load Dataset

df = pd.read_csv("student_performance.csv")

df.columns = [
    col.strip().lower().replace(" ", "_").replace("/", "_")
    for col in df.columns
]

df["result"] = df["math_score"].apply(lambda x: 1 if x >= 50 else 0)

X = df.drop(["math_score", "result"], axis=1)
y = df["result"]

categorical_cols = X.select_dtypes(include=["object", "string"]).columns.tolist()
numeric_cols = X.select_dtypes(include=["number"]).columns.tolist()

# Model
preprocessor = ColumnTransformer([
    ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_cols),
    ("num", StandardScaler(), numeric_cols)
])

model = Pipeline([
    ("preprocessor", preprocessor),
    ("classifier", LogisticRegression(max_iter=1000))
])

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

model.fit(X_train, y_train)
y_pred = model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred, zero_division=0)
recall = recall_score(y_test, y_pred, zero_division=0)
f1 = f1_score(y_test, y_pred, zero_division=0)
cm = confusion_matrix(y_test, y_pred)

BG = "#E8EEF6"
DARK = "#0B132B"
CARD = "#F8FAFC"
TEXT = "#0F172A"
MUTED = "#64748B"
BORDER = "#CBD5E1"
BLUE = "#2563EB"
GREEN = "#16A34A"
RED = "#DC2626"
CYAN = "#06B6D4"
PURPLE = "#8B5CF6"

# Functions

def predict_result():
    try:
        reading = float(reading_entry.get())
        writing = float(writing_entry.get())

        if not (0 <= reading <= 100 and 0 <= writing <= 100):
            messagebox.showerror("Input Error", "Scores must be between 0 and 100")
            return

        data = pd.DataFrame([{
            "gender": gender_var.get(),
            "race_ethnicity": race_var.get(),
            "parental_level_of_education": education_var.get(),
            "lunch": lunch_var.get(),
            "test_preparation_course": prep_var.get(),
            "reading_score": reading,
            "writing_score": writing
        }])

        prediction = model.predict(data)[0]
        probabilities = model.predict_proba(data)[0]
        fail_conf = probabilities[0] * 100
        pass_conf = probabilities[1] * 100
        confidence = probabilities[prediction] * 100

        update_confidence_chart(fail_conf, pass_conf)
        update_gauge(confidence, prediction)

        if prediction == 1:
            result_card.config(bg="#DCFCE7", highlightbackground="#86EFAC")
            result_title.config(text="PASS ✅", bg="#DCFCE7", fg="#15803D")
            result_confidence.config(text=f"Confidence: {confidence:.2f}%", bg="#DCFCE7", fg="#166534")
            result_hint.config(text="The student is predicted to pass based on the entered data.", bg="#DCFCE7", fg="#166534")
            recommendation_label.config(
                text="Recommendation: Keep the same study plan and continue improving problem-solving skills.",
                bg="#DCFCE7",
                fg="#166534"
            )
        else:
            result_card.config(bg="#FEE2E2", highlightbackground="#FCA5A5")
            result_title.config(text="FAIL ❌", bg="#FEE2E2", fg="#B91C1C")
            result_confidence.config(text=f"Confidence: {confidence:.2f}%", bg="#FEE2E2", fg="#991B1B")
            result_hint.config(text="The student may need academic support and improvement.", bg="#FEE2E2", fg="#991B1B")
            recommendation_label.config(
                text="Recommendation: Focus on reading and writing practice, review weak topics, and retake preparation exercises.",
                bg="#FEE2E2",
                fg="#991B1B"
            )

    except ValueError:
        messagebox.showerror("Input Error", "Please enter numeric scores only.")


def reset_fields():
    gender_combo.current(0)
    race_combo.current(0)
    education_combo.current(0)
    lunch_combo.current(0)
    prep_combo.current(0)

    reading_entry.delete(0, tk.END)
    writing_entry.delete(0, tk.END)

    result_card.config(bg="#E0F2FE", highlightbackground="#BAE6FD")
    result_title.config(text="Prediction Result", bg="#E0F2FE", fg="#075985")
    result_confidence.config(text="Enter student data and click Predict", bg="#E0F2FE", fg="#075985")
    result_hint.config(text="", bg="#E0F2FE", fg="#075985")
    recommendation_label.config(text="Recommendation will appear after prediction.", bg="#E0F2FE", fg="#075985")

    update_confidence_chart(50, 50)
    update_gauge(0, None)


def on_mousewheel(event):
    main_canvas.yview_scroll(int(-1 * (event.delta / 120)), "units")


def make_label(parent, text):
    tk.Label(parent, text=text, font=("Segoe UI", 10, "bold"), bg=CARD, fg=TEXT).pack(anchor="w", pady=(9, 4))


def make_combo(parent, variable, values):
    combo = ttk.Combobox(parent, textvariable=variable, values=values, state="readonly", font=("Segoe UI", 10))
    combo.current(0)
    combo.pack(fill="x", ipady=5)
    return combo


def make_entry(parent):
    entry = tk.Entry(parent, font=("Segoe UI", 11), bg="white", fg=TEXT, bd=1, relief="solid")
    entry.pack(fill="x", ipady=7)
    return entry


def make_button(parent, text, command, bg, hover):
    btn = tk.Button(
        parent,
        text=text,
        command=command,
        font=("Segoe UI", 12, "bold"),
        bg=bg,
        fg="white",
        activebackground=hover,
        activeforeground="white",
        bd=0,
        cursor="hand2",
        height=2
    )
    btn.bind("<Enter>", lambda e: btn.config(bg=hover))
    btn.bind("<Leave>", lambda e: btn.config(bg=bg))
    btn.pack(fill="x", pady=7)
    return btn


def metric_card(parent, title, value):
    card = tk.Frame(parent, bg="white", padx=12, pady=11, highlightbackground=BORDER, highlightthickness=1)
    card.pack(side="left", fill="both", expand=True, padx=6)

    tk.Label(card, text=title, font=("Segoe UI", 9, "bold"), bg="white", fg=MUTED).pack()
    tk.Label(card, text=value, font=("Segoe UI", 17, "bold"), bg="white", fg=TEXT).pack(pady=(3, 0))


def small_info_card(parent, title, value, color):
    card = tk.Frame(parent, bg=color, padx=10, pady=10)
    card.pack(side="left", fill="both", expand=True, padx=5)

    tk.Label(card, text=title, font=("Segoe UI", 8, "bold"), bg=color, fg="white").pack()
    tk.Label(card, text=value, font=("Segoe UI", 14, "bold"), bg=color, fg="white").pack(pady=(4, 0))


def update_gauge(confidence=0, prediction=None):
    gauge_canvas.delete("all")

    width = 620
    x1, y1 = 45, 42
    x2, y2 = 575, 74

    gauge_canvas.create_text(width // 2, 18, text="Confidence Gauge", font=("Segoe UI", 12, "bold"), fill=TEXT)
    gauge_canvas.create_rectangle(x1, y1, x2, y2, fill="#E5E7EB", outline=BORDER, width=1)

    fill_width = (x2 - x1) * (confidence / 100)

    if prediction == 1:
        color = "#22C55E"
        label = "PASS Confidence"
    elif prediction == 0:
        color = "#EF4444"
        label = "FAIL Confidence"
    else:
        color = "#94A3B8"
        label = "Waiting for prediction"

    gauge_canvas.create_rectangle(x1, y1, x1 + fill_width, y2, fill=color, outline="")
    gauge_canvas.create_text(width // 2, 96, text=f"{label}: {confidence:.2f}%", font=("Segoe UI", 11, "bold"), fill=TEXT)


def update_confidence_chart(fail_value=50, pass_value=50):
    confidence_fig.clear()
    ax = confidence_fig.add_subplot(111)

    bars = ax.bar(["Fail", "Pass"], [fail_value, pass_value], color=["#EF4444", "#22C55E"])
    ax.set_title("Dynamic Prediction Confidence", fontsize=11, fontweight="bold", pad=14)
    ax.set_ylabel("Confidence (%)")
    ax.set_ylim(0, 110)
    ax.grid(axis="y", alpha=0.22)

    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2, min(height + 3, 104), f"{height:.1f}%", ha="center", fontsize=9, fontweight="bold")

    confidence_fig.tight_layout(pad=2.0)
    confidence_canvas.draw()


def draw_confusion_matrix(parent):
    fig = Figure(figsize=(4.7, 2.9), dpi=100)
    ax = fig.add_subplot(111)
    image = ax.imshow(cm, cmap="Blues")

    ax.set_title("Confusion Matrix", fontsize=11, fontweight="bold")
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    ax.set_xticklabels(["Fail", "Pass"])
    ax.set_yticklabels(["Fail", "Pass"])

    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, cm[i, j], ha="center", va="center", color=TEXT, fontsize=12, fontweight="bold")

    fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout(pad=1.4)

    canvas = FigureCanvasTkAgg(fig, master=parent)
    canvas.draw()
    canvas.get_tk_widget().pack(fill="both", expand=True)


def draw_average_scores(parent):
    avg_math = df["math_score"].mean()
    avg_reading = df["reading_score"].mean()
    avg_writing = df["writing_score"].mean()

    fig = Figure(figsize=(4.7, 2.45), dpi=100)
    ax = fig.add_subplot(111)

    bars = ax.bar(["Math", "Reading", "Writing"], [avg_math, avg_reading, avg_writing], color=[BLUE, CYAN, PURPLE])

    ax.set_title("Average Student Scores", fontsize=11, fontweight="bold")
    ax.set_ylabel("Average Score")
    ax.set_ylim(0, 100)
    ax.grid(axis="y", alpha=0.22)

    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2, height + 1.5, f"{height:.1f}", ha="center", fontsize=9, fontweight="bold")

    fig.tight_layout(pad=1.4)
    canvas = FigureCanvasTkAgg(fig, master=parent)
    canvas.draw()
    canvas.get_tk_widget().pack(fill="both", expand=True)

# GUI Design
root = tk.Tk()
root.title("AI Student Performance Prediction System")
root.state("zoomed")
root.configure(bg=BG)

style = ttk.Style()
style.theme_use("clam")
style.configure("TCombobox", fieldbackground="white", background="white", foreground=TEXT, padding=5)

header = tk.Frame(root, bg=DARK, height=95)
header.pack(fill="x")
header.pack_propagate(False)

tk.Label(header, text="AI Student Performance Prediction System", font=("Segoe UI", 26, "bold"), bg=DARK, fg="white").pack(pady=(15, 0))
tk.Label(header, text="Machine Learning Project | Logistic Regression | Enhanced Tkinter GUI", font=("Segoe UI", 11), bg=DARK, fg="#CBD5E1").pack(pady=(3, 0))

scroll_container = tk.Frame(root, bg=BG)
scroll_container.pack(fill="both", expand=True)

main_canvas = tk.Canvas(scroll_container, bg=BG, highlightthickness=0)
main_canvas.pack(side="left", fill="both", expand=True)

scrollbar = ttk.Scrollbar(scroll_container, orient="vertical", command=main_canvas.yview)
scrollbar.pack(side="right", fill="y")

main_canvas.configure(yscrollcommand=scrollbar.set)

content = tk.Frame(main_canvas, bg=BG)
content_window = main_canvas.create_window((0, 0), window=content, anchor="nw")

def configure_scroll_region(event=None):
    main_canvas.configure(scrollregion=main_canvas.bbox("all"))

def configure_canvas_width(event):
    main_canvas.itemconfig(content_window, width=event.width)

content.bind("<Configure>", configure_scroll_region)
main_canvas.bind("<Configure>", configure_canvas_width)
main_canvas.bind_all("<MouseWheel>", on_mousewheel)

main = tk.Frame(content, bg=BG)
main.pack(fill="both", expand=True, padx=30, pady=22)

left = tk.Frame(main, bg=CARD, padx=24, pady=18, width=350, height=660, highlightbackground="#94A3B8", highlightthickness=1)
left.pack(side="left", fill="y", expand=False)
left.pack_propagate(False)

center = tk.Frame(main, bg=BG, width=700)
center.pack(side="left", fill="both", expand=True, padx=18)

right = tk.Frame(main, bg=CARD, padx=20, pady=18, width=520, height=980, highlightbackground="#94A3B8", highlightthickness=1)
right.pack(side="right", fill="y", expand=False)
right.pack_propagate(False)

gender_var = tk.StringVar()
race_var = tk.StringVar()
education_var = tk.StringVar()
lunch_var = tk.StringVar()
prep_var = tk.StringVar()

tk.Label(left, text="Student Information", font=("Segoe UI", 18, "bold"), bg=CARD, fg=TEXT).pack(anchor="w", pady=(0, 12))

make_label(left, "Gender")
gender_combo = make_combo(left, gender_var, sorted(df["gender"].unique()))

make_label(left, "Race / Ethnicity")
race_combo = make_combo(left, race_var, sorted(df["race_ethnicity"].unique()))

make_label(left, "Parental Level of Education")
education_combo = make_combo(left, education_var, sorted(df["parental_level_of_education"].unique()))

make_label(left, "Lunch Type")
lunch_combo = make_combo(left, lunch_var, sorted(df["lunch"].unique()))

make_label(left, "Test Preparation Course")
prep_combo = make_combo(left, prep_var, sorted(df["test_preparation_course"].unique()))

make_label(left, "Reading Score")
reading_entry = make_entry(left)

make_label(left, "Writing Score")
writing_entry = make_entry(left)

make_button(left, "🔮 Predict Result", predict_result, BLUE, "#1D4ED8")
make_button(left, "🔄 Reset", reset_fields, RED, "#B91C1C")

metrics_frame = tk.Frame(center, bg=BG)
metrics_frame.pack(fill="x", pady=(0, 18))

metric_card(metrics_frame, "Accuracy", f"{accuracy:.2f}")
metric_card(metrics_frame, "Precision", f"{precision:.2f}")
metric_card(metrics_frame, "Recall", f"{recall:.2f}")
metric_card(metrics_frame, "F1-score", f"{f1:.2f}")

result_card = tk.Frame(center, bg="#E0F2FE", padx=25, pady=22, height=210, highlightbackground="#BAE6FD", highlightthickness=2)
result_card.pack(fill="x", pady=(0, 18))
result_card.pack_propagate(False)

result_title = tk.Label(result_card, text="Prediction Result", font=("Segoe UI", 25, "bold"), bg="#E0F2FE", fg="#075985")
result_title.pack(pady=(0, 5))

result_confidence = tk.Label(result_card, text="Enter student data and click Predict", font=("Segoe UI", 13, "bold"), bg="#E0F2FE", fg="#075985")
result_confidence.pack(pady=(4, 0))

result_hint = tk.Label(result_card, text="", font=("Segoe UI", 10), bg="#E0F2FE", fg="#075985", wraplength=620)
result_hint.pack(pady=(5, 0))

recommendation_label = tk.Label(
    result_card,
    text="Recommendation will appear after prediction.",
    font=("Segoe UI", 10, "bold"),
    bg="#E0F2FE",
    fg="#075985",
    wraplength=650
)
recommendation_label.pack(pady=(8, 0))

gauge_card = tk.Frame(center, bg=CARD, padx=15, pady=10, height=150, highlightbackground=BORDER, highlightthickness=1)
gauge_card.pack(fill="x", pady=(0, 18))
gauge_card.pack_propagate(False)

gauge_canvas = tk.Canvas(gauge_card, width=620, height=115, bg=CARD, highlightthickness=0)
gauge_canvas.pack(fill="x")
update_gauge(0, None)

chart_card = tk.Frame(center, bg=CARD, padx=15, pady=12, height=395, highlightbackground=BORDER, highlightthickness=1)
chart_card.pack(fill="x", pady=(0, 18))
chart_card.pack_propagate(False)

tk.Label(chart_card, text="Dynamic Confidence Chart", font=("Segoe UI", 15, "bold"), bg=CARD, fg=TEXT).pack(anchor="w", pady=(0, 6))

confidence_fig = Figure(figsize=(6.0, 3.0), dpi=100)
confidence_canvas = FigureCanvasTkAgg(confidence_fig, master=chart_card)
confidence_canvas.get_tk_widget().pack(fill="both", expand=True, pady=4)
update_confidence_chart(50, 50)

tk.Label(right, text="Model Analysis", font=("Segoe UI", 18, "bold"), bg=CARD, fg=TEXT).pack(anchor="w", pady=(0, 12))

summary_cards = tk.Frame(right, bg=CARD, height=74)
summary_cards.pack(fill="x", pady=(0, 12))
summary_cards.pack_propagate(False)

small_info_card(summary_cards, "Students", str(len(df)), BLUE)
small_info_card(summary_cards, "Pass", str(int(df["result"].sum())), GREEN)
small_info_card(summary_cards, "Fail", str(int(len(df) - df["result"].sum())), RED)

cm_card = tk.Frame(right, bg="white", padx=10, pady=10, height=320, highlightbackground=BORDER, highlightthickness=1)
cm_card.pack(fill="x", pady=(0, 14))
cm_card.pack_propagate(False)
draw_confusion_matrix(cm_card)

avg_card = tk.Frame(right, bg="white", padx=10, pady=10, height=270, highlightbackground=BORDER, highlightthickness=1)
avg_card.pack(fill="x", pady=(0, 14))
avg_card.pack_propagate(False)
draw_average_scores(avg_card)

info_card = tk.Frame(right, bg="white", padx=16, pady=14, height=230, highlightbackground=BORDER, highlightthickness=1)
info_card.pack(fill="x")
info_card.pack_propagate(False)

tk.Label(info_card, text="Project Summary", font=("Segoe UI", 12, "bold"), bg="white", fg=TEXT).pack(anchor="w", pady=(0, 6))

info_text = (
    "This system predicts whether a student will pass or fail using machine learning.\n\n"
    "Input Features:\n"
    "Gender, Race / Ethnicity, Parental Education, Lunch Type, Test Preparation, Reading Score, and Writing Score.\n\n"
    "Target:\n"
    "Pass if Math Score >= 50, otherwise Fail."
)

tk.Label(
    info_card,
    text=info_text,
    font=("Segoe UI", 10),
    bg="white",
    fg="#334155",
    justify="left",
    anchor="nw",
    wraplength=455
).pack(anchor="nw", fill="both", expand=True)

footer = tk.Frame(content, bg=DARK, height=42)
footer.pack(fill="x", pady=(8, 0))
footer.pack_propagate(False)

tk.Label(
    footer,
    text="Developed for AI / Machine Learning Project  |  Student Performance Prediction",
    font=("Segoe UI", 9),
    bg=DARK,
    fg="#CBD5E1"
).pack(pady=10)

root.mainloop()
