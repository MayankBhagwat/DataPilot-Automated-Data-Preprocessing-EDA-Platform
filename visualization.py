import matplotlib.pyplot as plt
import seaborn as sns
import os


def correlation_heatmap(df):

    os.makedirs("static/charts", exist_ok=True)

    numeric_df = df.select_dtypes(include=['number'])

    plt.figure(figsize=(10, 6))

    sns.heatmap(
        numeric_df.corr(),
        annot=True,
        cmap='coolwarm'
    )

    chart_path = "static/charts/correlation_heatmap.png"

    plt.savefig(chart_path)

    plt.close()

    return chart_path


def missing_values_chart(df):

    os.makedirs("static/charts", exist_ok=True)

    missing = df.isnull().sum()

    plt.figure(figsize=(10, 5))

    missing.plot(kind='bar')

    plt.title("Missing Values")

    chart_path = "static/charts/missing_values.png"

    plt.savefig(chart_path)

    plt.close()

    return chart_path