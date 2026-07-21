import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st

from src.analysis_engine import (
    AnalysisConfiguration,
    calculate_default_rates,
    create_engineered_features,
    load_uploaded_file,
    profile_dataset,
    select_recommended_model,
    standardize_column_names,
    train_and_compare_models,
)


st.set_page_config(
    page_title="Intelligent Data Analysis Agent",
    page_icon="📊",
    layout="wide",
)


def format_percentage_columns(
    dataframe: pd.DataFrame,
) -> pd.DataFrame:
    """Round model metrics for clearer presentation."""

    formatted = dataframe.copy()

    metric_columns = [
        column
        for column in formatted.columns
        if column != "model"
    ]

    formatted[metric_columns] = formatted[
        metric_columns
    ].round(3)

    return formatted


def create_confusion_matrix_chart(
    matrix,
    model_name: str,
):
    """Create a simple confusion-matrix chart."""

    figure, axis = plt.subplots()

    image = axis.imshow(matrix)

    axis.set_title(f"{model_name}: Confusion Matrix")
    axis.set_xlabel("Predicted class")
    axis.set_ylabel("Actual class")

    axis.set_xticks([0, 1])
    axis.set_yticks([0, 1])

    for row in range(matrix.shape[0]):
        for column in range(matrix.shape[1]):
            axis.text(
                column,
                row,
                str(matrix[row, column]),
                ha="center",
                va="center",
            )

    figure.colorbar(image)

    return figure


st.title("Intelligent Data Analysis and Machine-Learning Agent")

st.write(
    """
    Upload a dataset, describe the assignment, review data quality,
    engineer features and compare classification models.
    """
)

with st.sidebar:
    st.header("Assignment information")

    assignment_description = st.text_area(
        "Describe the assignment",
        placeholder=(
            "Example: Predict which clients are likely to default "
            "and identify the most important risk factors."
        ),
        height=150,
    )

    expected_output = st.text_area(
        "What outputs do you expect?",
        placeholder=(
            "Example: Clean dataset, model comparison, recommended "
            "model and concise management report."
        ),
        height=120,
    )

    business_priority = st.selectbox(
        "Model-selection priority",
        options=[
            "Balanced performance",
            "Precision",
            "Recall",
        ],
        help=(
            "Select Recall when missing a true positive is especially "
            "costly. Select Precision when false alarms are costly."
        ),
    )


dataset_file = st.file_uploader(
    "Upload the dataset",
    type=["csv", "xlsx"],
)

codebook_file = st.file_uploader(
    "Upload a codebook or data dictionary, if available",
    type=["csv", "xlsx"],
)


if dataset_file is None:
    st.info("Upload a CSV or Excel dataset to begin.")
    st.stop()


try:
    raw_data = load_uploaded_file(dataset_file)
    data = standardize_column_names(raw_data)

except Exception as error:
    st.error(f"Could not load the dataset: {error}")
    st.stop()


profile = profile_dataset(data)

st.success(
    f"Dataset loaded successfully: "
    f"{profile['rows']:,} rows and "
    f"{profile['columns']:,} columns."
)


tab1, tab2, tab3, tab4 = st.tabs(
    [
        "Dataset",
        "Data quality",
        "Configure analysis",
        "Models and results",
    ]
)


with tab1:
    st.subheader("Dataset preview")

    st.dataframe(
        data.head(50),
        use_container_width=True,
    )

    st.subheader("Column information")

    st.dataframe(
        profile["column_summary"],
        use_container_width=True,
    )

    if codebook_file is not None:
        try:
            codebook = load_uploaded_file(codebook_file)
            codebook = standardize_column_names(codebook)

            st.subheader("Codebook preview")

            st.dataframe(
                codebook.head(50),
                use_container_width=True,
            )

            st.warning(
                "The current version displays the codebook. "
                "Detailed value-level codebook validation can be "
                "added once its variable-name and allowed-values "
                "columns are identified."
            )

        except Exception as error:
            st.error(f"Could not load the codebook: {error}")


with tab2:
    left_column, right_column = st.columns(2)

    with left_column:
        st.metric(
            "Duplicate rows",
            profile["duplicate_count"],
        )

        st.write("#### Missing values")

        st.dataframe(
            profile["missing_values"],
            use_container_width=True,
        )

    with right_column:
        st.write("#### Possible identifier columns")

        if profile["possible_identifier_columns"]:
            st.write(
                profile["possible_identifier_columns"]
            )
        else:
            st.write("No obvious unique identifier columns detected.")

        st.write("#### Constant columns")

        if profile["constant_columns"]:
            st.write(profile["constant_columns"])
        else:
            st.write("No constant columns detected.")

    st.write("#### Numeric summary")

    if profile["numeric_summary"].empty:
        st.info("No numeric variables were detected.")
    else:
        st.dataframe(
            profile["numeric_summary"],
            use_container_width=True,
        )


with tab3:
    st.subheader("Configure the classification analysis")

    target_column = st.selectbox(
        "Select the target variable",
        options=data.columns.tolist(),
    )

    age_options = ["Not available"] + data.columns.tolist()

    age_column = st.selectbox(
        "Select the age variable",
        options=age_options,
    )

    default_months_column = st.selectbox(
        "Select the number-of-months-defaulted variable",
        options=["Not available"] + data.columns.tolist(),
    )

    excluded_columns = st.multiselect(
        "Select identifiers or leakage variables to exclude",
        options=[
            column
            for column in data.columns
            if column != target_column
        ],
        default=[
            column
            for column in profile["possible_identifier_columns"]
            if column != target_column
        ],
    )

    test_size = st.slider(
        "Test-data percentage",
        min_value=10,
        max_value=40,
        value=20,
        step=5,
    )

    run_analysis = st.button(
        "Run machine-learning analysis",
        type="primary",
    )


if run_analysis:
    try:
        engineered_data = create_engineered_features(
            data,
            age_column=(
                None
                if age_column == "Not available"
                else age_column
            ),
            default_months_column=(
                None
                if default_months_column == "Not available"
                else default_months_column
            ),
        )

        configuration = AnalysisConfiguration(
            target_column=target_column,
            test_size=test_size / 100,
            business_priority=business_priority,
            age_column=(
                None
                if age_column == "Not available"
                else age_column
            ),
            default_months_column=(
                None
                if default_months_column == "Not available"
                else default_months_column
            ),
            excluded_columns=excluded_columns,
        )

        with st.spinner("Training and comparing models..."):
            comparison, fitted_models, model_details = (
                train_and_compare_models(
                    engineered_data,
                    configuration,
                )
            )

        recommended_model = select_recommended_model(
            comparison,
            business_priority,
        )

        st.session_state["engineered_data"] = engineered_data
        st.session_state["comparison"] = comparison
        st.session_state["model_details"] = model_details
        st.session_state["recommended_model"] = (
            recommended_model
        )
        st.session_state["target_column"] = target_column

        st.success("Analysis completed successfully.")

    except Exception as error:
        st.error(f"Analysis could not be completed: {error}")


with tab4:
    if "comparison" not in st.session_state:
        st.info(
            "Configure the analysis and select "
            "'Run machine-learning analysis'."
        )

    else:
        comparison = st.session_state["comparison"]
        engineered_data = st.session_state["engineered_data"]
        model_details = st.session_state["model_details"]
        recommended_model = st.session_state[
            "recommended_model"
        ]
        target_column = st.session_state["target_column"]

        st.subheader("Model comparison")

        st.dataframe(
            format_percentage_columns(comparison),
            use_container_width=True,
        )

        st.success(
            f"Recommended model: {recommended_model}"
        )

        st.write(
            """
            The recommendation reflects the selected business priority.
            Review precision, recall, F1-score, ROC-AUC, PR-AUC and
            log loss before making a final operational decision.
            """
        )

        st.subheader("Confusion matrices")

        for model_name, matrix in model_details[
            "confusion_matrices"
        ].items():
            chart = create_confusion_matrix_chart(
                matrix,
                model_name,
            )

            st.pyplot(chart)

        st.subheader("Default-rate summaries")

        grouping_columns = [
            "age_category",
            "gender",
            "marital_status",
            "default_duration_category",
        ]

        rate_tables = calculate_default_rates(
            engineered_data,
            target_column=target_column,
            grouping_columns=grouping_columns,
        )

        if not rate_tables:
            st.info(
                "No requested grouping variables were available."
            )

        for group_name, table in rate_tables.items():
            st.write(
                f"#### Default rate by "
                f"{group_name.replace('_', ' ').title()}"
            )

            st.dataframe(
                table,
                use_container_width=True,
            )

        st.subheader("Download results")

        cleaned_csv = engineered_data.to_csv(
            index=False
        ).encode("utf-8")

        comparison_csv = comparison.to_csv(
            index=False
        ).encode("utf-8")

        st.download_button(
            label="Download cleaned dataset",
            data=cleaned_csv,
            file_name="cleaned_dataset.csv",
            mime="text/csv",
        )

        st.download_button(
            label="Download model comparison",
            data=comparison_csv,
            file_name="model_comparison.csv",
            mime="text/csv",
        )