# Machine Learning for Data Classification and Clustering

## Project Overview

This Python project utilizes Streamlit to create an interactive web application for performing both supervised and unsupervised machine learning tasks on user-uploaded CSV or TXT datasets. The application offers two main functionalities:

1. **Supervised Learning (Random Forest Classifier):** Users can upload a labeled dataset and specify the maximum depth parameter for the Random Forest Classifier. The application then trains the classifier, evaluates its performance, and displays metrics such as accuracy, precision, recall, and F1 score.

2. **Unsupervised Learning (Hierarchical Clustering):** Users can upload an unlabeled dataset and determine the number of clusters for Hierarchical Clustering. The application performs clustering, calculates the silhouette score, and presents the Calinski-Harabasz index.

## Dependencies

Before running the project, ensure you have the following Python libraries installed:

- `numpy`
- `pandas`
- `streamlit`
- `scikit-learn`

You can install these libraries using pip:

```bash
pip install numpy pandas streamlit scikit-learn
```

## How to Use

Follow these steps to use the application:

1. Run the script by executing `streamlit run script_name.py` in your terminal.

2. Access the web application through your web browser.

3. Upload a dataset in CSV or TXT format. The application automatically detects the format based on the file extension.

4. Adjust the machine learning parameters using sliders:
   - For supervised learning, set the maximum depth for the Random Forest Classifier.
   - For unsupervised learning, determine the number of clusters for Hierarchical Clustering.

5. Click the "Run Machine Learning" button to start the selected machine learning process.

6. The application displays evaluation results, including accuracy, precision, recall, F1 score, silhouette score, and the Calinski-Harabasz index.

## Project Structure

The project comprises the following key components:

- `perform_ml()`: A function that performs both supervised and unsupervised machine learning tasks, including training a Random Forest Classifier and conducting Hierarchical Clustering.

- `app()`: The Streamlit application function that creates the user interface, handles file uploads, and presents evaluation results.

- File uploader: Users can upload datasets in CSV or TXT format.

- Parameter sliders: Sliders allow users to customize machine learning parameters.

- Evaluation results table: Displays metrics and scores from the machine learning process.

## Future Enhancements

This project serves as a versatile foundation for data classification and clustering tasks. Future enhancements could include:

- Support for additional machine learning algorithms, allowing users to choose different classifiers or clustering methods.

- Visualizations: Integration of data visualization tools to provide insights into the uploaded datasets and the results of machine learning processes.

- Improved user experience: Enhance the user interface with additional features and user-friendly design elements.

- Deployment: Deploy the application to a web server for broader accessibility.

- Error handling: Implement error handling for cases where users upload incompatible or erroneous datasets.

Feel free to explore and expand upon this project to make it even more powerful and user-friendly for various machine learning tasks.
