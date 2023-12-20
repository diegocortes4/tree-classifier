import weka.classifiers.trees.J48;  // Import Decision Tree classifier
import weka.core.Instance;

public class MachineLearningExample {

    public static void main(String[] args) {
        try {
            // Load dataset
            DataSource source = new DataSource("path/to/your/dataset.arff");
            Instances dataset = source.getDataSet();

            // Set the class index (assuming the last attribute is the class)
            if (dataset.classIndex() == -1) {
                dataset.setClassIndex(dataset.numAttributes() - 1);
            }

            // Create and build the linear regression model
            LinearRegression linearRegressionModel = new LinearRegression();
            linearRegressionModel.buildClassifier(dataset);

            // Create and build the decision tree model
            J48 decisionTreeModel = new J48();
            decisionTreeModel.buildClassifier(dataset);

            // Example: Make a prediction using linear regression
            Instance newInstance = dataset.get(0);  // Using the first instance as an example
            double linearRegressionPrediction = linearRegressionModel.classifyInstance(newInstance);
            System.out.println("Linear Regression Prediction: " + linearRegressionPrediction);

            // Example: Make a prediction using decision tree
            double decisionTreePrediction = decisionTreeModel.classifyInstance(newInstance);
            System.out.println("Decision Tree Prediction: " + decisionTreePrediction);

            // Evaluate the models
            Evaluation linearRegressionEvaluation = new Evaluation(dataset);
            linearRegressionEvaluation.crossValidateModel(linearRegressionModel, dataset, 10, new java.util.Random(1));

            Evaluation decisionTreeEvaluation = new Evaluation(dataset);
            decisionTreeEvaluation.crossValidateModel(decisionTreeModel, dataset, 10, new java.util.Random(1));

            // Output evaluation results
            System.out.println("===== Linear Regression Model Evaluation =====");
            System.out.println("Mean Absolute Error: " + linearRegressionEvaluation.meanAbsoluteError());
            System.out.println("Root Mean Squared Error: " + linearRegressionEvaluation.rootMeanSquaredError());
            System.out.println("Correlation Coefficient: " + linearRegressionEvaluation.correlationCoefficient());
            System.out.println("=================================================");

            System.out.println("===== Decision Tree Model Evaluation =====");
            System.out.println("Correctly Classified Instances: " + decisionTreeEvaluation.correct());
            System.out.println("Incorrectly Classified Instances: " + decisionTreeEvaluation.incorrect());
            System.out.println("Accuracy: " + decisionTreeEvaluation.pctCorrect() + "%");
            System.out.println("=================================================");

        } catch (Exception e) {
            e.printStackTrace();
        }
    }
}
