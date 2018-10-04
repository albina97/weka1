package classifiers.twoway;

import weka.classifiers.Classifier;
import weka.classifiers.bayes.NaiveBayesMultinomialText;
import weka.core.Instances;
import weka.classifiers.trees.J48;
import weka.classifiers.Evaluation;
import java.io.BufferedReader;
import java.io.FileReader;
import weka.core.converters.ConverterUtils.DataSource;

public class TwoClassTrain {
    private NaiveBayesMultinomialText Сlassifier;

    public static void main(String[] args) throws Exception {
        /*DataSource source = new DataSource("src/classifiers/train.arff");
        Instances train = source.getDataSet();
        DataSource source2 = new DataSource("src/classifiers/test.arff");
        Instances test = source2.getDataSet();*/
        Instances train = new Instances(new BufferedReader(new FileReader("train.arff")));
        Instances test = new Instances(new BufferedReader(new FileReader("test.arff")));
        // train classifier
        Classifier cls = new J48();
        cls.buildClassifier(train);
        // evaluate classifier and print some statistics
        Evaluation eval = new Evaluation(train);
        eval.evaluateModel(cls, test);
        System.out.println(eval.toSummaryString("\nResults\n======\n", false));
    }

}
