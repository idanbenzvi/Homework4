package hw4;

import java.io.BufferedReader;
import java.io.FileNotFoundException;
import java.io.FileReader;
import java.io.IOException;
import java.util.ArrayList;
import java.util.Arrays;

import weka.classifiers.lazy.IBk;
import weka.core.Instances;

public class MainHW4 {
	
	public static BufferedReader readDataFile(String filename) {
		BufferedReader inputReader = null;

		try {
			inputReader = new BufferedReader(new FileReader(filename));
		} catch (FileNotFoundException ex) {
			System.err.println("File not found: " + filename);
		}

		return inputReader;
	}

	/**
	 * Sets the class index as the last attribute.
	 * @param fileName
	 * @return Instances data
	 * @throws IOException
	 */
	public static Instances loadData(String fileName) throws IOException{
		BufferedReader datafile = readDataFile(fileName);
		Instances data = new Instances(datafile);
		data.setClassIndex(data.numAttributes() - 1);
		return data;
	}

	public static void main (String [] args) throws Exception{

		//load data
		Instances dataSet = loadData("glass.txt");

		//run the knn nearest neighbor classification process (find lower cross validation error, giving best performance)
		//using getclassvoteresult and getweightedclassvoteresult functions, all k values and all p values
		Knn ourKnn = new Knn();

		ourKnn.buildClassifier(dataSet);




		//output the cross validation error for both data sets
		String cverror_glass = "Cross validation error with K = <my_k>, p = <my_p>, vote function = <either weighted or uniform> for glass data is: <my_error>";

		String cverror_cancer = "Cross validation error with K = <my_k>, p = <my_p>, vote function = <either weighted or uniform> for cancer data is: <my_error>";
	}



}
