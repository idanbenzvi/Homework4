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
		Instances dataSet_glass = loadData("glass.txt");
		Instances dataSet_cancer = loadData("cancer.txt");

		//run the knn nearest neighbor classification process (find lower cross validation error, giving best performance)
		//using getclassvoteresult and getweightedclassvoteresult functions, all k values and all p values
		Knn ourKnn = new Knn();
		ourKnn.buildClassifier(dataSet_glass);

		//output the cross validation error for both data sets
		String cverror_glass = "Cross validation error with K = "+ourKnn.getM_bestK()+", p = "+ourKnn.getM_bestP()+", vote function = "+ourKnn.getM_bestFunc()+" for glass data is: "+ourKnn.getM_bestError();
		System.out.println(cverror_glass);

		//after selecting the best K, P and function, return the cross validation error of this set
        //// TODO: 20/04/2016

		//repeat process for cancer dataset
		ourKnn.buildClassifier(dataSet_cancer);

		String cverror_cancer = "Cross validation error with K = "+ourKnn.getM_bestK()+", p = "+ourKnn.getM_bestP()+", vote function = "+ourKnn.getM_bestFunc()+" for cancer data is: "+ourKnn.getM_bestError();
		System.out.println(cverror_cancer);
	}

	//// TODO: 24/04/2016
	// todolist:
	// finish the forward backward
	// test
	// create keeping values of k p and function
	// make sure i deal correctly with continuous values
	// compare results


}
