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

        //remove the ID property from the glass dataset
        dataSet_glass.deleteAttributeAt(0);

		//run the knn nearest neighbor classification process (find lower cross validation error, giving best performance)
		//using getclassvoteresult and getweightedclassvoteresult functions, all k values and all p values
		Knn ourKnn = new Knn();
		ourKnn.buildClassifier(dataSet_glass);

        String voteFunctionString = "";

        //assign correct meaning ot value of weight function
        if(ourKnn.getM_bestFunc()==0)
            voteFunctionString = "non-weighted";
        else
            voteFunctionString = "weighted";

		//output the cross validation error for both data sets
		String cverror_glass = "Cross validation error with K = "+ourKnn.getM_bestK()+", p = "+ourKnn.getM_bestP()+", vote function = "+voteFunctionString+" for glass data is: "+ourKnn.getM_bestError();
		System.out.println(cverror_glass);

		//repeat process for cancer dataset
		ourKnn.buildClassifier(dataSet_cancer);

        //assign correct meaning to value of weight function
        if(ourKnn.getM_bestFunc()==0)
            voteFunctionString = "non-weighted";
        else
            voteFunctionString = "weighted";

		String cverror_cancer = "Cross validation error with K = "+ourKnn.getM_bestK()+", p = "+ourKnn.getM_bestP()+", vote function = "+voteFunctionString+" for cancer data is: "+ourKnn.getM_bestError();
		System.out.println(cverror_cancer);


        //*****************************************************************************************************************************************

		// MOVING ONTO THE 2nd PART

		//run the KNN classifier on the glass data, using the 3 possible methods - no edit, backwards edit and forward edit
		//return for each method the error and the resulting computation speed after pruning the dataset
		ourKnn.buildClassifier(dataSet_glass); //this will result in the regular no edit model running (default)

		String CVEnoEdit = "Cross validation error of non-edited knn on glass dataset is "+ourKnn.getM_bestError()+" and the average elapsed time is "+ourKnn.m_calcTimeAvg;
		System.out.println(CVEnoEdit);


		//run the backwards edit KNN classifier after finding the best parameters in the non edited section
		ourKnn.setM_MODE("backward");
		ourKnn.buildClassifier(dataSet_glass);

		String CVEbackward = "Cross validation error of backwards-edited knn on glass dataset is "+ourKnn.getM_bestError() +" and the average elapsed time is "+ourKnn.m_calcTimeAvg;
		System.out.println(CVEbackward);


		//run the forward edit KNN classifier after finding the optimal parameters
		ourKnn.setM_MODE("forward");
		ourKnn.buildClassifier(dataSet_glass);

		String CVEforward = "Cross validation error of forwards-edited knn on glass dataset is "+ourKnn.getM_bestError() +" and the average elapsed time is "+ourKnn.m_calcTimeAvg;
		System.out.println(CVEforward);

	}

}
