package hw4;

import javafx.util.Pair;
import weka.classifiers.Classifier;
import weka.core.Instances;
import weka.core.Instance;
import weka.filters.Filter;
import weka.filters.unsupervised.instance.Normalize;

import java.util.*;


public class Knn extends Classifier {

    private String M_MODE = "";
    private int M_DISTFUNC;

    private int M_P_VALUE = 1; // by default


    private static final int M_FOLD_NUM = 10;
    private static final int LPSDISTANCE = 1;
    private static final int LINFINITYDISTANCE = 0;
    private static final int WEIGHTED = 1;
    private static final int NON_WEIGHTED = 0;

    private int m_distance_mode = 0; //
    Instances m_trainingInstances;
    Instances m_currentFolding_maj;

    //public since we want it to be available outside the scope of the class
    public double m_bestError;
    public double m_bestK = 1;
    public double m_bestP = 0;
    private int[] m_bestKatts;
    private int m_currK;


    public String getM_MODE() {
        return M_MODE;
    }

    public void setM_MODE(String m_MODE) {
        M_MODE = m_MODE;
    }

    @Override
    public void buildClassifier(Instances arg0) throws Exception {
        switch (M_MODE) {
            case "none":
                noEdit(arg0);
                break;
            case "forward":
                editedForward(arg0);
                break;
            case "backward":
                editedBackward(arg0);
                break;
            default:
                noEdit(arg0);
                break;
        }
    }

    private void noEdit(Instances instances) {
        m_trainingInstances = new Instances(instances);
        //normalize
        normalize(instances);


        double bestError = Double.MAX_VALUE;
        int currK = 1;
        int[] bestK ;
        int bestKNum = 0;
        int bestp;
        int bestMethod;
        double currError;


//        Loop over all combinations of k & p
//        In each combination:
//        Calculate the Cross Validation Error - crossValidationError:
//        Divide the data to 10 subsets. Train the kNN with 9 subsets and predict the last subset (every time different subset)
//        Calculate average error (#mistakes / #instances) with the calcAvgError method
//        The Cross Validation Error is the average error on all subsets

        //run through all k values and all p values, using the 2 classification methods

        //retain only the best (min error) classification and function
        //according to classification process

        trainModel(WEIGHTED);

        trainModel(NON_WEIGHTED);

    }

    /**
     *test all k,p values
     */
    private void trainModel(int functionType){
        double currError;

        //create all possible subsets of the given instances attributes(features)
        //ArrayList<int[]> Ksubsets = findSubsets(m_trainingInstances.numAttributes() - 1);


        for (int ksub = 0; ksub < 30; ksub++) {
            for (int currP = 0; currP <= 3; currP++) {
                //calc cross-valiation-error
                //note: the 0 value for currP, designates infinity
                //todo

                //set function to be weighted
                M_DISTFUNC = functionType;

//                //m_currKatts = Ksubsets.get(ksub);
//                m_currKatts = ksub;

                currError = crossValidationError(m_trainingInstances);

                if(currError<m_bestError) {
                    m_bestError = currError;
                    m_bestK = ksub;
                    m_bestP = currP;
                }
            }
        }

    }

    /**
     *  a simple function that counts the number of non zero elements within an int array
     *  @param arr the array
     *  @return the number of non zero elements
     */
    private static int countNonZeroElements(int[] arr){
        int count=0;

        for(int i = 0 ; i < arr.length; i++){
            if(arr[i]!=0)
                count++;
        }

        return count;
    }

    // Implementation of methods required

    /**
     * classify an input instance using the classifier after it has been trained by the KNN model
     *
     * @param instance an instance
     * @return the predicted target/class value of your algorithm for the input instance
     */
    public double classify(Instance instance) {
        //we will classify the given instances using the instances in the majority fold (in our case 9 out 10 instances)
        double resultClass ;

        Instances neighbors = findNearestNeighbors(instance,m_currK);

        if(m_distance_mode==NON_WEIGHTED)
            resultClass = getClassVoteResult(neighbors);
        else
            resultClass = getWeightedClassVoteResult(neighbors,instance);

        //return the resulting class
        return resultClass;
    }

    /**
     * Find the k nearest neighbors of instance given
     *
     * @param instance
     * @param k        - num of neighbors wanted
     * @return finds the K nearest neighbors (and perhaps their distances)
     */
    private Instances findNearestNeighbors(Instance instance, int k) {
        //// TODO: 20/04/2016 - check if the code works !!!! (debug it)
        
        //given an instance - find it's k nearest neighbors.
        //HOW: locate the k neighbors that have the minimal distance from the given instance.
        Instances neighbors = new Instances;
        ArrayList<Pair<Double,Instance>> distList = new ArrayList<Pair<Double, Instance>>(k);
        double currDist ; // the current distance
        boolean added = false; //did we add an element the k neighbors list

        // classify each instance according to its proximity to the instance in question. Do this for the 5 best instances,
        // in case one instance is closer than another - remove all instance farther away and keep it.

        Enumeration<Instance> instEnum = m_currentFolding_maj.enumerateInstances();

        //The solution to find the k closet elements to the instance, will be implemented using an arraylist of pairs
        //the pairs will be of double and instance classes. Meaning, each key will be the distance and the value will
        //be the instance associated with it.
        //we will iterate over all possible instances and set only the k most proximal.
        while(instEnum.hasMoreElements()){

            //reset the added boolean
            added = false;

            //get the next instance from the instances list
            Instance currInst = instEnum.nextElement();

            //get the distance according to the distance function being used
            currDist = distance(instance,currInst);

            //create an iterator to go over the arraylist of closest instances
            ListIterator listIter = distList.listIterator();

            //create a new pair to be used in case we will be adding this instance and its distance to the neighbors list
            Pair newp = new Pair(currDist, currInst);

            //first element - for the first time the list is being checked
            if(distList.isEmpty())
                distList.add(newp);
            else {
                //go over all instances and evaluate their distance from the current instance being checked
                while (listIter.hasNext()) {
                    //get the current pair
                    Pair currPair = (Pair) listIter.next();

                    //get current distance
                    double currDistofInst = (Double) currPair.getKey();

                    //check whether the current distance is smaller than one of the existing distances of instances in the
                    //arraylist we've created.
                    if (currDistofInst > currDist) {
                        //add the new instance and its distance to the list
                        distList.add(distList.indexOf(currPair) - 1, newp);

                        // remove pairs exceeding the k neighbors capacity limit
                        if (distList.size() > k)
                            distList.remove(k);
                            added = true;
                        break;
                    }

                }

                //if the list doesn't contain k elements yet
                //and we haven't replaced an element from the list
                //add this instance and its distance as the last ones in the list
                if(distList.size()<k && added == false)
                    distList.add(newp);
            }
        }

        //create neighbors instances object
        ListIterator<Pair<Double,Instance>> neiList = distList.listIterator();

        //add all the instances to the neighbors list
        while(neiList.hasNext()){
            neighbors.add(neiList.next().getValue());
        }

        return neighbors;
    }

    /**
     * should take a vote on what the class of the neighbors are and return the class value with the most votes
     *
     * @param instances a set of K nearest neighbors
     * @return
     */
    private double getClassVoteResult(Instances instances) {
        //find the class value which is most common
        int classValues = instances.classAttribute().numValues();
        int[] classValueFreq = new int[classValues];
        int maxFreqClass = 0;

        //count frequencies
        for (int j = 0 ; j < instances.numInstances();j++) {
            classValueFreq[(int) instances.instance(j).classValue()]++;
        }

        //find maximal frequency
        for (int k = 0 ; k < classValueFreq.length ; k++){
            if (classValueFreq[k] > maxFreqClass)
                maxFreqClass = k;
        }

        //return the most common class value
        return (double) maxFreqClass;
    }


    /**
     * this method should be the same as getClassVoteResult except instead of giving one vote to every class, you give a vote of 1/(distance from instance)^2.
     * should take a vote, weighted by each neighbor's distance from the instance being classified, on what the class of the neighbors are.
     *
     * @param instances a set of K nearest neighbors
     * @return the class value with the most votes
     */
    private double getWeightedClassVoteResult(Instances instances, Instance instanceClassified) {
        //give additional weight to each class according to its distance from the instance being classified.
        double currDistance ;

        int classValues = instances.classAttribute().numValues();
        double[] classValueFreq = new double[classValues];
        double maxFreqClass = 0;

        //go over all k instances
        for(int j = 0 ; j < instances.numInstances() ; j++){
            //measure the distance of the current instance from the evaluated instance
            currDistance = Math.pow((double) 1 / lPDistance(instances.instance(j),instanceClassified),2);
            //add the measured distance to the relevant class value
            classValueFreq[(int) instances.instance(j).classValue()] += currDistance ;
        }

        //find maximal frequency
        for (int k = 0 ; k < classValueFreq.length ; k++){
            if (classValueFreq[k] > maxFreqClass)
                maxFreqClass = k;
        }

        return maxFreqClass;
    }

    /**
     * calculate the input instances' distance according the distance function that the algorithm is configured to use.
     *
     * @param a first instance
     * @param b second instance
     * @return the distance between
     */
    private double distance(Instance a, Instance b) {
        double distance =0;

        switch (M_DISTFUNC) {
            case LPSDISTANCE: //distance according to L-P Distance function
                distance = lPDistance(a, b);
                break;
            case LINFINITYDISTANCE: //distance according to L Infinity function
                distance = lInfinityDistance(a, b);
                break;
        }

        return distance;
	}


	/**
	 * Calculate the l-p distance between the two instances
	 * p can be a variable of your class or you can set p some other way
	 * @param a
	 * @param b
     * @return the l-p distance
     */
	private double lPDistance(Instance a,Instance b){
        //iterate over every feature(attribute) and calculate the distance according to the p value selected
        // (except for the infinite of course)
        int numAttributes = m_trainingInstances.numAttributes();
        double distanceSum = 0;

        for(int i = 0 ; i < numAttributes ; i++){
            distanceSum += Math.pow(a.value(i)-b.value(i),M_P_VALUE);
        }

        distanceSum = Math.pow(distanceSum,(double) 1/M_P_VALUE);

        return distanceSum;
	}

	/**
	 *
	 * @param a
	 * @param b
	 * @return the l-infinity distance between two instances
     */
	private double  lInfinityDistance(Instance a, Instance b){
        //iterate over every feature(attribute) and calculate the distance according to the p value selected
        // (except for the infinite of course)
        int numAttributes = m_trainingInstances.numAttributes();

        double distanceSum = 0;
        double evaluateSum = 0;

        for(int i = 0 ; i < numAttributes ; i++){
            //evaulate the current vector dimension size
            evaluateSum = Math.abs(a.value(i)-b.value(i));
            //if it is larger than the distance previously measured on another dimension replace it.
            if(evaluateSum> distanceSum)
                distanceSum = evaluateSum;
        }

        return distanceSum;
	}

	/**
	 * calculate the average error
	 * NOTE: the error you should use is the number of mistakes divided by the number of input instances.
	 * @param instances
	 * @return the average error on the input instances.
	 */
	private double calcAvgError(Instances instances){
        Enumeration instEnum = instances.enumerateInstances();
        double classification ;
        double correctClass = 0;

        while(instEnum.hasMoreElements()) {
            Instance currentElement = (Instance) instEnum.nextElement();
            classification = classify(currentElement);
            if(currentElement.classValue()==classification)
                correctClass++;
        }

      return correctClass / (double) instances.numInstances();
	}

	/**
	 * Using 10 folds for the cross validation
	 * @param instances
	 * @return the cross validation error of your algorithm on the input instances
     */
	public double crossValidationError(Instances instances){
        //get splitting indices for folding
        int[] subsetIndices = foldIndices(instances,M_FOLD_NUM);
        Instances[] instArray;
        double[] errorArray = new double[M_FOLD_NUM];
        double cvError = 0;

        //calc l-p distance using 90% of the instances
        //test hypothesis on the remaining 10%

        for(int foldix = 0 ; foldix < 9 ; foldix++) {
            //get the division into 2 instance groups (9/10 ratio in our case)
            instArray = getFoldInstances(subsetIndices[foldix],subsetIndices[foldix+1],instances);

            //assign the current majority of the instances as the reference to the rest of them
            m_currentFolding_maj = instArray[0];

            //use the 2 created arrays in order to test the KNN model
            calcAvgError(instArray[1]);
            //for each instance of  the smaller group, locate the K nearest neighbors according to the selected
            //function. After doing so, classify according to these neighbors.
            //keep classification
        }

        return cvError;
	}

	/**
	 * should train your Knn algorithm using the edited Knn forwards algorithm shown in class
	 */
		public void	editedForward(Instances instances){

		}

	/**
	 * should train your Knn algorithm using the edited Knn backwards algorithm shown in class
	 * @param instances
     */
	 public void editedBackward(Instances instances){

	 }

	/**
	 * perform a normalization process over all instance features
	 */
	public void normalize(Instances instances){
        try {
            double means[] = new double[instances.numAttributes()];
            double tempSum = 0;

            Normalize norm = new Normalize();
            norm.setInputFormat(instances);
            Instances processed_training_data = Filter.useFilter(instances, norm);

            m_trainingInstances = processed_training_data;
        }
        catch(Exception e)
        {
            System.out.println("An error has occured while attempting to normalize the dataset");
        }
	}

	//Assistive methods - normalization of feature data

	/**
	 * generates an arraylist containing all the possible subsets of a given size (using bitmasks)
	 * @param k - the number of subsets we want to generate
	 * @return
	 */
	private static ArrayList<int[]> findSubsets(int k)
	{
		//create an int. array from 1 to k(included)
		int[] array = new int[k];
		for (int i = 1 ; i <= k ; i++){
			array[i-1] = i;
		}

		int numOfSubsets = 1 << array.length;
		ArrayList<int[]> elements = new ArrayList<int[]>();


		for(int i = 0; i < numOfSubsets; i++)
		{
			int pos = array.length - 1;
			int bitmask = i;
			int ix = 0;
			int[] currentArray = new int[array.length];

			while(bitmask > 0)
			{
				if((bitmask & 1) == 1) {
					currentArray[ix] = array[pos];
					ix++;
				}
				bitmask >>= 1;
				pos--;

			}
			elements.add(currentArray);
		}

		return elements;
	}


    /**
     * divide the instances into k folds, of equal size. possibly shuffle instances if required
     * @param instances
     * @return
     */
    private int[] foldIndices(Instances instances, int foldNum){
        //// TODO: 20/04/2016
        //divide to the nearest integer value possible (last set might be smaller than int value by remainder)
        int instCount = m_trainingInstances.numInstances() / foldNum; //rounded version of course
        int remainder = m_trainingInstances.numInstances() % foldNum;
        int addOne = (remainder > 0) ? 1 : 0;
        int[] foldIndexArray = new int[instCount + addOne];

        int ix = 0;

        //keep only the indices of the instances that are cutoff points between sets (each one composing of 10%)
        for(int i = 0;  i < instances.numInstances(); i++){
            if(i % instCount == 0) {
                foldIndexArray[ix] = i;
                ix++;
            }
        }

        return foldIndexArray;
    }

    /**
     * return instances that compose of the data outside the indices and the remaining instances
     * in the second index of the instances array.
     * @param start
     * @param end
     * @param instances
     * @return instancesArray composing of 9/10 division ratio between instances according to instructions above
     */
    private Instances[] getFoldInstances(int start,int end,Instances instances){
        Enumeration<Instance> instEnum = instances.enumerateInstances();
        int ix = 0;
        Instances[] instancesArray = new Instances[2];

        //as long as there are more elements to divide - keep on going
        while(instEnum.hasMoreElements()){
            if(ix<start || ix>end)
                instancesArray[0].add(instEnum.nextElement()); //add to the first instances group (majority of 90%)
            else
                instancesArray[1].add(instEnum.nextElement()); //the smaller group (in our case will be 10%)
        }

        return instancesArray;
    }



}
