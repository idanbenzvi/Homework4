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

    //weighted / non weighted distance function
    private int M_DISTFUNC;
    //function type switches
    private static final int WEIGHTED = 1;
    private static final int NON_WEIGHTED = 0;

    //p values field
    private int M_P_VALUE = 0; // by default

    //constants of the KNN implementation
    private static final int M_FOLD_NUM = 10;
    private static final int M_K_MAX = 30;

    Instances m_trainingInstances;

    //public since we want parameters and performance measurements to be available outside the scope of the class:
    private double m_bestError = Double.MAX_VALUE;
    private int m_bestK = 1; //best K value
    private int m_bestP = 0; //best P value (0 - infinity, other values are P values of powers
    private int m_bestFunc = 0; //best function - weighted / non-weighted
    public long m_calcTimeAvg =0; //error calculation average
    private int m_currK; //current K value (while training)

    //simple getters auto-generated
    public double getM_bestError() {
        return m_bestError;
    }

    public double getM_bestK() {
        return m_bestK;
    }

    public double getM_bestP() {
        return m_bestP;
    }

    public int getM_bestFunc() {
        return m_bestFunc;
    }

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

    /**
     * Create a KNN classifier and train it using cross-fold-validation while iterating over all possible
     * k,p, and function types (weighted,non-weighted)
     * @param instances
     */
    private void noEdit(Instances instances) {
        //reset all fields
        m_bestError = Double.MAX_VALUE;
        m_bestK = 1;
        m_bestP = 0;

//        Loop over all combinations of k & p
//        In each combination:
//        Calculate the Cross Validation Error - crossValidationError:
//        Divide the data to 10 subsets. Train the kNN with 9 subsets and predict the last subset (every time different subset)
//        Calculate average error (#mistakes / #instances) with the calcAvgError method
//        The Cross Validation Error is the average error on all subsets

        //run through all k values and all p values, using the 2 classification methods

        //retain only the best (min error) classification and function
        //according to classification process

        //to avoid code duplication, the method trainModel has been added
        trainModel(NON_WEIGHTED,instances);
        trainModel(WEIGHTED,instances);

        //set parameters after training (no need to set K value - model training is complete
        M_DISTFUNC = m_bestFunc;
        M_P_VALUE = m_bestP;
        m_currK = m_bestK;
    }

    /**
     *test all k,p values
     */
    private void trainModel(int functionType,Instances instances){
        double currError;

        //create all possible subsets of the given instances attributes(features)
        //ArrayList<int[]> Ksubsets = findSubsets(m_trainingInstances.numAttributes() - 1);
        for (int ksub = 1; ksub <= M_K_MAX; ksub++) {
            for (int currP = 0; currP <= 3; currP++) {
                //calc cross-valiation-error
                //note: the 0 value for currP, designates infinity

                //set function to be weighted
                M_DISTFUNC = functionType; //weighted vs. non-weigthed
                M_P_VALUE = currP;
                m_currK = ksub;

                // the current CV error
                currError = crossValidationError(instances);

                // update parameters if need be (better performing K / P / weight function / error value
                if(currError<m_bestError) {
                    m_bestError = currError;
                    m_bestK = ksub;
                    m_bestP = currP;
                    m_bestFunc = functionType;
                }
            }
        }

    }

    // Implementation of methods required

    /**
     * classify an input instance using the classifier after it has been trained by the KNN model
     *
     * @param instance an instance
     * @return the predicted target/class value of your algorithm for the input instance
     */
    public double classify(Instance instance,Instances classifySet) {
        //we will classify the given instances using the instances in the majority fold (in our case 9 out 10 instances)
        double resultClass ;

        //find neighbors of closest proximity
        Instances neighbors = findNearestNeighbors(instance,classifySet,m_currK);

        //get class voting (highsted freq.)
        if(M_DISTFUNC==0)
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
    private Instances findNearestNeighbors(Instance instance,Instances neighborhood, int k) {
        //insert k first instances into our neighbor array and start improvements afterward
            TreeMap<Double,Instance> neighborMap = new TreeMap<>();

        //init an empty neighbors instance class instance - will be returned later on
        Instances neighbors = new Instances(neighborhood,k);

        //insert regardless of distance values
        for(int i = 0 ;  i < k ; i++){
            neighborMap.put(distance(neighborhood.instance(i),instance),neighborhood.instance(i));
        }

        //sort map by simply creating a new treemap
        TreeMap<Double,Instance> neiMap = new TreeMap<>(neighborMap);

        //add only elements within the range of the first k elements
        for (int i = k ; i < neighborhood.numInstances() ; i++){
            double curDistance = distance(neighborhood.instance(i),instance);
            if (curDistance > neiMap.firstKey() || curDistance < neiMap.lastKey()) {
               if(!neiMap.containsKey(curDistance)) {
                   //insert the key and instance
                   neiMap.put(curDistance, neighborhood.instance(i));

                   //remove the last entry - as it is no longer within the constraints of k elements wanted
                   neiMap.remove(neiMap.lastKey());
               }
            }
        }

        //return instances set with the nearest neighbors :)
        Set<Double> keys = neiMap.keySet();

        //key (distances) iterator, which will be used in order to retrieve all the values (instances) and return the instances class
        Iterator<Double> keyIter = keys.iterator();

        //get all instances with the minimal distances
        while(keyIter.hasNext())
            neighbors.add(neiMap.get(keyIter.next()));

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

        //count frequencies of each class value
        for (int j = 0 ; j < instances.numInstances();j++) {
            classValueFreq[(int) instances.instance(j).classValue()]++;
        }

        //find maximal frequency class value (the one with largest value)
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
            if(distance(instances.instance(j),instanceClassified)==0)
                return instanceClassified.classValue();
            else {
                currDistance = 1.0 / Math.pow(distance(instances.instance(j), instanceClassified), 2);
                //add the measured distance to the relevant class value
                classValueFreq[(int) instances.instance(j).classValue()] += currDistance;
            }
        }

        //find maximal frequency by going over the array of sums, each index is a class value
        for (int k = 0 ; k < classValueFreq.length ; k++){
            if (classValueFreq[k] > maxFreqClass)
                maxFreqClass = k;
        }

        //return most frequent class value
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

        if(M_P_VALUE>0)
                distance = lPDistance(a, b);
            else
                distance = lInfinityDistance(a, b);

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
        int numAttributes = a.numAttributes();
        double distanceSum = 0;

        //calculate distance for each attribute of the instances
        for(int i = 0 ; i < numAttributes-1 ; i++){
            distanceSum += Math.pow(Math.abs(a.value(i)-b.value(i)),M_P_VALUE);
        }

        distanceSum = Math.pow(distanceSum, 1.0/M_P_VALUE);

        return distanceSum;
	}

	/**
	 * l-f infinity distance between two instances as taught in class
	 * @param a
	 * @param b
	 * @return the l-infinity distance between two instances
     */
	private double  lInfinityDistance(Instance a, Instance b){
        //iterate over every feature(attribute) and calculate the distance according to the p value selected
        // (except for the infinite of course)
        int numAttributes = a.numAttributes();

        double distanceSum = 0;
        double evaluateSum = 0;

        //go over all attributes (features) and calculate the distance for each of them
        for(int i = 0 ; i < numAttributes-1 ; i++){
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
	private double calcAvgError(Instances instances,Instances classifySet){
        Enumeration instEnum = instances.enumerateInstances();
        double classification ;
        double incorrectClass = 0;

        //classify and compare to actual class value, count the number of correct classifications
        while(instEnum.hasMoreElements()) {
            Instance currentElement = (Instance) instEnum.nextElement();
            classification = classify(currentElement,classifySet);

            if(currentElement.classValue()!=classification)
                incorrectClass++;
        }

        //return the percent of errors, from the total instance number required for classification
      return incorrectClass / instances.numInstances();
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

        long calcTimeAvg = 0;
        double cvError = 0;

        //calc l-p distance using 90% of the instances
        //test hypothesis on the remaining 10%

        for(int foldix = 0 ; foldix <=M_FOLD_NUM-1 ; foldix++) {
            //get the division into 2 instance groups (9/10 ratio in our case), as 2 different Instances class objects
            instArray = getFoldInstances(subsetIndices[foldix],subsetIndices[foldix+1],instances);

            //use the 2 created arrays in order to test the KNN model
            long startClock = System.nanoTime();

            //calculate avg error for current fold
            cvError += calcAvgError(instArray[1],instArray[0]);

            long endClock = System.nanoTime();

            //calculate the elapsed time
            calcTimeAvg += (endClock - startClock);

            //for each instance of  the smaller group, locate the K nearest neighbors according to the selected
            //function. After doing so, classify according to these neighbors.
            //keep classification
        }

        //divide the sum of errors by the number of folds to calculate the cross validation error
        cvError /= M_FOLD_NUM;
        m_calcTimeAvg = calcTimeAvg / (long) M_FOLD_NUM;

        return cvError;
	}

	/**
	 * should train your Knn algorithm using the edited Knn forwards algorithm shown in class
	 */
		public void	editedForward(Instances instances){
            //create an empty instances object (our T)
            m_trainingInstances = new Instances(instances,instances.numInstances());

            //the first instance will not be classified correctly, since T is empty, therefore we can add k instances as
            //the best k defines
            for(int i =0 ; i < getM_bestK() ; i++) {
                m_trainingInstances.add(instances.instance(i));
                instances.delete(i); // remove the instance from the instances class
            }


            //each one of the remaining instances is to be checked according to the above logic
            for(int i = 0; i < instances.numInstances(); i++){
                if(classify(instances.instance(i),m_trainingInstances)!=instances.instance(i).classValue()){
                    m_trainingInstances.add(instances.instance(i));
                }
            }

            m_bestError = crossValidationError(m_trainingInstances);
        }

	/**
	 * should train your Knn algorithm using the edited Knn backwards algorithm shown in class
	 * @param instances
     */
	 public void editedBackward(Instances instances){

         //general outline of algorithm:
        //         T = S
        //         For each instance x in T
        //         if x is classified correctly by T-{x}
        //         remove x from T
        //         Return T

         //after training the model and calculating the optimal functions and parameters
         //prune unrequired instances using the backwards method
         int endPoint = instances.numInstances();
         int currIx = 0;

         //set the given instances as the base set (starting point)
         m_trainingInstances = instances;

        for(int i = 0 ; i< instances.numInstances(); i++) {
            //classify each instance with the given training instances set
            if (classify(instances.instance(i), m_trainingInstances) == m_trainingInstances.instance(i).classValue()){
                m_trainingInstances.delete(i);
        }


         }

         //calculate the error and the time it takes on average to calculate the error for each fold
         m_bestError = crossValidationError(m_trainingInstances);

         }

	//Assistive methods - normalization of feature data

    /**
     * divide the instances into k folds, of equal size. possibly shuffle instances if required
     * @param instances
     * @return
     */
    private int[] foldIndices(Instances instances, int foldNum){
        //divide to the nearest integer value possible (last set might be smaller than int value by remainder)
        int instCount = (int) Math.floor( (double) instances.numInstances() / foldNum); //rounded version of course

        int[] foldIndexArray = new int[M_FOLD_NUM+1];
        int multiplier = 1;
        int ix = 0;
        foldIndexArray[0] = 0;

        //keep only the indices of the instances that are cutoff points between sets (each one composing of 10%)
        while(multiplier*instCount <= instances.numInstances()){
            foldIndexArray[multiplier] = multiplier * instCount;
            multiplier++;

            if(multiplier==10){
                    foldIndexArray[10] = instances.numInstances();
                    break;
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
        Instances[] instancesArray = new Instances[2] ;
        instancesArray[0] = new Instances(instances,instances.numInstances());
        instancesArray[1] = new Instances(instances,instances.numInstances());
        //as long as there are more elements to divide - keep on going
        while(instEnum.hasMoreElements()){
            if(ix<start || ix > end)
                instancesArray[0].add(instEnum.nextElement()); //add to the first instances group (majority of 90%)
            else
                instancesArray[1].add(instEnum.nextElement()); //the smaller group (in our case will be 10%)
            ix++; // increment index

        }

        return instancesArray;
    }

}
