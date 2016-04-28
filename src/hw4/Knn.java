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

    private void noEdit(Instances instances) {
        //normalize all instance data
        instances = normalize(instances);

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

        if(M_DISTFUNC==NON_WEIGHTED)
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
        //given an instance - find it's k nearest neighbors.
        //HOW: locate the k neighbors that have the minimal distance from the given instance.
        ArrayList<Pair<Double,Instance>> distList = new ArrayList<Pair<Double, Instance>>(k);
        double currDist ; // the current distance
        boolean added = false; //did we add an element the k neighbors list
        Instances neighbors = new Instances(neighborhood,neighborhood.numInstances());
        Enumeration<Instance> instEnum;
        // classify each instance according to its proximity to the instance in question. Do this for the 5 best instances,
        // in case one instance is closer than another - remove all instance farther away and keep it.

        //when training the KNN model to find the optimal parameters, we only use 9/10 instances (as the folds dictate)
        //when this process is done, the neighbors are to be calculated over the whole dataset.
        instEnum = neighborhood.enumerateInstances();

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
                        //add the new instance and its distance to the list, in the appropriate position in ascending distance order
                        if (distList.indexOf(currPair) != 0)
                            distList.add(distList.indexOf(currPair) - 1, newp);
                        else
                            distList.add(0, newp); //if it is the first position to be handled and replaced

                        added = true;
                        break;//instance added to our neighbors list, stop iterating over the current list
                    }

                    }
                // remove pairs exceeding the k neighbors capacity limit
                if (distList.size() > k) {
                    distList.remove(k);
                    added = true;
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

        for(int i = 0 ; i < numAttributes ; i++){
            distanceSum += Math.pow(Math.abs(a.value(i)-b.value(i)),M_P_VALUE);
        }

        distanceSum = Math.pow(distanceSum,(double) 1/M_P_VALUE);

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
	private double calcAvgError(Instances instances,Instances classifySet){
        Enumeration instEnum = instances.enumerateInstances();
        double classification ;
        double correctClass = 0;

        while(instEnum.hasMoreElements()) {
            Instance currentElement = (Instance) instEnum.nextElement();
            classification = classify(currentElement,classifySet);

            if(currentElement.classValue()==classification)
                correctClass++;
        }

      return 1.0d - (correctClass / instances.numInstances());
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

        for(int foldix = 0 ; foldix <= 9 ; foldix++) {
            //System.out.println("current fold being evaluated is:"+foldix);

            //get the division into 2 instance groups (9/10 ratio in our case)
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

//            Random randGen = new Random();
//            instances.randomize(randGen);

            //create an empty instances object (our T)
            Instances newInstances = new Instances(instances,instances.numInstances());

            //normalize the instances given as input to the KNN
            instances = normalize(instances);

            //the first instance will not be classified correctly, since T is empty, therefore we can add it
            newInstances.add(instances.instance(0));
            instances.delete(0); // remove the instance from the instances class

            //each one of the remaining instances is to be checked according to the above logic
            for(int i = 0; i < instances.numInstances(); i++){
                if(classify(instances.instance(i),newInstances)!=instances.instance(i).classValue()){
                    newInstances.add(instances.instance(i));
                }
            }

            m_bestError = crossValidationError(newInstances);
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

         //m_trainingInstances = normalize(instances);

         //after training the model and calculating the optimal functions and parameters
         //prune unrequired instances using the backwards method
         int endPoint = instances.numInstances();
         int currIx = 0;

         //// TODO: 28/04/2016 see this running
         while(currIx!=endPoint){
             Instance currInst = instances.instance(currIx);
             instances.delete(currIx);
             if(classify(currInst,instances)!=currInst.classValue()) {
                 instances.add(currInst);
                 currIx++;
             }else{
                 endPoint--;
             }
         }

         //calculate the error and the time it takes on average to calculate the error for each fold
         m_bestError = crossValidationError(instances);

         }

	/**
	 * perform a normalization process over all instance features
	 */
	public Instances normalize(Instances instances){
        try {
            double means[] = new double[instances.numAttributes()];
            double tempSum = 0;

            Normalize norm = new Normalize();
            norm.setInputFormat(instances);
            Instances processed_training_data = Filter.useFilter(instances, norm);

            return processed_training_data;
        }
        catch(Exception e)
        {
            System.out.println("An error has occured while attempting to normalize the dataset");
            return null;
        }
	}

	//Assistive methods - normalization of feature data

    /**
     * divide the instances into k folds, of equal size. possibly shuffle instances if required
     * @param instances
     * @return
     */
    private int[] foldIndices(Instances instances, int foldNum){
        //divide to the nearest integer value possible (last set might be smaller than int value by remainder)
        int instCount = (int) Math.ceil( (double) instances.numInstances() / foldNum); //rounded version of course
        //int remainder = m_trainingInstances.numInstances() % foldNum;
        //int addOne = (remainder > 0) ? 1 : 0;
        int[] foldIndexArray = new int[11];
        int multiplier = 1;
        int ix = 0;
        foldIndexArray[0] = 0;

        //keep only the indices of the instances that are cutoff points between sets (each one composing of 10%)
        while(multiplier*instCount <= instances.numInstances()){
            foldIndexArray[multiplier] = multiplier * instCount;
            multiplier++;

            if(multiplier==10){
                    foldIndexArray[10] = instances.numInstances();
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
