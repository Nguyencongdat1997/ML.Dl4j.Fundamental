import com.google.common.primitives.Doubles;
import java.io.File;
import java.io.IOException;
import java.util.ArrayList;
import java.util.Random;
import java.util.logging.Level;
import java.util.logging.Logger;
import org.datavec.api.io.labels.ParentPathLabelGenerator;
import org.datavec.api.split.FileSplit;
import org.datavec.image.loader.NativeImageLoader;
import org.datavec.image.recordreader.ImageRecordReader;
import org.deeplearning4j.datasets.datavec.RecordReaderDataSetIterator;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.deeplearning4j.eval.Evaluation;
import org.deeplearning4j.nn.api.OptimizationAlgorithm;
import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.Updater;
import org.deeplearning4j.nn.conf.inputs.InputType;
import org.deeplearning4j.nn.conf.layers.DenseLayer;
import org.deeplearning4j.nn.conf.layers.OutputLayer;
import org.deeplearning4j.nn.graph.ComputationGraph;
import org.deeplearning4j.nn.modelimport.keras.trainedmodels.TrainedModels;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.nn.weights.WeightInit;
import org.deeplearning4j.optimize.listeners.ScoreIterationListener;
import org.deeplearning4j.util.ModelSerializer;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.dataset.api.preprocessor.DataNormalization;
import org.nd4j.linalg.dataset.api.preprocessor.ImagePreProcessingScaler;
import org.nd4j.linalg.dataset.api.preprocessor.VGG16ImagePreProcessor;
import org.nd4j.linalg.learning.config.Nesterovs;
import org.nd4j.linalg.lossfunctions.LossFunctions.LossFunction;

/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */

/**
 *
 * @author DatGatto
 */
public class MNIST {
    //number of rows and columns in the input pictures    
    static final int outputNum = 10; // number of output classes
    static final int batchSize = 128; // batch size for each epoch
    static final int rngSeed = 123; // random number seed for reproducibility
    static final int numEpochs = 15; // number of epochs to perform
    static final double rate = 0.0015; // learning rate

    //set information: 28*28image with single chanel
    static final int height =28;
    static final int width = 28;
    static final int channels =1;
    static final int rngseed = 123;
    static Random randNumGen = new Random(rngseed);           

    
    static File trainData;
    static File testData;
    static ImageRecordReader recordReader;
    static FileSplit train;
    static FileSplit test;
    static DataNormalization scaler = new ImagePreProcessingScaler(0,1);
    static MultiLayerNetwork model;
    
    public void trainAndSave() throws IOException {        
        //Initialize the reocrd reader
        //add a listener to extract the name 
        //recordReader.reset();
        recordReader.initialize(train);
        //recordReader.setListeners(new LogRecordListener());
        
        //Load Dataset Iterator
        DataSetIterator mnistTrain = new RecordReaderDataSetIterator(recordReader, batchSize,1,outputNum);
        
        //Preprocessor
        
        scaler.fit(mnistTrain);     
        mnistTrain.setPreProcessor(scaler);
        
        //Set layers configuration
        MultiLayerConfiguration layerConfiguration = new NeuralNetConfiguration.Builder()
            .seed(rngSeed)
            .optimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT)
            .iterations(1)                
            .learningRate(rate)
            .updater(Updater.NESTEROVS).momentum(0.9)
            .regularization(true).l2(1e-4)
            .list()            
            .layer(0, new DenseLayer.Builder()                
                .nIn(height * width) // Number of input datapoints.
                .nOut(1000) // Number of output datapoints.
                .activation("relu") // Activation function.
                .weightInit(WeightInit.XAVIER) // Weight initialization.
                .build())
            .layer(1, new OutputLayer.Builder(LossFunction.NEGATIVELOGLIKELIHOOD)
                    .nIn(1000)
                    .nOut(outputNum)
                    .activation("softmax")
                    .weightInit(WeightInit.XAVIER)
                    .build())            
            .pretrain(false).backprop(true)  
            .setInputType(InputType.convolutional(height, width,channels))
            .build();
        
        //Set model
        model = new MultiLayerNetwork(layerConfiguration);        
        model.init();
        model.setListeners(new ScoreIterationListener(5));  //print the score with every iteration

        //Train
        System.out.println("Train model....");
        for( int i=0; i<numEpochs; i++ ){
            System.out.println("Epoch " + i);
            model.fit(mnistTrain);
        }

        //Save the model
        File locationToSave = new File("C:\\Users\\DatGatto\\.deeplearning4j\\mnist.zip");            
        ModelSerializer.writeModel(model, locationToSave,true); //true here is to agree to save the Updater
        System.err.println("Model has been saved.");

    }
    public void test() throws IOException{
        //Load the model
        File locationToSave = new File("C:\\Users\\DatGatto\\.deeplearning4j\\mnist.zip");            
        model = ModelSerializer.restoreMultiLayerNetwork(locationToSave); 
        System.err.println("Model has been loaded.");

        //Test
        System.out.println("Evaluate model....");
        
        //recordReader.reset();
        recordReader.initialize(test);
        
        DataSetIterator mnistTest = new RecordReaderDataSetIterator(recordReader, batchSize,1,outputNum);
        
        scaler.fit(mnistTest);
        mnistTest.setPreProcessor(scaler);
 
        Evaluation eval = new Evaluation(outputNum); //create an evaluation object with 10 possible classes        
        while(mnistTest.hasNext()){
            DataSet next = mnistTest.next();
            INDArray output = model.output(next.getFeatureMatrix()); //get the networks prediction            
            eval.eval(next.getLabels(), output); //check the prediction against the true class
        }
        System.err.println("Evaluated reuslt: " + eval.stats());
               
    }
    public String predictOneImage(String fileLink) throws IOException{
        
        // Load the trained model
        File locationToSave = new File("C:\\Users\\DatGatto\\.deeplearning4j\\mnist.zip");            
        model = ModelSerializer.restoreMultiLayerNetwork(locationToSave); 
        //Set image loader        
        NativeImageLoader loader = new NativeImageLoader(height, width, channels);
        //Predict
        File file = new File(fileLink);
        try{
            INDArray imageMatrix = loader.asMatrix(file);                    
            scaler.transform(imageMatrix);
            INDArray output = model.output(imageMatrix);
            //return most possible value
            String outputString = output.toString();
            String[] splitedTxtOutput = outputString.replaceAll("\\,|\\[|\\]", " ").trim().split(" +");            
            double max = Doubles.tryParse(splitedTxtOutput[0]);
            int maxIndex = 0;
            for (int i =1;i<10;i++){
                if (max< Doubles.tryParse(splitedTxtOutput[i])){
                    max = Doubles.tryParse(splitedTxtOutput[i]);
                    maxIndex = i;
                }
            }            
            return "Prediction: " + maxIndex +" with "+max*100 +"%" ;
        }
        catch (IOException ex){
            System.err.println("Error at loading image."+fileLink);
            return "Error at loading image";
        }
    }
    public MNIST() throws IOException{
              
        //Prepare to Image Pipeline loading
        trainData = new File("E:\\environment\\git\\AI\\ML\\Data\\MNIST\\mnistasjpg\\largeSet\\trainingSet");
        testData = new File("E:\\environment\\git\\AI\\ML\\Data\\MNIST\\mnistasjpg\\largeSet\\testSet");
    
        //Define the FIleSplit(PATH,ALLOWED FORMATS, randomNum to randomize the data)
        train = new FileSplit(trainData,NativeImageLoader.ALLOWED_FORMATS,randNumGen);
        test = new FileSplit(testData,NativeImageLoader.ALLOWED_FORMATS,randNumGen);
        
        //Set label as Parent Folder's Name
        ParentPathLabelGenerator labelMaker = new ParentPathLabelGenerator();        
        recordReader = new ImageRecordReader(height,width,channels,labelMaker);
        
        
    }
}
