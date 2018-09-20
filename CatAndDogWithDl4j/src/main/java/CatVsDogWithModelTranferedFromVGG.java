import java.io.File;
import java.io.IOException;
import java.util.Random;
import org.datavec.api.io.labels.ParentPathLabelGenerator;
import org.datavec.api.split.FileSplit;
import org.datavec.image.loader.NativeImageLoader;
import org.datavec.image.recordreader.ImageRecordReader;
import org.deeplearning4j.datasets.datavec.RecordReaderDataSetIterator;
import org.deeplearning4j.eval.Evaluation;
import org.deeplearning4j.nn.api.OptimizationAlgorithm;
import org.deeplearning4j.nn.conf.Updater;
import org.deeplearning4j.nn.conf.inputs.InputType;
import org.deeplearning4j.nn.conf.layers.DenseLayer;
import org.deeplearning4j.nn.conf.layers.OutputLayer;
import org.deeplearning4j.nn.graph.ComputationGraph;
import org.deeplearning4j.nn.modelimport.keras.trainedmodels.TrainedModelHelper;
import org.deeplearning4j.nn.modelimport.keras.trainedmodels.TrainedModels;
import org.deeplearning4j.nn.transferlearning.FineTuneConfiguration;
import org.deeplearning4j.nn.transferlearning.TransferLearning;
import org.deeplearning4j.nn.weights.WeightInit;
import org.deeplearning4j.util.ModelSerializer;
import org.deeplearning4j.zoo.ZooModel;
import org.deeplearning4j.zoo.model.VGG16;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.nd4j.linalg.dataset.api.preprocessor.DataNormalization;
import org.nd4j.linalg.dataset.api.preprocessor.ImagePreProcessingScaler;
import org.nd4j.linalg.lossfunctions.LossFunctions;
import org.slf4j.Logger;

/**
 *
 * @author DatGatto
 */
public class CatVsDogWithModelTranferedFromVGG {
    protected static final int numClasses = 2;

    protected static final long rngseed = 123;
    private static final int trainPerc = 80;
    private static final int batchSize = 128;
    private static final String featureExtractionLayer = "block5_pool";
    
    //set information: 28*28image with single chanel
    static int height =224;
    static int width = 224;
    static int channels =3;      
    static Random randNumGen = new Random(rngseed);    
    
    private static DataSetIterator testIter;
    private static DataSetIterator trainIter;
    public static void loadDataSet() throws Exception{           
        
        //Prepare to Image Pipeline loading
        File trainData = new File("E:\\environment\\git\\AI\\ML\\Data\\CatVsDog\\dogscats\\dogscats\\train");
        File testData = new File("E:\\environment\\git\\AI\\ML\\Data\\CatVsDog\\dogscats\\dogscats\\test");
        
        //Define the FIleSplit(PATH,ALLOWED FORMATS, randomNum to randomize the data)
        FileSplit train = new FileSplit(trainData,NativeImageLoader.ALLOWED_FORMATS,randNumGen);
        FileSplit test = new FileSplit(testData,NativeImageLoader.ALLOWED_FORMATS,randNumGen);
        
        //Set label as Parent Folder's Name
        ParentPathLabelGenerator labelMaker = new ParentPathLabelGenerator();        
        ImageRecordReader recordReader = new ImageRecordReader(height,width,channels,labelMaker);
        
        //Preprocessor
        DataNormalization scaler = new ImagePreProcessingScaler(0,1);
        
        //Load Train Dataset 
        recordReader.initialize(train);
        trainIter = new RecordReaderDataSetIterator(recordReader, batchSize,1,numClasses);                     
        scaler.fit(trainIter);     
        trainIter.setPreProcessor(scaler);
        //Load Test DataSet
        recordReader.reset();
        recordReader.initialize(test);
        testIter = new RecordReaderDataSetIterator(recordReader, batchSize,1,numClasses);
        scaler.fit(testIter);
        testIter.setPreProcessor(scaler);
        
    }
    public static void main(String[] args) throws Exception{
        System.out.print("Start Project! ");      
        // Load the saved model
        File locationToLoad = new File("C:\\Users\\DatGatto\\.deeplearning4j\\vgg.zip");
        ComputationGraph vgg16 = (ComputationGraph)ModelSerializer.restoreComputationGraph(locationToLoad);
        System.out.println(vgg16.summary());
        
        //Decide on a fine tune configuration.
        //*note: In cases where there already exists a setting the fine tune setting will
        //  override the setting for all layers that are not "frozen".
        FineTuneConfiguration fineTuneConf = new FineTuneConfiguration.Builder()
            .activation(Activation.LEAKYRELU)
            .weightInit(WeightInit.RELU)
            .learningRate(5e-5)
            .optimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT)
            .updater(Updater.NESTEROVS)
            .dropOut(0.5)
            .seed(rngseed)
            .build();
        
        //Restructure model
        //  -Frozen layers "block5_pool" and previous ones
        //  -Change "fc2" layer nOit
        //  -Add more layers
        ComputationGraph vgg16Transfer = new TransferLearning.GraphBuilder(vgg16)
           .fineTuneConfiguration(fineTuneConf)
           .setFeatureExtractor(featureExtractionLayer) //"block5_pool" and below are frozen
           .nOutReplace("fc2",1024, WeightInit.XAVIER) //modify nOut of the "fc2" vertex
           .removeVertexAndConnections("predictions") //remove the final vertex and it's connections
           .addLayer("fc3",new DenseLayer.Builder().activation(Activation.TANH).nIn(1024).nOut(256).build(),"fc2") 
           .addLayer("newpredictions",new OutputLayer
                .Builder(LossFunctions.LossFunction.NEGATIVELOGLIKELIHOOD)
                .activation(Activation.SOFTMAX)
                .nIn(256)
                .nOut(numClasses)
                .build(),"fc3") 
           .setOutputs("newpredictions") //specify new outputs for the graph to replace the deleted one
           .setInputTypes(InputType.convolutional(height, width,channels))
           .build();
        System.out.println("Model restructured.");
        
        //Dataset iterations
        loadDataSet();
        System.out.println("Data Loaded");
        
        //Train
        Evaluation eval;
        eval = vgg16Transfer.evaluate(testIter);
        System.out.println("Eval stats BEFORE fit.....");
        System.out.println(eval.stats());
        testIter.reset();
        int iter = 0;
        while(trainIter.hasNext()) {
            vgg16Transfer.fit(trainIter.next());
            if (iter % 10 == 0) {
                System.out.println("Evaluate model at iter "+iter +" ....");
                eval = vgg16Transfer.evaluate(testIter);
                System.out.println(eval.stats());
                testIter.reset();
            }
            iter++;
        }
        
        //Save the model
        File locationToSave = new File("C:\\Users\\DatGatto\\.deeplearning4j\\vggForDogVsCat.zip");
        boolean saveUpdater = false;
        ModelSerializer.writeModel(vgg16Transfer, locationToSave, saveUpdater);
    }
}

