import java.io.*;
import org.datavec.image.loader.NativeImageLoader;
import org.deeplearning4j.nn.api.*;
import org.deeplearning4j.nn.graph.ComputationGraph;
import org.deeplearning4j.nn.modelimport.keras.trainedmodels.TrainedModels;
import org.deeplearning4j.util.ModelSerializer;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.api.preprocessor.DataNormalization;
import org.nd4j.linalg.dataset.api.preprocessor.VGG16ImagePreProcessor;
import org.deeplearning4j.zoo.model.*;
import org.deeplearning4j.zoo.*;
import java.io.File;
import java.io.InputStream;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.StandardCopyOption;
/**
 *
 * @author DatGatto
 */
public class VGGTestOnCommandLine {
    public static void main(String[] args) {  
        System.out.print("Start Project! ");
        try{
            // Load the trained model
            File locationToSave = new File("C:\\Users\\DatGatto\\.deeplearning4j\\vgg.zip");
            ComputationGraph vgg16 = (ComputationGraph)ModelSerializer.restoreComputationGraph(locationToSave);
            //Set processor
            DataNormalization scaler = new VGG16ImagePreProcessor();
            NativeImageLoader loader = new NativeImageLoader(224, 224, 3);
            //Test
            String[] paths = {"C:\\Users\\DatGatto\\.deeplearning4j\\imgs\\dog1.jpg",
                          "C:\\Users\\DatGatto\\.deeplearning4j\\imgs\\dog2.jpg",
                          "C:\\Users\\DatGatto\\.deeplearning4j\\imgs\\cat1.jpg",
                          "C:\\Users\\DatGatto\\.deeplearning4j\\imgs\\cat2.jpg"};
            for (String i : paths){
                File file = new File(i);                
                try{
                    INDArray image = loader.asMatrix(file);                    
                    scaler.transform(image);
                    INDArray[] output = vgg16.output(false,image);
                    System.out.println(i+""+TrainedModels.VGG16.decodePredictions(output[0]));
                }
                catch (IOException ex){
                    System.err.println("Error at loading image."+i);
                }
            }     
        }        
        catch(IOException ex){
            System.out.println("Error at loading pretrained vgg model"+ex.getMessage());
        }      
    }
}
