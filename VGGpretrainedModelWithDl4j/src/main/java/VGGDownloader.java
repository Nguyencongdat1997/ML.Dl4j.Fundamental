import java.io.File;
import java.io.IOException;
import org.deeplearning4j.nn.graph.ComputationGraph;
import org.deeplearning4j.nn.api.*;
import org.deeplearning4j.util.ModelSerializer;
import org.deeplearning4j.zoo.PretrainedType;
import org.deeplearning4j.zoo.ZooModel;
import org.deeplearning4j.zoo.model.VGG16;
/**
 *
 * @author DatGatto
 */
public class VGGDownloader {
    public static void main(String[] args) {                  
        ZooModel zooModel = new VGG16();
        try{
            System.out.println("Start Download!");    
            Model x = zooModel.initPretrained(PretrainedType.IMAGENET);
            System.out.println("End download");
            ComputationGraph vgg16 = (ComputationGraph)x;            
            File locationToSave = new File("C:\\Users\\DatGatto\\.deeplearning4j\\vgg.zip");            
            ModelSerializer.writeModel(vgg16, locationToSave,true);
        }
        catch(IOException ex){
            System.out.println("Error at loading pretrained vgg model: "+ex.getMessage());
        }                
        
    }
}
