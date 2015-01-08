package l90;

import java.io.*;
import java.util.*;

import org.apache.commons.io.FileUtils;

import weka.core.Attribute;
import weka.core.FastVector;
import weka.core.Instance;
import weka.core.Instances;
import weka.core.SparseInstance;
import edu.stanford.nlp.ling.CoreLabel;
import edu.stanford.nlp.process.CoreLabelTokenFactory;
import edu.stanford.nlp.process.PTBTokenizer;

public class FeatureExtractor {
	private String hashFile;  
	private ArrayList<String> featureWords;
	private FastVector attributeList;
	private FastVector sentimentClassList;
	private boolean useFrequency;
	
	@SuppressWarnings({ "rawtypes", "unchecked" })
	public FeatureExtractor(String dir, boolean useFrequency) {
		featureWords = new ArrayList<String>();
		attributeList = new FastVector();
		this.hashFile = "_featureWords.dat";
		this.useFrequency = useFrequency;
		
		sentimentClassList = new FastVector(2);
		sentimentClassList.addElement("pos");
		sentimentClassList.addElement("neg");
		
		Initialize(dir);
	}
	
	@SuppressWarnings({ "rawtypes", "resource" })
	private void Initialize(String dir) {
		if(new File(hashFile).isFile()) {
			Deserialise();
		} else {
			HashMap<String, Integer> map = new HashMap<String, Integer>();
			Iterator iter =  FileUtils.iterateFiles(new File(dir), new String[]{"txt"}, true);
			while(iter.hasNext()) {
			    File file = (File) iter.next();
			    
				try {
					Scanner scan = new Scanner(file);
					while (scan.hasNextLine()) {
				         String line = scan.nextLine();
				         for (String word : line.split(" ")) {
				        	 word = word.toLowerCase();
				        	 if(word.length() > 2) {
								if(!map.containsKey(word)) map.put(word, 1);
								else map.put(word, map.get(word) + 1);
				        	 }
				         }
				      }
				} catch (Exception e) {
					// TODO Auto-generated catch block
					e.printStackTrace();
				}
			}
			System.out.println(map);
			for (Map.Entry<String, Integer> entry : map.entrySet()) {
			    if(entry.getValue() > 4) featureWords.add(entry.getKey()); 
			}
		}
		
		Serialise();
	
		for(String featureWord : featureWords)
        {
            attributeList.addElement(new Attribute(featureWord));
        }
        //the last attribute reprsents ths CLASS (Sentiment) of the review
        attributeList.addElement(new Attribute("@class",sentimentClassList));
	}
	
    private void Serialise() {
    	try {
			FileOutputStream fos = new FileOutputStream(this.hashFile);
			ObjectOutputStream oos = new ObjectOutputStream(fos);
			oos.writeObject(this.featureWords);
			oos.close();
		} catch (IOException e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		}
    }
    
    @SuppressWarnings("unchecked")
	private void Deserialise() {
    	FileInputStream fis;
		try {
			fis = new FileInputStream(hashFile);
			ObjectInputStream ois = new ObjectInputStream(fis);
			featureWords = (ArrayList<String>) ois.readObject();
	        ois.close();
		} catch (Exception e) {
			e.printStackTrace();
		}
    }
    
    @SuppressWarnings("rawtypes")
	public Instances CreateInstances(String instanceName, String dir, String sentimentClass) {
        Instances instances = new Instances(instanceName,attributeList,0);
        instances.setClassIndex(instances.numAttributes()-1);   
        
        Iterator iter =  FileUtils.iterateFiles(new File(dir), new String[]{"txt"}, true);
		while(iter.hasNext()) {
		    File file = (File) iter.next();
            Instance currentFeatureVector = DocumentToVec(file.getAbsolutePath(), sentimentClass);
            currentFeatureVector.setDataset(instances);
            instances.add(currentFeatureVector);
		}
        
        return instances;
    }
    
		
	@SuppressWarnings({ "rawtypes" })
	public Instance DocumentToVec(String file, String sentimentClass) {
		Map<Integer,Double> featureMap = new TreeMap<Integer, Double>();		
		
		try {
			Scanner scan = new Scanner(new File(file));
			while (scan.hasNextLine()) {
		         String line = scan.nextLine();
		         for (String word : line.split(" ")) {
		        	 word = word.toLowerCase();
		        	 if(word.length() > 1 && featureWords.contains(word)) {
		        		 int idx = featureWords.indexOf(word);
		                 //adding 1.0 to the featureMap represents that the feature word is present in the input data
	 			
		        		 if(featureMap.containsKey(idx) && useFrequency) {
		 					featureMap.put(idx,featureMap.get(idx) + 1.0);
		 				} else featureMap.put(idx,1.0);
		        	 }
		         }
		      }
		} catch (Exception e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		}
				
		int indices[] = new int[featureMap.size()+1];
	    double values[] = new double[featureMap.size()+1];
	    int i=0;
	    for(Map.Entry<Integer,Double> entry : featureMap.entrySet())
	    {
	        indices[i] = entry.getKey();
	        values[i] = entry.getValue();
	        i++;
	    }
	    indices[i] = featureWords.size();
	    values[i] = (double)sentimentClassList.indexOf(sentimentClass);
		
	    SparseInstance instance = new SparseInstance(1.0,values,indices,featureWords.size());
	    return instance;
	}
}
