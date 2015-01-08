package l90;

import java.io.File;
import java.io.FileInputStream;
import java.io.FileNotFoundException;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.ObjectInputStream;
import java.io.ObjectOutputStream;
import java.util.HashMap;
import java.util.List;
import java.util.Properties;
import java.util.Scanner;

import edu.mit.jwi.item.POS;
import edu.stanford.nlp.ling.CoreLabel;
import edu.stanford.nlp.ling.CoreAnnotations.LemmaAnnotation;
import edu.stanford.nlp.ling.CoreAnnotations.PartOfSpeechAnnotation;
import edu.stanford.nlp.ling.CoreAnnotations.SentencesAnnotation;
import edu.stanford.nlp.ling.CoreAnnotations.TextAnnotation;
import edu.stanford.nlp.ling.CoreAnnotations.TokensAnnotation;
import edu.stanford.nlp.pipeline.Annotation;
import edu.stanford.nlp.pipeline.StanfordCoreNLP;
import edu.stanford.nlp.util.CoreMap;
import weka.core.Attribute;
import weka.core.Instances;
import weka.core.SelectedTag;
import weka.filters.Filter;
import weka.filters.unsupervised.attribute.Add;


public class SentiWordBuilder {
	private String db;
	private HashMap<String, Double> map;
	private HashMap<String, Integer> mapSynsets;
	private String serFile = "db.ser";
	private String serFileId = "dbId.ser";
	private Properties props = new Properties();
	private StanfordCoreNLP pipeline;
	
	public SentiWordBuilder(String db) throws IOException, ClassNotFoundException {
		this.db = db;
		map = new HashMap<String, Double>();
		mapSynsets = new HashMap<String, Integer>();
		File f = new File(serFile);
		if(f.exists() && !f.isDirectory()) {
			loadMap();
			return;
		}
		
		Scanner scan = new Scanner(new File(db));
		while (scan.hasNextLine()) {
			String[] words = scan.nextLine().split("\\s+");
			String pos = words[0];
			int synId = Integer.parseInt(words[1]);
			double negScore = Double.parseDouble(words[2]);
			double posScore = Double.parseDouble(words[3]);
			double score = posScore > negScore ? posScore : negScore - 2*negScore;
			String[] terms = words[4].split(" ");
			for(int i =0;i<terms.length;i++) {
				if(!terms[i].contains("#")) continue;
				String key = terms[i].split("#")[0] + "_" + pos;
				
				//add key to map
				if(!map.containsKey(key)) map.put(key, score);
				if(!mapSynsets.containsKey(key)) mapSynsets.put(key, synId);
			}
		}
		scan.close();
		saveMap();
	}
	
	private void saveMap() throws IOException {
		FileOutputStream fout = new FileOutputStream(serFile);
        ObjectOutputStream oos = new ObjectOutputStream(fout);
        oos.writeObject(map);
        fout = new FileOutputStream(serFileId);
        oos = new ObjectOutputStream(fout);
        oos.writeObject(map);
	}
	
	@SuppressWarnings("unchecked")
	private void loadMap() throws IOException, ClassNotFoundException {
		FileInputStream fin = new FileInputStream(serFile);
        ObjectInputStream oos = new ObjectInputStream(fin);
        map = (HashMap<String, Double>) oos.readObject();
        fin = new FileInputStream(serFileId);
        oos = new ObjectInputStream(fin);
        mapSynsets = (HashMap<String, Integer>) oos.readObject();
	}
        
	private String getPos(String pos) {
		if(pos.contains("NN")) return "n";
		else if(pos.contains("JJ")) return "a";
		else if(pos.contains("VB")) return "v";
		else if(pos.contains("RB")) return "r";
		else return pos;
	}
	
	public Instances useSentiWordVals(Instances vec, boolean useScore) throws Exception {
		Instances instances = new Instances(vec);
		
		for(int i=0;i<instances.numInstances(); i++) {
			for(int j=1; j<instances.numAttributes(); j++) {
				Attribute att = instances.instance(i).attribute(j);
				String name = att.name();
				double val = instances.instance(i).value(att);
				if(map.containsKey(name) && val != 0) {
					double sentVal = useScore == true ? map.get(name) : mapSynsets.get(name);
					instances.instance(i).setValue(att, 1.0 + sentVal);
				}
			}	
		}
		return instances;
	}
	
	public Instances getPosTagged(Instances data) {
		props.put("annotators", "tokenize,ssplit, pos, lemma");
		pipeline = new StanfordCoreNLP(props);
		Instances instances = new Instances(data);
		
		for(int i=0;i<instances.numInstances(); i++) {
			String text = instances.instance(i).stringValue(0);
			String tagged = getTagged(text);
			instances.instance(i).setValue(0, tagged);
		}
		return instances;
	}
	
	private String getTagged(String text){ 
		Annotation document = new Annotation(text);
		// run all Annotators on this text
		pipeline.annotate(document); 
		List<CoreMap> sentences = document.get(SentencesAnnotation.class);
		StringBuilder sb = new StringBuilder();
		
		for(CoreMap sentence: sentences) {
			for (CoreLabel token: sentence.get(TokensAnnotation.class)) {
				String word = token.get(TextAnnotation.class);
				word = word.replaceAll("[^a-zA-Z0-9\\s]","");
				String pos = token.get(PartOfSpeechAnnotation.class);
				pos = getPos(pos);
				sb.append(word + "_" + pos + " ");
			}
		}
		return sb.toString();
	}	
}
